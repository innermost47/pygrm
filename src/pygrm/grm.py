# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Anthony Charretier
"""
Global Relational Model engine - log-space relational solver.

Nodes
    positive quantities (default) are handled as x = log10(value);
    signed quantities ({"scale": "lin"}) are handled as x = value (charges, curvature, w0...).

Relations (sigma in the relation's own residual unit; dex for log-space relations)
    log-linear (default)  sum(coeff * log10(node)) = rhs        products, ratios, powers
                          E = m c^2  ->  {"terms": {"E": 1, "m": -1, "c": -2}, "rhs": 0}
                          rhs may be a callable of the current values
    sum                   {"type": "sum", "terms": {node: coeff}, "pow": {node: p}, "K": k}
                          sum(coeff * value^p) = K              charges, fractions of the universe
    fn                    {"type": "fn", "nodes": [...], "f": callable(values) -> residual,
                           "grad": optional callable(values) -> {node: d residual / d value}}
                          any equation, written as f(values) = 0  (Koide, cosmological ages...)

What the solver returns for each node:
    values          best estimate
    sigma_dex       1-sigma uncertainty in log10 units (relative uncertainty ~ 2.303 * sigma_dex)
    undetermined    nodes that the relations and measurements do not fix (their value is only the guess)

Measurements are data with an uncertainty, not starting points:
    solve(measured_dict={"G": 6.674e-11, "H0": (70.0, 0.01)})   # value, or (value, sigma in dex)

Numerics: weighted least squares solved by QR factorisation (no normal equations), so relations
as tight as 1e-9 dex and values from 1e-44 to 1e122 coexist. Uses NumPy when it is installed,
pure Python otherwise. Purely log-linear problems are solved in a few exact steps; sums, functions
and callable rhs switch to Levenberg-Marquardt (analytic Jacobian for sums, finite differences
for functions).
"""

import math
import random

try:
    from . import units as _units
except ImportError:  # grm.py used on its own, next to units.py
    import units as _units

try:
    import numpy as _np
except ImportError:  # pragma: no cover - pure Python fallback
    _np = None

LN10 = math.log(10.0)


# ---------------------------------------------------------------------- dense linear algebra
def _qr_solve(A, b):
    """Least squares min |A d - b| for a full-column-rank A (m x n, m >= n).
    Returns (d, R) where R is the n x n upper triangular factor."""
    if _np is not None:
        A = _np.asarray(A, dtype=float)
        Q, R = _np.linalg.qr(A, mode="reduced")
        if _np.any(_np.diag(R) == 0.0):
            raise ArithmeticError("rank-deficient system")
        d = _np.linalg.solve(R, Q.T @ _np.asarray(b, dtype=float))
        return [float(v) for v in d], R
    m, n = len(A), len(A[0]) if A else 0
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for k in range(n):
        norm = math.sqrt(sum(M[i][k] ** 2 for i in range(k, m)))
        if norm == 0.0:
            raise ArithmeticError("rank-deficient system")
        alpha = -norm if M[k][k] >= 0 else norm
        v = [0.0] * m
        v[k] = M[k][k] - alpha
        for i in range(k + 1, m):
            v[i] = M[i][k]
        vv = sum(v[i] ** 2 for i in range(k, m))
        if vv == 0.0:
            continue
        for j in range(k, n + 1):
            s = sum(v[i] * M[i][j] for i in range(k, m)) * 2.0 / vv
            if s:
                for i in range(k, m):
                    M[i][j] -= s * v[i]
    R = [[M[i][j] if j >= i else 0.0 for j in range(n)] for i in range(n)]
    return _back_substitution(R, [M[i][n] for i in range(n)]), R


def _back_substitution(R, y):
    n = len(y)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = float(y[i])
        Ri = R[i]
        for j in range(i + 1, n):
            s -= float(Ri[j]) * x[j]
        if Ri[i] == 0.0:
            raise ArithmeticError("rank-deficient system")
        x[i] = s / float(Ri[i])
    return x


def _covariance_from_R(R):
    """(R^T R)^-1 = R^-1 R^-T."""
    if _np is not None:
        Rinv = _np.linalg.inv(_np.asarray(R, dtype=float))
        return (Rinv @ Rinv.T).tolist()
    n = len(R)
    Rinv = [[0.0] * n for _ in range(n)]
    for j in range(n):
        e = [1.0 if i == j else 0.0 for i in range(n)]
        col = _back_substitution(R, e)
        for i in range(n):
            Rinv[i][j] = col[i]
    return [
        [sum(Rinv[i][k] * Rinv[j][k] for k in range(n)) for j in range(n)]
        for i in range(n)
    ]


# ---------------------------------------------------------------------- network
RESIDUAL_FLOOR = 1e-9   # dex: a residual below this is rounding (solved values, float log10), not a difference
EXACT_SIGMA = 1e-6      # dex: an equation made of relations this precise is exact; within 2 of its sigmas, its
                        # residual is the precision its values were solved to, not a difference


def _merit(span, residual, sigma=None):
    """Span of scales over residual: merit = span / (1 + 1e7 * r), r = max(residual, RESIDUAL_FLOOR), and
    r = RESIDUAL_FLOOR when the equation is exact (sigma <= EXACT_SIGMA) and its residual within 2 sigma."""
    if sigma is not None and sigma <= EXACT_SIGMA and residual <= 2.0 * sigma:
        residual = 0.0
    return span / (1.0 + 1e7 * max(residual, RESIDUAL_FLOOR))


class GlobalRelationalNetwork:
    def __init__(
        self,
        nodes_spec=None,
        relations_spec=None,
        prior_sigma=1e3,
        undetermined_sigma=1.0,
        units="si",
    ):
        """
        nodes_spec      {name: {"guess": value, "fixed": bool, "scale": "log" | "lin", "min": val, "max": val,
                                "dim": [M, L, T, Q, K]}}   (dim: powers of mass, length, time, charge, temperature)
        relations_spec  list of relations (see module docstring); each may have "name" and "sigma"
                        (default 1e-3), log-linear ones may have "depends" (nodes a callable rhs reads)
        prior_sigma     width of a very weak ridge that keeps the problem solvable when some direction
                        is not determined by anything (it does not bias determined nodes)
        undetermined_sigma  a free node whose uncertainty exceeds this (dex, or value units for a
                        signed node, relative to max(1, |value|)) is reported as undetermined
        units           the system the network is SOLVED in: "si" (default), "atomic", "planck" or "ticks"
                        (see pygrm.units). Specs and measurements are always given in SI; with another
                        system they are converted (copies: the caller's dicts are left untouched) and the
                        values returned by solve() are numbers in that system. result["si"] gives them back
                        in SI. To only READ an SI result in another system, keep units="si" and call express().
        """
        nodes_spec = nodes_spec if nodes_spec is not None else {}
        relations_spec = relations_spec if relations_spec is not None else []
        if units not in _units.MODES:
            raise ValueError(f"units must be one of {', '.join(_units.MODES)}")
        self.units = units
        self._unit_net = None
        if units != "si":
            self._unit_net = _units.convert_network(nodes_spec, relations_spec, units)
            nodes_spec, relations_spec = self._unit_net.nodes, self._unit_net.relations
        self.nodes_names = list(nodes_spec)
        self.node_to_idx = {name: i for i, name in enumerate(self.nodes_names)}
        self.nodes_specs: dict = nodes_spec
        self.prior_sigma = float(prior_sigma)
        self.undetermined_sigma = float(undetermined_sigma)
        self.relations_spec = relations_spec
        self.discovered_relations = {
            "multivariate": {},
            "agnostic": {},
            "topological": {},
        }
        self.x0, self.lin, self.fixed_nodes = [], [], set()
        self.node_mins, self.node_maxs = [], []

        if self.nodes_specs:
            for name, spec in self.nodes_specs.items():
                scale = spec.get("scale", "log")
                if scale not in ("log", "lin"):
                    raise ValueError(f"node '{name}': scale must be 'log' or 'lin'")
                is_lin = scale == "lin"
                self.lin.append(is_lin)

                guess = spec.get("guess", 0.0 if is_lin else 1.0)
                self.x0.append(
                    float(guess)
                    if is_lin
                    else self._log10_checked(guess, f"guess of node '{name}'")
                )
                if spec.get("fixed", False):
                    self.fixed_nodes.add(name)

                # optional bounds of the node, in its own scale (a log node's bounds must be positive)
                raw_min, raw_max = spec.get("min"), spec.get("max")
                mn = None if raw_min is None else (
                    float(raw_min) if is_lin else self._log10_checked(raw_min, f"min of node '{name}'"))
                mx = None if raw_max is None else (
                    float(raw_max) if is_lin else self._log10_checked(raw_max, f"max of node '{name}'"))
                if mn is not None and mx is not None and mn > mx:
                    raise ValueError(f"node '{name}': min is above max")

                self.node_mins.append(mn)
                self.node_maxs.append(mx)

        # weak priors of the hidden nodes added by auto_expand (kept in the node specs, so they survive a rebuild
        # or an export to JSON)
        self.hidden_priors = {n: tuple(spec["hidden_prior"]) for n, spec in self.nodes_specs.items()
                              if "hidden_prior" in spec}
        self.relations = []

        if self.relations_spec:
            for k, rel in enumerate(self.relations_spec):
                name = rel.get("name", f"relation_{k}")
                kind = rel.get("type", "log")
                sigma = rel.get("sigma", 1e-3)
                if not callable(sigma) and not float(sigma) > 0:
                    raise ValueError(f"relation '{name}' needs a positive sigma")
                if kind == "fn":
                    nodes = rel.get("nodes")
                    if not nodes or not callable(rel.get("f")):
                        raise ValueError(
                            f"relation '{name}' (fn) needs 'nodes' and a callable 'f'"
                        )
                    self._check_names(name, nodes)
                    grad = rel.get("grad")
                    if grad is not None and not callable(grad):
                        raise ValueError(f"relation '{name}': 'grad' must be callable")
                    self.relations.append(
                        {
                            "name": name,
                            "type": "fn",
                            "f": rel["f"],
                            "sigma": sigma,
                            "grad": grad,
                            "nodes": [self.node_to_idx[n] for n in nodes],
                        }
                    )
                    continue
                terms = rel.get("terms")
                if not terms:
                    raise ValueError(f"relation '{name}' has no terms")
                self._check_names(name, terms)
                if kind == "sum":
                    pw = rel.get("pow", {})
                    self._check_names(name, pw)
                    self.relations.append(
                        {
                            "name": name,
                            "type": "sum",
                            "sigma": sigma,
                            "K": float(rel.get("K", 0.0)),
                            "terms": [
                                (self.node_to_idx[n], float(c), float(pw.get(n, 1.0)))
                                for n, c in terms.items()
                            ],
                        }
                    )
                    continue
                if kind != "log":
                    raise ValueError(f"relation '{name}': unknown type '{kind}'")
                signed = [n for n in terms if self.lin[self.node_to_idx[n]]]
                if signed:
                    raise ValueError(
                        f"relation '{name}' is log-linear but uses signed nodes {signed}: use type 'sum' or 'fn'"
                    )
                depends = rel.get("depends")
                if depends is not None:
                    self._check_names(name, depends)
                self.relations.append(
                    {
                        "name": name,
                        "type": "log",
                        "sigma": sigma,
                        "rhs": rel.get("rhs", 0.0),
                        "terms": [
                            (self.node_to_idx[n], float(c)) for n, c in terms.items()
                        ],
                        "depends": (
                            None
                            if depends is None
                            else [self.node_to_idx[n] for n in depends]
                        ),
                    }
                )
        self._nonlinear = any(
            r["type"] != "log" or callable(r["rhs"]) for r in self.relations
        )
        self._has_callables = self._nonlinear or any(
            callable(r["sigma"]) for r in self.relations
        )
        self.x = list(self.x0)  # last solution (solver space), kept for inspection

    # ------------------------------------------------------------------ tuning between solves
    def set_sigma(self, name, sigma):
        """Change the tolerance of a relation between two solves (smaller sigma = the relation weighs more).
        Used for continuation (a constraint tightened step by step) and for reweighting the relations
        that stay in tension."""
        if not (callable(sigma) or float(sigma) > 0):
            raise ValueError("sigma must be positive")
        if not hasattr(self, "_rel_index"):
            self._rel_index = {r["name"]: k for k, r in enumerate(self.relations)}
        if name not in self._rel_index:
            raise KeyError(f"unknown relation '{name}'")
        self.relations[self._rel_index[name]]["sigma"] = sigma
        self._has_callables = self._nonlinear or any(
            callable(r["sigma"]) for r in self.relations
        )

    def get_sigma(self, name):
        if not hasattr(self, "_rel_index"):
            self._rel_index = {r["name"]: k for k, r in enumerate(self.relations)}
        return self.relations[self._rel_index[name]]["sigma"]

    # ------------------------------------------------------------------ helpers
    def _check_names(self, rel_name, names):
        unknown = [n for n in names if n not in self.node_to_idx]
        if unknown:
            raise KeyError(f"relation '{rel_name}' uses unknown nodes: {unknown}")

    @staticmethod
    def _log10_checked(value, what):
        value = float(value)
        if not value > 0 or math.isinf(value):
            raise ValueError(
                f"{what} must be a finite positive number (log-space), got {value}"
            )
        return math.log10(value)

    def _clip(self, x):
        """x with every node kept inside its bounds (a copy; nodes without bounds unchanged)."""
        x = list(x)
        for i, (mn, mx) in enumerate(zip(self.node_mins, self.node_maxs)):
            if mn is not None and x[i] < mn:
                x[i] = mn
            if mx is not None and x[i] > mx:
                x[i] = mx
        return x

    def _to_value(self, i, xi):
        return xi if self.lin[i] else 10.0**xi

    def _values(self, x):
        return {
            name: self._to_value(i, x[i]) for i, name in enumerate(self.nodes_names)
        }

    @staticmethod
    def _eval(param, values):
        return float(param(values)) if callable(param) else float(param)

    def _residual(self, rel, x, values):
        if rel["type"] == "log":
            r = -self._eval(rel["rhs"], values)
            for i, c in rel["terms"]:
                r += c * x[i]
            return r
        if rel["type"] == "sum":
            r = -rel["K"]
            for i, c, p in rel["terms"]:
                r += c * self._to_value(i, x[i]) ** p
            return r
        return float(rel["f"](values))

    def _fd_step(self, i, xi):
        return 1e-6 * max(1.0, abs(xi)) if self.lin[i] else 1e-6

    def _derivative(self, rel, x, h, rhs_only, values, i):
        """Central difference on node i, falling back to a one-sided one at the edge of a domain.
        x and values are changed in place and restored (no copy: cheap on large networks).
        """
        name = self.nodes_names[i]
        x0 = x[i]
        v0 = values[name] if values is not None else None

        def g(xi):
            x[i] = xi
            try:
                if values is not None:
                    values[name] = self._to_value(i, xi)
                    vz = values
                else:
                    vz = self._values(x)
                if rhs_only:
                    return -self._eval(rel["rhs"], vz)
                return self._residual(rel, x, vz)
            except (ValueError, ZeroDivisionError, OverflowError, TypeError):
                return None

        try:
            fp, fm = g(x0 + h), g(x0 - h)
            f0 = None if (fp is not None and fm is not None) else g(x0)
        finally:
            x[i] = x0
            if values is not None:
                values[name] = v0
        if fp is not None and fm is not None:
            return (fp - fm) / (2 * h)
        if fp is not None and f0 is not None:
            return (fp - f0) / h
        if fm is not None and f0 is not None:
            return (f0 - fm) / h
        return 0.0

    def _rows(self, x, measurements, col, jacobian=True):
        """Residual rows: list of (entries [(column, derivative)], residual, weight), and the cost."""
        values = self._values(x) if self._has_callables else None
        rows, cost = [], 0.0
        for rel in self.relations:
            r = self._residual(rel, x, values)
            w = 1.0 / max(self._eval(rel["sigma"], values), 1e-300)
            cost += (w * r) ** 2
            entries = {}
            if jacobian:
                if rel["type"] == "log":
                    for i, c in rel["terms"]:
                        if i in col:
                            entries[col[i]] = entries.get(col[i], 0.0) + c
                    probe = (
                        (rel["depends"] if rel["depends"] is not None else list(col))
                        if callable(rel["rhs"])
                        else []
                    )
                    rhs_only = True
                elif rel["type"] == "sum":
                    for i, c, p in rel["terms"]:
                        if i in col:
                            v = self._to_value(i, x[i])
                            dv = 1.0 if self.lin[i] else v * LN10
                            entries[col[i]] = (
                                entries.get(col[i], 0.0) + c * p * v ** (p - 1) * dv
                            )
                    probe, rhs_only = [], False
                elif (
                    rel["grad"] is not None
                ):  # analytic gradient given with the relation
                    for name, dfdv in rel["grad"](values).items():
                        i = self.node_to_idx[name]
                        if i in col:
                            dv = 1.0 if self.lin[i] else values[name] * LN10
                            entries[col[i]] = entries.get(col[i], 0.0) + dfdv * dv
                    probe, rhs_only = [], False
                else:
                    probe, rhs_only = rel["nodes"], False
                for i in probe:
                    if i not in col:
                        continue
                    h = self._fd_step(i, x[i])
                    d = self._derivative(rel, x, h, rhs_only, values, i)
                    if d:
                        entries[col[i]] = entries.get(col[i], 0.0) + d
            rows.append((list(entries.items()), r, w))
        for i, (target, sigma) in measurements:
            r = x[i] - target
            w = 1.0 / sigma
            cost += (w * r) ** 2
            rows.append(([(col[i], 1.0)] if (jacobian and i in col) else [], r, w))
        return rows, cost

    def _cost(self, x, measurements):
        """Cost at x, or infinity when x is outside the domain of some relation (sqrt of a negative...)."""
        try:
            c = self._rows(x, measurements, {}, jacobian=False)[1]
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            return math.inf
        return c if math.isfinite(c) else math.inf

    def _step(self, rows, n, lam):
        """Solve the damped, ridge-regularised least-squares step. Returns (delta, R)."""
        if _np is not None:  # assemble directly into arrays (fast for large networks)
            live = [(e, r, w) for e, r, w in rows if e]
            m = len(live)
            A = _np.zeros((m + n * (2 if lam > 0 else 1), n))
            b = _np.zeros(A.shape[0])
            for k, (entries, r, w) in enumerate(live):
                for a, d in entries:
                    A[k, a] = w * d
                b[k] = -w * r
            idx = _np.arange(n)
            A[m + idx, idx] = 1.0 / self.prior_sigma
            if lam > 0:
                colnorm = _np.sqrt(_np.sum(A[:m] ** 2, axis=0))
                colnorm[colnorm == 0] = 1.0
                A[m + n + idx, idx] = math.sqrt(lam) * colnorm
            return _qr_solve(A, b)
        A, b = [], []
        colnorm = [0.0] * n
        for entries, r, w in rows:
            if not entries:
                continue
            row = [0.0] * n
            for a, d in entries:
                row[a] = w * d
                colnorm[a] += (w * d) ** 2
            A.append(row)
            b.append(-w * r)
        ridge = 1.0 / self.prior_sigma
        for a in range(n):
            row = [0.0] * n
            row[a] = ridge
            A.append(row)
            b.append(0.0)
        if lam > 0:
            for a in range(n):
                row = [0.0] * n
                row[a] = math.sqrt(lam) * (
                    math.sqrt(colnorm[a]) if colnorm[a] > 0 else 1.0
                )
                A.append(row)
                b.append(0.0)
        return _qr_solve(A, b)

    def solve(
        self,
        measured_dict=None,
        max_iterations=50,
        damping_lambda=1e-3,
        tolerance=1e-12,
        measurement_sigma=1e-4,
        compute_uncertainty=True,
        start=None,
        continuation=True,
    ):
        """
        measured_dict      {node: value} or {node: (value, sigma_dex)}
        max_iterations     upper bound for the nonlinear case (a linear problem needs 1 to 4 steps)
        damping_lambda     initial Levenberg-Marquardt damping (adapted during the solve)
        measurement_sigma  default sigma (dex) of a measurement given without its own
        start              optional {node: value} starting point (defaults to the guesses)
        continuation       if the direct solve does not converge and a sum or fn relation is very tight, solve again
                           with that relation loosened then tightened stage by stage, and keep the better end
        """
        measured_dict = dict(measured_dict or {})
        if self._unit_net is not None:
            measured_dict = self._unit_net.measured(measured_dict)
            start = self._unit_net.from_si(start) if start else start
        for name, prior in getattr(self, "hidden_priors", {}).items():
            if name in self.node_to_idx and name not in measured_dict:
                measured_dict[name] = prior
        measurements = []
        for name, m in measured_dict.items():
            if name not in self.node_to_idx:
                raise KeyError(f"unknown measured node '{name}'")
            value, sigma = m if isinstance(m, (tuple, list)) else (m, measurement_sigma)
            if not float(sigma) > 0:
                raise ValueError(f"measurement of '{name}' needs a positive sigma")
            i = self.node_to_idx[name]
            target = (
                float(value)
                if self.lin[i]
                else self._log10_checked(value, f"measurement of '{name}'")
            )
            measurements.append((i, (target, float(sigma))))

        x = list(self.x0)
        if start:
            for name, v in start.items():
                i = self.node_to_idx[name]
                x[i] = (
                    float(v)
                    if self.lin[i]
                    else self._log10_checked(v, f"start value of '{name}'")
                )
        x = self._clip(x)

        for i, (
            target,
            _,
        ) in (
            measurements
        ):  # a fixed node that is also measured takes the measured value
            if self.nodes_names[i] in self.fixed_nodes:
                x[i] = target
        x = self._clip(x)

        free = [
            i for i, name in enumerate(self.nodes_names) if name not in self.fixed_nodes
        ]
        col = {i: a for a, i in enumerate(free)}
        n = len(free)
        iterations, converged = 0, True
        R = None

        if n and not self._nonlinear:
            for iterations in range(1, 5):
                rows, _ = self._rows(x, measurements, col)
                delta, R = self._step(rows, n, 0.0)
                x_prev = list(x)
                for a, i in enumerate(free):
                    x[i] += delta[a]
                x = self._clip(x)
                if max(abs(x[i] - x_prev[i]) for i in free) < 1e-13:
                    break
        elif n:
            x = self._warm_start(x, measurements, free, col)
            x = self._clip(x)
            cost = self._cost(x, measurements)
            if not math.isfinite(cost):
                raise ValueError(
                    "the starting point is outside the domain of a relation "
                    "(e.g. the square root or arccos of an invalid value): change the guesses"
                )
            x_start = list(x)
            x, iterations, converged = self._descend(
                x, measurements, free, col, n, max_iterations, damping_lambda, tolerance
            )
            x = self._clip(x)
            if not converged and continuation:
                x, iterations, converged = self._continuation(
                    x_start,
                    x,
                    measurements,
                    free,
                    col,
                    n,
                    max_iterations,
                    damping_lambda,
                    tolerance,
                    iterations,
                )
                x = self._clip(x)

        self.x = x
        _, cost = self._rows(x, measurements, {}, jacobian=False)
        dof = len(self.relations) + len(measurements) - n
        result = {
            "values": self._values(x),
            "cost": cost,
            "iterations": iterations,
            "converged": converged,
            "dof": dof,
            "reduced_chi2": cost / dof if dof > 0 else None,
            "units": self.units,
        }
        if self._unit_net is not None:
            result["si"] = self._unit_net.to_si(result["values"])
        pulls = self._rows(x, [], {}, jacobian=False)[0]
        result["relation_pulls"] = {
            rel["name"]: r * w for rel, (_, r, w) in zip(self.relations, pulls)
        }
        # the solved residual and pull are also written back into the relation specs (the dicts given by the caller)
        pulls_dict = result["relation_pulls"]
        values_dict = self._values(x)
        for rel_spec, rel_internal in zip(self.relations_spec, self.relations):
            rel_spec["solved_residual"] = float(self._residual(rel_internal, x, values_dict))
            if rel_internal["name"] in pulls_dict:
                rel_spec["solved_pull"] = float(pulls_dict[rel_internal["name"]])

        if compute_uncertainty:
            sigma_dex = {name: 0.0 for name in self.nodes_names}
            covariance = {}
            if n:
                if R is None or self._nonlinear:
                    rows, _ = self._rows(x, measurements, col)
                    _, R = self._step(rows, n, 0.0)
                cov = _covariance_from_R(R)
                for a, i in enumerate(free):
                    sigma_dex[self.nodes_names[i]] = math.sqrt(max(cov[a][a], 0.0))
                    for b, j in enumerate(free):
                        covariance[(self.nodes_names[i], self.nodes_names[j])] = cov[a][
                            b
                        ]
            result["sigma_dex"] = sigma_dex
            result["uncertainty"] = {
                k: (s if self.lin[self.node_to_idx[k]] else LN10 * s)
                for k, s in sigma_dex.items()
            }
            result["covariance"] = covariance
            result["undetermined"] = sorted(
                k
                for k, s in sigma_dex.items()
                if s
                > self.undetermined_sigma
                * (
                    max(1.0, abs(x[self.node_to_idx[k]]))
                    if self.lin[self.node_to_idx[k]]
                    else 1.0
                )
            )
        for i, name in enumerate(self.nodes_names):
            if name in self.nodes_specs:
                val = x[i]
                resolved_val = val if self.lin[i] else 10.0**val
                self.nodes_specs[name]["solved"] = resolved_val
                if compute_uncertainty and name in sigma_dex:
                    self.nodes_specs[name]["sigma_dex"] = sigma_dex[name]
        return result

    def _descend(
        self, x, measurements, free, col, n, max_iterations, damping_lambda, tolerance
    ):
        """Gauss-Newton with backtracking, then Levenberg-Marquardt damping. Returns (x, iterations, converged)."""
        lam = damping_lambda
        converged = False
        iterations = 0
        cost = self._cost(x, measurements)
        self._step_sizes = []
        for iterations in range(1, max_iterations + 1):
            rows, cost = self._rows(x, measurements, col)
            accepted = False
            try:
                gn, _R = self._step(rows, n, 0.0)
            except ArithmeticError:
                gn = None
            if gn is not None:
                alpha = 1.0
                for _ in range(25):
                    delta = [alpha * d for d in gn]
                    x_new = list(x)
                    for a, i in enumerate(free):
                        x_new[i] += delta[a]
                    x_new = self._clip(x_new)
                    new_cost = self._cost(x_new, measurements)
                    if new_cost <= cost:
                        accepted = True
                        self._step_sizes.append(alpha)
                        break
                    alpha *= 0.5
            if not accepted:
                for _ in range(30):
                    try:
                        delta, _R = self._step(rows, n, lam)
                    except ArithmeticError:
                        lam *= 10.0
                        continue
                    x_new = list(x)
                    for a, i in enumerate(free):
                        x_new[i] += delta[a]
                    x_new = self._clip(x_new)
                    new_cost = self._cost(x_new, measurements)
                    if new_cost <= cost:
                        accepted = True
                        self._step_sizes.append(0.0)
                        break
                    lam *= 10.0
            if not accepted:
                converged = True
                break
            step = max(abs(x_new[i] - x[i]) for i in free)
            drop = cost - new_cost
            x, cost = x_new, new_cost
            lam = max(lam / 10.0, 1e-15)
            if step < 1e-12 or drop <= tolerance * max(1.0, cost):
                converged = True
                break
        return x, iterations, converged

    CONTINUATION_START = 1e-2
    CONTINUATION_FACTOR = 100.0

    def _continuation(
        self,
        x_start,
        x_direct,
        measurements,
        free,
        col,
        n,
        max_iterations,
        damping_lambda,
        tolerance,
        iterations_direct,
    ):
        tight = [
            r
            for r in self.relations
            if r["type"] != "log"
            and not callable(r["sigma"])
            and r["sigma"] < self.CONTINUATION_START
        ]
        if not tight:
            return x_direct, iterations_direct, False
        true_sigma = [r["sigma"] for r in tight]
        levels = [self.CONTINUATION_START]
        while levels[-1] > min(true_sigma):
            levels.append(levels[-1] / self.CONTINUATION_FACTOR)
        budget = max(5, max_iterations // len(levels))
        x = list(x_start)
        total, converged = iterations_direct, False
        try:
            for level in levels:
                for r, s in zip(tight, true_sigma):
                    r["sigma"] = max(s, level)
                if not math.isfinite(self._cost(x, measurements)):
                    break
                x, it, converged = self._descend(
                    x, measurements, free, col, n, budget, damping_lambda, tolerance
                )
                total += it
        finally:
            for r, s in zip(tight, true_sigma):
                r["sigma"] = s
        if converged and self._cost(x, measurements) <= self._cost(
            x_direct, measurements
        ):
            return x, total, True
        return x_direct, total, False

    def _warm_start(self, x, measurements, free, col):
        linear = [
            r for r in self.relations if r["type"] == "log" and not callable(r["rhs"])
        ]
        if not linear:
            return x
        saved = self.relations, self._has_callables
        self.relations = linear
        self._has_callables = any(callable(r["sigma"]) for r in linear)
        try:
            xw = list(x)
            for _ in range(3):
                rows, _ = self._rows(xw, measurements, col)
                delta, _R = self._step(rows, len(free), 0.0)
                for a, i in enumerate(free):
                    xw[i] += delta[a]
        except ArithmeticError:
            xw = None
        finally:
            self.relations, self._has_callables = saved
        if xw is None:
            return x
        cw, c0 = self._cost(xw, measurements), self._cost(x, measurements)
        return xw if (math.isfinite(cw) and (cw <= c0 or not math.isfinite(c0))) else x

    def _format_scientific_equation(self, terms: list, coefficients: list) -> str:
        """
        Formats discovered terms and coefficients into explicit log10 space notation.
        """
        formatted_parts = []
        for name, coeff in zip(terms, coefficients):
            if abs(coeff) >= 1e4 or (abs(coeff) < 1e-3 and coeff != 0):
                coef_str = f"{coeff:+.4e}"
            else:
                coef_str = f"{coeff:+.4f}"

            if name == "1":
                formatted_parts.append(f"{coef_str}")
            else:
                formatted_parts.append(f"{coef_str} \\cdot \\log_{10}({name})")

        return "f(x) = " + " ".join(formatted_parts)

    def express(self, result, mode):
        """The values of a solve() written in another system of units ("si", "atomic", "planck", "ticks").
        Works whatever system the network was solved in. Each node gets value, sigma_dex (the solve's and the
        units' own uncertainty together; for a signed node, the sigma in the new units) and unit_sigma_dex;
        in mode "ticks" also power k and ticks_log10 (the value is N ticks to the power k).
        Needs "dim" in the node specs (a node without it is a pure number and keeps its value)."""
        values = result.get("si") or result["values"]
        sig = dict(result.get("sigma_dex") or {})
        for n in sig:
            if self._unit_net is not None and self.lin[self.node_to_idx[n]]:
                sig[n] = self._unit_net._back(n, sig[n])     # a signed node's sigma is in value units
        return _units.express(values, self.nodes_specs, mode, sig)

    def leave_one_out(self, measured_dict, **solve_kwargs):
        default_sigma = solve_kwargs.get("measurement_sigma", 1e-4)
        report = {}
        for hidden, m in measured_dict.items():
            value, sigma_m = m if isinstance(m, (tuple, list)) else (m, default_sigma)
            rest = {k: v for k, v in measured_dict.items() if k != hidden}
            out = self.solve(measured_dict=rest, **solve_kwargs)
            pred, s = out["values"][hidden], out["sigma_dex"][hidden]
            d = (
                (pred - value)
                if self.lin[self.node_to_idx[hidden]]
                else math.log10(pred) - math.log10(value)
            )
            report[hidden] = {
                "measured": value,
                "predicted": pred,
                "sigma_dex": s,
                "pull": d / math.sqrt(s * s + sigma_m * sigma_m),
                "determined": hidden not in out["undetermined"],
            }
        return report

    def get_relation_tensions(self):
        values = self._values(self.x) if self._has_callables else None
        tensions = []

        for rel in self.relations:
            name = rel.get("name", "unnamed")
            r = self._residual(rel, self.x, values)
            sigma_val = self._eval(rel["sigma"], values)
            w = 1.0 / max(sigma_val, 1e-300)
            weighted_tension = abs(r * w)

            tensions.append(
                {
                    "name": name,
                    "type": rel.get("type"),
                    "residual": float(r),
                    "sigma": float(sigma_val),
                    "tension": float(weighted_tension),
                }
            )

        return sorted(tensions, key=lambda x: x["tension"], reverse=True)

    def _format_with_latex(
        self, terms: dict, rhs: float, display_mode: str = "latex_block", powers=None, kind="log"
    ) -> str:
        """A relation as text. kind "log": sum(c * log10(node)) = rhs; kind "sum": sum(c * node^p) = rhs."""
        if display_mode == "raw_terms":
            return terms
        powers = powers or {}
        if display_mode == "compact":
            eq_str = " + ".join([f"{c}·{t}" + (f"^{powers[t]}" if powers.get(t, 1) != 1 else "") for t, c in terms.items()])
            return f"{eq_str} = {rhs:.6g}"

        def name(t):
            return "\\text{" + str(t).replace("_", "\\_") + "}"

        latex_body = ""
        for k, (t, c) in enumerate(terms.items()):
            c = float(c)
            if kind == "sum":
                p = float(powers.get(t, 1.0))
                body = name(t) + ("" if p == 1.0 else f"^{{{p:g}}}")
            else:
                body = f"\\log_{{10}}\\left({name(t)}\\right)"
            coef = "" if abs(c) == 1.0 else f"{abs(c):g}\\,"
            sign = ("-" if c < 0 else "") if k == 0 else (" - " if c < 0 else " + ")
            latex_body += sign + coef + body
        # a log relation: sum(c * log10(node)) = rhs, i.e. the product of the powers is 10^rhs
        right = f"{rhs:.6g}"
        if display_mode == "latex_block":
            return f"{latex_body} = {right}"
        return f"$${latex_body} = {right}$$"

    def derive(self, equation, keep=(), max_steps=16):
        """The demonstration of `equation` ('S * Equant = Estock') from this network's product relations, step by
        step (pygrm.derive): d.found, d.exact, d.steps, d.text(lang="fr" | "en", fmt="markdown" | "latex",
        templates={...})."""
        from .derive import derive as _derive
        return _derive(self, equation, keep=keep, max_steps=max_steps)

    def global_equations(self, keep, tol=1e-9):
        """
        Equations the network holds between the nodes in `keep` only, with every other node eliminated.

        The log relations are linear in log10: A x = b (one row per relation, sigma per row). A combination y of
        relations with y^T A zero on every eliminated node is itself a relation between the kept nodes:
            sum_k (y^T A)_k log10(node_k) = y^T b,    sigma = sqrt(sum_i (y_i sigma_i)^2)
        It is implied by the network although no single relation writes it. Each independent combination gives
        one equation (none if the kept nodes are not linked by a chain of relations). sum and fn relations are
        not used (they are not linear in log10). The coefficients are scaled so the smallest one is 1 in size.
        Returns a list of {"terms", "rhs", "sigma", "value", "relations", "text"}; "value" is 10^rhs, the product
        of the kept nodes' powers, and "relations" the weight of each relation in the combination.
        """
        if _np is None:
            raise RuntimeError("global_equations needs NumPy")
        keep = list(keep)
        for n in keep:
            if n not in self.node_to_idx:
                raise KeyError(f"unknown node '{n}'")
        A, b, sig, names = self._log_system()
        if A is None:
            return []
        return self._equations_of(A, b, sig, names, keep, tol)

    def _log_system(self):
        """The log relations as a linear system in log10: A x = b, a sigma and a name per row (None if none)."""
        rows, rhs, sig, names = [], [], [], []
        values = None
        for rel in self.relations:
            if rel["type"] != "log":
                continue
            if callable(rel["rhs"]):
                continue
            row = _np.zeros(len(self.nodes_names))
            for i, c in rel["terms"]:
                row[i] += c
            rows.append(row)
            rhs.append(float(rel["rhs"]))
            sigma = rel["sigma"]
            if callable(sigma):
                values = values or self._values(self.x)
                sigma = self._eval(sigma, values)
            sig.append(float(sigma))
            names.append(rel["name"])
        if not rows:
            return None, None, None, None
        return _np.array(rows), _np.array(rhs), _np.array(sig), names

    def _equations_of(self, A, b, sig, names, keep, tol=1e-9):
        """global_equations on a system already built."""
        kept = [self.node_to_idx[n] for n in keep]
        elim = [i for i in range(A.shape[1]) if i not in kept and _np.any(A[:, i] != 0)]
        if elim:
            u, sv, vt = _np.linalg.svd(A[:, elim].T)
            rank = int(_np.sum(sv > tol * max(sv.max(), 1.0)))
            basis = vt[rank:]
        else:
            basis = _np.eye(A.shape[0])
        out = []
        for y in basis:
            coeffs = y @ A[:, kept]
            if _np.max(_np.abs(coeffs)) < tol:
                continue
            nonzero = _np.abs(coeffs)[_np.abs(coeffs) > 1e-6 * _np.max(_np.abs(coeffs))]
            scale = 1.0 / nonzero.min()
            if coeffs[_np.argmax(_np.abs(coeffs) > 1e-6 * _np.max(_np.abs(coeffs)))] < 0:
                scale = -scale
            coeffs, y_s = coeffs * scale, y * scale
            terms = {keep[k]: round(float(c), 9) for k, c in enumerate(coeffs) if abs(c) > 1e-9}
            right = float(y_s @ b)
            parts = []
            for n, c in terms.items():
                power = f"{c:g}"
                parts.append(n if power == "1" else f"{n}^{power}")
            out.append({
                "terms": terms,
                "rhs": right,
                "sigma": float(_np.sqrt(_np.sum((y_s * sig) ** 2))),
                "value": 10.0 ** right if abs(right) < 300 else None,   # None: beyond double precision
                # the relations it comes from: weights below 1e-4 of the largest are loops of the network added by
                # the decomposition (they cancel), not part of the chain
                "relations": {names[i]: round(float(w), 9) for i, w in enumerate(y_s)
                              if abs(w) > max(1e-9, 1e-4 * float(_np.max(_np.abs(y_s))))},
                "text": " · ".join(parts) + (f" = {10.0 ** right:.6g}" if abs(right) < 300 else f" = 10^{right:.6f}"),
            })
        return out

    def scale_classes(self, tol=1e-7):
        """The nodes that are the same scale: a pure-number multiple of each other, a = k * b with k a number the
        network implies (no constant in it, no measurement: its relations alone). vacRatio = (8/3) S is one.
        Returns {node: [the nodes of its class]} for the nodes in a class of two or more.

        a / b is implied by the log relations when e_a - e_b lies in the row space of A (x = log10 of the nodes);
        with P the projector on that row space, it is when the columns a and b of I - P are equal."""
        if _np is None:
            return {}
        A, _, _, _ = self._log_system()
        if A is None:
            return {}
        _, sv, vt = _np.linalg.svd(A, full_matrices=False)
        rank = int(_np.sum(sv > 1e-9 * max(sv.max(), 1.0)))
        V = vt[:rank]
        Q = _np.eye(A.shape[1]) - V.T @ V
        used = _np.any(A != 0, axis=0)
        out = {}
        for i in range(A.shape[1]):
            if not used[i] or self.nodes_names[i] in out:
                continue
            d = _np.max(_np.abs(Q - Q[:, [i]]), axis=0)
            same = [j for j in _np.nonzero(d < tol)[0] if used[j]]
            if len(same) > 1:
                group = [self.nodes_names[j] for j in same]
                for n in group:
                    out[n] = group
        return out

    def _scales(self, resolved_vals):
        """The scale of each positive node, in decades: its log10, or, for a node in a class of pure-number
        multiples, the mean log10 of its class (so that a node 8/3 of another does not stretch a span)."""
        own = {n: math.log10(v) for n, v in resolved_vals.items()
               if n in self.node_to_idx and isinstance(v, (int, float)) and v > 0}
        scale = dict(own)
        for n, group in self.scale_classes().items():
            logs = [own[m] for m in group if m in own]
            if n in own and logs:
                scale[n] = sum(logs) / len(logs)
        return scale

    def explain_hidden_nodes(self, result, tolerance_dex=0.02, max_power=3, decoys=(0.1372, -0.2113, 0.0731),
                             hops=3):
        """
        What the hidden nodes added by auto_expand turned out to be, after a solve (result = solve()).

        For each hidden node: its value and uncertainty, whether it is really needed (how many sigmas it stands
        from "nothing missing"), and what it could be:
          - a simple number (2, 1/2, pi, 2 pi, 4 pi...);
          - a power of one existing node, or a product of powers of two, of the same dimension when the nodes
            carry "dim" (powers -max_power..max_power, halves included),
        within the node's own uncertainty (at least tolerance_dex). Many combinations are tried, so the same search
        is run against decoy values (shifted by amounts that are not simple numbers): "chance" is how many
        matches chance alone gives. A candidate is worth a look when chance is well below 1.
        Only the nodes within `hops` relations of the tense relation are used (None: the whole network), so a
        missing factor in an atomic relation is not "explained" by a guitar string or a vowel. Nodes equal to 1
        (to 1e-3 dex) are left out: every power of them is the same. A factor close to 1 is also compared with
        (1 + x)^k for the small dimensionless nodes x around it (a correction such as a reduced mass).
        Signed (lin) hidden nodes are reported with their value only.
        """
        simple = {"2": 2.0, "3": 3.0, "4": 4.0, "sqrt 2": math.sqrt(2), "sqrt 3": math.sqrt(3), "pi": math.pi,
                  "2 pi": 2 * math.pi, "4 pi": 4 * math.pi, "8 pi": 8 * math.pi, "pi^2": math.pi ** 2,
                  "sqrt pi": math.sqrt(math.pi), "sqrt(2 pi)": math.sqrt(2 * math.pi), "e": math.e}
        numbers = [(0.0, "1")]
        for label, x in simple.items():
            numbers += [(math.log10(x), label), (-math.log10(x), f"1/({label})" if " " in label else f"1/{label}")]
        values = result["values"]
        sig = result.get("sigma_dex", {})
        hidden = [n for n in self.nodes_names if self.nodes_specs.get(n, {}).get("hidden_for")]
        known = [n for n in self.nodes_names if n not in hidden and not self.lin[self.node_to_idx[n]]
                 and values.get(n, 0) > 0]
        known = [n for n in known if abs(math.log10(values[n])) > 1e-3]
        powers = sorted({p / 2 for p in range(-2 * max_power, 2 * max_power + 1)} - {0.0}, key=abs)
        relations_of = {}
        for rel in self.relations_spec:
            for n in rel.get("terms", {}) or rel.get("nodes", []):
                relations_of.setdefault(n, []).append(rel)

        def around(relation_name):
            if hops is None:
                return set(known)
            rel = next((r for r in self.relations_spec if r.get("name") == relation_name), None)
            if rel is None:
                return set(known)
            reach = set(rel.get("terms", {}) or rel.get("nodes", []))
            frontier = set(reach)
            for _ in range(hops - 1):
                nxt = set()
                for n in frontier:
                    for r in relations_of.get(n, []):
                        nxt |= set(r.get("terms", {}) or r.get("nodes", []))
                frontier = nxt - reach
                reach |= nxt
            return reach & set(known)

        def dim_of(n):
            d = self.nodes_specs.get(n, {}).get("dim")
            return None if d is None else [float(v) for v in d]

        report = []
        for h in hidden:
            v = values[h]
            entry = {"node": h, "relation": self.nodes_specs[h]["hidden_for"], "value": v}
            if self.lin[self.node_to_idx[h]]:
                entry["sigma"] = sig.get(h)
                entry["needed_sigmas"] = abs(v) / sig[h] if sig.get(h) else None
                report.append(entry)
                continue
            lh, sh = math.log10(v), sig.get(h, 0.0)
            entry["sigma_dex"] = sh
            entry["needed_sigmas"] = abs(lh) / sh if sh > 0 else None
            # a factor close to 1 needs a tolerance finer than its own distance to 1
            tol = max(min(tolerance_dex, 0.1 * abs(lh)), 2 * sh)
            dh = dim_of(h)
            candidates, tried, chance = [], 0, 0

            def consider(log_value, label, dims):
                nonlocal tried, chance
                if dh is not None and dims is not None and any(abs(a - b) > 1e-9 for a, b in zip(dims, dh)):
                    return
                tried += 1
                if abs(log_value - lh) < tol:
                    candidates.append({"form": label, "error_dex": log_value - lh})
                chance += sum(abs(log_value - (lh + off)) < tol for off in decoys)

            zero = [0.0] * len(dh) if dh is not None else None
            own = set(next((r.get("terms", {}) for r in self.relations_spec if r.get("name") == entry["relation"]), {}))
            near = [n for n in known if n in around(entry["relation"]) and n not in own]   # not its own relation
            for lv, label in numbers:
                consider(lv, label, zero)
            if abs(lh) < 0.05:          # close to 1: a small correction (1 + x)^k
                for a in near:
                    x, da = values[a], dim_of(a)
                    if x < 0.1 and (da is None or all(abs(v) < 1e-9 for v in da)):
                        for k in (1, -1, 2, -2, 0.5, -0.5, 3, -3):
                            consider(k * math.log10(1 + x), f"(1 + {a})^{k:g}", zero)
            for a in near:
                la, da = math.log10(values[a]), dim_of(a)
                for pa in powers:
                    consider(pa * la, f"{a}^{pa:g}", None if da is None else [pa * x for x in da])
            for i, a in enumerate(near):
                la, da = math.log10(values[a]), dim_of(a)
                for b in near[i + 1:]:
                    lb, db = math.log10(values[b]), dim_of(b)
                    for pa in powers:
                        for pb in powers:
                            dims = None if da is None or db is None else [pa * x + pb * y for x, y in zip(da, db)]
                            consider(pa * la + pb * lb, f"{a}^{pa:g} {b}^{pb:g}", dims)
            candidates.sort(key=lambda c: (c["form"].count("^"), abs(c["error_dex"])))
            entry.update({"candidates": candidates[:10], "combinations_tried": tried,
                          "chance": chance / len(decoys)})
            report.append(entry)
        return report

    def extract_master_invariant(
        self, resolved_vals: dict, display_mode="latex_block", unify=True, max_pairs=60, max_triples=200
    ) -> dict:
        """
        The master invariant of the network: the equation that ties together the most distant scales with the
        smallest residual on the solved values (merit = span of scales in decades / (1 + 1e7 * max(residual, 1e-9))).

        Candidates:
          - the relations given (log and sum relations);
          - with unify=True, the unifying equations the network implies without writing them: for the pairs of
            positive nodes farthest apart in scale (up to max_pairs), every other node is eliminated
            (global_equations) except the constants (nodes declared "fixed", or with "constant": True in their
            spec). What is left links the two ends of the network through all the relations in between.
          - with max_triples > 0, the same for triples of nodes: the pairs farthest apart, each with every third node
            linked to them (at most max_triples triples), the constants kept and, separately, eliminated too (an
            equation where every constant cancels, such as S * Equant = Estock, has three nodes and no constant).
        A residual below RESIDUAL_FLOOR (1e-9 dex) is rounding, not a difference: all such residuals count as equal.
        Among equal merits, the equation with fewer terms wins. The result says whether the invariant was derived
        ("derived": True, with the relations it combines) or given.
        """
        candidates = []
        scale = self._scales(resolved_vals) if unify and _np is not None else {}

        for rel in self.relations_spec:
            terms = rel.get("terms", {})
            if not terms or len(terms) < 2 or rel.get("hidden"):
                continue

            rel_type = rel.get("type", "log")
            if rel_type == "fn":
                continue
            if rel_type == "sum":
                rhs = float(rel.get("K", 0.0))
            else:
                if callable(rel.get("rhs", 0.0)):
                    continue                      # a computed right-hand side has no fixed value to show
                rhs = float(rel.get("rhs", 0.0))
            powers = rel.get("pow", {})

            valid = True
            node_logs = []

            if rel_type == "sum":
                sum_val = 0.0
                for node_name, coeff in terms.items():
                    if node_name not in resolved_vals:
                        valid = False
                        break
                    v = resolved_vals[node_name]
                    sum_val += float(coeff) * v ** float(powers.get(node_name, 1.0))
                    node_logs.append(math.log10(max(abs(v), 1e-300)))

                if not valid:
                    continue
                residual = abs(sum_val - rhs)
                span = max(node_logs) - min(node_logs) if node_logs else 0.0
                merit = _merit(span, residual)

            else:
                lhs = 0.0
                for node_name, coeff in terms.items():
                    if node_name not in resolved_vals:
                        valid = False
                        break
                    v = resolved_vals[node_name]
                    if v <= 0:
                        valid = False
                        break
                    l_val = math.log10(v)
                    node_logs.append(scale.get(node_name, l_val))
                    lhs += float(coeff) * l_val

                if not valid or len(node_logs) < 2:
                    continue

                residual = abs(lhs - rhs)
                span = max(node_logs) - min(node_logs)
                sig = rel.get("sigma")
                merit = _merit(span, residual, sig if isinstance(sig, (int, float)) else None)

            candidates.append(
                {
                    "name": rel.get("name", "rel"),
                    "terms": terms,
                    "rhs": rhs,
                    "type": rel_type,
                    "pow": powers,
                    "residual": residual,
                    "span": span,
                    "merit": merit,
                }
            )

        if not candidates:
            return {
                "title": "Global Equilibrium Invariant",
                "equation_text": "No closed relation found",
                "ratio_text": "1.00000000",
                "solver_logs": ["No candidates found."],
            }

        if unify and _np is not None:
            candidates += self._unifying_candidates(resolved_vals, max_pairs, max_triples, scale)
        if not candidates:
            return {
                "title": "Global Equilibrium Invariant",
                "equation_text": "No closed relation found",
                "ratio_text": "1.00000000",
                "solver_logs": ["No candidates found."],
            }
        # equal merits: fewer terms, then smaller powers, then a right side closer to 1 (the simpler equation)
        candidates.sort(key=lambda c: (-round(c["merit"], 9), len(c["terms"]),
                                       round(sum(abs(float(x)) for x in c["terms"].values()), 6),
                                       abs(float(c["rhs"]))))
        best = candidates[0]

        dynamic_latex = self._format_with_latex(
            best["terms"], best["rhs"], display_mode=display_mode, powers=best["pow"], kind=best["type"]
        )

        model_signature = f"master_invariant_{best['name']}"
        if "topological" not in self.discovered_relations:
            self.discovered_relations["topological"] = {}

        self.discovered_relations["topological"][model_signature] = {
            "equation": dynamic_latex,
            "terms": best["terms"],
            "rhs": best["rhs"],
            "residual": best["residual"],
            "span": best["span"],
            "merit": best["merit"],
            "derived": best.get("derived", False),
            "relations": best.get("relations"),
        }

        return self.discovered_relations["topological"][model_signature]

    def _unifying_candidates(self, resolved_vals, max_pairs, max_triples=0, scale=None):
        """Equations implied by the log relations between the pairs of nodes farthest apart in scale and, with
        max_triples > 0, between triples of nodes (the farthest pairs, each with a third node linked to them).
        scale: {node: decades} (_scales); nodes that are pure-number multiples of each other are one scale."""
        constants = [n for n in self.nodes_names
                     if n in self.fixed_nodes or self.nodes_specs.get(n, {}).get("constant")]
        positive = [n for n in self.nodes_names
                    if not self.lin[self.node_to_idx[n]] and n not in constants
                    and resolved_vals.get(n, 0) > 0 and not self.nodes_specs.get(n, {}).get("hidden_for")]
        scale = dict(scale or {})
        for n in positive:
            scale.setdefault(n, math.log10(resolved_vals[n]))
        A, b, sig, names = self._log_system()
        if A is None:
            return []
        ranked = sorted(((a, c) for i, a in enumerate(positive) for c in positive[i + 1:]),
                        key=lambda ab: -abs(scale[ab[0]] - scale[ab[1]]))
        out, seen = [], {}

        def take(eq, name, needed):
            if not all(n in eq["terms"] for n in needed):
                return
            vals = [resolved_vals.get(n, 0.0) for n in eq["terms"]]
            if any(v <= 0 for v in vals):
                return
            node_logs = [math.log10(v) for v in vals]
            residual = abs(sum(c * l for c, l in zip(eq["terms"].values(), node_logs)) - eq["rhs"])
            spans = [scale.get(n, l) for n, l in zip(eq["terms"], node_logs)]
            span = max(spans) - min(spans)
            sigma = eq.get("sigma")
            cand = {"name": name, "terms": eq["terms"], "rhs": eq["rhs"], "type": "log", "pow": {},
                    "residual": residual, "span": span, "merit": _merit(span, residual, sigma),
                    "derived": True, "relations": eq["relations"], "sigma": sigma}
            # the same equation from another combination of relations (a network in tension gives it with
            # slightly different right sides): the one that holds best is kept
            key = tuple(sorted(eq["terms"].items()))
            if key in seen:
                k = seen[key]
                if residual < out[k]["residual"]:
                    out[k] = cand
                return
            seen[key] = len(out)
            out.append(cand)

        for a, c in ranked[:max_pairs]:
            for eq in self._equations_of(A, b, sig, names, [a, c] + constants):
                take(eq, f"unified_{a}_{c}", (a, c))
        if max_triples > 0 and len(positive) >= 3:
            # the triples: only nodes linked by a chain of log relations that does not pass through a constant
            # (through a constant everything would be linked)
            group = {n: n for n in positive}

            def root(n):
                while group[n] != n:
                    group[n] = group[group[n]]
                    n = group[n]
                return n
            for rel in self.relations:
                if rel["type"] != "log":
                    continue
                linked = [self.nodes_names[i] for i, _ in rel["terms"] if self.nodes_names[i] in group]
                for n in linked[1:]:
                    group[root(n)] = root(linked[0])
            # the farthest pairs (all those as far as the last one kept), and the third nodes, the most extreme
            # scales first; a third node is tried with every pair before the next one is
            per_pair = max(1, len(positive) - 2)
            n_pairs = max(1, max_triples // per_pair)
            if n_pairs < len(ranked):
                last = abs(scale[ranked[n_pairs - 1][0]] - scale[ranked[n_pairs - 1][1]])
                while n_pairs < len(ranked) and \
                        abs(scale[ranked[n_pairs][0]] - scale[ranked[n_pairs][1]]) >= last - 1e-9:
                    n_pairs += 1
            pairs = [(a, c) for a, c in ranked[:n_pairs] if root(a) == root(c)]
            thirds = sorted(positive, key=lambda n: -abs(scale[n]))
            tried = set()
            for third in thirds:
                for a, c in pairs:
                    if third in (a, c) or root(third) != root(a):
                        continue
                    trio = frozenset((a, c, third))
                    if trio in tried:
                        continue
                    tried.add(trio)
                    for keep in ([a, c, third], [a, c, third] + constants):
                        for eq in self._equations_of(A, b, sig, names, keep):
                            take(eq, f"unified_{a}_{c}_{third}", (a, c, third))
                    if len(tried) >= max_triples:
                        return out
        return out

    def build_grm_table(
        self, input_data: dict = None, display_mode="latex_mode"
    ) -> dict:
        input_data = input_data or {}

        nodes_spec = input_data.get("nodes_spec", input_data.get("nodes", {}))
        relations_spec = input_data.get("relations_spec", [])

        if not nodes_spec and self.nodes_specs:
            nodes_spec = self.nodes_specs
        if (
            not relations_spec
            and hasattr(self, "relations_spec")
            and self.relations_spec
        ):
            relations_spec = self.relations_spec

        # the measurements are what tie the network to the data: without them every unmeasured node drifts freely
        measured = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in input_data.get("measured", {}).items()
        }
        print(f"Solving Full System: {len(nodes_spec)} nodes, {len(measured)} measurements via GRM...")
        grm_instance = GlobalRelationalNetwork(nodes_spec, relations_spec)
        sol = grm_instance.solve(measured_dict=measured)
        resolved_vals = sol["values"]
        state = "converged" if sol.get("converged", True) else "NOT converged"
        print(f"✓ Solve {state} (cost: {sol['cost']:.4g})")

        grm_instance.relations_spec = relations_spec
        master_eq_data = grm_instance.extract_master_invariant(
            resolved_vals, display_mode=display_mode
        )

        report_lines = [
            "=== GRM SYSTEM ANALYSIS REPORT ===",
            f"Total resolved nodes: {len(resolved_vals)}",
            f"Global residual tension: {sol['cost']:.2e}",
            f"Discovered master invariant signature: {master_eq_data.get('equation', 'N/A')}",
            "",
            "--- MAIN RESOLVED VALUES ---",
        ]

        for name, val in list(resolved_vals.items())[:15]:
            report_lines.append(f" - {name}: {val:.6e}")

        # where the tension sits: the measurements and relations that are pulled hardest
        pulls = []
        for name, (value, sigma) in measured.items():
            if name not in resolved_vals or not sigma > 0:
                continue
            if grm_instance.lin[grm_instance.node_to_idx[name]]:
                pulls.append((name, "measured", (resolved_vals[name] - value) / sigma))
            elif value > 0 and resolved_vals[name] > 0:
                pulls.append((name, "measured", (math.log10(resolved_vals[name]) - math.log10(value)) / sigma))
        for name, p in (sol.get("relation_pulls") or {}).items():
            pulls.append((name, "relation", p))
        pulls.sort(key=lambda t: -abs(t[2]))
        if pulls:
            report_lines += ["", "--- LARGEST PULLS (sigmas) ---"]
            for name, kind, p in pulls[:15]:
                report_lines.append(f" - {name:22s} {kind:9s} {p:+.2f}")

        report = {
            "resolved_values": resolved_vals,
            "master_equation": master_eq_data,
            "system_cost": sol["cost"],
            "text_report": "\n".join(report_lines),
        }

        return report

    def _solved_values(self, measured_dict=None):
        """The values of the last solve (solving now if the network was never solved, or if measurements are given)."""
        if measured_dict is not None or not all("solved" in self.nodes_specs.get(n, {}) for n in self.nodes_names):
            return self.solve(measured_dict=measured_dict, compute_uncertainty=False)["values"]
        return {n: self.nodes_specs[n]["solved"] for n in self.nodes_names}

    # constants used when the network has no closed log relation to take the invariant from
    BASE_EXP_S, EXP_EQUANT, EXP_ESTOCK = 122.0, -53.0, 70.0

    def evaluate_grm_master_invariant(self, scale_shift_s: float = 0.0, measured_dict=None) -> dict:
        """
        The master invariant of this network under a scale shift.

        The invariant is the closed log relation the network holds best across the widest span of scales
        (extract_master_invariant on the solved values): sum(c_i * log10(node_i)) = rhs. Its left side is evaluated
        on the solved nodes and shifted by scale_shift_s decades; the residual tension is |lhs - rhs|, the torsion
        rhs - lhs, the duality ratio 10^(lhs - rhs). Without such a relation, the fixed constants are used:
        lhs = BASE_EXP_S + s + EXP_EQUANT, rhs = EXP_ESTOCK.
        """
        source, lhs_base, rhs_exponent = "constants", self.BASE_EXP_S + self.EXP_EQUANT, self.EXP_ESTOCK
        equation = None
        if self.relations and self.nodes_names:
            values = self._solved_values(measured_dict)
            best = self._best_log_invariant(values)
            if best is not None:
                lhs_base = sum(float(c) * math.log10(values[n]) for n, c in best["terms"].items())
                rhs_exponent = best["rhs"]
                equation = best["name"]
                source = "network"
        lhs_exponent = lhs_base + scale_shift_s
        residual_tension = abs(lhs_exponent - rhs_exponent)
        torsion_factor = rhs_exponent - lhs_exponent
        if residual_tension < 1e-12:
            status = "Zero Tension (Perfect Equilibrium)"
        elif torsion_factor > 0:
            status = f"Active Positive Tension (Compression / Under-capacity by {torsion_factor:+.3g} dec)"
        else:
            status = f"Active Negative Tension (Expansion / Over-capacity by {torsion_factor:+.3g} dec)"
        return {
            "source": source,
            "invariant": equation,
            "scale_shift_s": scale_shift_s,
            "lhs_exponent": lhs_exponent,
            "rhs_exponent": rhs_exponent,
            "residual_tension": residual_tension,
            "torsion_factor": torsion_factor,
            "duality_ratio": 10.0 ** max(min(lhs_exponent - rhs_exponent, 300.0), -300.0),
            "status": status,
        }

    def _best_log_invariant(self, values):
        """The closed log relation (constant right side, 2 terms or more, positive nodes) with the best merit:
        widest span of scales for the smallest residual."""
        best = None
        for k, rel in enumerate(self.relations_spec):
            if rel.get("type", "log") != "log" or callable(rel.get("rhs", 0.0)):
                continue
            terms = rel.get("terms", {})
            if len(terms) < 2 or any(n not in values or values[n] <= 0 for n in terms):
                continue
            logs = [math.log10(values[n]) for n in terms]
            residual = abs(sum(float(c) * l for c, l in zip(terms.values(), logs)) - float(rel.get("rhs", 0.0)))
            merit = (max(logs) - min(logs)) / (1.0 + 1e7 * residual)
            if best is None or merit > best["merit"]:
                best = {"name": rel.get("name", f"relation_{k}"), "terms": terms, "rhs": float(rel.get("rhs", 0.0)),
                        "merit": merit}
        return best

    def evaluate_dual_invariant(self, scale_shift_s: float, measured_dict=None) -> dict:
        """
        How the master invariant answers a scale shift: the residual it takes at shift s (the dual residual) and
        the correction factor k = dual residual / |s| (1 when the invariant holds exactly at s = 0, so that every
        decade of shift is a decade of tension; other values when the network already holds some tension).
        """
        dual_residual = self.evaluate_grm_master_invariant(scale_shift_s, measured_dict)["residual_tension"]
        s_abs = abs(scale_shift_s)
        return {
            "scale_shift_s": scale_shift_s,
            "correction_factor_k": dual_residual / s_abs if s_abs > 0 else None,
            "dual_residual": dual_residual,
        }

    def fit_dual_invariant(
        self,
        empirical_scales,
        node_guess=0.0,
        node_scale="lin",
        node_min=0.0,
        node_max=2.0,
        relation_sigma=1e-2,
        prior_sigma=1e3,
    ):
        """
        Fits one node k_dual so that k_dual * |s| matches the residual tension of this network's master invariant
        (evaluate_grm_master_invariant) at the given scale shifts, with a GRM sub-network.
        """
        native_residuals = [
            self.evaluate_grm_master_invariant(scale_shift_s=s)["residual_tension"]
            for s in empirical_scales
        ]

        nodes_spec = {
            "k_dual": {
                "guess": node_guess,
                "scale": node_scale,
                "min": node_min,
                "max": node_max,
            }
        }

        relations_spec = []
        for idx, (s_val, res_target) in enumerate(
            zip(empirical_scales, native_residuals)
        ):
            s_abs = abs(s_val)

            def make_dual_relation(target, magnitude):
                def f(values):
                    k = values["k_dual"]
                    return target - (k * magnitude)

                return f

            relations_spec.append(
                {
                    "name": f"native_dual_fit_{idx}",
                    "type": "fn",
                    "nodes": ["k_dual"],
                    "f": make_dual_relation(res_target, s_abs),
                    "sigma": relation_sigma,
                }
            )

        sub_network = GlobalRelationalNetwork(
            nodes_spec=nodes_spec,
            relations_spec=relations_spec,
            prior_sigma=prior_sigma,
        )

        result = sub_network.solve()
        optimal_k = result["values"]["k_dual"]

        return {
            "optimal_k": optimal_k,
            "converged": result["converged"],
            "reduced_chi2": result["reduced_chi2"],
            "status": "Agnostic native node-and-relation solver successfully optimized the dual invariant",
        }

    @staticmethod
    def _forward_select(design, y, max_terms, min_gain=1e-6):
        """Greedy forward selection: adds, one at a time, the candidate term that lowers the residual the most
        (columns compared after scaling to unit norm, so a term is not favoured for its units), and stops when a new
        term no longer explains at least min_gain of the variance. Returns (indices, weights, predictions)."""
        if _np is None:
            raise RuntimeError("equation discovery needs NumPy")
        design = _np.asarray(design, dtype=float)
        y = _np.asarray(y, dtype=float)
        norms = _np.linalg.norm(design, axis=0)
        usable = [j for j in range(design.shape[1]) if norms[j] > 0 and _np.all(_np.isfinite(design[:, j]))]
        total = float(_np.sum((y - y.mean()) ** 2)) or float(_np.sum(y ** 2)) or 1.0
        chosen, weights = [], _np.array([])
        predictions = _np.zeros_like(y)
        best_ss = float(_np.sum(y ** 2))
        for _ in range(min(max_terms, len(usable))):
            best = None
            for j in usable:
                if j in chosen:
                    continue
                cols = chosen + [j]
                sub = design[:, cols] / norms[cols]
                w, *_ = _np.linalg.lstsq(sub, y, rcond=None)
                ss = float(_np.sum((y - sub @ w) ** 2))
                if best is None or ss < best[0]:
                    best = (ss, j)
            if best is None or (best_ss - best[0]) < min_gain * total:
                break
            best_ss = best[0]
            chosen.append(best[1])
        if chosen:
            sub = design[:, chosen]
            weights, *_ = _np.linalg.lstsq(sub, y, rcond=None)
            predictions = sub @ weights
        return chosen, weights, predictions

    def discover_equation_autonomously(
        self, x_data, y_data, max_terms: int = 4
    ) -> dict:
        """
        Autonomously discovers the underlying analytical equation linking x and y
        by constructing and evaluating a rich dictionary of non-linear candidate functions,
        retaining only the most statistically significant terms (Sparse Regression / LASSO-like approach).
        """
        x = _np.array(x_data, dtype=float)
        y = _np.array(y_data, dtype=float)

        if len(x) != len(y) or len(x) == 0:
            raise ValueError(
                "Input data arrays must be non-empty and of identical length."
            )

        dictionary_builders = [
            lambda val: _np.ones_like(val),
            lambda val: val,
            lambda val: val**2,
            lambda val: val**3,
            lambda val: val**4,
            lambda val: _np.log10(_np.maximum(val, 1e-9)),
            lambda val: _np.sqrt(_np.maximum(val, 0.0)),
            lambda val: _np.exp(-val),
            lambda val: _np.sin(val),
        ]

        symbol_names = [
            "1",
            "x",
            "x^2",
            "x^3",
            "x^4",
            "log10(x)",
            "sqrt(x)",
            "exp(-x)",
            "sin(x)",
        ]

        design_matrix = _np.column_stack(
            [builder(x) for builder in dictionary_builders]
        )

        selected_indices, optimized_weights, predictions = self._forward_select(design_matrix, y, max_terms)
        selected_names = [symbol_names[i] for i in selected_indices]

        ss_residual = _np.sum((y - predictions) ** 2)
        ss_total = _np.sum((y - _np.mean(y)) ** 2)
        r_squared = 1.0 - (ss_residual / ss_total) if ss_total > 0 else 0.0

        console_terms = []
        latex_terms = []

        for name, weight in zip(selected_names, optimized_weights):
            if abs(weight) >= 1e4 or (abs(weight) < 1e-3 and weight != 0):
                coef_str = f"{weight:+.4e}"
            else:
                coef_str = f"{weight:+.4f}"

            if name == "1":
                console_terms.append(f"{coef_str}")
                latex_terms.append(f"{coef_str}")
            else:
                console_terms.append(f"{coef_str}*{name}")
                latex_terms.append(f"{coef_str} \\cdot {name}")

        console_eq = (
            "f(x) = " + " ".join(console_terms) if console_terms else "f(x) = 0"
        )
        latex_eq = (
            "$$f(x) = " + " ".join(latex_terms) + "$$"
            if latex_terms
            else "$$f(x) = 0$$"
        )

        discovered_equation_str = f"{console_eq} | LaTeX: {latex_eq}"

        model_signature = f"autonomous_discovery_{len(selected_indices)}_terms"
        self.discovered_relations["agnostic"][model_signature] = {
            "equation": discovered_equation_str,
            "terms": selected_names,
            "coefficients": optimized_weights.tolist(),
            "r_squared": r_squared,
            "sample_count": len(x),
        }

        return self.discovered_relations["agnostic"][model_signature]

    def discover_multivariate_equation_autonomously(
        self, X_data, y_data, max_terms: int = 4
    ) -> dict:
        """
        Autonomously discovers multi-variable analytical relations
        from a matrix of independent features (e.g., a and b for Pythagoras).
        """
        X = _np.array(X_data, dtype=float)
        y = _np.array(y_data, dtype=float)

        if X.ndim == 1:
            X = X[:, _np.newaxis]
        if X.shape[1] > 2:
            raise ValueError("discover_multivariate_equation_autonomously uses two features (a, b); got more")

        n_samples, n_features = X.shape

        if n_samples != len(y) or n_samples == 0:
            raise ValueError(
                "Input data arrays must be non-empty and matching in size."
            )

        a = X[:, 0]
        b = X[:, 1] if n_features > 1 else _np.zeros_like(a)

        dictionary_builders = [
            lambda a_val, b_val: _np.ones_like(a_val),
            lambda a_val, b_val: a_val,
            lambda a_val, b_val: b_val,
            lambda a_val, b_val: a_val**2,
            lambda a_val, b_val: b_val**2,
            lambda a_val, b_val: a_val * b_val,
            lambda a_val, b_val: _np.sqrt(_np.maximum(a_val**2 + b_val**2, 0.0)),
            lambda a_val, b_val: _np.log10(_np.maximum(a_val, 1e-9)),
            lambda a_val, b_val: _np.log10(_np.maximum(b_val, 1e-9)),
        ]

        symbol_names = [
            "1",
            "a",
            "b",
            "a^2",
            "b^2",
            "a*b",
            "sqrt(a^2 + b^2)",
            "log10(a)",
            "log10(b)",
        ]

        design_matrix = _np.column_stack(
            [builder(a, b) for builder in dictionary_builders]
        )

        selected_indices, optimized_weights, predictions = self._forward_select(design_matrix, y, max_terms)
        selected_names = [symbol_names[i] for i in selected_indices]

        ss_residual = _np.sum((y - predictions) ** 2)
        ss_total = _np.sum((y - _np.mean(y)) ** 2)
        r_squared = 1.0 - (ss_residual / ss_total) if ss_total > 0 else 0.0

        console_terms = []
        latex_terms = []

        for name, weight in zip(selected_names, optimized_weights):
            if abs(weight) >= 1e4 or (abs(weight) < 1e-3 and weight != 0):
                coef_str = f"{weight:+.4e}"
            else:
                coef_str = f"{weight:+.4f}"

            if name == "1":
                console_terms.append(f"{coef_str}")
                latex_terms.append(f"{coef_str}")
            else:
                console_terms.append(f"{coef_str}*{name}")
                latex_terms.append(f"{coef_str} \\cdot {name}")

        console_eq = (
            "c(a, b) = " + " ".join(console_terms) if console_terms else "c(a, b) = 0"
        )
        latex_eq = (
            "$$c(a, b) = " + " ".join(latex_terms) + "$$"
            if latex_terms
            else "$$c(a, b) = 0$$"
        )

        discovered_equation_str = f"{console_eq} | LaTeX: {latex_eq}"

        model_signature = f"multivariate_discovery_{len(selected_indices)}_terms"
        self.discovered_relations["multivariate"][model_signature] = {
            "equation": discovered_equation_str,
            "terms": selected_names,
            "coefficients": optimized_weights.tolist(),
            "r_squared": r_squared,
            "sample_count": len(y),
        }

        return self.discovered_relations["multivariate"][model_signature]

    def auto_expand(self, threshold_tension=3.0, max_additions=2, seed=None, measured_dict=None,
                    hidden_sigma=1.0):
        """
        Expands the network where it is in tension: a relation the solved network cannot satisfy (pull above
        threshold_tension, in sigmas) is missing something. For each of the most tense relations (at most
        max_additions), a hidden node is added inside it:
            log relation   sum(c_i log10(n_i)) + log10(h) = rhs     h: a missing factor (dimension completing the
                                                                    relation when the nodes carry "dim")
            sum relation   sum(c_i n_i^p_i) + h = K                 h: a missing term
        The hidden node starts at "nothing missing" (1 for a factor, 0 for a term) with a weak prior of hidden_sigma
        (dex for a factor, value units for a term), so it moves only as far as the data demand. After the next
        solve, its value says what is missing and where: it is a quantity to look for, not a result.
        fn relations are left as they are. The network is rebuilt; returns the number of nodes added.
        """
        rng = random.Random(seed)
        self.solve(measured_dict=measured_dict, compute_uncertainty=False)
        tense = [t for t in self.get_relation_tensions() if t["tension"] > threshold_tension]
        new_nodes = {n: dict(spec) for n, spec in self.nodes_specs.items()}
        new_relations = [dict(r) for r in self.relations_spec]
        index = {r.get("name", f"relation_{k}"): k for k, r in enumerate(new_relations)}
        added = 0
        for t in tense:
            if added >= max_additions:
                break
            k = index.get(t["name"])
            if k is None:
                continue
            rel = new_relations[k]
            kind = rel.get("type", "log")
            if kind == "fn" or rel.get("hidden"):
                continue
            hidden = f"hidden_{t['name']}_{rng.randint(0, 99999):05d}"
            while hidden in new_nodes:
                hidden = f"hidden_{t['name']}_{rng.randint(0, 99999):05d}"
            spec = {"guess": 1.0 if kind == "log" else 0.0, "scale": "log" if kind == "log" else "lin",
                    "hidden_for": t["name"]}
            dims = [new_nodes[n].get("dim") for n in rel["terms"]]
            if all(dims):
                if kind == "log":     # the factor carries what the relation lacks to be dimensionless
                    raw = [-sum(float(c) * d[i] for c, d in zip(rel["terms"].values(), dims))
                           for i in range(len(dims[0]))]
                    spec["dim"] = [int(v) if float(v).is_integer() else v for v in raw]
                    spec["dim"] = [0 if v == 0 else v for v in spec["dim"]]
                else:                 # a term has the dimension of the terms it is added to
                    spec["dim"] = list(dims[0])
            new_nodes[hidden] = spec
            rel["terms"] = dict(rel["terms"], **{hidden: 1.0})
            rel["hidden"] = hidden
            spec["hidden_prior"] = [spec["guess"], hidden_sigma]
            added += 1
        if added:
            discovered = self.discovered_relations
            self.__init__(new_nodes, new_relations, self.prior_sigma, self.undetermined_sigma)
            self.discovered_relations = discovered
        return added

    def mine_bridges(self, tolerance_dex=0.005, min_scale_distance=0.0, max_hits=128):
        """Searches combinations of 2 or 3 positive nodes (small integer powers) that are dimensionless and whose
        value is close to a simple number (1, 2, pi...). Many combinations are tried, so some match by chance: the
        same search is run against decoy targets (the same numbers shifted by amounts that are not simple numbers),
        and self.last_bridge_search reports how many combinations were tried and how many decoy hits came out.
        A real bridge should stand out from that chance level; every hit carries "chance_hits" for comparison."""
        CANONICAL_COMBOS = [
            (1, -1, 0),
            (1, -2, 0),
            (2, -1, 0),
            (1, -3, 0),
            (3, -1, 0),
            (1, 1, -1),
            (1, -1, 1),
            (-1, 1, 1),
            (1, -1, -1),
            (-1, 1, -1),
            (-1, -1, 1),
            (1, -1, -2),
            (1, -2, -1),
            (-2, 1, -1),
            (-1, 1, -2),
            (-1, -2, 1),
            (-2, -1, 1),
            (2, -1, -1),
            (-1, 2, -1),
            (-1, -1, 2),
            (1, 2, -1),
            (1, -1, 2),
            (-1, 1, 2),
            (2, 1, -1),
        ]

        MATHEMATICAL_TARGETS = [
            (0.0, "1.0"),
            (0.301029995, "2.0"),
            (-0.301029995, "1/2"),
            (0.602059991, "4.0"),
            (-0.602059991, "1/4"),
            (0.49714987, "π"),
            (-0.49714987, "1/π"),
            (0.798179868, "2π"),
            (-0.798179868, "1/(2π)"),
            (1.099209864, "4π"),
            (-1.099209864, "1/(4π)"),
            (-0.176091259, "2/3"),
        ]

        nodes = self.nodes_specs
        node_names = self.nodes_names
        count = len(node_names)
        discovered = []
        seen_signatures = set()
        decoy_offsets = (0.1372, -0.2113, 0.0731)      # dex: shifts that land on no simple number
        tested, decoy_hits = 0, 0

        node_cache = []
        for name in node_names:
            spec = nodes.get(name, {})
            val = spec.get("solved", spec.get("guess", 1.0))
            is_lin = spec.get("scale", "log") == "lin"
            log_val = 0.0 if is_lin else math.log10(max(abs(val), 1e-120))
            dim = spec.get("dim", [0, 0, 0, 0, 0])
            node_cache.append(
                {
                    "name": name,
                    "logVal": log_val,
                    "scale": 0.0 if is_lin else 1.0,
                    "dim": dim,
                }
            )

        for i in range(count):
            for j in range(i + 1, count):
                for k in range(j, count):
                    if len(discovered) >= max_hits:
                        break

                    nodeA = node_cache[i]
                    nodeB = node_cache[j]
                    nodeC = node_cache[k]

                    if (
                        nodeA["scale"] < 0.5
                        or nodeB["scale"] < 0.5
                        or (k != j and nodeC["scale"] < 0.5)
                    ):
                        continue

                    for wA, wB, wC in CANONICAL_COMBOS:
                        if wC == 0 and k != j:
                            continue

                        d0 = (
                            wA * nodeA["dim"][0]
                            + wB * nodeB["dim"][0]
                            + (wC * nodeC["dim"][0] if wC != 0 else 0)
                        )
                        d1 = (
                            wA * nodeA["dim"][1]
                            + wB * nodeB["dim"][1]
                            + (wC * nodeC["dim"][1] if wC != 0 else 0)
                        )
                        d2 = (
                            wA * nodeA["dim"][2]
                            + wB * nodeB["dim"][2]
                            + (wC * nodeC["dim"][2] if wC != 0 else 0)
                        )
                        d3 = (
                            wA * nodeA["dim"][3]
                            + wB * nodeB["dim"][3]
                            + (wC * nodeC["dim"][3] if wC != 0 else 0)
                        )
                        d4 = (
                            wA * nodeA["dim"][4]
                            + wB * nodeB["dim"][4]
                            + (wC * nodeC["dim"][4] if wC != 0 else 0)
                        )

                        if d0 != 0 or d1 != 0 or d2 != 0 or d3 != 0 or d4 != 0:
                            continue

                        max_dex = max(nodeA["logVal"], nodeB["logVal"])
                        min_dex = min(nodeA["logVal"], nodeB["logVal"])
                        if wC != 0:
                            max_dex = max(max_dex, nodeC["logVal"])
                            min_dex = min(min_dex, nodeC["logVal"])
                        if (max_dex - min_dex) < min_scale_distance:
                            continue

                        combo_log = (
                            wA * nodeA["logVal"]
                            + wB * nodeB["logVal"]
                            + (wC * nodeC["logVal"] if wC != 0 else 0)
                        )
                        tested += 1
                        for off in decoy_offsets:
                            if any(abs(combo_log - (t + off)) < tolerance_dex for t, _ in MATHEMATICAL_TARGETS):
                                decoy_hits += 1

                        for target_val, k_display in MATHEMATICAL_TARGETS:
                            res = abs(combo_log - target_val)
                            if res < tolerance_dex:
                                terms = {nodeA["name"]: wA, nodeB["name"]: wB}
                                if wC != 0:
                                    terms[nodeC["name"]] = (
                                        terms.get(nodeC["name"], 0) + wC
                                    )
                                terms = {k: v for k, v in terms.items() if v != 0}

                                active_keys = sorted(terms.keys())
                                if len(active_keys) < 2:
                                    continue

                                sig_key = "|".join(
                                    [f"{k}:{terms[k]}" for k in active_keys]
                                )
                                if sig_key in seen_signatures:
                                    continue
                                seen_signatures.add(sig_key)

                                numerators = []
                                for k, p in terms.items():
                                    if p > 0:
                                        numerators.append(
                                            k if p == 1 else f"{k}^{int(p)}"
                                        )
                                    else:
                                        numerators.append(
                                            f"{k}^(-1)" if p == -1 else f"{k}^{int(p)}"
                                        )

                                formula_str = f"{' × '.join(numerators)} = {k_display}"

                                discovered.append(
                                    {
                                        "name": f"invariant_{'_'.join(active_keys)}",
                                        "type": "log",
                                        "terms": terms,
                                        "rhs": target_val,
                                        "sigma": max(res, 0.005),
                                        "residualDex": res,
                                        "formula": formula_str,
                                    }
                                )
                                break

        chance = decoy_hits / len(decoy_offsets)
        self.last_bridge_search = {"combinations_tested": tested, "hits": len(discovered),
                                   "chance_hits": chance}
        for d in discovered:
            d["chance_hits"] = chance
        return discovered
