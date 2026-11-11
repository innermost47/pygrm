# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Anthony Charretier
"""What a network can find by itself in what it has seen: new relations, and nodes nobody mentioned.

    from pygrm import discover
    rows = [{"t": 1.0, "h": 4.9}, {"t": 2.0, "h": 19.6}, ...]
    discover.find_relations(rows)          # [{"target": "h", "terms": {"t": 2.0}, "constant": 4.9, ...}]
    discover.hidden_node(residuals)        # {"needed": True, "value": ..., "sigma": ..., "t": ...}
    discover.find_dynamics(rows)           # x(t) = a x(t-1) + b x(t-2) + c: growth, decay, oscillations
    discover.hidden_dynamics(residuals)    # a missing node that changes from moment to moment

find_relations looks, among the quantities seen together row by row, for the relations a GRM network writes:
  - power laws between positive quantities: y = K * x^p (* z^q), with p and q simple numbers (1, 2, 1/2, -1,
    3/2...), K a constant: a straight relation in log;
  - straight lines between any quantities: y = a * x + b.
A relation is kept only if it holds on the rows (r^2 at least 0.999 by default) and if its powers, made simple, still
hold as well as the free fit (a relation with p = 1.93 is not h = K t^2 unless 2 fits nearly as well). Among the
relations of the same quantities, the simplest that holds wins (a score that pays for every term: the Bayesian
information criterion). The constant comes with its uncertainty, so a network can start from it and go on learning.

hidden_node says whether the residuals of a relation (how far it was from holding, row after row, in its sigmas)
show something missing: a residual that keeps the same sign, far beyond what chance gives (|mean| / (sd / sqrt(n))
above 3, and the mean beyond 1 sigma). It also refuses to invent one when nothing is missing.

Pure Python: no NumPy needed.
"""
import itertools
import math

SIMPLE_POWERS = (1.0, 2.0, 3.0, 0.5, 1.5, 1.0 / 3.0, 2.0 / 3.0, -1.0, -2.0, -3.0, -0.5, -1.5, -1.0 / 3.0, -2.0 / 3.0)


def _fit(ys, xs):
    """Least squares ys ~ c + sum_j b_j xs[j]. Returns (coefficients [c, b1, ...], rmse). Pure Python normal
    equations (the systems here have 2 or 3 unknowns, centred)."""
    n, k = len(ys), len(xs)
    my = sum(ys) / n
    mx = [sum(x) / n for x in xs]
    X = [[x[i] - mx[j] for j, x in enumerate(xs)] for i in range(n)]
    Y = [y - my for y in ys]
    A = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    r = [sum(X[i][a] * Y[i] for i in range(n)) for a in range(k)]
    b = _solve(A, r)
    if b is None:
        return None, math.inf
    c = my - sum(bj * mj for bj, mj in zip(b, mx))
    res = [ys[i] - c - sum(b[j] * xs[j][i] for j in range(k)) for i in range(n)]
    return [c] + b, math.sqrt(sum(e * e for e in res) / n)


def _solve(A, r):
    n = len(A)
    M = [row[:] + [r[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda i: abs(M[i][c]))
        if abs(M[p][c]) < 1e-300:
            return None
        M[c], M[p] = M[p], M[c]
        for i in range(n):
            if i != c:
                f = M[i][c] / M[c][c]
                for j in range(c, n + 1):
                    M[i][j] -= f * M[c][j]
    return [M[i][n] / M[i][i] for i in range(n)]


def _fixed(ys, xs, powers):
    """The constant and rmse of ys ~ c + sum p_j xs[j] with the powers given."""
    n = len(ys)
    res = [ys[i] - sum(p * x[i] for p, x in zip(powers, xs)) for i in range(n)]
    c = sum(res) / n
    rmse = math.sqrt(sum((e - c) ** 2 for e in res) / n)
    return c, rmse


def _r2(ys, rmse):
    n = len(ys)
    m = sum(ys) / n
    var = sum((y - m) ** 2 for y in ys) / n
    return 1.0 - rmse ** 2 / var if var > 0 else 0.0


def _bic(n, rmse, k):
    return n * math.log(max(rmse, 1e-12) ** 2) + k * math.log(n)


def _simplest(p):
    return min(SIMPLE_POWERS, key=lambda s: abs(s - p))


def _fmt_power(p):
    for den in (1, 2, 3):
        for num in range(-6, 7):
            if num and abs(p - num / den) < 1e-9:
                return str(num) if den == 1 else f"{num}/{den}"
    return f"{p:g}"


def _columns(rows):
    names = sorted({k for r in rows for k in r})
    cols = {}
    for k in names:
        vals = [r.get(k) for r in rows]
        if all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
            cols[k] = [float(v) for v in vals]
    return cols


def find_relations(rows, max_terms=2, min_rows=8, r2_min=0.999, powers_slack=2.0, exclude=(), sums=True):
    """The relations that hold among the quantities of `rows` (a list of {name: value}, one per moment seen).

    max_terms: 1 (pairs) or 2 (a quantity from two others). Returns a list, the best first, of
    {"kind", "target", "terms", "constant", "sigma", "r2", "rows", "constant_sigma", "text"}:
      "power"        y = K * prod x^p             terms {x: p}, constant K, sigma the scatter in dex
      "linear"       y = a x + b                  terms {x: a}, constant b, sigma in the unit of y
      "exponential"  y = A * e^(k x)              terms {x: k}, constant A (y positive, x any), sigma in dex
      "log"          y = a ln(x) + b              terms {x: a}, constant b (x positive), sigma in the unit of y
      "sum"          y = sum_j a_j m_j + b        terms [(a_j, {x: p})] (monomials m_j of positive quantities, or a
                                                  signed quantity), constant b: a sum built on products
    The simplest kind that holds is kept for each quantity, in this order (a sum only when nothing simpler holds).
    exclude: sets of names already tied together (no relation is proposed among them). sums: look for sums."""
    cols = _columns(rows)
    n = min((len(v) for v in cols.values()), default=0)
    if n < min_rows:
        return []
    moving = {k: v for k, v in cols.items() if max(v) - min(v) > 1e-12 * max(1.0, max(abs(x) for x in v))}
    positive = {k for k, v in moving.items() if min(v) > 0}
    logs = {k: [math.log10(x) for x in moving[k]] for k in positive}
    excluded = [set(e) for e in exclude]
    found = []

    def allowed(names):
        return not any(set(names) <= e for e in excluded)

    for y in sorted(moving):
        best = None
        # power laws: log y = log K + sum p log x
        if y in positive:
            for k in range(1, max_terms + 1):
                for xs in itertools.combinations(sorted(positive - {y}), k):
                    if not allowed((y,) + xs):
                        continue
                    coef, free = _fit(logs[y], [logs[x] for x in xs])
                    if coef is None:
                        continue
                    ps = [_simplest(p) for p in coef[1:]]
                    c, rmse = _fixed(logs[y], [logs[x] for x in xs], ps)
                    if rmse > powers_slack * free + 1e-9 or _r2(logs[y], rmse) < r2_min:
                        continue
                    score = _bic(n, rmse, k + 1)
                    if best is None or score < best[0] - 2.0 * math.log(n) * (k - len(best[1]["terms"])):
                        terms = dict(zip(xs, ps))
                        best = (score, {
                            "kind": "power", "target": y, "terms": terms, "constant": 10 ** c,
                            "constant_sigma": rmse / math.sqrt(n), "sigma": rmse, "r2": _r2(logs[y], rmse), "rows": n,
                            "text": f"{y} = {10 ** c:.6g} * " + " * ".join(
                                f"{x}^{_fmt_power(p)}" if p != 1 else x for x, p in terms.items()),
                        })
        # straight lines: y = a x + b
        for x in sorted(set(moving) - {y}):
            if not allowed((y, x)):
                continue
            coef, rmse = _fit(moving[y], [moving[x]])
            if coef is None or _r2(moving[y], rmse) < r2_min:
                continue
            score = _bic(n, rmse, 2) + (0.0 if best is None or best[1]["kind"] == "linear" else 1e9)
            # in log and in value the scores are not comparable: a power law found is kept; a line is for what is
            # not a power law (signed quantities, or a line that is not through zero)
            if best is None or (best[1]["kind"] == "linear" and score < best[0]):
                a, b = coef[1], coef[0]
                sx = math.sqrt(sum((v - sum(moving[x]) / n) ** 2 for v in moving[x]))
                best = (score, {
                    "kind": "linear", "target": y, "terms": {x: a}, "constant": b,
                    "constant_sigma": rmse / math.sqrt(n), "slope_sigma": rmse / sx if sx > 0 else math.inf,
                    "sigma": rmse, "r2": _r2(moving[y], rmse), "rows": n,
                    "text": f"{y} = {a:.6g} * {x} {'+' if b >= 0 else '-'} {abs(b):.6g}",
                })
        # exponentials and logarithms: a straight line between a quantity and the log of another
        if best is None:
            best = _semilog(y, moving, positive, logs, n, r2_min, allowed)
        # sums built on products (E = a m v^2 + b m h): only when nothing simpler holds
        if best is None and sums:
            best = _sum_of_products(y, moving, positive, logs, n, r2_min, allowed)
        if best:
            found.append(best)
    # one relation per set of quantities (y from x and x from y are one), written the simplest way: whole powers
    # rather than roots, positive powers rather than negative (F = m a rather than a = F m^-1)
    def plainness(rel):
        ps = list(rel["terms"].values()) if rel["kind"] == "power" else [1.0]
        if rel["kind"] == "sum":
            ps = [1.0] * len(rel["terms"])
        return (sum(abs(p - round(p)) > 1e-9 for p in ps), sum(p < 0 for p in ps), sum(abs(p) for p in ps))

    order = {}
    for r in rows:
        for k in r:
            order.setdefault(k, len(order))
    seen, out = set(), []
    # ... and, between equals, the quantity that comes last in the rows is the one written from the others
    kinds = {"power": 0, "linear": 1, "exponential": 2, "log": 2, "sum": 4}   # y = A e^(kx) and x = ln(y/A)/k are one
    for score, rel in sorted(found, key=lambda t: (kinds[t[1]["kind"]], plainness(t[1]),
                                                   -order.get(t[1]["target"], 0), t[0])):
        key = frozenset([rel["target"], *_names(rel)])
        if key in seen:
            continue
        seen.add(key)
        out.append(rel)
    return out


def _names(rel):
    if rel["kind"] == "sum":
        return [x for _, mono in rel["terms"] for x in mono]
    return list(rel["terms"])


def _semilog(y, moving, positive, logs, n, r2_min, allowed):
    best = None
    if y in positive:                                  # log y = k x + c: y = A e^(k ln10 x)
        for x in sorted(set(moving) - {y}):
            if not allowed((y, x)):
                continue
            coef, rmse = _fit(logs[y], [moving[x]])
            if coef is None or _r2(logs[y], rmse) < r2_min:
                continue
            score = _bic(n, rmse, 2)
            if best is None or score < best[0]:
                k, A = coef[1] * math.log(10), 10 ** coef[0]
                best = (score, {"kind": "exponential", "target": y, "terms": {x: k}, "constant": A,
                                "constant_sigma": rmse / math.sqrt(n), "sigma": rmse, "r2": _r2(logs[y], rmse),
                                "rows": n, "text": f"{y} = {A:.6g} * e^({k:.6g} * {x})"})
    if best is None:                                   # y = a ln x + b
        for x in sorted(positive - {y}):
            if not allowed((y, x)):
                continue
            lnx = [v * math.log(10) for v in logs[x]]
            coef, rmse = _fit(moving[y], [lnx])
            if coef is None or _r2(moving[y], rmse) < r2_min:
                continue
            score = _bic(n, rmse, 2)
            if best is None or score < best[0]:
                a, b = coef[1], coef[0]
                best = (score, {"kind": "log", "target": y, "terms": {x: a}, "constant": b,
                                "constant_sigma": rmse / math.sqrt(n), "sigma": rmse, "r2": _r2(moving[y], rmse),
                                "rows": n, "text": f"{y} = {a:.6g} * ln({x}) {'+' if b >= 0 else '-'} {abs(b):.6g}"})
    return best


def _monomials(names, positive, degree=3):
    """Products of positive quantities with powers 1 or 2 (total degree up to `degree`), and the signed ones alone."""
    pos = sorted(positive & set(names))
    out = [{x: 1.0} for x in sorted(set(names) - positive)]
    for k in range(1, degree + 1):
        for combo in itertools.combinations(pos, k):
            for pw in itertools.product((1.0, 2.0), repeat=k):
                if sum(pw) <= degree:
                    out.append(dict(zip(combo, pw)))
    return out


def _sum_of_products(y, moving, positive, logs, n, r2_min, allowed, max_terms=3):
    """y = sum_j a_j m_j + b, the m_j products of the other quantities (a sum built on products: stacked)."""
    lib = [m for m in _monomials(set(moving) - {y}, positive) if allowed((y, *m))]
    cols = []
    for m in lib:
        col = [1.0] * n
        for x, p in m.items():
            col = [c * v ** p for c, v in zip(col, moving[x])]
        cols.append(col)
    # forward selection: the term that explains the most, then the one that explains most of what is left...
    chosen, best = [], None
    for k in range(1, max_terms + 1):
        step = None
        for i in range(len(lib)):
            if i in chosen:
                continue
            coef, rmse = _fit(moving[y], [cols[j] for j in chosen + [i]])
            if coef is not None and (step is None or rmse < step[1]):
                step = (i, rmse, coef)
        if step is None:
            break
        chosen.append(step[0])
        rmse, coef = step[1], step[2]
        if k < 2 or len({x for i in chosen for x in lib[i]}) < 2 or _r2(moving[y], rmse) < r2_min:
            continue
        # every term must count: dropping any one of them must make the fit clearly worse
        if any(_fit(moving[y], [cols[i] for j, i in enumerate(chosen) if j != drop])[1] < 3.0 * rmse
               for drop in range(len(chosen))):
            continue
        terms = [(coef[1 + j], lib[i]) for j, i in enumerate(chosen)]
        b = coef[0]
        text = f"{y} = " + " + ".join(
            f"{a:.6g} * " + " * ".join(f"{x}^{_fmt_power(p)}" if p != 1 else x for x, p in m.items())
            for a, m in terms) + f" {'+' if b >= 0 else '-'} {abs(b):.6g}"
        best = (_bic(n, rmse, k + 1), {"kind": "sum", "target": y, "terms": terms, "constant": b,
                                       "constant_sigma": rmse / math.sqrt(n), "sigma": rmse,
                                       "r2": _r2(moving[y], rmse), "rows": n, "text": text.replace("+ -", "- ")})
        break                                           # the fewest terms that hold
    return best


def find_dynamics(rows, min_rows=12, r2_min=0.999, exclude=()):
    """What carries each quantity from one moment to the next (`rows` in the order they were seen):
        x(t) = a x(t-1) + c                 growth or decay by a factor a each moment
        x(t) = a x(t-1) + b x(t-2) + c      an oscillation (damped or not) or two growths
    Returns {"kind": "dynamics", "target", "lags": [a] or [a, b], "constant": c, "sigma", "r2", "rows", "text",
    "meaning": {...}}: for an oscillation, its period (in moments) and what is left of it after one moment (damping);
    for one lag, the factor and the half-life or doubling time."""
    cols = _columns(rows)
    n = min((len(v) for v in cols.values()), default=0)
    if n < min_rows:
        return []
    excluded = {x for e in exclude for x in e if len(e) == 1}
    out = []
    for x, v in sorted(cols.items()):
        if x in excluded or max(v) - min(v) <= 1e-12 * max(1.0, max(abs(t) for t in v)):
            continue
        y = v[2:]
        one = _fit(y, [v[1:-1]])
        two = _fit(y, [v[1:-1], v[:-2]])
        best = None
        for coef, rmse, lags in ((one[0], one[1], 1), (two[0], two[1], 2)):
            if coef is None or _r2(y, rmse) < r2_min:
                continue
            score = _bic(len(y), rmse, lags + 1)
            if best is None or score < best[0] - math.log(len(y)):
                best = (score, coef, rmse, lags)
        if best is None:
            continue
        _, coef, rmse, lags = best
        c, ls = coef[0], coef[1:]
        meaning = {}
        if lags == 1:
            a = ls[0]
            meaning["factor"] = a
            if 0 < a < 1:
                meaning["half_life"] = math.log(0.5) / math.log(a)
            elif a > 1:
                meaning["doubling"] = math.log(2.0) / math.log(a)
            text = f"{x} = {a:.6g} * {x}(t-1) {'+' if c >= 0 else '-'} {abs(c):.6g}"
        else:
            a, b = ls
            if a * a + 4 * b < 0 and b < 0:
                r = math.sqrt(-b)
                theta = math.acos(max(-1.0, min(1.0, a / (2 * r))))
                meaning["period"] = 2 * math.pi / theta if theta > 0 else math.inf
                meaning["damping"] = r
            text = (f"{x} = {a:.6g} * {x}(t-1) {'+' if b >= 0 else '-'} {abs(b):.6g} * {x}(t-2) "
                    f"{'+' if c >= 0 else '-'} {abs(c):.6g}")
        out.append({"kind": "dynamics", "target": x, "lags": list(ls), "constant": c, "sigma": rmse,
                    "r2": _r2(y, rmse), "rows": len(y), "text": text, "meaning": meaning})
    return out


def hidden_dynamics(residuals, threshold=3.0, min_rows=20):
    """Is a node missing that changes from moment to moment? Residuals of a relation that follow each other (one
    moment's residual tells the next) call for a hidden variable with its own law, h(t) = rho h(t-1).
    Returns {"needed", "rho", "sigma" (what each moment adds), "rows"}."""
    n = len(residuals)
    if n < min_rows:
        return {"needed": False, "rho": 0.0, "sigma": math.inf, "rows": n}
    m = sum(residuals) / n
    d = [r - m for r in residuals]
    var = sum(v * v for v in d) / n
    if var <= 0:
        return {"needed": False, "rho": 0.0, "sigma": 0.0, "rows": n}
    rho = sum(d[i] * d[i - 1] for i in range(1, n)) / (n * var)
    innov = math.sqrt(max(var * (1 - rho * rho), 1e-300))
    needed = abs(rho) > 0.5 and abs(rho) * math.sqrt(n) > threshold and math.sqrt(var + m * m) > 1.0
    return {"needed": bool(needed), "rho": rho, "sigma": innov, "rows": n}


def hidden_node(residuals, threshold=3.0, min_rows=10):
    """Is something missing? residuals: how far a relation was from holding, row after row, in its sigmas.
    Returns {"needed", "value" (the mean residual), "sigma" (its uncertainty), "t", "rows"}."""
    n = len(residuals)
    if n < min_rows:
        return {"needed": False, "value": 0.0, "sigma": math.inf, "t": 0.0, "rows": n}
    m = sum(residuals) / n
    sd = math.sqrt(sum((r - m) ** 2 for r in residuals) / max(1, n - 1))
    s = sd / math.sqrt(n) if sd > 0 else 0.0
    t = abs(m) / s if s > 0 else (math.inf if m != 0 else 0.0)
    return {"needed": bool(t > threshold and abs(m) > 1.0), "value": m, "sigma": s, "t": t, "rows": n}
