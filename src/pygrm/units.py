# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Anthony Charretier
"""Four systems of units for a GRM network: SI, atomic, Planck, and Planck ticks.

A node says what it measures with "dim": its powers of mass, length, time, charge and temperature,
[M, L, T, Q, K] (a speed is [0, 1, -1, 0, 0], an energy [1, 2, -2, 0, 0]). Without "dim" a node is a pure number.

    mode      unit of each dimension                                  precision of the units
    "si"      kilogram, metre, second, coulomb, kelvin                exact
    "atomic"  electron mass, Bohr radius, hbar / E_h, e, E_h / k_B    ~1e-10 (the electron is well measured)
    "planck"  m_P, l_P, t_P, q_P, T_P   (c = G = hbar = k_B = 1)      ~1e-5  (G is the least known constant)
    "ticks"   the Planck tick t_P only: with c = hbar = k_B = 1 every dimension is a power of time,
              t_P^k with k = -M + L + T - K (a length is a light-travel time, a mass or an energy beats
              once every hbar / E). The number is the Planck one; it reads as N ticks to the power k.

Two ways to use them:
    express(values, nodes_spec, mode, sigma_dex=None)
        the result of an ordinary (SI) solve, written in another system, with the uncertainty of the units
        themselves added (in Planck units, G's uncertainty reaches every mass and energy).
    convert_network(nodes_spec, relations_spec, mode)  ->  UnitNetwork
        the whole network rewritten in another system, so that it is SOLVED in it ("hardcore"): node
        values, bounds and measurements, the right side of log relations, the coefficients of sum
        relations, and fn relations (wrapped so that their own function still sees SI values).
        UnitNetwork.measured(...) converts measurements, UnitNetwork.to_si(values) brings results back.

Constants: CODATA 2018 (h, e, k_B, c exact since the 2019 SI; G, alpha, m_e with their uncertainties).
"""
import math

BASES = ("M", "L", "T", "Q", "K")
MODES = ("si", "atomic", "planck", "ticks")

# CODATA 2018: value, relative standard uncertainty
CONSTANTS = {
    "c": (299792458.0, 0.0),
    "hbar": (1.054571817e-34, 0.0),
    "e": (1.602176634e-19, 0.0),
    "kB": (1.380649e-23, 0.0),
    "G": (6.67430e-11, 2.2e-5),
    "alpha": (7.2973525693e-3, 1.5e-10),
    "me": (9.1093837015e-31, 3.0e-10),
}


def _units(mode, k):
    """Value in SI of the unit of each base dimension, for constants k = {name: value}."""
    c, hbar, e, kB, G, alpha, me = (k[n] for n in ("c", "hbar", "e", "kB", "G", "alpha", "me"))
    if mode == "si":
        return (1.0, 1.0, 1.0, 1.0, 1.0)
    if mode == "atomic":
        a0 = hbar / (me * c * alpha)
        Eh = me * (c * alpha) ** 2
        return (me, a0, hbar / Eh, e, Eh / kB)
    if mode in ("planck", "ticks"):
        mP = math.sqrt(hbar * c / G)
        tP = math.sqrt(hbar * G / c ** 5)
        return (mP, c * tP, tP, e / math.sqrt(alpha), mP * c * c / kB)
    raise ValueError(f"unknown mode '{mode}' (one of {', '.join(MODES)})")


def _dim(spec):
    """The dim of a node spec as a 5-tuple (accepts a list, or a dict like {"M": 1, "L": 2, "T": -2})."""
    d = spec.get("dim") if isinstance(spec, dict) else spec
    if d is None:
        return (0.0,) * 5
    if isinstance(d, dict):
        return tuple(float(d.get(b, 0.0)) for b in BASES)
    d = list(d) + [0.0] * (5 - len(d))
    return tuple(float(x) for x in d[:5])


def unit_log10(dim, mode):
    """log10 of the unit of a quantity of dimension dim in this mode (SI value of '1' in that mode)."""
    k = {n: v for n, (v, _) in CONSTANTS.items()}
    return sum(p * math.log10(u) for p, u in zip(_dim(dim), _units(mode, k)) if p)


def unit_sigma_dex(dim, mode):
    """Uncertainty (dex) of that unit, from the uncertainties of the constants it is made of."""
    k = {n: v for n, (v, _) in CONSTANTS.items()}
    base = unit_log10(dim, mode)
    var = 0.0
    for name, (v, rel) in CONSTANTS.items():
        if rel <= 0.0:
            continue
        kk = dict(k)
        kk[name] = v * (1.0 + rel)
        shifted = sum(p * math.log10(u) for p, u in zip(_dim(dim), _units(mode, kk)) if p)
        var += (shifted - base) ** 2
    return math.sqrt(var)


def tick_power(dim):
    """k such that a quantity of this dimension is a number of Planck ticks to the power k."""
    M, L, T, Q, K = _dim(dim)
    return -M + L + T - K


def to_mode(value, dim, mode):
    """A value given in SI, written in the units of `mode`."""
    return value / 10 ** unit_log10(dim, mode)


def from_mode(value, dim, mode):
    """A value given in the units of `mode`, written in SI."""
    return value * 10 ** unit_log10(dim, mode)


def express(values, nodes_spec, mode, sigma_dex=None):
    """The values of a solve (SI), written in `mode`. For every node with a positive value:
        value      the number in that system
        sigma_dex  its uncertainty: the solve's (if given) and the units' own, added in quadrature
                   (a signed "lin" node has "sigma" in value units instead)
        unit_sigma_dex  the part that comes from the units (G in Planck units, for instance)
    and in mode "ticks" also
        power      k, and  ticks_log10  log10 N  (the value is N ticks to the power k; None when k = 0)."""
    out = {}
    for name, v in values.items():
        spec = nodes_spec.get(name, {})
        dim = _dim(spec)
        lin = spec.get("scale", "log") == "lin"
        u = unit_log10(dim, mode)
        us = unit_sigma_dex(dim, mode)
        row = {"value": v / 10 ** u, "unit_sigma_dex": us}
        s = None if sigma_dex is None else sigma_dex.get(name)
        if not lin and v > 0:
            row["sigma_dex"] = math.sqrt((s or 0.0) ** 2 + us ** 2) if s is not None or us else s
            if mode == "ticks":
                k = tick_power(dim)
                row["power"] = k
                row["ticks_log10"] = math.log10(row["value"]) / k if k else None
        else:
            row["sigma"] = None if s is None else s / 10 ** u     # a signed node: sigma in value units
            row["sigma_dex"] = None
        out[name] = row
    return out


class UnitNetwork:
    """A network rewritten in another system of units (see convert_network)."""

    def __init__(self, nodes, relations, offsets, lin, mode):
        self.nodes, self.relations, self.mode = nodes, relations, mode
        self._offset = offsets            # log10 of each node's unit
        self._lin = lin

    def _to(self, name, v):
        u = 10 ** self._offset.get(name, 0.0)
        return v / u

    def _back(self, name, v):
        return v * 10 ** self._offset.get(name, 0.0)

    def measured(self, measured_dict):
        """Measurements (SI) written in this system: a log node keeps its sigma in dex, a signed node's sigma
        is scaled with its value."""
        out = {}
        for name, m in (measured_dict or {}).items():
            if isinstance(m, (tuple, list)):
                v, s = m
                s = s / 10 ** self._offset.get(name, 0.0) if self._lin.get(name) else s
                out[name] = (self._to(name, v), s)
            else:
                out[name] = self._to(name, m)
        return out

    def to_si(self, values):
        """Solved values brought back to SI."""
        return {n: self._back(n, v) for n, v in values.items()}

    def from_si(self, values):
        return {n: self._to(n, v) for n, v in values.items()}


def convert_network(nodes_spec, relations_spec, mode):
    """The network rewritten in `mode` (to be solved in it). Returns a UnitNetwork."""
    offsets = {n: unit_log10(_dim(s), mode) for n, s in nodes_spec.items()}
    lin = {n: s.get("scale", "log") == "lin" for n, s in nodes_spec.items()}
    nodes = {}
    for n, s in nodes_spec.items():
        s2 = dict(s)
        f = 10 ** offsets[n]
        for key in ("guess", "min", "max", "solved", "value"):
            if s2.get(key) is not None:
                s2[key] = s2[key] / f
        nodes[n] = s2

    def si_values(vals):
        return {n: v * 10 ** offsets.get(n, 0.0) for n, v in vals.items()}

    relations = []
    for r in relations_spec:
        r2 = dict(r)
        kind = r.get("type", "log")
        if kind == "log":
            shift = sum(c * offsets.get(n, 0.0) for n, c in r["terms"].items())
            rhs = r.get("rhs", 0.0)
            if callable(rhs):
                r2["rhs"] = (lambda f, sh: (lambda vals: f(si_values(vals)) - sh))(rhs, shift)
            else:
                r2["rhs"] = float(rhs) - shift
        elif kind == "sum":
            pw = r.get("pow", {}) or {}
            r2["terms"] = {n: c * (10 ** offsets.get(n, 0.0)) ** pw.get(n, 1.0) for n, c in r["terms"].items()}
        elif kind == "fn":
            f = r["f"]
            r2["f"] = (lambda g: (lambda vals: g(si_values(vals))))(f)
            if callable(r.get("grad")):
                g0 = r["grad"]
                r2["grad"] = (lambda g: (lambda vals: {n: d * 10 ** offsets.get(n, 0.0)
                                                        for n, d in g(si_values(vals)).items()}))(g0)
        relations.append(r2)
    return UnitNetwork(nodes, relations, offsets, lin, mode)
