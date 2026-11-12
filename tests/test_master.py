# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Anthony Charretier
"""The master invariant: pairs and triples of nodes, and residuals that are only rounding."""
import math

import pytest

from pygrm import GlobalRelationalNetwork
from pygrm.grm import RESIDUAL_FLOOR, _merit

pytest.importorskip("numpy")

C, HBAR, KB, G = 299792458.0, 1.054571817e-34, 1.380649e-23, 6.67430e-11
H0, OL = 2.184e-18, 0.6847


def log(x):
    return math.log10(x)


def universe(g_constant=True):
    """The seven relations of the chain (grm-core README), in log10: sum of coeff * log10(node) = rhs."""
    nodes = {n: {"guess": 1.0} for n in ("lP", "Hend", "rhoL", "Estock", "Tfloor", "Equant", "S", "H0", "OmegaL")}
    for n, v in (("c", C), ("hbar", HBAR), ("kB", KB)) + ((("G", G),) if g_constant else ()):
        nodes[n] = {"guess": v, "fixed": True}
    if not g_constant:
        nodes["G"] = {"guess": G}
    pi = math.pi
    rels = [
        ("planck", {"lP": 2, "c": 3, "hbar": -1, "G": -1}, 0.0),
        ("end", {"Hend": 2, "H0": -2, "OmegaL": -1}, 0.0),
        ("vacuum", {"rhoL": 1, "G": 1, "c": -2, "H0": -2, "OmegaL": -1}, log(3 / (8 * pi))),
        ("stock", {"Estock": 1, "Hend": 3, "rhoL": -1, "c": -3}, log(4 * pi / 3)),
        ("floor", {"Tfloor": 1, "kB": 1, "hbar": -1, "Hend": -1}, log(1 / (2 * pi))),
        ("quantum", {"Equant": 1, "kB": -1, "Tfloor": -1}, 0.0),
        ("entropy", {"S": 1, "Hend": 2, "lP": 2, "c": -2}, log(pi)),
    ]
    spec = [{"name": n, "terms": t, "rhs": r, "sigma": 1e-9} for n, t, r in rels]
    net = GlobalRelationalNetwork(nodes, spec)
    v = {"c": C, "hbar": HBAR, "kB": KB, "G": G, "H0": H0, "OmegaL": OL}
    v["lP"] = math.sqrt(HBAR * G / C ** 3)
    v["Hend"] = H0 * math.sqrt(OL)
    v["rhoL"] = 3 * C ** 2 * H0 ** 2 * OL / (8 * pi * G)
    v["Estock"] = 4 * pi * v["rhoL"] * C ** 3 / (3 * v["Hend"] ** 3)
    v["Tfloor"] = HBAR * v["Hend"] / (2 * pi * KB)
    v["Equant"] = KB * v["Tfloor"]
    v["S"] = pi * C ** 2 / (v["Hend"] ** 2 * v["lP"] ** 2)
    return net, v


@pytest.mark.parametrize("g_constant", [True, False])
def test_a_three_node_equation_where_every_constant_cancels(g_constant):
    net, v = universe(g_constant)
    m = net.extract_master_invariant(v, display_mode="compact")
    assert m["terms"] == {"Equant": 1.0, "S": 1.0, "Estock": -1.0} or \
        {k: -c for k, c in m["terms"].items()} == {"Equant": 1.0, "S": 1.0, "Estock": -1.0}
    assert abs(m["rhs"]) < 1e-9 and m["derived"] and m["span"] == pytest.approx(175.0, abs=0.5)


def test_pairs_only_cannot_bring_it_up():
    net, v = universe(True)
    m = net.extract_master_invariant(v, display_mode="compact", max_triples=0)
    assert len(m["terms"]) > 3                 # a pair and the constants: S Equant^2 = hbar c^5 / (4 pi G)


def test_a_wrong_relation_does_not_give_it():
    net, v = universe(True)
    for rel in net.relations_spec:
        if rel["name"] == "stock":
            rel["terms"] = {"Estock": 1, "Hend": 2, "rhoL": -1, "c": -3}     # Hend^2: wrong on purpose
    net = GlobalRelationalNetwork(net.nodes_specs, net.relations_spec)
    m = net.extract_master_invariant(v, display_mode="compact")
    assert set(m["terms"]) != {"Equant", "S", "Estock"}


def test_rounding_is_not_a_difference():
    assert _merit(10.0, 1e-14) == _merit(10.0, RESIDUAL_FLOOR)
    assert _merit(10.0, 3e-9, sigma=2e-9) == _merit(10.0, 0.0)       # exact relations, within 2 of their sigmas
    assert _merit(10.0, 3e-9, sigma=1e-3) < _merit(10.0, 0.0)        # not exact: the residual counts
    assert _merit(10.0, 1e-3) < _merit(9.0, 0.0)
