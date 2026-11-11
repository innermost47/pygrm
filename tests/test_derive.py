# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Anthony Charretier
import math
from fractions import Fraction

from pygrm import GlobalRelationalNetwork, derive


def universe():
    nodes = {n: {"guess": v, "fixed": True} for n, v in
             dict(c=299792458, hbar=1.054571817e-34, kB=1.380649e-23, G=6.6743e-11).items()}
    for n in ["H0", "OmegaL", "lP", "Hend", "rhoL", "Estock", "Tfloor", "Equant", "S", "x", "y"]:
        nodes[n] = {"guess": 1.0}
    L = math.log10
    rels = [
        {"name": "planck", "terms": {"lP": 2, "c": 3, "hbar": -1, "G": -1}, "rhs": 0.0},
        {"name": "end", "terms": {"Hend": 2, "H0": -2, "OmegaL": -1}, "rhs": 0.0},
        {"name": "vacuum", "terms": {"rhoL": 1, "G": 1, "c": -2, "H0": -2, "OmegaL": -1}, "rhs": L(3 / (8 * math.pi))},
        {"name": "stock", "terms": {"Estock": 1, "Hend": 3, "rhoL": -1, "c": -3}, "rhs": L(4 * math.pi / 3)},
        {"name": "floor", "terms": {"Tfloor": 1, "kB": 1, "hbar": -1, "Hend": -1}, "rhs": L(1 / (2 * math.pi))},
        {"name": "quantum", "terms": {"Equant": 1, "kB": -1, "Tfloor": -1}, "rhs": 0.0},
        {"name": "entropy", "terms": {"S": 1, "Hend": 2, "lP": 2, "c": -2}, "rhs": L(math.pi)},
        {"name": "noise", "terms": {"x": 1, "y": 1}, "rhs": 0.3, "type": "log"},
    ]
    return GlobalRelationalNetwork(nodes, rels)


def test_the_equation_follows_from_seven_relations():
    d = universe().derive("S * Equant = Estock")
    assert d.found and d.exact and d.holds
    assert sorted(s.relation for s in d.steps) == sorted(
        ["entropy", "quantum", "stock", "planck", "floor", "end", "vacuum"])
    assert all(isinstance(s.weight, Fraction) for s in d.steps)
    assert d.conclusion() == "S · Equant = Estock"
    fr = d.text(lang="fr")
    assert "D'où `S · Equant = Estock`" in fr and "7 relation(s)" in fr
    assert r"\frac{S \cdot E_{quant}}{E_{stock}}" in d.text(fmt="latex")


def test_a_wrong_equation_is_said_wrong():
    d = universe().derive("S * Equant = 2 * Estock")
    assert d.exact and not d.holds
    assert "off by a factor `1 / 2`" in d.text()


def test_numbers_and_pi():
    d = universe().derive("Tfloor * kB = hbar * Hend / 2 / pi")
    assert d.holds and len(d.steps) == 1
    d = universe().derive("Tfloor * kB = hbar * Hend")
    assert d.exact and not d.holds and d.conclusion() == "Tfloor · kB = hbar · Hend / (2 · π)"


def test_keep_and_not_found():
    d = universe().derive("S = 1", keep=("Hend",))
    assert d.exact and d.conclusion() == "S = π · c^5 / (Hend^2 · hbar · G)"
    d = universe().derive("S = x", max_steps=6)
    assert not d.found and "No chain" in d.text()


def test_templates_are_the_callers():
    d = universe().derive("Equant = kB * Tfloor")
    out = d.text(templates={"title": "# {equation}", "start": "From {lhs}.",
                            "step": "({n}) {node} <- {definition}", "end": "{result}",
                            "conclusion": "QED: {conclusion}", "exact": ""})
    assert out.startswith("# Equant = kB * Tfloor") and "QED: `Equant = kB · Tfloor`" in out


def test_number_recognition():
    assert derive.number(math.log10(3 / (8 * math.pi))) == (Fraction(3, 8), -1)
    assert derive.number(math.log10(math.sqrt(2)))[0] is None
