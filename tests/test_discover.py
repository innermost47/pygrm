# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Anthony Charretier
import math
import random

from pygrm import discover


def rows_of(n=40, seed=3):
    r = random.Random(seed)
    rows = []
    for _ in range(n):
        t, m, a, x = r.uniform(0.5, 3), r.uniform(1, 10), r.uniform(1, 5), r.uniform(-5, 5)
        rows.append({"t": t, "h": 4.905 * t * t * (1 + r.gauss(0, 0.002)), "m": m, "a": a,
                     "F": m * a * (1 + r.gauss(0, 0.001)), "x": x, "y": 3 * x + 2 + r.gauss(0, 0.05),
                     "noise": r.uniform(1, 9)})
    return rows


def test_finds_power_laws_and_lines():
    found = {frozenset([f["target"], *f["terms"]]): f for f in discover.find_relations(rows_of())}
    fall = found[frozenset(["h", "t"])]
    assert fall["kind"] == "power" and fall["target"] == "h" and fall["terms"] == {"t": 2.0}
    assert fall["constant"] == pytest_approx(4.905, 0.01)
    newton = found[frozenset(["F", "m", "a"])]
    assert newton["target"] == "F" and newton["terms"] == {"m": 1.0, "a": 1.0}
    line = found[frozenset(["x", "y"])]
    assert line["kind"] == "linear" and line["target"] == "y"
    assert line["terms"]["x"] == pytest_approx(3, 0.01) and line["constant"] == pytest_approx(2, 0.05)
    assert not any("noise" in f["terms"] or f["target"] == "noise" for f in found.values())


def test_nothing_in_noise():
    r = random.Random(1)
    rows = [{"a": r.uniform(1, 9), "b": r.uniform(1, 9), "c": r.uniform(-9, 9)} for _ in range(60)]
    assert discover.find_relations(rows) == []


def test_a_power_that_is_not_simple_is_not_made_simple():
    r = random.Random(2)
    rows = [{"x": x, "y": 2.0 * x ** 1.8} for x in [r.uniform(1, 20) for _ in range(40)]]
    assert discover.find_relations(rows) == []          # 1.8 is not 2: no h = K x^2 invented


def test_exclude():
    assert discover.find_relations(rows_of(), exclude=[{"h", "t"}, {"F", "m", "a"}, {"x", "y"}]) == []


def test_hidden_node():
    r = random.Random(4)
    assert discover.hidden_node([1.5 + r.gauss(0, 1) for _ in range(40)])["needed"]
    assert not discover.hidden_node([r.gauss(0, 1) for _ in range(40)])["needed"]
    assert not discover.hidden_node([5.0] * 5)["needed"]             # too few rows to say


def pytest_approx(v, rel):
    class A:
        def __eq__(self, o):
            return abs(o - v) <= rel * abs(v)
    return A()


def test_exponentials_logs_and_sums():
    r = random.Random(5)
    rows = []
    for _ in range(40):
        x, z, m, v, h = r.uniform(-2, 2), r.uniform(1, 50), r.uniform(1, 5), r.uniform(0.5, 4), r.uniform(0.1, 3)
        rows.append({"x": x, "growth": 2 * math.exp(0.5 * x) * (1 + r.gauss(0, 0.0005)), "z": z,
                     "loud": 3 * math.log(z) + 1 + r.gauss(0, 0.001), "m": m, "v": v, "h": h,
                     "E": 0.5 * m * v * v + 9.81 * m * h + r.gauss(0, 0.001)})
    found = {f["target"]: f for f in discover.find_relations(rows)}
    assert found["growth"]["kind"] == "exponential"
    assert found["growth"]["terms"]["x"] == pytest_approx(0.5, 0.01) and found["growth"]["constant"] == pytest_approx(2, 0.01)
    assert found["loud"]["kind"] == "log" and found["loud"]["terms"]["z"] == pytest_approx(3, 0.01)
    energy = found["E"]
    assert energy["kind"] == "sum"
    terms = {frozenset(m.items()): a for a, m in energy["terms"]}
    assert terms[frozenset({("m", 1.0), ("v", 2.0)})] == pytest_approx(0.5, 0.01)
    assert terms[frozenset({("m", 1.0), ("h", 1.0)})] == pytest_approx(9.81, 0.01)


def test_dynamics():
    r = random.Random(6)
    xs, theta, rr = [1.0, 0.9], 2 * math.pi / 20, 0.98
    for _ in range(80):
        xs.append(2 * rr * math.cos(theta) * xs[-1] - rr * rr * xs[-2] + r.gauss(0, 1e-4))
    decay = [10 * 0.9 ** i for i in range(len(xs))]
    found = {f["target"]: f for f in discover.find_dynamics([{"wave": a, "decay": b} for a, b in zip(xs, decay)])}
    assert found["wave"]["meaning"]["period"] == pytest_approx(20, 0.01)
    assert found["wave"]["meaning"]["damping"] == pytest_approx(0.98, 0.005)
    assert found["decay"]["meaning"]["half_life"] == pytest_approx(math.log(0.5) / math.log(0.9), 0.01)
    assert discover.find_dynamics([{"noise": r.gauss(0, 1)} for _ in range(60)]) == []


def test_hidden_dynamics():
    r = random.Random(7)
    ar = [0.0]
    for _ in range(60):
        ar.append(0.9 * ar[-1] + r.gauss(0, 2))
    got = discover.hidden_dynamics(ar)
    assert got["needed"] and 0.6 < got["rho"] < 1.0
    assert not discover.hidden_dynamics([r.gauss(0, 2) for _ in range(60)])["needed"]
