"""The four systems of units: one network, the same physics in each."""
import math

from pygrm import GlobalRelationalNetwork, units

C, HBAR, G = 299792458.0, 1.054571817e-34, 6.67430e-11
H = 2.27e-18
T_P = math.sqrt(HBAR * G / C ** 5)
ENERGY, TIME, INVERSE_TIME, SPEED, ACTION, GRAVITY = (
    [1, 2, -2, 0, 0], [0, 0, 1, 0, 0], [0, 0, -1, 0, 0], [0, 1, -1, 0, 0], [1, 2, -1, 0, 0], [-1, 3, -2, 0, 0])


def nodes():
    return {
        "c": {"guess": C, "fixed": True, "dim": SPEED},
        "hbar": {"guess": HBAR, "fixed": True, "dim": ACTION},
        "G": {"guess": G, "dim": GRAVITY},
        "H": {"guess": 2e-18, "dim": INVERSE_TIME},
        "tP": {"guess": 1e-43, "dim": TIME},
        "S": {"guess": 1e120},
        "Equant": {"guess": 1e-52, "dim": ENERGY},
        "Estock": {"guess": 1e70, "dim": ENERGY},
        "w": {"guess": -0.9, "scale": "lin"},
    }


def relations():
    log2pi = math.log10(2 * math.pi)
    return [
        {"name": "planck_time", "terms": {"tP": 2, "hbar": -1, "G": -1, "c": 5}, "rhs": 0.0, "sigma": 1e-9},
        {"name": "grains", "terms": {"S": 1, "H": 2, "tP": 2}, "rhs": math.log10(math.pi), "sigma": 1e-9},
        {"name": "quantum", "terms": {"Equant": 1, "hbar": -1, "H": -1}, "rhs": -log2pi, "sigma": 1e-9},
        {"name": "stock", "terms": {"Estock": 1, "c": -5, "G": 1, "H": 1}, "rhs": -math.log10(2), "sigma": 1e-9},
        {"name": "equation", "terms": {"S": 1, "Equant": 1, "Estock": -1}, "rhs": 0.0, "sigma": 1e-9},
        {"name": "w_sum", "type": "sum", "terms": {"w": 1.0}, "K": -1.0, "sigma": 1e-3},
        {"name": "w_fn", "type": "fn", "nodes": ["w", "H"], "f": lambda v: (v["w"] + 1.0) * 1e-3, "sigma": 1e-3},
    ]


MEASURED = {"G": (G, 1e-5), "H": (H, 1e-3)}


def solve(mode):
    net = GlobalRelationalNetwork(nodes(), relations(), units=mode)
    return net, net.solve(MEASURED)


def test_same_physics_in_every_system():
    _, ref = solve("si")
    for mode in units.MODES:
        net, res = solve(mode)
        assert res["converged"] and res["units"] == mode
        si = res.get("si", res["values"])
        for n, v in ref["values"].items():
            if n == "w":
                assert abs(si[n] - v) < 1e-6
            else:
                assert abs(math.log10(si[n]) - math.log10(v)) < 1e-9, (mode, n)
        assert abs(res["values"]["S"] - ref["values"]["S"]) / ref["values"]["S"] < 1e-9  # a pure number


def test_planck_numbers():
    _, res = solve("planck")
    v = res["values"]
    assert abs(v["c"] - 1) < 1e-12 and abs(v["hbar"] - 1) < 1e-12
    assert abs(math.log10(v["G"])) < 1e-6 and abs(math.log10(v["tP"])) < 1e-6
    assert abs(v["H"] - H * T_P) / (H * T_P) < 1e-6


def test_express_and_ticks():
    net, res = solve("si")
    out = net.express(res, "ticks")
    assert out["tP"]["power"] == 1 and abs(out["tP"]["ticks_log10"]) < 1e-6
    assert out["H"]["power"] == -1                       # a rate: one beat every N ticks
    assert abs(out["H"]["ticks_log10"] + math.log10(H * T_P)) < 1e-6
    assert out["Equant"]["power"] == -1                  # an energy too
    assert out["S"]["power"] == 0 and out["S"]["ticks_log10"] is None
    assert out["Equant"]["unit_sigma_dex"] > 0           # G's uncertainty enters the Planck energy
    assert out["S"]["unit_sigma_dex"] == 0
    assert units.unit_sigma_dex(TIME, "si") == 0


def test_atomic_units():
    a0 = 5.29177210903e-11
    assert abs(units.to_mode(a0, [0, 1, 0, 0, 0], "atomic") - 1) < 1e-9
    assert abs(units.to_mode(9.1093837015e-31, [1, 0, 0, 0, 0], "atomic") - 1) < 1e-12
    eh = 4.3597447222071e-18
    assert abs(units.to_mode(eh, ENERGY, "atomic") - 1) < 1e-9
    assert abs(units.from_mode(1.0, ENERGY, "atomic") - eh) / eh < 1e-9


def test_caller_specs_untouched():
    n, r = nodes(), relations()
    GlobalRelationalNetwork(n, r, units="planck").solve(MEASURED)
    assert n["c"]["guess"] == C and "solved" not in n["c"]
