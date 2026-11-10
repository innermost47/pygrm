# pygrm

The **Global Relational Model** engine for Python: a solver for networks of relations.

A node is a quantity; it has no value of its own, only the relations that hold it. pygrm solves the whole network
at once and tells you:

- the best value of every node, with its uncertainty, and the nodes nothing determines;
- where the network is in tension (the pull of every relation and measurement);
- the equations the network implies without writing them (`global_equations`), and which relations they come from;
- the invariant that ties the most distant scales together (`extract_master_invariant`).

Relations are written in log space, so values from 1e-53 to 1e122 live side by side. Products, ratios and powers are
linear there (`log`), sums have their own type (`sum`), and any other equation can be given as a function (`fn`).
The solver is a weighted least squares by QR factorisation, with Levenberg-Marquardt for the non-linear cases.
It runs in pure Python; NumPy makes it faster.

## Install

```
pip install pygrm            # pure Python
pip install pygrm[fast]      # with NumPy
```

## Example

```python
import math
from pygrm import GlobalRelationalNetwork

nodes = {
    "c": {"guess": 299792458.0, "fixed": True},
    "G": {"guess": 6.67430e-11, "fixed": True},
    "hbar": {"guess": 1.054571817e-34, "fixed": True},
    "H": {"guess": 2e-18},
    "S": {"guess": 1e122},
    "Equant": {"guess": 1e-53},
    "Estock": {"guess": 1e70},
}
relations = [  # sum of coeff * log10(node) = rhs
    {"name": "entropy", "terms": {"S": 1, "H": 2, "hbar": 1, "G": 1, "c": -5}, "rhs": math.log10(math.pi), "sigma": 1e-9},
    {"name": "quantum", "terms": {"Equant": 1, "hbar": -1, "H": -1}, "rhs": -math.log10(2 * math.pi), "sigma": 1e-9},
    {"name": "content", "terms": {"Estock": 1, "c": -5, "G": 1, "H": 1}, "rhs": -math.log10(2), "sigma": 1e-9},
]

net = GlobalRelationalNetwork(nodes, relations)
res = net.solve(measured_dict={"H": (1.89e-18, 0.005)})   # value, sigma in dex

for n in ("S", "Equant", "Estock"):
    print(f"{n:7s} {res['values'][n]:.4e}  +- {res['sigma_dex'][n]:.4f} dex")

for eq in net.global_equations(["S", "Equant", "Estock"]):
    print(eq["text"])
```

```
S       3.0259e+122  +- 0.0100 dex
Equant  3.1722e-53  +- 0.0050 dex
Estock  9.5986e+69  +- 0.0050 dex
S · Equant · Estock^-1 = 1
```

The three relations never mention one another, yet the network implies `S x Equant = Estock`, for any H.

## Relations

| type  | form                                                     | use                               |
|-------|----------------------------------------------------------|-----------------------------------|
| `log` | `{"terms": {node: coeff}, "rhs": r}`: sum coeff·log10(node) = r | products, ratios, powers |
| `sum` | `{"type": "sum", "terms": {node: coeff}, "pow": {node: p}, "K": k}`: sum coeff·node^p = k | charges, fractions |
| `fn`  | `{"type": "fn", "nodes": [...], "f": callable}`: f(values) = 0 | anything else |

Each relation may carry a `sigma` (its tolerance, in dex for `log` relations). Nodes are positive by default
(handled as log10); `{"scale": "lin"}` makes a signed node.

## Showcase

What the GRM is used for, from cosmology to a physically modelled guitar:
[grm-core](https://github.com/innermost47/grm-core).

## License

GNU Affero General Public License v3.0 or later. Copyright (C) 2026 Anthony Charretier.
