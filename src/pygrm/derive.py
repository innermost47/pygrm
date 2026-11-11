# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Anthony Charretier
"""From an equation to its demonstration, through the network's own relations.

    from pygrm import GlobalRelationalNetwork
    net = GlobalRelationalNetwork(nodes, relations)
    d = net.derive("S * Equant = Estock")
    print(d.text(lang="fr"))             # a written demonstration (Markdown), from templates with placeholders
    print(d.text(fmt="latex"))
    d.steps, d.exact, d.factor           # the steps, whether everything cancels exactly, what is left

How. A product relation is a straight line in log10: sum_i p_i log(n_i) = b. The equation to show is one too
(the left side over the right side, sum_i t_i log(n_i)). The demonstration replaces, one at a time, a quantity of
the expression by its definition taken from one relation of the network (the relation solved for that quantity),
until no quantity is left but the fixed constants of the network (and those the caller keeps). Each replacement is
a step. The powers are kept as exact fractions: when every quantity cancels exactly, what is left (a number, a power
of pi, constants) is the right side, and the equation is shown inside the network: from its relations, by
substitutions alone. Among the ways to get there, the one with the fewest steps is taken (iterative deepening).

What it does not do: relations with sums or functions (not straight in log) are not used; a relation whose right
side is computed (a callable) is not used; the demonstration holds inside the network, it does not show that its
relations are true (that is what measurements are for).

The words are the caller's: `templates` holds the sentences, with placeholders the network fills ({lhs},
{relation}, {node}, {definition}, {expression}, {rhs}, {weight}, {n}). French and English templates are given.
"""
import math
from fractions import Fraction

TEMPLATES = {
    "en": {
        "title": "## Demonstration: {equation}",
        "start": "Start from the left side over the right side: {lhs}.",
        "step": "{n}. Replace {node} by its definition ({relation}): {definition}. The expression becomes "
                "{expression}.",
        "end": "No quantity of the network is left: {result}.",
        "conclusion": "Hence {conclusion}.",
        "exact": "Every power cancels exactly: the equation follows from {count} relation(s) of the network, by "
                 "substitution alone.",
        "differs": "Every power cancels, but what is left is not 1: the network gives {conclusion}. The equation as "
                   "written is off by a factor {ratio}.",
        "inexact": "The powers do not all cancel: {rest} remain. The equation does not follow from these relations "
                   "alone.",
        "none": "No chain of the network's product relations leads from {lhs} to its constants within {max_steps} "
                "steps.",
    },
    "fr": {
        "title": "## Démonstration : {equation}",
        "start": "Le point de départ est le membre de gauche divisé par le membre de droite : {lhs}.",
        "step": "{n}. {node} est remplacé par sa définition ({relation}) : {definition}. L'expression devient "
                "{expression}.",
        "end": "Il ne reste plus aucune grandeur du réseau : {result}.",
        "conclusion": "D'où {conclusion}.",
        "exact": "Toutes les puissances s'annulent exactement : l'équation découle de {count} relation(s) du "
                 "réseau, par substitution seule.",
        "differs": "Les puissances s'annulent, mais il reste un facteur : le réseau donne {conclusion}. L'équation "
                   "telle qu'écrite est fausse d'un facteur {ratio}.",
        "inexact": "Les puissances ne s'annulent pas toutes : il reste {rest}. L'équation ne découle pas de ces "
                   "seules relations.",
        "none": "Aucune chaîne de relations de produit du réseau ne mène de {lhs} à ses constantes en moins de "
                "{max_steps} étapes.",
    },
}


def _frac(x):
    return Fraction(x).limit_denominator(1000)


def parse_equation(text):
    """'S * Equant = Estock' -> {S: 1, Equant: 1, Estock: -1} (left over right), and the log10 of the numbers."""
    if "=" not in text:
        raise ValueError("an equation: left = right")
    left, right = text.split("=", 1)
    terms, const = {}, 0.0
    for side, sign in ((left, 1), (right, -1)):
        t, c = _side(side)
        const += sign * c
        for n, p in t.items():
            terms[n] = terms.get(n, Fraction(0)) + sign * p
    return {n: p for n, p in terms.items() if p != 0}, const


def _side(text):
    import re
    text = text.replace(" ", "")
    if not text:
        raise ValueError("an empty side")
    terms, const, sign = {}, 0.0, 1
    for part in re.split(r"([*/])", text):
        if part in ("*", "/"):
            sign = 1 if part == "*" else -1
            continue
        if not part:
            continue
        base, _, exp = part.partition("^")
        p = _frac(eval(exp.strip("()"), {"__builtins__": {}}, {})) if exp else Fraction(1)
        base = base.strip("()")
        if base == "pi":
            const += sign * float(p) * math.log10(math.pi)
        elif re.fullmatch(r"[0-9.eE+-]+", base):
            const += sign * float(p) * math.log10(float(base))
        else:
            terms[base] = terms.get(base, Fraction(0)) + sign * p
        sign = 1
    return terms, const


def number(log10_value, tol=1e-9):
    """A number given by its log10, written as a simple rational times a power of pi when it is one."""
    v = 10 ** log10_value
    for k in (0, 1, -1, 2, -2, 3, -3):
        r = v / math.pi ** k
        f = Fraction(r).limit_denominator(1000)
        if f and abs(float(f) - r) <= tol * abs(r):
            return f, k
    return None, None


def _num_parts(log10_value, fmt):
    """The number as (top, bottom) pieces: 3/(8 pi) -> (["3"], ["8", "pi"])."""
    if abs(log10_value) <= 1e-12:
        return [], []
    f, k = number(log10_value)
    if f is None:
        return [f"{10 ** log10_value:.6g}"], []
    pi = r"\pi" if fmt == "latex" else "π"
    top, bottom = [], []
    if f.numerator != 1:
        top.append(str(f.numerator))
    if f.denominator != 1:
        bottom.append(str(f.denominator))
    if k:
        piece = pi if abs(k) == 1 else (f"{pi}^{{{abs(k)}}}" if fmt == "latex" else f"{pi}^{abs(k)}")
        (top if k > 0 else bottom).append(piece)
    return top, bottom


def _pow_text(name, p, fmt):
    if p == 1:
        return _name(name, fmt)
    if fmt == "latex":
        e = str(p.numerator) if p.denominator == 1 else rf"{p.numerator}/{p.denominator}"
        return f"{_name(name, fmt)}^{{{e}}}"
    e = str(p.numerator) if p.denominator == 1 else f"({p.numerator}/{p.denominator})"
    return f"{name}^{e}"


def _name(name, fmt):
    if fmt != "latex":
        return name
    head, _, tail = name.partition("_")
    greek = {"hbar": r"\hbar", "rhoL": r"\rho_\Lambda", "OmegaL": r"\Omega_\Lambda", "pi": r"\pi"}
    if name in greek:
        return greek[name]
    if len(name) > 1 and name[0].isalpha() and name[1:].isalnum() and not tail:
        # S, Hend -> H_{end}, lP -> l_{P}, kB -> k_{B}
        return f"{name[0]}_{{{name[1:]}}}" if len(name) > 1 else name
    return rf"\mathrm{{{head}}}_{{{tail}}}" if tail else name


def product_text(terms, log10_const=0.0, fmt="markdown"):
    """{node: power} and a number -> '3 · c^2 · H0^2 / (8 · π · G)' (or LaTeX)."""
    ntop, nbottom = _num_parts(log10_const, fmt)
    num = ntop + [_pow_text(n, p, fmt) for n, p in terms.items() if p > 0]
    den = nbottom + [_pow_text(n, -p, fmt) for n, p in terms.items() if p < 0]
    dot = r" \cdot " if fmt == "latex" else " · "
    top = dot.join(num) or "1"
    if not den:
        return top
    bottom = dot.join(den)
    if fmt == "latex":
        return rf"\frac{{{top}}}{{{bottom}}}"
    return f"{top} / ({bottom})" if len(den) > 1 else f"{top} / {bottom}"


class Step:
    def __init__(self, node, relation, weight, definition, expression, log10_const):
        self.node, self.relation, self.weight = node, relation, weight
        self.definition = definition              # (node, {node: power}, log10 constant): node = K * prod
        self.expression = expression              # {node: power} after the step
        self.log10_const = log10_const            # the number in front of the expression after the step


class Derivation:
    def __init__(self, equation, target, target_const, steps, rest, log10_const, constants, max_steps):
        self.equation, self.target, self.target_const = equation, target, target_const
        self.steps, self.rest, self.log10_const = steps, rest, log10_const
        self.constants, self.max_steps = constants, max_steps
        self.found = steps is not None
        # exact: every power of a quantity of the network cancels (fractions), only kept constants may remain
        self.exact = self.found and all(p == 0 for n, p in rest.items() if n not in constants)
        # holds: what is left is exactly 1 (no constant, no number): the equation as written follows
        self.holds = self.exact and all(p == 0 for p in rest.values()) and abs(log10_const) < 1e-9
        self.factor = (rest, log10_const)

    @property
    def relations(self):
        return [] if not self.steps else [(s.relation, s.weight) for s in self.steps]

    def rhs_text(self, fmt="markdown"):
        left = {n: p for n, p in self.rest.items() if p != 0}
        return product_text(left, self.log10_const, fmt) if (left or abs(self.log10_const) > 1e-12) else "1"

    def conclusion(self, fmt="markdown"):
        """The equation as shown: left = (what is left) x right."""
        left = {n: p for n, p in self.target.items() if p > 0}
        right = {n: -p for n, p in self.target.items() if p < 0}
        for n, p in self.rest.items():
            if p != 0:
                right[n] = right.get(n, Fraction(0)) + p
        return product_text(left, 0.0, fmt) + " = " + product_text(
            {n: p for n, p in right.items() if p != 0}, self.log10_const - self.target_const, fmt)

    def text(self, lang="en", fmt="markdown", templates=None):
        """The written demonstration: the caller's sentences (templates, with placeholders), the network's
        quantities. fmt: "markdown" (plain formulas) or "latex" (formulas in $...$)."""
        t = dict(TEMPLATES.get(lang, TEMPLATES["en"]))
        t.update(templates or {})
        wrap = (lambda s: f"${s}$") if fmt == "latex" else (lambda s: f"`{s}`")
        lhs = wrap(product_text(self.target, self.target_const, fmt))
        lines = [t["title"].format(equation=self.equation)]
        if not self.found:
            lines.append(t["none"].format(lhs=lhs, max_steps=self.max_steps))
            return "\n\n".join(lines)
        lines.append(t["start"].format(lhs=lhs))
        for k, s in enumerate(self.steps, 1):
            node, defn, c = s.definition
            definition = wrap(f"{_name(node, fmt)} = {product_text(defn, c, fmt)}")
            lines.append(t["step"].format(n=k, node=wrap(_name(s.node, fmt)), relation=s.relation,
                                          definition=definition, weight=s.weight,
                                          expression=wrap(product_text(s.expression, s.log10_const, fmt))))
        lines.append(t["end"].format(lhs=lhs, rhs=wrap(self.rhs_text(fmt)), result=wrap(
            product_text(self.target, self.target_const, fmt) + " = " + self.rhs_text(fmt))))
        if self.holds:
            lines.append(t["conclusion"].format(conclusion=wrap(self.conclusion(fmt))))
            lines.append(t["exact"].format(count=len(self.steps)))
        elif self.exact:
            lines.append(t["differs"].format(conclusion=wrap(self.conclusion(fmt)), ratio=wrap(self.rhs_text(fmt))))
        else:
            rest = ", ".join(wrap(_pow_text(n, p, fmt)) for n, p in self.rest.items() if p != 0 and n not in self.constants)
            lines.append(t["inexact"].format(rest=rest))
        return "\n\n".join(lines)


def _relations(network_or_specs):
    if hasattr(network_or_specs, "relations_spec"):
        nodes, rels = network_or_specs.nodes_specs, network_or_specs.relations_spec
    else:
        nodes, rels = network_or_specs
    out = []
    for k, r in enumerate(rels):
        if r.get("type", "log") != "log" or callable(r.get("rhs", 0.0)) or not r.get("terms"):
            continue
        terms = {n: _frac(p) for n, p in r["terms"].items() if _frac(p) != 0}
        out.append((r.get("name", f"relation_{k}"), terms, float(r.get("rhs", 0.0))))
    fixed = {n for n, s in nodes.items() if s.get("fixed")}
    return out, fixed


def derive(network_or_specs, equation, keep=(), max_steps=16, max_nodes=200000):
    """The demonstration of `equation` ('S * Equant = Estock', or {node: power} for left over right) from the
    product relations of a network (a GlobalRelationalNetwork, or (nodes_spec, relations_spec)).
    keep: quantities allowed to remain on the right side, with the network's fixed constants.
    Returns a Derivation (d.found, d.exact, d.steps, d.text())."""
    if isinstance(equation, str):
        target, target_const = parse_equation(equation)
        text = equation
    else:
        target, target_const = {n: _frac(p) for n, p in equation.items()}, 0.0
        text = product_text(target) + " = 1"
    rels, fixed = _relations(network_or_specs)
    constants = set(fixed) | set(keep)
    by_node = {}
    for i, (_, terms, _) in enumerate(rels):
        for n in terms:
            by_node.setdefault(n, []).append(i)
    budget = [max_nodes]

    def open_nodes(expr):
        return [n for n, p in expr.items() if p != 0 and n not in constants]

    def search(expr, const, used, depth, path):
        todo = open_nodes(expr)
        if not todo:
            return path, expr, const
        if depth == 0 or budget[0] <= 0:
            return None
        # the quantity with the fewest relations left to replace it, the target's own first
        todo.sort(key=lambda n: (n not in target, len([i for i in by_node.get(n, []) if i not in used])))
        n = todo[0]
        cands = [i for i in by_node.get(n, []) if i not in used]
        # the relations that bring in the fewest new quantities first
        cands.sort(key=lambda i: len([m for m in rels[i][1] if m != n and m not in constants and expr.get(m, 0) == 0]))
        for i in cands:
            budget[0] -= 1
            name, terms, b = rels[i]
            w = expr[n] / terms[n]
            new = dict(expr)
            for m, p in terms.items():
                new[m] = new.get(m, Fraction(0)) - w * p
            new = {m: p for m, p in new.items() if p != 0}
            # node = 10^(b/p_n) * prod others^(-p_m/p_n): the relation solved for the node
            definition = (n, {m: -p / terms[n] for m, p in terms.items() if m != n}, b / float(terms[n]))
            got = search(new, const + float(w) * b, used | {i}, depth - 1,
                         path + [Step(n, name, w, definition, new, const + float(w) * b)])
            if got:
                return got
        return None

    expr0 = dict(target)
    for depth in range(1, max_steps + 1):
        got = search(expr0, target_const, frozenset(), depth, [])
        if got:
            steps, rest, const = got
            return Derivation(text, target, target_const, steps, rest, const, constants, max_steps)
        if budget[0] <= 0:
            break
    return Derivation(text, target, target_const, None, {}, 0.0, constants, max_steps)
