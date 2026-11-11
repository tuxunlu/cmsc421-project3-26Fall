"""Public tests for Part 2 (bayesian_network.py)."""

import itertools

from bayesian_network import generate_bayesnet, find_best_overtake_condition, load_data

PART = "2: Bayesian network"
T, F = True, False
VARIABLES = ("MuchFaster", "Early", "Overtake", "Crash", "Win")


def points(n, part):
    def deco(fn):
        fn.points, fn.part = n, part
        return fn
    return deco


def seeds(base, per_case=1):
    return [base + i for i in range(per_case)]


@points(2, PART)
def test_structure():
    bn = generate_bayesnet()
    assert set(bn.variables) == set(VARIABLES), "variables must be named " + ", ".join(VARIABLES)
    order = list(bn.variables)
    for v in VARIABLES:
        for p in bn.variable_node(v).parents:
            assert order.index(p) < order.index(v), "parents must come before their children"
    assert not bn.variable_node("MuchFaster").parents and not bn.variable_node("Early").parents, \
        "MuchFaster and Early are independent and have no parents"


@points(3, PART)
def test_cpts_are_probabilities():
    data = load_data()
    bn = generate_bayesnet()
    for v in VARIABLES:
        node = bn.variable_node(v)
        for values in itertools.product((T, F), repeat=len(node.parents)):
            got = node.p(True, dict(zip(node.parents, values)))
            assert 0.0 <= got <= 1.0, "P({} = T | ...) must be a probability".format(v)
    expect = sum(d.muchfaster for d in data) / len(data)
    got = bn.variable_node("MuchFaster").p(True, {})
    assert abs(got - expect) < 1e-9, "P(MuchFaster = T) = {:.4f}, expected the frequency {:.4f}".format(got, expect)


@points(3, PART)
def test_best_overtake_condition():
    best = tuple(find_best_overtake_condition(generate_bayesnet()))
    assert len(best) == 2 and all(isinstance(x, bool) for x in best), \
        "return the condition as a pair (MuchFaster, Early) of booleans"
