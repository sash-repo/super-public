#!/usr/bin/env python3
"""test_ratfun.py -- checks of the ground field K(mu); run under BOTH pythons."""

from fractions import Fraction as F
import ratfun
import stab
import t3

print("backend:", ratfun.backend())

# ---- field axioms on random elements
mu = ratfun.setup(2)
a, b = mu[0], mu[1]
one = ratfun.const(1)
assert (a + b) * (a - b) == a * a - b * b
assert (a / b) * b == a
assert a - a == 0 and not (a - a)
assert bool(a) and bool(a + one)
assert (one / (a + 1)) * (a + 1) == one
assert ((a * a - 1) / (a - 1)) == a + 1          # cancellation is exact
assert hash((a * a - 1) / (a - 1)) == hash(a + 1)
assert a * 2 == 2 * a and a + F(1, 2) == F(1, 2) + a
assert (a + 1).at([F(3), F(5)]) == 4
assert ((one / a).at([F(0), F(1)])) is None

# ---- the trace records exactly what was inverted
ratfun.trace_reset()
_ = one / (a + 1)
_ = one / (a * b - 3)
ps = [str(p) for p in ratfun.trace_polys()]
assert len(ps) == 2, ps

# ---- linear algebra over K(mu): the generic engine agrees with the special one
mu = ratfun.setup(1)
m = mu[0]
M = [[ratfun.const(1), m], [ratfun.const(2), ratfun.const(3)]]
assert stab.rank_rows([r[:] for r in M]) == (2 if True else 0)
# rank drops exactly on the inverted locus 3 - 2 mu = 0
ratfun.trace_reset()
r = stab.rank_rows([row[:] for row in M])
polys = ratfun.trace_polys()
assert r == 2
bad = [F(3, 2)]
for x in [F(0), F(1), F(5)]:
    num = stab.rank_rows([[F(1), x], [F(2), F(3)]])
    assert num == 2 or x in bad
assert stab.rank_rows([[F(1), F(3, 2)], [F(2), F(3)]]) == 1
assert any(p.eval([F(3, 2)]) == 0 for p in polys), [str(p) for p in polys]

# ---- t3.rref and nullspace over K(mu)
R = t3.rref([[ratfun.const(1), m, ratfun.const(0)],
             [ratfun.const(2), 2 * m, ratfun.const(1)]])
assert len(R) == 2
cols = [{0: ratfun.const(1)}, {0: m}, {0: ratfun.const(0)}]
ns = stab.nullspace(cols, {0})
assert len(ns) == 2
for s in ns:
    assert not (s[0] * ratfun.const(1) + s[1] * m)

# ---- an orbit computation over K(mu) reproduces the numeric one generically
import orb
A = [[[ratfun.const(0), ratfun.const(1)], [ratfun.const(0), ratfun.const(0)]]]
v = [m, ratfun.const(1)]
assert orb.orbit_dim(A, v) == 1
reps, why = t3.orbit_reps([[[F(0), F(1)], [F(0), F(0)]]], 2)
print("numeric orbit certificate:", why)
repsK, whyK = t3.orbit_reps(A, 2)
print("generic orbit certificate:", whyK)
assert (reps is None) == (repsK is None)
if reps is not None:
    assert len(reps) == len(repsK)

print("all checks pass")
