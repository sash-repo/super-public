"""Calibration of the operator engine against operators known in advance."""

import os, sys
from fractions import Fraction as Fr
from contact import *
from gen import *

FAIL, N = [], 0


def chk(name, cond, extra=""):
    global N
    N += 1
    print(("  ok   " if cond else "  FAIL ") + name + (" " + extra if extra else ""))
    if not cond:
        FAIL.append(name)


print("== k_- invariance is automatic ==")
ok = True
for (k1, k2, k3, W) in ((0, 0, 0, 2), (1, 1, 0, 0), (2, 2, 2, 2)):
    terms = term_basis(k1, k2, k3, W)
    M1, M2 = Module(k1, Fr(1, 3)), Module(k2, Fr(-2, 5))
    M3 = Module(k3, target_weight(k1, Fr(1, 3), k2, Fr(-2, 5), k3, W))
    for f in KMINUS:
        for F in monomial_tests(k1, 3):
            for G in monomial_tests(k2, 3):
                if rows_for(terms, f, F, G, M1, M2, M3):
                    ok = False
chk("every term of the ansatz is k_{-2}+k_{-1}-invariant", ok)

print("== operators known in advance ==")

# 1) multiplication
l3, terms, ns = solve(0, Fr(1, 3), 0, Fr(-2, 5), 0, 0)
chk("W=0 on densities: multiplication only", len(ns) == 1 and l3 == Fr(1, 3) - Fr(2, 5),
    "dim=%d, lam3=%s" % (len(ns), l3))

# 2) the contact bracket lives at W=2 on F_{-1} x F_{-1} -> F_{-1}
l3, terms, ns = solve(0, -1, 0, -1, 0, 2)
chk("W=2, lam=mu=-1 gives lam3=-1", l3 == -1)
# is the bracket in the span?  {f,g} = w^{ab}(e_a f)(e_b g) + f R g - (R f) g,
# with w^{12} fixed by the calibration below.
vec = [Fr(0)] * len(terms)
idx = {t: i for i, t in enumerate(terms)}
brk = {((0, 0, 0), (0, 0, 1)): Fr(1), ((0, 0, 1), (0, 0, 0)): Fr(-1)}
for D1, D2 in (((1, 0, 0), (0, 1, 0)), ((0, 1, 0), (1, 0, 0))):
    pass
# read the bracket off directly instead of guessing signs
F = {(0, 0, 0, 0): Fr(1)}
tst = [pmono(a, b, c) for a in range(3) for b in range(3) for c in range(3)
       if a + b + c <= 2]
# express {f,g} in the frame basis by fitting
import itertools
cand = [((0, 0, 0), (0, 0, 1)), ((0, 0, 1), (0, 0, 0)),
        ((1, 0, 0), (0, 1, 0)), ((0, 1, 0), (1, 0, 0))]
rowsA, rhs = [], []
keys = set()
data = []
for f in tst:
    for g in tst:
        vals = {}
        for D1, D2 in cand:
            vals[(D1, D2)] = pmul(apply_D(D1, f), apply_D(D2, g))
        tgt = kbracket(f, g)
        data.append((vals, tgt))
        keys |= set(tgt)
        for v in vals.values():
            keys |= set(v)
allkeys = sorted(keys)
A, b = [], []
for vals, tgt in data:
    for k in allkeys:
        A.append([vals[c].get(k, Fr(0)) for c in cand])
        b.append(tgt.get(k, Fr(0)))
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import solve as lsolve
sol = lsolve(A, b)
chk("the contact bracket is a constant-coefficient frame expression", sol is not None,
    str(sol))
if sol is not None:
    v = [Fr(0)] * len(terms)
    for c, D in zip(sol, cand):
        v[idx[(0, 0, D[0], D[1], 0)]] = c
    # is it in the span of ns?
    from exactlin import rank as lrank
    inspan = lrank([list(x) for x in ns] + [v]) == lrank([list(x) for x in ns])
    chk("the contact bracket is found by the engine", inspan,
        "dim of the W=2 space = %d" % len(ns))

# 3) the transvectant: w^{ab} F_a G_b, T(1,0)xT(1,0) -> T(0,1), W=0
l3, terms, ns = solve(1, 0, 1, 0, 0, 0)
chk("T(1,0)xT(1,0)->T(0,1) at W=0: the symplectic pairing", len(ns) == 1 and l3 == 1,
    "dim=%d, lam3=%s" % (len(ns), l3))

# 4) the symmetric product T(1,0)xT(1,0)->T(2,0), W=0
l3, terms, ns = solve(1, 0, 1, 0, 2, 0)
chk("T(1,0)xT(1,0)->T(2,0) at W=0: the symmetric product", len(ns) == 1 and l3 == 0,
    "dim=%d, lam3=%s" % (len(ns), l3))

# 5) an operator must exist nowhere below its weight
l3, terms, ns = solve(0, 0, 0, 0, 2, 0)
chk("T(0,0)xT(0,0)->T(2,lam) at W=0: nothing", len(ns) == 0,
    "dim=%d" % len(ns))

print("\n%d checks, %d failed" % (N, len(FAIL)))
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
