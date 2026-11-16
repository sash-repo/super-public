#!/usr/bin/env python3
"""THE classification: all binary i(osp(2|2))-invariant differential operators

    J : F_{lam_1,nu_1} (x) F_{lam_2,nu_2} --> F_{lam,nu}   over k(1|2),

in the xi-eta realisation, with ALL FOUR weights symbolic.

Because the weights (lam,nu) of the target are forced by the degree s and the
charge w of the operator (lam = lam_1+lam_2-s, nu = nu_1+nu_2-w), it is enough
to run over (s,w).  Invariance under K_t and K_{xi eta} is then automatic, and
invariance under g_- is automatic because the D's supercommute with g_-; so the
equations come from K_{t xi} and K_{t eta} alone (they generate the rest of
i(osp(2|2)) together with g_- and g_0).  Everything found here is verified in
verify_k12.py against ALL EIGHT generators of i(osp(2|2)).
"""
import sys
from sympy import symbols, S, Matrix, expand, simplify, factor, together, cancel
from invar import gen_binary, eqs_binary, matrix_of
from k12 import OSP, kb, div2, d_t
from super12 import parity

l1, n1, l2, n2 = symbols('lambda_1 nu_1 lambda_2 nu_2')
FIELDS = ('txi', 'teta')

print("# preliminary: [K_txi,K_teta] is a nonzero multiple of K_t2, and")
print("#              d_t(f) = div_2(f) = 0 for f = 1, xi, eta")
print("  {t xi, t eta} =", kb(OSP['txi'], OSP['teta']))
for nm in ('1', 'xi', 'eta'):
    print(f"  f={nm:3s}: d_t f = {d_t(OSP[nm])}, div_2 f = {div2(OSP[nm])}")
print()

SMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 5
for s in range(-1, -SMAX - 1, -1):
    for w in (-2, -1, 0, 1, 2):
        terms = gen_binary(s, w)
        if not terms:
            continue
        pB = w % 2
        lt, nt = l1 + l2 - s, n1 + n2 - w
        cs, eqs = eqs_binary(terms, l1, n1, l2, n2, lt, nt, pB,
                             nmax=(-s) // 2 + 2, fields=FIELDS)
        M = matrix_of(eqs, cs)
        ns = M.nullspace()
        names = [t[0] for t in terms]
        print(f"s={s:3d} w={w:+d}  (order {-s}, parity {'odd' if pB else 'even'}): "
              f"{len(names):2d} terms, rank {M.rank()}, dim {len(ns)}"
              f"   -> F_{{{lt}, {nt}}}")
        for v in ns:
            den = 1
            for x in v:
                if x != 0:
                    den = together(x).as_numer_denom()[1] if den == 1 else den
            vv = [cancel(x * den) for x in v]
            g = 0
            print("   {")
            for i, x in enumerate(vv):
                if x != 0:
                    print(f"     {names[i]:24s} : {factor(x)}")
            print("   }")
        sys.stdout.flush()
