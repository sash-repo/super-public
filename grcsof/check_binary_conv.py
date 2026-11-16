#!/usr/bin/env python3
"""Decisive validation of the binary invariance convention: three bilinear
operators that are invariant *by construction* (composites of invariant maps)
must come out invariant, and the Koszul-twisted variants must not.

  (1) multiplication            F_{l1,n1} x F_{l2,n2} -> F_{l1+l2, n1+n2}
  (2) (D_xi (x) 1)              needs n1 = -l1  [BLS1 Thm 1.5(ii): B_xi]
  (3) (1 (x) D_xi)              needs n2 = -l2
  (4) (D_eta (x) 1), (1 (x) D_eta)  need n_i = +l_i
"""
from sympy import symbols, S, expand
from invar import eqs_binary, opterm, DOPS
from k12 import D_xi, D_eta

l1, n1, l2, n2 = symbols('lambda_1 nu_1 lambda_2 nu_2')
idop = lambda u: u
ok = True


def test(name, terms, L1, N1, L2, N2, lt, nt, pB, nmax=3):
    global ok
    cs, eqs = eqs_binary(terms, L1, N1, L2, N2, lt, nt, pB, nmax=nmax)
    # every term gets coefficient 1
    res = [expand(e.subs({c: 1 for c in cs})) for e in eqs]
    res = [r for r in res if r != 0]
    good = not res
    print(f"  {'OK  ' if good else 'FAIL'}  {name}"
          + ('' if good else f"   residuals e.g. {res[:2]}"))
    ok = ok and good
    return good


print("Koszul convention  (P (x) Q)(phi,psi) = (-1)^{p(Q)p(phi)} P(phi) Q(psi):")
test("multiplication",
     [('1x1', idop, idop, 0, 0)], l1, n1, l2, n2, l1 + l2, n1 + n2, 0)
test("D_xi (x) 1   on F_{l1,-l1} x F_{l2,n2}",
     [('Dx x 1', D_xi, idop, 1, 0)], l1, -l1, l2, n2, l1 + l2 + 1, -l1 - 1 + n2, 1)
test("1 (x) D_xi   on F_{l1,n1} x F_{l2,-l2}",
     [('1 x Dx', idop, D_xi, 1, 1)], l1, n1, l2, -l2, l1 + l2 + 1, n1 - l2 - 1, 1)
test("D_eta (x) 1  on F_{l1,l1} x F_{l2,n2}",
     [('De x 1', D_eta, idop, 1, 0)], l1, l1, l2, n2, l1 + l2 + 1, l1 + 1 + n2, 1)
test("1 (x) D_eta  on F_{l1,n1} x F_{l2,l2}",
     [('1 x De', idop, D_eta, 1, 1)], l1, n1, l2, l2, l1 + l2 + 1, n1 + l2 + 1, 1)

ok_pos = ok
print("control: the same term WITHOUT the Koszul sign -- MUST fail:")
bad = test("1 (x) D_xi, untwisted",
           [('1 x Dx', idop, D_xi, 1, 0)], l1, n1, l2, -l2, l1 + l2 + 1, n1 - l2 - 1, 1)
ok = ok_pos and not bad

print()
print("CONVENTION VALIDATED" if ok else "CONVENTION STILL WRONG")
