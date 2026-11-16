#!/usr/bin/env python3
"""Section 8, odd Type A: the closed formula of Theorem 8.4 of the paper
checked against an independent solution of the invariance equations.

The Theorem's ansatz is
  J = sum_i A_i X^i (x) X^{k-i-1} Q + sum_i B_i X^i (x) X^{k-i-2} T
    + sum_i E_i X^{i-1} Q (x) X^{k-i} + sum_i F_i X^{i-2} T (x) X^{k-i},
  X = D_xi, Q = D_q, T = d_tau,
and it claims, at generic weights,
  A_i = (-1)^i C(k-1,i) prod_{0<=r<=i-1}(2 l2 -k+r+1) / prod_{0<=r<=i-1}(2 l1 - r) A_0,
  B_i = (k-i-1) A_i,   E_i = i/(k-i) A_i,   F_i = (i-1) E_i.

Here the space is solved for from scratch (bjtypes.dim_sector, i.e. the invariance
equations of classify_m1) and the resulting single vector is compared with the
formula, coefficient by coefficient.
"""
import sys
from sympy import Rational, binomial, simplify, nsimplify, S

from bjtypes import dim_sector, terms_of


def formula(k, L1, L2):
    """the coefficients of the Theorem, normalized by A_0 = 1."""
    A = {}
    for i in range(0, k):
        num = den = S(1)
        for r in range(0, i):
            num *= (2 * L2 - k + r + 1)
            den *= (2 * L1 - r)
        A[i] = (-1) ** i * binomial(k - 1, i) * num / den
    out = {}
    for i in range(0, k):
        out[f"X{i}|X{k-i-1}Q"] = A[i]
    for i in range(0, k - 1):
        out[f"X{i}|X{k-i-2}T"] = (k - i - 1) * A[i]
    E = {}
    for i in range(1, k):
        E[i] = Rational(i, k - i) * A[i]
    E[k] = 2 * L2 / (k - 1 - 2 * L1) * A[k - 1]          # stated separately
    for i in range(1, k + 1):
        out[f"X{i-1}Q|X{k-i}"] = E[i]
    for i in range(2, k + 1):
        out[f"X{i-2}T|X{k-i}"] = (i - 1) * E[i]
    return out


def norm(name):
    """bjtypes' monomial names use X (not X1) for a single D_xi and 1 for none."""
    a, b = name.split('|')
    return f"{a}|{b}"


def canon(s):
    """'X0|X2Q' -> '1|X2Q',  'X1|XQ' -> 'X|XQ'"""
    out = []
    for part in s.split('|'):
        part = part.replace('X0', '')
        part = part.replace('X1', 'X') if not part.startswith('X1') or \
            len(part) < 3 or not part[2:3].isdigit() else part
        out.append(part or '1')
    return '|'.join(out)


if __name__ == '__main__':
    L1 = Rational(7, 3)
    L2 = Rational(11, 5)
    KS = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [3, 4, 5]
    for k in KS:
        d, tt, ns = dim_sector(k, 1, L1, L2)
        print(f"k={k}: dim = {d}", flush=True)
        if d != 1:
            print("   not 1-dimensional, skipping the comparison", flush=True)
            continue
        vec = ns[0]
        sol = {tt[i][0]: vec[i] for i in range(len(tt))}
        fm = {canon(nm): c for nm, c in formula(k, L1, L2).items()}
        # normalize both by the same monomial: the one with the largest |coeff|
        key = 'X0|' + ('X' + str(k - 1) if k - 1 > 1 else ('X' if k - 1 == 1 else '')) + 'Q'
        key = canon(key)
        assert key in sol, (key, sorted(sol))
        assert sol[key] != 0, "A_0 vanishes at these weights"
        s0, f0 = sol[key], fm[key]
        bad = []
        for nm in sorted(set(sol) | set(fm)):
            a = simplify(sol.get(nm, 0) / s0)
            b = simplify(fm.get(nm, 0) / f0)
            if simplify(a - b) != 0:
                bad.append((nm, a, b))
        if bad:
            print(f"   MISMATCH in {len(bad)} of {len(set(sol)|set(fm))} monomials:")
            for nm, a, b in bad[:8]:
                print(f"      {nm}: solved {a}   formula {b}")
        else:
            print(f"   all {len(set(sol)|set(fm))} coefficients agree", flush=True)
