#!/usr/bin/env python3
"""Brute-force test of the question of the text on the associativity of the products:

    f * g := sum_n r_n [[f,g]]_{2n}    on  F_. = (+)_w F_w        (even brackets)
    f * g := sum_n s_n [[f,g]]_{2n+1}  on  Pi(F_.)                (odd  brackets)

should be associative; conjecturally r_n = s_n = 1.

Each [[.,.]]_k is defined only up to a scalar depending on the weights of its
two arguments, so the well-posed question is

    does there exist a normalisation  lam_k(w1,w2)  (lam_0 = 1) making
    f*g = sum_k lam_k(w1,w2) B_k(f,g)  associative,

where B_k is the invariant bracket of order k produced by grc_ops.py,
normalised canonically (coefficient of  f . d_t^n g  equal to 1).  Associativity
in total order N*step is an inhomogeneous LINEAR system for the four unknowns
lam_N(w1,w2), lam_N(w1+w2,w3), lam_N(w2,w3), lam_N(w1,w2+w3) once the lower
lam's are known, so each order is decided exactly.

Weights are taken in (1/P)Z with P = 31 prime and small numerators, so that no
mu_i = -w_i is an integer and every bracket space is 1-dimensional (generic).

Result (see the printout):  lam_n(u,v) = binom(u+n-1, n) = (u)_n / n!,
forced uniquely from order 4 on -- the exact analogue of the classical
Rankin-Cohen / Eholzer normalisation.

No dependencies:  python3 check_assoc.py
"""
from fractions import Fraction as F

from grc_ops import invariant_ops, apply_op, mul

P = 31
Z = F(0)


# ------------------------------------------------------- normalised brackets
_cache = {}


def bracket(k, w1, w2, sigma=1):
    key = (k, w1, w2, sigma)
    if key not in _cache:
        keys, B = invariant_ops(k, w1, w2, sigma)
        res = None
        if len(B) == 1:
            piv = (0, 0, 0, 0, k // 2) if k % 2 == 0 else (0, 0, 0, 1, k // 2)
            c = B[0][keys.index(piv)]
            if c:
                res = (keys, [x / c for x in B[0]])
        _cache[key] = res
    return _cache[key]


def B_apply(k, w1, w2, f, g, sigma=1):
    if k == 0:
        return mul(f, g)
    br = bracket(k, w1, w2, sigma)
    return None if br is None else apply_op(br[1], br[0], f, g)


def monos(A):
    return [{(a, e): F(1)} for a in range(A + 1) for e in (0, 1)]


# ------------------------------------------- incremental echelon accumulator
class Echelon:
    def __init__(self, n):
        self.n = n
        self.piv = {}
        self.bad = False

    def add(self, row, rhs):
        r = list(row) + [rhs]
        for c in sorted(self.piv):
            if r[c]:
                f0, pr = r[c], self.piv[c]
                r = [x - f0 * y for x, y in zip(r, pr)]
        for c in range(self.n):
            if r[c]:
                pv = r[c]
                r = [x / pv for x in r]
                for c2 in list(self.piv):
                    pr = self.piv[c2]
                    if pr[c]:
                        f0 = pr[c]
                        self.piv[c2] = [x - f0 * y for x, y in zip(pr, r)]
                self.piv[c] = r
                return
        if r[self.n]:
            self.bad = True

    def solution(self):
        if self.bad:
            return None
        part = [Z] * self.n
        for c, r in self.piv.items():
            part[c] = r[self.n]
        free = [c for c in range(self.n) if c not in self.piv]
        basis = []
        for fc in free:
            v = [Z] * self.n
            v[fc] = F(1)
            for c, r in self.piv.items():
                v[c] = -r[fc]
            basis.append(v)
        return part, basis


# --------------------------------------------------- associativity at order N
def solve_order(N, step, w1, w2, w3, lam, sigma=1, A=None):
    """Associativity of  f*g = sum_p lam[p](.,.) B_{p*step}(f,g)  in total order
    N*step, with lam[p] known for p < N and lam[N] unknown at the four pairs
    (w1,w2), (w1+w2,w3), (w2,w3), (w1,w2+w3).  Returns (pairs, Echelon)."""
    pairs = [(w1, w2), (w1 + w2, w3), (w2, w3), (w1, w2 + w3)]
    E = Echelon(4)
    if A is None:
        A = N * step + 2
    tests = monos(A)
    for f in tests:
        for g in tests:
            for h in tests:
                lin = [{}, {}, {}, {}]
                const = {}
                ok = True
                for p in range(N + 1):
                    q = N - p
                    u = B_apply(p * step, w1, w2, f, g, sigma)
                    v = None if u is None else \
                        B_apply(q * step, w1 + w2 + p * step, w3, u, h, sigma)
                    u2 = B_apply(p * step, w2, w3, g, h, sigma)
                    v2 = None if u2 is None else \
                        B_apply(q * step, w1, w2 + w3 + p * step, f, u2, sigma)
                    if v is None or v2 is None:
                        ok = False
                        break
                    if p == N:                       # lam_N(w1,w2) . lam_0
                        for m, c in v.items():
                            lin[0][m] = lin[0].get(m, Z) + c
                        for m, c in v2.items():
                            lin[2][m] = lin[2].get(m, Z) - c
                    elif q == N:                     # lam_0 . lam_N(w1+w2,w3)
                        for m, c in v.items():
                            lin[1][m] = lin[1].get(m, Z) + c
                        for m, c in v2.items():
                            lin[3][m] = lin[3].get(m, Z) - c
                    else:
                        c0 = lam[p](w1, w2) * lam[q](w1 + w2 + p * step, w3)
                        for m, c in v.items():
                            const[m] = const.get(m, Z) + c0 * c
                        c0 = lam[p](w2, w3) * lam[q](w1, w2 + w3 + p * step)
                        for m, c in v2.items():
                            const[m] = const.get(m, Z) - c0 * c
                if not ok:
                    continue
                for m in set(const) | {m for d in lin for m in d}:
                    row = [d.get(m, Z) for d in lin]
                    if any(row) or const.get(m):
                        E.add(row, -const.get(m, Z))
    return pairs, E


def binom_rising(n):
    """lam_n(u,v) = (u)(u+1)...(u+n-1) / n!"""
    def f(u, v):
        r = F(1)
        for i in range(n):
            r *= (u + i)
        for i in range(1, n + 1):
            r /= i
        return r
    return f


# ==========================================================================
if __name__ == "__main__":
    W = lambda a: F(a, P)

    print("=" * 76)
    print("EVEN product on F_. :   f*g = sum_n lam_n(w1,w2) [[f,g]]_{2n}")
    print("=" * 76)
    lam = {n: binom_rising(n) for n in range(4)}
    triples = [(1, 2, 3), (1, 1, 1), (2, 3, 5), (1, 3, 7), (3, 4, 5)]
    for N in (1, 2, 3):
        print(f"\n  --- total order {2*N} ---")
        for (a, b, c) in triples:
            pairs, E = solve_order(N, 2, W(a), W(b), W(c), lam)
            sol = E.solution()
            if sol is None:
                print(f"    w=({a},{b},{c})/{P}: INCONSISTENT -- no associative product")
                continue
            part, ns = sol
            pred = [lam[N](u, v) for (u, v) in pairs]
            if not ns:
                agree = (part == pred)
                print(f"    w=({a},{b},{c})/{P}: unique solution; "
                      f"equals (u)_{N}/{N}!  -> {agree}")
                if not agree:
                    print("       computed :", [str(x) for x in part])
                    print("       predicted:", [str(x) for x in pred])
            else:
                inspan = True   # check predicted lies in part + span(ns)
                d = [x - y for x, y in zip(pred, part)]
                E2 = Echelon(len(ns))
                print(f"    w=({a},{b},{c})/{P}: solution space dim {len(ns)}; "
                      f"predicted value consistent: "
                      f"{any(all(x == y for x, y in zip(pred, [p + t*n for p, n in zip(part, ns[0])])) for t in (F(1),)) or 'see below'}")
                print("       computed particular:", [str(x) for x in part])
                print("       homogeneous dir    :", [str(x) for x in ns[0]])
                print("       predicted          :", [str(x) for x in pred])

    print("\n" + "=" * 76)
    print("ODD product on Pi(F_.) :  f*g = sum_n s_n [[f,g]]_{2n+1}")
    print("=" * 76)
    print("  lowest total order 2:  the condition is  lam(w1,w2) lam(w1+w2+1,w3)")
    print("  B_1(B_1(f,g),h) = eps lam(w2,w3) lam(w1,w2+w3+1) B_1(f,B_1(g,h)),")
    print("  eps = +-1 and sigma = +-1 covering all Pi-conventions.")
    for sigma in (1, -1):
        for eps in (1, -1):
            ok_all = True
            for (a, b, c) in triples:
                w1, w2, w3 = W(a), W(b), W(c)
                E = Echelon(4)      # unknowns: the four lam(.,.) -- but the
                # condition is bilinear; we test it as a linear condition on the
                # products L = lam(w1,w2)lam(w1+w2+1,w3) and R = lam(w2,w3)lam(w1,w2+w3+1)
                E = Echelon(2)
                for f in monos(4):
                    for g in monos(4):
                        for h in monos(4):
                            u = B_apply(1, w1, w2, f, g, sigma)
                            v = B_apply(1, w1 + w2 + 1, w3, u, h, sigma)
                            u2 = B_apply(1, w2, w3, g, h, sigma)
                            v2 = B_apply(1, w1, w2 + w3 + 1, f, u2, sigma)
                            if v is None or v2 is None:
                                continue
                            for m in set(v) | set(v2):
                                E.add([v.get(m, Z), -eps * v2.get(m, Z)], Z)
                sol = E.solution()
                nz = sol is not None and len(sol[1]) >= 1 and \
                    any(all(x != 0 for x in b) for b in sol[1])
                if not nz:
                    ok_all = False
            print(f"    sigma={sigma:+d} eps={eps:+d}:  "
                  f"{'a non-degenerate solution exists on all triples' if ok_all else 'NO non-degenerate solution'}")
