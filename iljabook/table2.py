#!/usr/bin/env python3
"""The graded components of the supersymmetry of four equations (Table of Statement 4A.6.1) and the
two wider definitions (Remark 4A.6.2):
dim of {X in g_m, coefficients polynomial of degree <= d in the even coordinates of E, [X,Q]=0}
(and [X,Q] in K Q, [X,Q] in O Q for the last equation)."""
import itertools, sympy as sp
import flint

def monos(vars_, d):
    out = []
    for deg in range(d + 1):
        for c in itertools.combinations_with_replacement(vars_, deg):
            out.append(sp.Mul(*c))
    return out

class SAlg:
    """elements: dict {tuple(sorted theta indices): sympy expr}"""
    def __init__(self, ev, nth):
        self.ev, self.n = ev, nth
    @staticmethod
    def mul_mono(I, J):
        if set(I) & set(J): return 0, None
        L = list(I) + list(J); s = 1
        # bubble sort sign
        L2 = L[:]
        for i in range(len(L2)):
            for j in range(len(L2) - 1 - i):
                if L2[j] > L2[j + 1]: L2[j], L2[j + 1] = L2[j + 1], L2[j]; s = -s
        return s, tuple(L2)
    def mul(self, a, b):
        r = {}
        for I, f in a.items():
            for J, g in b.items():
                s, K = self.mul_mono(I, J)
                if s: r[K] = sp.expand(r.get(K, 0) + s * f * g)
        return {k: v for k, v in r.items() if v != 0}
    def add(self, a, b, c=1):
        r = dict(a)
        for k, v in b.items(): r[k] = sp.expand(r.get(k, 0) + c * v)
        return {k: v for k, v in r.items() if v != 0}

def apply(A, X, par, elem):
    """X: dict gen -> element (gen = ('e',var) or ('t',i)); par = parity of X; elem: element"""
    res = {}
    for I, f in elem.items():
        # X(f) * theta_I
        for v in A.ev:
            df = sp.diff(f, v)
            if df != 0:
                res = A.add(res, A.mul(A.mul(X[('e', v)], {(): df}), {I: 1}))
        # f * X(theta_I)
        for k, i in enumerate(I):
            sign = (-1) ** (par * k)
            left = {I[:k]: 1}; right = {I[k + 1:]: 1}
            term = A.mul(A.mul(left, X[('t', i)]), right)
            res = A.add(res, A.mul({(): f}, term), sign)
    return res

def gens(A):
    return [('e', v) for v in A.ev] + [('t', i) for i in range(A.n)]

def gen_elem(g):
    return {(): g[1]} if g[0] == 'e' else {(g[1],): 1}

def bracket_vals(A, X, px, Q):
    out = {}
    for g in gens(A):
        e = gen_elem(g)
        a = apply(A, X, px, apply(A, Q, 1, e))
        b = apply(A, Q, 1, apply(A, X, px, e))
        out[g] = A.add(a, b, -((-1) ** px))
    return out

def run(ev, nth, Qd, m, d, mode="cent"):
    A = SAlg(ev, nth)
    Q = {g: Qd.get(g, {}) for g in gens(A)}
    ms = monos(ev, d)
    unknowns = []; X = {g: {} for g in gens(A)}
    syms = []
    for g in gens(A):
        tdeg = m if g[0] == 'e' else m + 1
        if tdeg < 0 or tdeg > nth: continue
        for I in itertools.combinations(range(nth), tdeg):
            for mo in ms:
                c = sp.Symbol('c%d' % len(syms)); syms.append(c)
                X[g] = A.add(X[g], {I: c * mo})
    if not syms: return 0, 0
    px = m % 2
    B = bracket_vals(A, X, px, Q)
    extra = []
    if mode == "line":   # [X,Q] = lam Q
        lam = sp.Symbol('lam'); extra = [lam]
        for g in gens(A): B[g] = A.add(B[g], Q[g], -lam)
    if mode == "module":  # [X,Q] = F Q, F in O of theta-degree m (even coeff deg <= d+? ) -- F arbitrary of degree m: coefficients up to degree d+1
        for I in (itertools.combinations(range(nth), m) if m >= 0 else []):
            for mo in monos(ev, d + 1):
                c = sp.Symbol('f%d' % len(extra)); extra.append(c)
                F = {I: c * mo}
                for g in gens(A): B[g] = A.add(B[g], A.mul(F, Q[g]), -1)
    allv = syms + extra
    rows = []
    for g in gens(A):
        for I, f in B[g].items():
            P = sp.Poly(f, *ev) if ev else None
            for _, coeff in sp.Poly(f, *ev).terms():
                rows.append([coeff.coeff(v) for v in allv])
    if not rows: return len(syms), len(syms)
    M = flint.fmpq_mat(len(rows), len(allv), [flint.fmpq(int(sp.fraction(x)[0]), int(sp.fraction(x)[1])) for r in rows for x in r])
    # kernel dimension projected to the X-variables: dim ker - dim(ker cap {X=0})
    rk = M.rank()
    kdim = len(allv) - rk
    if extra:
        # solutions with X = 0: F Q = 0 or lam Q = 0 -> only zero (Q has independent components), check
        Msub = flint.fmpq_mat(len(rows), len(extra), [flint.fmpq(int(sp.fraction(r[len(syms) + j])[0]), int(sp.fraction(r[len(syms) + j])[1])) for r in rows for j in range(len(extra))])
        kdim -= len(extra) - Msub.rank()
    return kdim, len(syms)

x, y, u, p, q, p1, p2 = sp.symbols('x y u p q p1 p2')
E = {
 "u''=0": ([x, u, p], 1, {('e', x): {(0,): 1}, ('e', u): {(0,): p}}),
 "u'''=0": ([x, u, p1, p2], 1, {('e', x): {(0,): 1}, ('e', u): {(0,): p1}, ('e', p1): {(0,): p2}}),
 "u_x=u_y=u": ([x, y, u], 2, {('e', x): {(0,): 1}, ('e', y): {(1,): 1}, ('e', u): {(0,): u, (1,): u}}),
 "u_xx=u_xy=u_yy=0": ([x, y, u, p, q], 2, {('e', x): {(0,): 1}, ('e', y): {(1,): 1}, ('e', u): {(0,): p, (1,): q}}),
}
import sys
for d in (1, 2):
    print("degree <=", d)
    for name, (ev, nth, Qd) in E.items():
        row = []
        for m in range(-1, nth + 1):
            a, b = run(ev, nth, Qd, m, d)
            row.append("%d of %d" % (a, b))
        print("  %-20s %s" % (name, " | ".join(row))); sys.stdout.flush()
ev, nth, Qd = E["u_xx=u_xy=u_yy=0"]
for mode in ("line", "module"):
    print(mode, [run(ev, nth, Qd, m, 2, mode)[0] for m in (-1, 0, 1, 2)])
