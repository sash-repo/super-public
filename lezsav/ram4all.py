#!/usr/bin/env python3
"""ram4all.py -- Conjecture Statement 9.2.2 for EVERY weight.

On the double cover, generating functions t^a theta_S, a in (1/2)Z, weight w=2a+|S|.
At weight w the 16 monomials are t^{(w-|S|)/2} theta_S; the conditions Xi=XiBar=0
(n4shift.py) have coefficients polynomial in w.  M (eq. (Monf)) acts on
t^a theta_S by (-1)^{w-|S|} (-1)^{[4 in S]}, so on a parity class of w it is a fixed
coordinate projection.  We treat w as a SYMBOL within each parity class: generic rank
over Q(i)(w), and the finitely many exceptional w = roots of det(A^dag A).
Cross-checked against n4shift.Xi/XiBar at numeric weights.
"""
import itertools, sys
import sympy as sp
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
w = sp.Symbol("w", real=True); t = sp.Symbol("t", positive=True)
I = sp.I
IDX = (1, 2, 3, 4)

def add(f, g):
    h = dict(f)
    for k, c in g.items():
        h[k] = sp.expand(h.get(k, 0) + c)
    return {k: c for k, c in h.items() if c != 0}

def smul(c, f):
    return {k: sp.expand(c * v) for k, v in f.items() if sp.expand(c * v) != 0}

def d_odd(i, f):
    out = {}
    for S, c in f.items():
        if i in S:
            p = S.index(i)
            T = S[:p] + S[p + 1:]
            out = add(out, {T: (-1) ** p * c})
    return out

def mul_theta(i, f):
    out = {}
    for S, c in f.items():
        if i in S:
            continue
        T = tuple(sorted(S + (i,)))
        p = T.index(i)
        out = add(out, {T: (-1) ** p * c})
    return out

def d_t(f):
    return {S: sp.diff(c, t) for S, c in f.items() if sp.diff(c, t) != 0}

def D(i, f):
    return add(d_odd(i, f), smul(-1, mul_theta(i, d_t(f))))

def cD(a, f):
    j = 2 * a - 1
    return add(D(j, f), smul(-I, D(j + 1, f)))

def bD(a, f):
    j = 2 * a - 1
    return add(D(j, f), smul(I, D(j + 1, f)))

def Xi(f): return smul(sp.Rational(1, 2), cD(1, cD(2, f)))
def XiBar(f): return smul(sp.Rational(1, 2), bD(1, bD(2, f)))

SUBSETS = [S for k in range(5) for S in itertools.combinations(IDX, k)]

def mono(S, wv=w):
    return {S: t ** ((wv - len(S)) / sp.Integer(2))}

def Msign(S, wpar):
    return (-1) ** ((wpar - len(S)) % 2) * (-1 if 4 in S else 1)

def matrix(cols, wv=w):
    rows = {}
    for j, S in enumerate(cols):
        for tag, op in (("X", Xi), ("B", XiBar)):
            for T, c in op(mono(S, wv)).items():
                # strip the t-power: all terms of weight wv
                c = sp.simplify(c / t ** ((wv - 2 - len(T)) / sp.Integer(2)))
                rows.setdefault((tag, T), {})[j] = c
    keys = sorted(rows, key=str)
    return sp.Matrix([[rows[k].get(j, 0) for j in range(len(cols))] for k in keys])

def analyse(wpar):
    cols = [S for S in SUBSETS if Msign(S, wpar) > 0]
    A = matrix(cols)
    Ad = A.conjugate().T  # w real
    G = sp.simplify(Ad * A)
    det = sp.factor(G.det())
    print("parity", wpar, "columns", len(cols), "rows", A.rows)
    print("  det(A^dag A) =", det)
    roots = sp.real_roots(sp.Poly(sp.numer(det), w)) if det != 0 else "identically 0"
    print("  roots:", roots)
    return cols, roots

def kernel_at(wv):
    wpar = wv % 2
    cols = [S for S in SUBSETS if Msign(S, wpar) > 0]
    A = matrix(cols, sp.Integer(wv))
    ns = A.nullspace()
    out = []
    for v in ns:
        f = {cols[j]: sp.nsimplify(v[j]) for j in range(len(cols)) if v[j] != 0}
        out.append(f)
    return out

def full_dim(wv):
    A = matrix(SUBSETS, sp.Integer(wv))
    return 16 - A.rank()

if __name__ == "__main__":
    # cross-check with n4shift at numeric weights: full value comparison
    from fractions import Fraction as Q
    from n4emb import mono as m0
    from n4shift import Xi as X0, XiBar as B0
    def conv(R0):
        out = {}
        for (aa, T), c in R0.items():
            out[T] = sp.nsimplify(str(c).replace("i", "*I")) if False else c
        return out
    bad = n = 0
    for wv in range(-4, 6):
        for S in SUBSETS:
            a = Q(wv - len(S), 2)
            for op0, op1 in ((X0, Xi), (B0, XiBar)):
                n += 1
                R0 = op0(m0(a, S)); R1 = op1(mono(S, sp.Integer(wv)))
                k1 = {}
                for T, c in R1.items():
                    e = sp.Rational(wv - 2 - len(T), 2)
                    k1[(e, T)] = sp.simplify(c / t ** e)
                k0 = {}
                for (aa, T), c in R0.items():
                    k0[(sp.Rational(aa.numerator, aa.denominator), T)] = sp.Rational(c.re.numerator, c.re.denominator) + I * sp.Rational(c.im.numerator, c.im.denominator)
                if set(k0) != set(k1) or any(sp.simplify(k0[k] - k1[k]) != 0 for k in k0):
                    bad += 1
    print("cross-check vs n4shift:", n, "pairs,", bad, "mismatches")
    for p in (0, 1):
        cols, roots = analyse(p)
    print("generic full kernel dims (no M):", {wv: full_dim(wv) for wv in range(-6, 9)})
    for wv in range(-6, 9):
        k = kernel_at(wv)
        if k:
            print("w=%d M-invariant kernel:" % wv, k)

def full_generic():
    """Without M: the kernel has dimension 8 for every weight but w=2, where it is 9."""
    A = matrix(SUBSETS)
    r = A.rank(simplify=True)
    print("generic rank of the 16-column system:", r)
    # exceptional w: rank drop of A, via the 8x8 minors of a fraction-free echelon form
    E = A.echelon_form(simplify=True)
    piv = [next((E[i, j] for j in range(E.cols) if E[i, j] != 0), 0) for i in range(E.rows)]
    piv = [sp.factor(p) for p in piv if p != 0]
    print("echelon pivots:", piv)

if __name__ == "__main__" and "--full" in sys.argv:
    full_generic()
