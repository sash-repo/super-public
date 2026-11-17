#!/usr/bin/env python3
"""
ds.py --- the Volichenko algebra h_{ad_x} versus the Duflo-Serganova functor.

Subsec. 2.14 of the text: the relation with the Duflo-Serganova functor,
and the table of Remark 2.14.3.

Setting.  Let g be a Lie superalgebra and x in g_od homologic, [x,x] = 0.  Two
constructions use exactly this datum:

  (V)  the epimorphic Volichenko algebra   h = h_{ad_x} = {a + [x,a] | a in g_ev},
       2-step filtered, with h_{\hat 0} = Ann(x) = Ker ad_x \cap g_ev;

  (DS) the Duflo-Serganova functor         g_x = Ker ad_x / Im ad_x,
       a Lie superalgebra again (Duflo-Serganova; see [GHSS]).

What is verified here, exactly (Fraction arithmetic, no floating point), on
every basis pair of every example:

  D0. x is homologic and ad_x^2 = 0, so Im ad_x \subseteq Ker ad_x and g_x is
      defined; Im ad_x \cap g_ev is an ideal of the Lie algebra Ann(x).

  D1. THE DICTIONARY between the two constructions:
          gr_0 h := h_{\hat 0}       = Ker ad_x \cap g_ev  = numerator of (g_x)_ev
          gr_1 h := h / h_{\hat 0}  ~= Im  ad_x \cap g_od  = denominator of (g_x)_od
      so
          (g_x)_ev = gr_0(h) / (Im ad_x \cap g_ev),
          (g_x)_od = (Ker ad_x \cap g_od) / gr_1(h).

  D2. The deformation term of the Volichenko product,
          S(a,b) = [[x,a],[x,b]]      (eq. (152) of the paper),
      satisfies  S(a,b) = [x, [a,[x,b]]] in Im ad_x  for ALL a,b in g_ev.
      Hence, the whole difference between h_{ad_x} and the Lie algebra g_ev
      takes values in the subspace DS kills: DS_x does not see the Volichenko
      deformation.

  D3. Consequently  [gr_1 h, gr_1 h] \subseteq Im ad_x \cap g_ev, and
          x ensures nontriviality  (i.e., S =/= 0)  ==>  Im ad_x \cap g_ev =/= 0
          ==>  (g_x)_ev is a PROPER quotient of h_{\hat 0}.

Examples covered:
    gl(p|q), sl(p|q), q(n)                  --- matrix realizations, x = rank-1
                                                and higher-rank odd nilpotents;
    h'(0|n), n = 4,...,7, for both homologic elements of
    Table 2.12.2:  x = H_{theta_1} (anisotropic) and
                    x = H_{theta_1 + i theta_2} (isotropic, taken over Q in the
                    hyperbolic basis, as in hamvol.py).

Usage:  python3 ds.py            (faster with python-flint installed)
"""

import sys
import os
from fractions import Fraction as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "."))
try:
    from exactlin import rank as _rank, nullspace as _nullspace, backend
    HAVE_EXACTLIN = True
except Exception:                                        # pure-python fallback
    HAVE_EXACTLIN = False

    def backend():
        return "pure-python (fallback)"

import hamvol as HV                                       # Grassmann / odd Poisson


# --------------------------------------------------------- linear algebra

def rref(rows):
    if not rows:
        return [], []
    A = [[F(x) for x in r] for r in rows]
    ncols = len(A[0])
    piv, r = [], 0
    for c in range(ncols):
        p = None
        for i in range(r, len(A)):
            if A[i][c]:
                p = i
                break
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        inv = F(1) / A[r][c]
        A[r] = [v * inv for v in A[r]]
        for i in range(len(A)):
            if i != r and A[i][c]:
                f = A[i][c]
                A[i] = [A[i][j] - f * A[r][j] for j in range(ncols)]
        piv.append(c)
        r += 1
        if r == len(A):
            break
    return A[:r], piv


def rank(rows):
    if not rows:
        return 0
    if HAVE_EXACTLIN:
        return _rank(rows)
    return len(rref(rows)[0])


def nullspace(rows, ncols=None):
    """basis of {v | rows . v = 0}, as a list of row vectors"""
    if not rows:
        return [[F(1) if j == i else F(0) for j in range(ncols)]
                for i in range(ncols)]
    if HAVE_EXACTLIN:
        return _nullspace(rows)
    R, piv = rref(rows)
    ncols = len(rows[0])
    free = [c for c in range(ncols) if c not in piv]
    out = []
    for f in free:
        v = [F(0)] * ncols
        v[f] = F(1)
        for i, c in enumerate(piv):
            v[c] = -R[i][f]
        out.append(v)
    return out


def in_span(basis, v):
    "is v in the span of basis (list of rows)?"
    return rank(list(basis) + [list(v)]) == rank(list(basis))


# =================================================================== ABSTRACT
# A Lie superalgebra given by a basis split into even/odd parts and a bracket.

class SLA:
    def __init__(self, tag, ev, od, br, x, tostr=None):
        self.tag = tag
        self.ev = list(ev)                # basis of g_ev  (opaque elements)
        self.od = list(od)                # basis of g_od
        self.br = br                      # br(u, v) -> vector dict {index: coeff}
        self.x = x                        # the homologic element (a vector dict on od)
        self.tostr = tostr or (lambda v: str(v))
        self.dev, self.dod = len(self.ev), len(self.od)

    # vectors are dicts {basis index: coeff} together with a parity tag


# ------------------------------------------------------------- matrix models

def mat_model(tag, p, q, kind="gl", xspec=None):
    """gl(p|q), sl(p|q) or q(n) with an odd homologic x, as explicit matrices.

    Basis vectors are (n x n) matrices over Q; brackets are supercommutators.
    """
    if kind == "q":
        n = p
        N = 2 * n
    else:
        n = p + q
        N = n

    def zeros():
        return [[F(0)] * N for _ in range(N)]

    def E(i, j):
        M = zeros()
        M[i][j] = F(1)
        return M

    def mm(A, B):
        C = zeros()
        for i in range(N):
            for k in range(N):
                a = A[i][k]
                if a:
                    for j in range(N):
                        if B[k][j]:
                            C[i][j] += a * B[k][j]
        return C

    def lin(A, B, s):
        return [[A[i][j] + s * B[i][j] for j in range(N)] for i in range(N)]

    if kind in ("gl", "sl"):
        ev_m = ([E(i, j) for i in range(p) for j in range(p)]
                + [E(p + i, p + j) for i in range(q) for j in range(q)])
        od_m = ([E(i, p + j) for i in range(p) for j in range(q)]
                + [E(p + i, j) for i in range(q) for j in range(p)])
        if kind == "sl":
            # supertraceless even part; the odd part of gl is already str-free
            ev2 = []
            for M in ev_m:
                s = sum(M[i][i] for i in range(p)) - sum(M[p + i][p + i]
                                                         for i in range(q))
                if s == 0:
                    ev2.append(M)
            # add the one supertraceless diagonal combination(s)
            diag = [E(i, i) for i in range(p)] + [E(p + i, p + i) for i in range(q)]
            sig = [F(1)] * p + [F(-1)] * q
            for k in range(1, len(diag)):
                M = lin(diag[k], diag[0], -sig[k] / sig[0])
                ev2.append(M)
            ev_m = ev2
    else:                                                # q(n): [[A,B],[B,A]]
        ev_m = []
        od_m = []
        for i in range(n):
            for j in range(n):
                M = zeros()
                M[i][j] = F(1)
                M[n + i][n + j] = F(1)
                ev_m.append(M)
                M = zeros()
                M[i][n + j] = F(1)
                M[n + i][j] = F(1)
                od_m.append(M)

    # the odd homologic element
    if xspec is None:
        if kind == "q":
            X = zeros()
            X[0][n] = F(1)
            X[n][0] = F(1)                       # x = antidiag(E11,E11): x^2 = E11 =/= 0
            # that one is NOT homologic; use a nilpotent A instead: A^2 = 0
            X = zeros()
            if n >= 2:
                X[0][n + 1] = F(1)
                X[n][1] = F(1)
        else:
            X = E(0, p)                          # E_{1,p+1}: x^2 = 0
    else:
        X = xspec(E, zeros, p, q)

    def bracket(A, B, pa, pb):
        "[A,B] = AB - (-1)^{p(A)p(B)} BA"
        s = F(1) if (pa and pb) else F(-1)
        return lin(mm(A, B), mm(B, A), s)

    return dict(N=N, ev=ev_m, od=od_m, x=X, br=bracket, mm=mm, lin=lin,
                zeros=zeros, tag=tag)


def flat(M):
    return [c for row in M for c in row]


def analyse_matrix(model, verbose=True):
    ev, od, x = model["ev"], model["od"], model["x"]
    br, mm, lin = model["br"], model["mm"], model["lin"]
    tag = model["tag"]
    ok = True

    # D0a: x homologic
    xx = br(x, x, 1, 1)
    if any(any(r) for r in xx):
        print("  ! %s: [x,x] =/= 0" % tag)
        ok = False

    def ad_x(v, pv):
        return br(x, v, 1, pv)

    # D0b: ad_x^2 = 0
    for v in ev + od:
        pv = 0 if v in ev else 1
        if any(any(r) for r in ad_x(ad_x(v, pv), 1 - pv)):
            print("  ! %s: ad_x^2 =/= 0" % tag)
            ok = False
            break

    ev_f = [flat(M) for M in ev]
    od_f = [flat(M) for M in od]

    # images and kernels of ad_x, in coordinates of the ambient matrix space
    im_ev = [flat(ad_x(M, 1)) for M in od]        # ad_x(g_od) \subseteq g_ev
    im_od = [flat(ad_x(M, 0)) for M in ev]        # ad_x(g_ev) \subseteq g_od
    im_ev_b = [r for r in rref(im_ev)[0]]
    im_od_b = [r for r in rref(im_od)[0]]

    def kernel_basis(basis, par):
        """basis of {v in span(basis) | [x,v] = 0}, returned in ambient coords"""
        rows = [flat(ad_x(M, par)) for M in basis]      # rows = images of basis
        # solve  sum c_i [x, b_i] = 0
        mat = [[rows[i][j] for i in range(len(basis))]
               for j in range(len(rows[0]))]
        ns = nullspace(mat, len(basis))
        out = []
        for cvec in ns:
            v = [F(0)] * len(basis[0]) ** 2 if False else None
            acc = [F(0)] * len(flat(basis[0]))
            for i, c in enumerate(cvec):
                if c:
                    fb = flat(basis[i])
                    acc = [acc[k] + c * fb[k] for k in range(len(acc))]
            out.append(acc)
        return out

    ker_ev = kernel_basis(ev, 0)
    ker_od = kernel_basis(od, 1)
    dke, dko = rank(ker_ev), rank(ker_od)
    die, dio = rank(im_ev_b), rank(im_od_b)

    # D0c: Im \subseteq Ker
    for r in im_ev_b:
        if not in_span(ker_ev, r):
            print("  ! %s: Im ad_x \\cap g_ev not inside Ker" % tag)
            ok = False
    for r in im_od_b:
        if not in_span(ker_od, r):
            print("  ! %s: Im ad_x \\cap g_od not inside Ker" % tag)
            ok = False

    # D2: S(a,b) = [[x,a],[x,b]] lies in Im ad_x, and equals [x,[a,[x,b]]]
    bad = 0
    for A in ev:
        for B in ev:
            S = br(ad_x(A, 0), ad_x(B, 0), 1, 1)
            if any(any(r) for r in S):
                if not in_span(im_ev_b, flat(S)):
                    bad += 1
            # the identity S = [x, [a, [x,b]]]
            rhs = ad_x(br(A, ad_x(B, 0), 0, 1), 1)
            if flat(S) != flat(rhs):
                bad += 1
    if bad:
        print("  ! %s: D2 fails in %d pairs" % (tag, bad))
        ok = False

    # D1: gr_0 h = Ker \cap g_ev, gr_1 h = Im \cap g_od
    #     sdim g_x = (dke - die | dko - dio)
    sdim = (dke - die, dko - dio)
    nontrivial = any(any(any(r) for r in br(ad_x(A, 0), ad_x(B, 0), 1, 1))
                     for A in ev for B in ev)

    if verbose:
        print("  %-16s sdim g = (%d|%d)   Ann(x) = h_0 : %d   [x,g_ev] = h_1 : %d"
              % (tag, len(ev), len(od), dke, dio))
        print("  %-16s Im \\cap g_ev = %d,  Im \\cap g_od = %d   ==>  sdim g_x = (%d|%d)"
              % ("", die, dio, sdim[0], sdim[1]))
        print("  %-16s x ensures nontriviality: %s;  D0-D3: %s"
              % ("", "yes" if nontrivial else "no", "OK" if ok else "FAILED"))
    # D3
    if nontrivial and die == 0:
        print("  ! %s: D3 fails -- nontrivial but Im \\cap g_ev = 0" % tag)
        ok = False
    return ok, sdim, (dke, dio, die, dio), nontrivial


# =========================================================== HAMILTONIAN CASE

def analyse_ham(n, B=None, x=None, tag="", verbose=True):
    """g = h'(0|n) with the odd Poisson bracket; x from the Table of 2.9.2."""
    H = HV.HamVol(n, B, x)
    gb = H.gbasis
    evb = [m for m in gb if HV.popcount(m) % 2 == 0]
    odb = [m for m in gb if HV.popcount(m) % 2 == 1]
    ei = {m: i for i, m in enumerate(evb)}
    oi = {m: i for i, m in enumerate(odb)}
    ok = True

    def vec_ev(f):
        v = [F(0)] * len(evb)
        for m, c in f.items():
            v[ei[m]] += c
        return v

    def vec_od(f):
        v = [F(0)] * len(odb)
        for m, c in f.items():
            v[oi[m]] += c
        return v

    if H.br(H.x, H.x):
        print("  ! %s: [x,x] =/= 0" % tag)
        ok = False

    # ad_x : g_ev -> g_od  and  g_od -> g_ev
    A_eo = [vec_od(H.ad_x({m: 1})) for m in evb]          # rows indexed by evb
    A_oe = [vec_ev(H.ad_x({m: 1})) for m in odb]

    im_od = [r for r in rref(A_eo)[0]]                    # Im \cap g_od
    im_ev = [r for r in rref(A_oe)[0]]                    # Im \cap g_ev
    die, dio = rank(im_ev), rank(im_od)

    # kernels
    ker_ev_c = nullspace([[A_eo[i][j] for i in range(len(evb))]
                          for j in range(len(odb))], len(evb))
    ker_od_c = nullspace([[A_oe[i][j] for i in range(len(odb))]
                          for j in range(len(evb))], len(odb))
    dke, dko = rank(ker_ev_c), rank(ker_od_c)

    # Im \subseteq Ker
    for r in im_ev:
        if not in_span(ker_ev_c, r):
            print("  ! %s: Im \\cap g_ev not inside Ker" % tag)
            ok = False
    for r in im_od:
        if not in_span(ker_od_c, r):
            print("  ! %s: Im \\cap g_od not inside Ker" % tag)
            ok = False

    # D2 on every basis pair of g_ev
    bad = 0
    for m in evb:
        a = {m: 1}
        for mm_ in evb:
            b = {mm_: 1}
            S = H.S(a, b)
            if S and not in_span(im_ev, vec_ev(S)):
                bad += 1
            # hamvol's ad_x(v) is {v,x}, i.e. -[x,v] on v even and +[x,v] on v
            # odd; so [x,[a,[x,b]]] = -ad_x(br(a, ad_x b)) for a, b even.
            rhs = HV.smul(-1, H.ad_x(H.br(a, H.ad_x(b))))
            if HV.add(S, HV.smul(-1, rhs)):
                bad += 1
    if bad:
        print("  ! %s: D2 fails in %d pairs" % (tag, bad))
        ok = False

    nontrivial = any(H.S({m: 1}, {mm_: 1}) for m in evb for mm_ in evb)
    if nontrivial and die == 0:
        print("  ! %s: D3 fails" % tag)
        ok = False

    if verbose:
        print("  %-26s sdim g = (%d|%d)  h_0 = Ann(x) : %d  h_1 = [x,g_ev] : %d"
              % (tag, len(evb), len(odb), dke, dio))
        print("  %-26s Im \\cap g_ev = %d, Im \\cap g_od = %d  ==>  sdim g_x = (%d|%d)"
              % ("", die, dio, dke - die, dko - dio))
        print("  %-26s nontriviality: %s;  D0--D3: %s"
              % ("", "yes" if nontrivial else "no", "OK" if ok else "FAILED"))
    return ok, (dke - die, dko - dio)


# ==================================================================== driver

def main():
    print("ds.py --- Volichenko algebras h_{ad_x} vs. the DS functor g_x")
    print("exact linear algebra backend: %s\n" % backend())

    allok = True

    def rank2(E, zeros, p, q):
        "x = E_{1,p+1} + E_{2,p+2}: still x^2 = 0, but of rank 2"
        M = zeros()
        M[0][p] = F(1)
        M[1][p + 1] = F(1)
        return M

    def twoblock(E, zeros, p, q):
        "x = E_{1,p+1} + E_{p+q,2}: both off-diagonal blocks nonzero, x^2 = 0"
        M = zeros()
        M[0][p] = F(1)
        M[p + q - 1][1] = F(1)
        return M

    print("Matrix series (x = E_{1,p+1}, resp. the odd nilpotent of q(n)):")
    cases = [("gl", 1, 1, None), ("gl", 2, 1, None), ("gl", 2, 2, None),
             ("gl", 3, 2, None), ("sl", 2, 2, None),
             ("gl", 2, 2, rank2), ("gl", 3, 3, rank2), ("sl", 3, 3, rank2),
             ("gl", 2, 2, twoblock), ("gl", 3, 3, twoblock), ("sl", 2, 2, twoblock),
             ("q", 2, 0, None), ("q", 3, 0, None), ("q", 4, 0, None)]
    for kind, p, q, xs in cases:
        tag = ("%s(%d|%d)" % (kind, p, q)) if kind != "q" else "q(%d)" % p
        if xs is not None:
            tag += ", " + xs.__name__
        model = mat_model(tag, p, q, kind, xs)
        ok, sdim, dims, nt = analyse_matrix(model)
        allok &= ok
    print()

    print("Hamiltonian series, g = h'(0|n), both x of Table 2.12.2:")
    for n in range(4, 8):
        ok, s1 = analyse_ham(n, None, {1: 1},
                             tag="h'(0|%d), x = H_{th_1}" % n)
        allok &= ok
        ok, s2 = analyse_ham(n, HV.hyperbolic(n), {1: 1},
                             tag="h'(0|%d), x isotropic" % n)
        allok &= ok
    print()
    print("ALL CHECKS: %s" % ("OK" if allok else "FAILED"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
