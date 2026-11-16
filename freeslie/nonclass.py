#!/usr/bin/env python3
"""nonclass.py -- non-classical restrictedness (p = 2) and the PBW criteria.

Two points: 1) the comparison with the PBW theorems of Braverman-Gaitsgory and of
Etingof (Verlinde categories); 2) the "classical" restrictedness of [BLLSh] =
Bouarroudj, Lebedev, Leites, Shchepochkina, arXiv:1407.1695, alone loses the only
simple 3-dimensional Lie algebra in characteristic 2, and much else.

Everything below is exact: integers, F_2, F_4 and F_3, no floating point.
Linear algebra over F_p goes through ../lib/exactlin.py (FLINT when available).

Parts
-----
A  o^{(1)}(3) over F_2 in the matrix form of [BLLSh, Subsec. 2.6]: it is a
   3-dimensional simple Lie algebra, it has NO 2-structure, and it does have a
   (2,4)-structure whose [4] is the fourth matrix power.

B  Exhaustive check that o^{(1)}(3) is the ONLY 3-dimensional simple Lie algebra
   over F_2 and over F_4, up to isomorphism (brute force over all q^9 brackets,
   then GL(3,q)-orbits).

C  The additivity rule for the [4]-map of a (2,4)-structure, computed in the free
   restricted Lie algebra on two generators over F_2. [BLLSh] states the axioms
   for centerless g only, where x^[4] is determined; a variety needs the rule.

D  The free object at p = 2 in the three framings: no restrictedness (our
   Theorem 5.2), the 2|2-structure and the 2|4-structure. The last two coincide,
   and their free object is Jacobson's free restricted Lie algebra, which is
   strictly bigger than ours.

E  p = 3: the collapse of U(g) for the free object of the category WITHOUT the
   identity [x,[x,x]] = 0 -- the failure of condition (J) of Braverman-Gaitsgory.
"""

import sys
import os
from itertools import product

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import rank, rref, backend                        # noqa: E402


def show(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# tiny finite fields: F_2 and F_4 = F_2[w]/(w^2+w+1), elements 0,1,2=w,3=w+1
# ---------------------------------------------------------------------------

class GF:
    def __init__(self, q):
        assert q in (2, 4)
        self.q = q
        self.els = list(range(q))
        if q == 2:
            self._mul = [[0, 0], [0, 1]]
        else:
            # w^2 = w+1 = 3, w*3 = w^2+w = 1, 3*3 = w^2+1 = w = 2
            self._mul = [[0, 0, 0, 0],
                         [0, 1, 2, 3],
                         [0, 2, 3, 1],
                         [0, 3, 1, 2]]

    def add(self, a, b):
        return a ^ b                      # characteristic 2, both fields

    def mul(self, a, b):
        return self._mul[a][b]

    def inv(self, a):
        assert a
        for b in self.els:
            if self.mul(a, b) == 1:
                return b


# ---------------------------------------------------------------------------
# 3-dimensional algebras over a field of characteristic 2, as bracket tables
# ---------------------------------------------------------------------------
#
# A bracket on K^3 with [x,x] = 0 is fixed by the three vectors
#     c12 = [e1,e2], c13 = [e1,e3], c23 = [e2,e3]
# (in characteristic 2, [e_j,e_i] = [e_i,e_j]).  We store it as the triple
# (c12, c13, c23) of 3-tuples over the field.

PAIRS = [(0, 1), (0, 2), (1, 2)]


def br_basis(F, C, i, j):
    """[e_i, e_j] for basis vectors."""
    if i == j:
        return (0, 0, 0)
    k = PAIRS.index((min(i, j), max(i, j)))
    return C[k]


def scal(F, a, v):
    return tuple(F.mul(a, c) for c in v)


def vadd(F, u, v):
    return tuple(F.add(a, b) for a, b in zip(u, v))


def bracket(F, C, u, v):
    out = (0, 0, 0)
    for i in range(3):
        if not u[i]:
            continue
        for j in range(3):
            if not v[j] or i == j:
                continue
            out = vadd(F, out, scal(F, F.mul(u[i], v[j]), br_basis(F, C, i, j)))
    return out


def jacobi_ok(F, C):
    """In dimension 3 the only non-trivial Jacobi triple is (e1,e2,e3)."""
    e = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    s = (0, 0, 0)
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        s = vadd(F, s, bracket(F, C, bracket(F, C, e[a], e[b]), e[c]))
    return s == (0, 0, 0)


def subspaces(F, dim):
    """All subspaces of K^3 of the given dimension, as frozensets of vectors."""
    vecs = [v for v in product(F.els, repeat=3)]
    out = set()
    if dim == 1:
        for v in vecs:
            if any(v):
                out.add(frozenset(scal(F, a, v) for a in F.els))
    elif dim == 2:
        for u in vecs:
            if not any(u):
                continue
            for v in vecs:
                if not any(v):
                    continue
                sp = frozenset(vadd(F, scal(F, a, u), scal(F, b, v))
                               for a in F.els for b in F.els)
                if len(sp) == F.q ** 2:
                    out.add(sp)
    return out


_SUBSP = {}


def all_subspaces(F):
    if F.q not in _SUBSP:
        _SUBSP[F.q] = [sp for d in (1, 2) for sp in subspaces(F, d)]
    return _SUBSP[F.q]


def is_simple(F, C):
    """No proper non-zero ideal, and the algebra is not abelian."""
    e = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    if all(C[k] == (0, 0, 0) for k in range(3)):
        return False
    for sp in all_subspaces(F):
        if all(bracket(F, C, v, e[i]) in sp for v in sp for i in range(3)):
            return False
    return True


def transport(F, C, g, gi):
    """Push the bracket C forward along the invertible matrix g (rows = images
    of the basis vectors): the new bracket C' has [e_i,e_j]' = g^{-1}[ge_i,ge_j].
    We instead compute the bracket pulled back, which is what orbit-hunting
    needs; both give the same orbit."""
    Cn = []
    for (i, j) in PAIRS:
        u = g[i]
        v = g[j]
        w = bracket(F, C, u, v)
        Cn.append(mat_apply(F, gi, w))
    return tuple(Cn)


def mat_apply(F, g, v):
    """v * g  (v a row vector)."""
    out = (0, 0, 0)
    for i in range(3):
        if v[i]:
            out = vadd(F, out, scal(F, v[i], g[i]))
    return out


def mat_inv(F, g):
    n = 3
    a = [list(g[i]) + [1 if j == i else 0 for j in range(n)] for i in range(n)]
    row = 0
    for col in range(n):
        piv = None
        for r in range(row, n):
            if a[r][col]:
                piv = r
                break
        if piv is None:
            return None
        a[row], a[piv] = a[piv], a[row]
        iv = F.inv(a[row][col])
        a[row] = [F.mul(iv, x) for x in a[row]]
        for r in range(n):
            if r != row and a[r][col]:
                f = a[r][col]
                a[r] = [F.add(x, F.mul(f, y)) for x, y in zip(a[r], a[row])]
        row += 1
    return tuple(tuple(a[i][n:]) for i in range(n))


def gl3(F):
    """All (g, g^{-1}) with g invertible."""
    out = []
    for g in product(product(F.els, repeat=3), repeat=3):
        gi = mat_inv(F, g)
        if gi is not None:
            out.append((g, gi))
    return out


# ---------------------------------------------------------------------------
# A: o^{(1)}(3) over F_2 in the matrix realization of [BLLSh, Subsec. 2.6]
# ---------------------------------------------------------------------------
#
#            / A  X  B \        A in gl(n);  B,C in ZD(n) (symmetric, zero
#            | Y  0  X |        diagonal);  X,Y column vectors.
#            \ C  Y  A /        For n = 1: B = C = 0, so dim = 3.

def m_mul(a, b, p=2):
    n = len(a)
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(n)) % p
                       for j in range(n)) for i in range(n))


def m_add(a, b, p=2):
    n = len(a)
    return tuple(tuple((a[i][j] + b[i][j]) % p for j in range(n)) for i in range(n))


def m_scal(c, a, p=2):
    return tuple(tuple((c * x) % p for x in row) for row in a)


def m_comm(a, b, p=2):
    return m_add(m_mul(a, b, p), m_mul(b, a, p), p)      # char 2


def o13_matrices():
    """H, X, Y spanning o^{(1)}(3) over F_2."""
    def M(a, x, y):
        return ((a, x, 0), (y, 0, x), (0, y, a))
    return M(1, 0, 0), M(0, 1, 0), M(0, 0, 1)


def part_A():
    show("A.  o^{(1)}(3) over F_2: simple, no 2-structure, a (2,4)-structure")
    H, X, Y = o13_matrices()
    names = {H: "H", X: "X", Y: "Y"}
    basis = [H, X, Y]

    def coords(m):
        """Express a matrix in the basis H,X,Y (or None)."""
        for a, b, c in product(range(2), repeat=3):
            t = m_add(m_add(m_scal(a, H), m_scal(b, X)), m_scal(c, Y))
            if t == m:
                return (a, b, c)
        return None

    print("basis (matrices over F_2):")
    for b in basis:
        print(f"  {names[b]} = {b}")

    print("\nbrackets [u,v] = uv + vu:")
    C = []
    for (i, j) in PAIRS:
        c = coords(m_comm(basis[i], basis[j]))
        assert c is not None, "bracket left the span"
        C.append(c)
        print(f"  [{names[basis[i]]},{names[basis[j]]}] = {c}")
    C = tuple(C)

    F = GF(2)
    print(f"\n  Jacobi identity holds: {jacobi_ok(F, C)}")
    print(f"  [x,x] = 0 for all x   : "
          f"{all(bracket(F, C, v, v) == (0,0,0) for v in product(F.els, repeat=3))}")
    print(f"  simple                : {is_simple(F, C)}")

    # -- no 2-structure ----------------------------------------------------
    #    a 2-structure needs, for EVERY x, an element x^[2] with ad_{x^[2]} =
    #    (ad_x)^2.  Test it on the basis: Jacobson's criterion (a p-structure
    #    exists iff it exists on one basis) makes this decisive.
    print("\n  is (ad_x)^2 inner, x running over a basis?")
    inner_ok = {}
    for v, nm in ((1, 0, 0), "H"), ((0, 1, 0), "X"), ((0, 0, 1), "Y"):
        target = [bracket(F, C, v, bracket(F, C, v, e))
                  for e in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
        found = None
        for u in product(F.els, repeat=3):
            if all(bracket(F, C, u, e) == t
                   for e, t in zip(((1, 0, 0), (0, 1, 0), (0, 0, 1)), target)):
                found = u
                break
        inner_ok[nm] = found
        print(f"    x = {nm}: (ad_x)^2 = ad_u with u = {found}")
    print("  => a 2-structure exists iff every entry above is not None:",
          all(v is not None for v in inner_ok.values()))

    # -- the (2,4)-structure ------------------------------------------------
    #    g_+ = <H>, g_- = <X,Y>;  [2] on g_+ is the matrix square,
    #    [4] on g_- is the fourth matrix power.
    print("\n  (2,4)-structure with g_+ = <H>, g_- = <X,Y>:")
    Hsq = m_mul(H, H)
    print(f"    H^[2] = H^2 = {coords(Hsq)} (in the basis H,X,Y)")
    target = [bracket(F, C, (1, 0, 0), bracket(F, C, (1, 0, 0), e))
              for e in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    print(f"    ad_{{H^[2]}} = (ad_H)^2 : "
          f"{all(bracket(F, C, coords(Hsq), e) == t for e, t in zip(((1,0,0),(0,1,0),(0,0,1)), target))}")

    ok4 = True
    for x in product(range(2), repeat=2):                    # x = bX + cY
        if not any(x):
            continue
        Mx = m_add(m_scal(x[0], X), m_scal(x[1], Y))
        M4 = m_mul(m_mul(Mx, Mx), m_mul(Mx, Mx))
        c4 = coords(M4)
        assert c4 is not None
        xv = (0, x[0], x[1])
        for e in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            lhs = bracket(F, C, c4, e)
            rhs = e
            for _ in range(4):
                rhs = bracket(F, C, xv, rhs)
            if lhs != rhs:
                ok4 = False
        print(f"    x = {xv}: x^[4] = {c4}, [x^[4],y] = (ad_x)^4(y) for all y: {ok4}")
    print(f"    x^[4] lands in g_+ = <H> for every x in g_-: "
          f"{all(coords(m_mul(m_mul(m,m), m_mul(m,m)))[1:] == (0,0) for m in [m_add(m_scal(a,X), m_scal(b,Y)) for a,b in product(range(2),repeat=2) if a or b])}")
    print(f"    additivity (x+y)^[4] = x^[4]+y^[4]+s(x,y) is Part C.")
    return C


# ---------------------------------------------------------------------------
# B: the only 3-dimensional simple Lie algebra in characteristic 2
# ---------------------------------------------------------------------------

def part_B(C_o13):
    show("B.  All 3-dimensional simple Lie algebras over F_2 and F_4")
    for q in (2, 4):
        F = GF(q)
        allC = []
        for C in product(product(F.els, repeat=3), repeat=3):
            if jacobi_ok(F, C):
                allC.append(C)
        simples = [C for C in allC if is_simple(F, C)]
        print(f"\n  q = {q}: {q**9} brackets, {len(allC)} satisfy Jacobi, "
              f"{len(simples)} are simple")
        if not simples:
            continue
        # orbits under GL(3,q)
        left = set(simples)
        orbits = []
        G = gl3(F)
        while left:
            rep = next(iter(left))
            orb = set()
            for g, gi in G:
                orb.add(transport(F, rep, g, gi))
            orbits.append((rep, orb & set(simples)))
            left -= orb
        print(f"  |GL(3,{q})| = {len(G)}; number of isomorphism classes: {len(orbits)}")
        for rep, orb in orbits:
            print(f"    class of size {len(orb)}, representative "
                  f"[e1,e2],[e1,e3],[e2,e3] = {rep}"
                  + ("   <-- o^{(1)}(3)" if C_o13 in orb else ""))


# ---------------------------------------------------------------------------
# C: the additivity rule for the [4]-map
# ---------------------------------------------------------------------------
#
# Work in the free associative algebra F_2<x,y,z>.  Both (ad_{x+y})^4 and
# (ad_x)^4 + (ad_y)^4 are derivations (char 2: the square of a derivation is a
# derivation), so their sum O is a derivation, and O = ad_u iff O and ad_u agree
# on the generators x, y, z.  We look for u in the span of the degree-4 elements
# of the free restricted Lie algebra on x,y that are legitimate for a
# (2,4)-structure: Lie monomials of degree 4, and the square [x,y]^[2].

def w_mul(a, b):
    out = {}
    for u, ca in a.items():
        for v, cb in b.items():
            w = u + v
            out[w] = (out.get(w, 0) + ca * cb) % 2
    return {w: c for w, c in out.items() if c}


def w_add(*xs):
    out = {}
    for x in xs:
        for w, c in x.items():
            out[w] = (out.get(w, 0) + c) % 2
    return {w: c for w, c in out.items() if c}


def w_br(a, b):
    return w_add(w_mul(a, b), w_mul(b, a))              # char 2


def part_C():
    show("C.  The additivity rule for the [4]-map of a (2,4)-structure")
    X = {("x",): 1}
    Y = {("y",): 1}
    Z = {("z",): 1}
    XY = w_br(X, Y)

    def ad(u, v):
        return w_br(u, v)

    def ad4(u, v):
        for _ in range(4):
            v = ad(u, v)
        return v

    XplusY = w_add(X, Y)
    # the derivation O = (ad_{x+y})^4 + (ad_x)^4 + (ad_y)^4
    O = {nm: w_add(ad4(XplusY, g), ad4(X, g), ad4(Y, g)) for nm, g in
         (("x", X), ("y", Y), ("z", Z))}

    # candidate values of u: a basis of the degree-4 part of the free restricted
    # Lie algebra on x,y that a (2,4)-structure can produce
    cand = {
        "[x,[x,[x,y]]]": w_br(X, w_br(X, w_br(X, Y))),
        "[y,[y,[y,x]]]": w_br(Y, w_br(Y, w_br(Y, X))),
        "[x,[y,[x,y]]]": w_br(X, w_br(Y, XY)),
        "[y,[x,[x,y]]]": w_br(Y, w_br(X, XY)),
        "[x,y]^[2]":     w_mul(XY, XY),
    }
    names = list(cand)

    # solve  sum_i a_i ad_{cand_i}(g) = O(g)  for g = x,y,z
    words = set()
    cols = {nm: {} for nm in names}
    rhs = {}
    for gnm, g in (("x", X), ("y", Y), ("z", Z)):
        for nm in names:
            for w, c in ad(cand[nm], g).items():
                cols[nm][(gnm, w)] = c
                words.add((gnm, w))
        for w, c in O[gnm].items():
            rhs[(gnm, w)] = c
            words.add((gnm, w))
    words = sorted(words)
    A = [[cols[nm].get(w, 0) for nm in names] + [rhs.get(w, 0)] for w in words]
    R, piv = rref(A, p=2)
    if len(names) in piv:
        print("  no solution: the correction term is NOT inner")
        return
    sol = [0] * len(names)
    for r, c in zip(R, piv):
        if c < len(names):
            sol[c] = r[len(names)] % 2
    used = [names[i] for i in range(len(names)) if sol[i]]
    print("  (ad_{x+y})^4 + (ad_x)^4 + (ad_y)^4 = ad_s  with")
    print("     s = " + (" + ".join(used) if used else "0"))
    # independent verification
    s = {}
    for i, nm in enumerate(names):
        if sol[i]:
            s = w_add(s, cand[nm])
    ok = all(ad(s, g) == O[gnm] for gnm, g in (("x", X), ("y", Y), ("z", Z)))
    print(f"  verified on the generators x, y, z: {ok}")
    print("  so, for x,y in g_- of a (2,4)-structure,")
    print("     (x+y)^[4] = x^[4] + y^[4] + " + (" + ".join(used) if used else "0"))
    print("  (every term lies in g_+, as it must: [x,y] is in g_+ and a bracket")
    print("   of an even number of elements of g_- is in g_+).")
    print("  Note [x,[y,[x,y]]] = [y,[x,[x,y]]], since their sum is")
    print("  [[x,y],[x,y]] = 0; so the two candidates are the same element.")

    # -- the rule, checked on o^{(1)}(3) over F_2 -----------------------------
    F = GF(2)
    H, Xm, Ym = o13_matrices()
    C = ((0, 1, 0), (0, 0, 1), (1, 0, 0))       # [H,X]=X, [H,Y]=Y, [X,Y]=H
    hh = (1, 0, 0)

    def sq_plus(v):                              # [2] on g_+ = <H>
        assert v[1] == v[2] == 0
        return v                                 # H^[2] = H, 0^[2] = 0

    def p4(v):                                   # [4] on g_-: fourth power
        m = ((0, v[1], 0), (v[2], 0, v[1]), (0, v[2], 0))
        m4 = m_mul(m_mul(m, m), m_mul(m, m))
        return (m4[0][0], m4[0][1], m4[1][0])

    ok = True
    for a in product(range(2), repeat=2):
        for b in product(range(2), repeat=2):
            x = (0, a[0], a[1])
            y = (0, b[0], b[1])
            w = bracket(F, C, x, y)
            s_val = vadd(F, vadd(F,
                bracket(F, C, x, bracket(F, C, x, bracket(F, C, x, y))),
                bracket(F, C, y, bracket(F, C, y, bracket(F, C, y, x)))),
                vadd(F, bracket(F, C, x, bracket(F, C, y, w)), sq_plus(w)))
            lhs = p4(vadd(F, x, y))
            rhs = vadd(F, vadd(F, p4(x), p4(y)), s_val)
            if lhs != rhs:
                ok = False
    print(f"  the rule holds on o^(1)(3) for all 16 pairs (x,y) in g_- x g_-: {ok}")
    # semilinearity is immediate: (ax)^[4] = a^4 x^[4]
    return used


# ---------------------------------------------------------------------------
# D: the free object at p = 2 in the three framings
# ---------------------------------------------------------------------------

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from freelie import Tensor, witt, mdegs                          # noqa: E402


def part_D(cases=((1, 1), (0, 2), (2, 0), (0, 1)), D=6):
    show("D.  p = 2: three framings, three free objects")
    print("""
  S_2      Lie superalgebras, no restrictedness  [Theorem 5.2]:
             Free(V) = L(V) + span{b^2 | b in B_od},
             dim_m = W(m) + [2|m] dim (L_{m/2})_od
  2|2      Lie superalgebras with a 2|2-structure = Z/2-graded restricted Lie
           algebras: the squaring is defined on all of g, so the free object is
           Jacobson's free restricted Lie algebra,
             dim_m = sum_{k >= 0, 2^k | m} W(m/2^k)
  2|4      classical restrictedness: same objects as 2|2 (Lemma 9.6), so the
           same free object.
""")
    for par in cases:
        n = sum(par)
        parities = [0] * par[0] + [1] * par[1]
        T = Tensor(parities, D)
        L = T.lie(p=2)
        print(f"  sdim V = ({par[0]}|{par[1]}):")
        print("    multideg |   W(m) | S_2 dim | pred | 2|2 dim | pred")
        for d in range(1, D + 1):
            for m in mdegs(n, d):
                if m not in T.words:
                    continue
                Lm = L.get(m, [])
                Wm = len(Lm)
                # --- S_2 model: L_m + squares of L_{m/2} odd
                extra = []
                if all(a % 2 == 0 for a in m):
                    half = tuple(a // 2 for a in m)
                    if T.parity(half) == 1:
                        for v in L.get(half, []):
                            extra.append(T.mul(v, v))
                s2 = T.rank_of(m, [dict(v) for v in Lm] + extra, 2)
                pred2 = Wm + (len(L.get(tuple(a // 2 for a in m), []))
                              if all(a % 2 == 0 for a in m)
                              and T.parity(tuple(a // 2 for a in m)) == 1 else 0)
                # --- 2|2 model: L_m + all 2^k-th powers of L_{m/2^k}
                pw = []
                predr = Wm
                k = 1
                while True:
                    q = 2 ** k
                    if any(a % q for a in m):
                        break
                    sub = tuple(a // q for a in m)
                    for v in L.get(sub, []):
                        u = dict(v)
                        for _ in range(k):
                            u = T.mul(u, u)
                            u = {w: c % 2 for w, c in u.items() if c % 2}
                        pw.append(u)
                    predr += len(L.get(sub, []))
                    k += 1
                r22 = T.rank_of(m, [dict(v) for v in Lm] + pw, 2)
                flag = "" if (s2 == pred2 and r22 == predr) else "   <-- MISMATCH"
                print(f"    {str(m):>9} | {Wm:6d} | {s2:7d} | {pred2:4d} "
                      f"| {r22:7d} | {predr:4d}{flag}")
        # closure of the 2|2 model under the bracket and the squaring
        ok = True
        for d in range(1, D // 2 + 1):
            for m in mdegs(n, d):
                for v in L.get(m, []):
                    sq = T.mul(dict(v), dict(v))
                    sq = {w: c % 2 for w, c in sq.items() if c % 2}
                    m2 = tuple(2 * a for a in m)
                    if m2 in T.words:
                        base = [dict(u) for u in L.get(m2, [])]
                        k = 1
                        while True:
                            q = 2 ** k
                            if any(a % q for a in m2):
                                break
                            sub = tuple(a // q for a in m2)
                            for u0 in L.get(sub, []):
                                u = dict(u0)
                                for _ in range(k):
                                    u = T.mul(u, u)
                                    u = {w: c % 2 for w, c in u.items() if c % 2}
                                base.append(u)
                            k += 1
                        if T.rank_of(m2, base + [sq], 2) != T.rank_of(m2, base, 2):
                            ok = False
        print(f"    squares of L stay inside the 2|2-model: {ok}")


# ---------------------------------------------------------------------------
# E: p = 3, the collapse of U(g) -- Braverman-Gaitsgory condition (J)
# ---------------------------------------------------------------------------

def part_E():
    show("E.  p = 3: U(g) kills the cube -- the Braverman-Gaitsgory obstruction")
    p = 3
    # g = <y odd, h = [y,y] even, z = [y,h] odd central>, the free object on one
    # odd generator of the category WITHOUT the identity [x,[x,x]] = 0.
    names = ["y", "h", "z"]
    par = [1, 0, 1]
    brk = {(0, 0): {1: 1}, (0, 1): {2: 1}, (1, 0): {2: p - 1}}

    def br(i, j):
        return brk.get((i, j), {})

    # U(g) truncated at degree Dmax: T(g)/(x_i x_j - (-1)^{..} x_j x_i - [x_i,x_j])
    Dmax = 4
    wordsd = {d: list(product(range(3), repeat=d)) for d in range(Dmax + 1)}
    allw = [w for d in range(Dmax + 1) for w in wordsd[d]]
    idx = {w: i for i, w in enumerate(allw)}

    rels = []
    for i in range(3):
        for j in range(3):
            r = {}
            r[(i, j)] = r.get((i, j), 0) + 1
            s = -1 if (par[i] * par[j]) % 2 else 1
            r[(j, i)] = r.get((j, i), 0) - s
            for k, c in br(i, j).items():
                r[(k,)] = r.get((k,), 0) - c
            r = {w: c % p for w, c in r.items() if c % p}
            if r:
                rels.append(r)
    # two-sided ideal, truncated
    gens = []
    for r in rels:
        for du in range(Dmax + 1):
            for u in wordsd[du]:
                for dv in range(Dmax + 1):
                    for v in wordsd[dv]:
                        if du + dv + 2 > Dmax:
                            continue
                        g = {}
                        for w, c in r.items():
                            g[u + w + v] = (g.get(u + w + v, 0) + c) % p
                        g = {w: c for w, c in g.items() if c}
                        if g:
                            gens.append(g)
    rows = []
    for g in gens:
        row = [0] * len(allw)
        for w, c in g.items():
            row[idx[w]] = c % p
        rows.append(row)
    rk = rank(rows, p=p)
    print(f"  T(g) truncated at degree {Dmax}: {len(allw)} words")
    print(f"  ideal of relations, truncated: rank {rk}")
    print(f"  dim U(g)_(<= {Dmax}) = {len(allw) - rk}")

    # is z (a degree-1 word) in the ideal?
    for gen_name, k in (("y", 0), ("h", 1), ("z", 2)):
        row = [0] * len(allw)
        row[idx[(k,)]] = 1
        rk2 = rank(rows + [row], p=p)
        print(f"    {gen_name} lies in the ideal (i.e. dies in U(g)): {rk2 == rk}")
    print("""
  So the natural map g -> U(g) has kernel <z> = <T(y)>: PBW fails for the
  category WITHOUT the identity, as Subsection 7.3 says.
  In the Braverman-Gaitsgory picture (J. Algebra 181 (1996) 315-328) the
  quadratic algebra Sym(g) is Koszul in sVec, so their criterion applies and
  the failure is the failure of their condition (J) in degree 3: the two
  reductions of y.y.y differ by [[y,y],y], which the super Jacobi identity does
  not kill when 3 = 0.  Etingof (Adv. Math. 327 (2018) 128-160, arXiv:1603.08133)
  turns this into a definition: his p-Jacobi identity, which for p = 3 in
  Ver_3 = sVec reads [[x,x],x] = 0 for odd x, is what restores PBW.
""")


def main():
    print(f"exactlin backend: {backend()}")
    C = part_A()
    part_B(C)
    part_C()
    part_D()
    part_E()


if __name__ == "__main__":
    main()
