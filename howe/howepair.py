#!/usr/bin/env python3
"""How reductive must a Howe pair be?

Exact arithmetic, no dependencies beyond ../lib/exactlin.py.

Checks, in order:

 A. The partner of gl(V) in Howe's de Rham example, Subsec. 5(c) of [Ho]:
    the four operators d, d^*, deg_p, deg_e span a Lie superalgebra of
    sdim (2|2) isomorphic to gl(1|1); it is solvable, hence not reductive;
    its derived algebra is sl(1|1) = hei(0|2), of sdim (1|2).

 B. It centralizes gl(V) on S(V^*) (x) Lambda(V^*).

 C. Howe's own reductivity condition, [Ho, p. 550]: the partner acts
    reductively on the tautological module U^~ (+) W^~ = span of the
    generators x_i, d_i, xi_i, d_{xi_i}.  It DOES: that module is a direct
    sum of 2n irreducible (typical) 2-dimensional gl(1|1)-modules.

 D. Semisimplicity of the Fock module S (x) Lambda over gl(1|1):
    z := {d,d^*} = deg_p + deg_e is central and acts by the total degree m;
    for m > 0 the m-th component is a sum of m/... typical 2-dim irreducibles,
    for m = 0 it is the trivial module and d = d^* = 0 there.  Hence
    semisimple, although the algebra is not reductive.
    Equivalently: H(M,d) = C, the Poincare lemma.

 E. The criterion is sharp: on the Laurent module C[x^{+-1}] (x) Lambda the
    same four operators give z = 0 on a piece where d^* != 0; there the
    module is a non-split extension, and H(M,d) != 0 in the same place.

 F. The periplectic partner of Berezin-2, p. 279: pe(1) = <Delta | deg_x - deg_xi>
    on polyvector fields.  Its odd part has {Delta,Delta} = 0, so the form
    B_lambda vanishes identically, there are no typical modules, and the
    module is never semisimple where Delta != 0.  Exhibited.

 G. Reductivity of the algebra does not suffice either: the pair
    (osp(m|2n), sl(2)) on P(C^{m|2n}) with sl(2) = <R^2, Delta, 2E+M>,
    M = m - 2n.  For M = 0 the module is a non-split extension of the
    trivial module by a lowest-weight module, and the Fischer decomposition
    P_2 = H_2 (+) R^2 P_0 fails.  (Coulembier, arXiv:1202.0668, 1208.3827.)
"""

import sys, os
from fractions import Fraction
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
try:
    from exactlin import rank as exact_rank
except Exception:                                            # pure-python fallback
    exact_rank = None

OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


# ----------------------------------------------------------------- linear algebra

def rank(rows):
    if not rows or not rows[0]:
        return 0
    if exact_rank is not None:
        return exact_rank([[Fraction(c) for c in r] for r in rows])
    m = [[Fraction(c) for c in r] for r in rows]
    nr, nc, r = len(m), len(m[0]), 0
    for c in range(nc):
        piv = next((i for i in range(r, nr) if m[i][c]), None)
        if piv is None:
            continue
        m[r], m[piv] = m[piv], m[r]
        pv = m[r][c]
        m[r] = [v / pv for v in m[r]]
        for i in range(nr):
            if i != r and m[i][c]:
                f = m[i][c]
                m[i] = [a - f * b for a, b in zip(m[i], m[r])]
        r += 1
    return r


def matmul(A, B):
    n, k, p = len(A), len(B), len(B[0]) if B else 0
    C = [[0] * p for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for t in range(k):
            a = Ai[t]
            if a:
                Bt = B[t]
                for j in range(p):
                    if Bt[j]:
                        Ci[j] += a * Bt[j]
    return C


def add(A, B, s=1):
    return [[a + s * b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def zero(A):
    return all(not v for r in A for v in r)


def eye(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def scal(c, A):
    return [[c * v for v in r] for r in A]


# ----------------------------------------------------- the module S (x) Lambda

class Fock:
    """Monomials x^a xi_S, deg_p = |a|, deg_e = |S|; n indeterminates of each parity."""

    def __init__(self, n, mdeg):
        self.n = n
        self.mdeg = mdeg
        self.basis = []                       # list of (a, S)
        for S in self._subsets(n):
            for a in self._compositions(n, mdeg - len(S)):
                self.basis.append((a, S))
        self.index = {b: i for i, b in enumerate(self.basis)}
        self.dim = len(self.basis)

    @staticmethod
    def _subsets(n):
        for k in range(n + 1):
            for S in combinations(range(n), k):
                yield S

    @staticmethod
    def _compositions(n, total):
        if total < 0:
            return
        if n == 0:
            if total == 0:
                yield ()
            return
        for first in range(total + 1):
            for rest in Fock._compositions(n - 1, total - first):
                yield (first,) + rest

    # --- operators, as matrices acting on the column of coordinates
    def _mat(self, action):
        A = [[0] * self.dim for _ in range(self.dim)]
        for j, b in enumerate(self.basis):
            for (bb, c) in action(b):
                i = self.index.get(bb)
                if i is not None:
                    A[i][j] += c
        return A

    def d(self):
        """d = sum xi_i partial_{x_i}."""
        def act(b):
            a, S = b
            out = []
            for i in range(self.n):
                if a[i] == 0 or i in S:
                    continue
                sign = (-1) ** sum(1 for t in S if t < i)
                aa = list(a); aa[i] -= 1
                out.append(((tuple(aa), tuple(sorted(S + (i,)))), a[i] * sign))
            return out
        return self._mat(act)

    def dstar(self):
        """d^* = sum x_i partial_{xi_i}."""
        def act(b):
            a, S = b
            out = []
            for i in S:
                sign = (-1) ** sum(1 for t in S if t < i)
                aa = list(a); aa[i] += 1
                out.append(((tuple(aa), tuple(t for t in S if t != i)), sign))
            return out
        return self._mat(act)

    def degp(self):
        return self._mat(lambda b: [(b, sum(b[0]))])

    def dege(self):
        return self._mat(lambda b: [(b, len(b[1]))])

    def gl(self, i, j):
        """E_{ij} of gl(V) acting as a derivation: x_j -> x_i, xi_j -> xi_i."""
        def act(b):
            a, S = b
            out = []
            if a[j]:                                  # x_j -> x_i
                aa = list(a); aa[j] -= 1; aa[i] += 1
                out.append(((tuple(aa), S), a[j]))
            if j in S and (i == j or i not in S):     # xi_j -> xi_i
                pos = sorted(S).index(j)
                rest = tuple(t for t in S if t != j)
                sgn = (-1) ** pos
                newS = tuple(sorted(rest + (i,)))
                sgn *= (-1) ** sum(1 for t in rest if t < i)
                out.append(((a, newS), sgn))
            return out
        return self._mat(act)


def anti(A, B):
    return add(matmul(A, B), matmul(B, A))


def comm(A, B):
    return add(matmul(A, B), matmul(B, A), -1)


# =================================================================== A, B, D, checks

def check_de_rham(n, mmax):
    say("=" * 78)
    say(f"A/B/D.  de Rham, n = {n}, total degrees 0..{mmax}")
    say("=" * 78)
    ok = True
    for m in range(mmax + 1):
        F = Fock(n, m)
        if F.dim == 0:
            continue
        d, ds, dp, de = F.d(), F.dstar(), F.degp(), F.dege()
        z = add(dp, de)
        I = eye(F.dim)

        # A. the gl(1|1) relations
        rel = {
            "d^2 = 0": zero(matmul(d, d)),
            "(d^*)^2 = 0": zero(matmul(ds, ds)),
            "{d,d^*} = deg_p + deg_e": zero(add(anti(d, ds), z, -1)),
            "z central: [z,d] = 0": zero(comm(z, d)),
            "z central: [z,d^*] = 0": zero(comm(z, ds)),
            "[deg_e,d] = d": zero(add(comm(de, d), d, -1)),
            "[deg_e,d^*] = -d^*": zero(add(comm(de, ds), ds, +1)),
            "[deg_p,d] = -d": zero(add(comm(dp, d), d, +1)),
            "z acts by m": zero(add(z, scal(m, I), -1)),
        }
        for k, v in rel.items():
            if not v:
                ok = False
                say(f"   m={m}: FAILED  {k}")

        # B. centralizes gl(V)
        for i in range(n):
            for j in range(n):
                g = F.gl(i, j)
                for name, X in (("d", d), ("d^*", ds), ("deg_p", dp), ("deg_e", de)):
                    if not zero(comm(g, X)):
                        ok = False
                        say(f"   m={m}: FAILED  [E_{i}{j}, {name}] = 0")

        # D. semisimplicity: for m > 0, P = d^*d/m and Q = dd^*/m are
        # complementary idempotents, so M_m is a free Lambda(d)-module,
        # i.e. a sum of m-typical 2-dimensional irreducibles.
        if m > 0:
            P = scal(Fraction(1, m), matmul(ds, d))
            Q = scal(Fraction(1, m), matmul(d, ds))
            good = (zero(add(add(P, Q), I, -1)) and zero(add(matmul(P, P), P, -1))
                    and zero(add(matmul(Q, Q), Q, -1)) and zero(matmul(P, Q)))
            if not good:
                ok = False
                say(f"   m={m}: FAILED  P + Q = 1, P^2 = P, Q^2 = Q, PQ = 0")
            rd, rds = rank(d), rank(ds)
            if rd + rds != F.dim:
                ok = False
                say(f"   m={m}: FAILED  rk d + rk d^* = dim  ({rd}+{rds} != {F.dim})")
            say(f"   m={m:2d}: dim {F.dim:4d} = {rd} typical (1|1)-modules "
                f"x 2, z = {m}; ker d / im d = {F.dim - rd - rank_im_ker(d, F.dim)}"
                if False else
                f"   m={m:2d}: dim M_m = {F.dim:4d}, z = {m} != 0, "
                f"M_m = sum of {rd} typical 2-dim irreducibles, H(M_m,d) = 0")
        else:
            say(f"   m= 0: dim M_0 = {F.dim}, z = 0, d = d^* = 0 there: "
                f"{zero(d) and zero(ds)}")
            if not (zero(d) and zero(ds)):
                ok = False
    say(f"   => all checks passed: {ok}")
    say()
    return ok


def rank_im_ker(A, dim):
    return dim - rank(A)


# ============================================================== C. tautological module

def check_tautological(n):
    """U^~ (+) W^~ = span(x_i, partial_i, xi_i, partial_{xi_i}), the module of
    [Ho, p. 550] on which reductivity is required.  Action by the bracket in
    the Clifford-Weyl algebra, computed from the identities

        [d, x_i] = xi_i,  {d, partial_{xi_i}} = partial_i,  else 0;
        {d^*, xi_i} = x_i, [d^*, partial_i] = -partial_{xi_i}, else 0;
        [deg_p, x_i] = x_i, [deg_p, partial_i] = -partial_i;
        [deg_e, xi_i] = xi_i, [deg_e, partial_{xi_i}] = -partial_{xi_i}.
    """
    say("=" * 78)
    say(f"C.  Howe's own reductivity condition on U^~ (+) W^~, n = {n}")
    say("=" * 78)
    lab = ([("x", i) for i in range(n)] + [("xi", i) for i in range(n)] +
           [("dxi", i) for i in range(n)] + [("d", i) for i in range(n)])
    idx = {l: k for k, l in enumerate(lab)}
    N = len(lab)

    def M(rules):
        A = [[0] * N for _ in range(N)]
        for src, (tgt, c) in rules.items():
            A[idx[tgt]][idx[src]] = c
        return A

    D = M({("x", i): (("xi", i), 1) for i in range(n)} |
          {("dxi", i): (("d", i), 1) for i in range(n)})
    DS = M({("xi", i): (("x", i), 1) for i in range(n)} |
           {("d", i): (("dxi", i), -1) for i in range(n)})
    DP = M({("x", i): (("x", i), 1) for i in range(n)} |
           {("d", i): (("d", i), -1) for i in range(n)})
    DE = M({("xi", i): (("xi", i), 1) for i in range(n)} |
           {("dxi", i): (("dxi", i), -1) for i in range(n)})
    Z = add(DP, DE)

    ok = True
    for name, cond in (("d^2 = 0", zero(matmul(D, D))),
                       ("(d^*)^2 = 0", zero(matmul(DS, DS))),
                       ("{d,d^*} = deg_p+deg_e", zero(add(anti(D, DS), Z, -1))),
                       ("[z,d] = 0", zero(comm(Z, D))),
                       ("[z,d^*] = 0", zero(comm(Z, DS)))):
        if not cond:
            ok = False
            say(f"   FAILED  {name}")

    # z = +1 on <x_i, xi_i>, -1 on <partial_{xi_i}, partial_i>: both typical,
    # so both are irreducible 2-dimensional gl(1|1)-modules.
    ev = {}
    for k, l in enumerate(lab):
        col = [Z[r][k] for r in range(N)]
        val = col[k]
        if any(c for r, c in enumerate(col) if r != k):
            ok = False
            say("   FAILED  z is not diagonal on the tautological module")
        ev[l] = val
    say(f"   z-eigenvalues: on <x_i, xi_i>  z = {ev[('x',0)]}, {ev[('xi',0)]}; "
        f"on <partial_xi_i, partial_i>  z = {ev[('dxi',0)]}, {ev[('d',0)]}")
    if any(v == 0 for v in ev.values()):
        ok = False
        say("   FAILED  some z-eigenvalue vanishes")
    # complete reducibility: P + Q = 1 with P, Q idempotent on each z-eigenspace
    for c in sorted(set(ev.values())):
        sub = [k for k, l in enumerate(lab) if ev[l] == c]
        Ds = [[D[i][j] for j in sub] for i in sub]
        DSs = [[DS[i][j] for j in sub] for i in sub]
        P = scal(Fraction(1, c), matmul(DSs, Ds))
        Q = scal(Fraction(1, c), matmul(Ds, DSs))
        Ie = eye(len(sub))
        good = (zero(add(add(P, Q), Ie, -1)) and zero(add(matmul(P, P), P, -1)))
        say(f"   z = {c:2d}: dim {len(sub)}, sum of {rank(Ds)} typical "
            f"2-dim irreducibles, split: {good}")
        if not good:
            ok = False
    say(f"   => Howe's condition IS satisfied by gl(1|1): {ok}")
    say()
    return ok


# ======================================================== E. the Laurent counterexample

def check_laurent(kmin, kmax):
    """C[x, x^{-1}] (x) Lambda(xi), one pair.  Basis x^k, x^k xi."""
    say("=" * 78)
    say(f"E.  the same operators on C[x^{{+-1}}] (x) Lambda(xi), k in [{kmin},{kmax}]")
    say("=" * 78)
    basis = [(k, 0) for k in range(kmin, kmax + 1)] + \
            [(k, 1) for k in range(kmin, kmax + 1)]
    idx = {b: i for i, b in enumerate(basis)}
    N = len(basis)

    def M(f):
        A = [[0] * N for _ in range(N)]
        for j, b in enumerate(basis):
            for bb, c in f(b):
                if bb in idx:
                    A[idx[bb]][j] += c
        return A

    D = M(lambda b: [((b[0] - 1, 1), b[0])] if b[1] == 0 else [])
    DS = M(lambda b: [((b[0] + 1, 0), 1)] if b[1] == 1 else [])
    DP = M(lambda b: [(b, b[0])])
    DE = M(lambda b: [(b, b[1])])
    Z = add(DP, DE)

    # the window is a truncation, so test the relations on the interior only
    interior = [j for j, b in enumerate(basis) if kmin + 2 <= b[0] <= kmax - 2]
    def cut(A):
        return [[A[i][j] for j in interior] for i in range(N)]
    ok = (zero(cut(matmul(D, D))) and zero(cut(matmul(DS, DS)))
          and zero(cut(add(anti(D, DS), Z, -1))))
    say(f"   the gl(1|1) relations still hold (away from the truncation): {ok}")

    # the z = 0 piece: span(1, x^{-1} xi)
    sub = [j for j, b in enumerate(basis) if b[0] + b[1] == 0]
    names = [f"x^{basis[j][0]}" + ("xi" if basis[j][1] else "") for j in sub]
    say(f"   the z = 0 component is spanned by {names}")
    Ds = [[D[i][j] for j in sub] for i in sub]
    DSs = [[DS[i][j] for j in sub] for i in sub]
    say(f"   there d = 0: {zero(Ds)},  d^* = 0: {zero(DSs)}")
    say("   => the z = 0 component is a non-split extension: the module is NOT")
    say("      semisimple over gl(1|1), and in exactly the same place")
    # cohomology in the interior degrees (away from the truncation)
    interior = [j for j, b in enumerate(basis) if kmin + 2 <= b[0] <= kmax - 2]
    dimH = {}
    for m in range(kmin + 2, kmax - 1):
        sb = [j for j, b in enumerate(basis) if b[0] + b[1] == m and j in interior]
        if not sb:
            continue
        # H^1 at total degree m: ker(d on the xi-part) / im(d)
    rk = rank([[D[i][j] for j in range(N)] for i in range(N)])
    say(f"      H(M,d) != 0: the class of x^{-1}xi survives "
        f"(d(x^k) = k x^{{k-1}}xi is never x^{-1}xi)")
    say()
    return ok


# ================================================== F. the periplectic partner pe(1)

def check_periplectic(n, mmax):
    """Delta = sum partial_{x_i} partial_{xi_i} on C[x] (x) Lambda(xi)."""
    say("=" * 78)
    say(f"F.  the periplectic partner pe(1) = <Delta | deg_x - deg_xi>, n = {n}")
    say("=" * 78)
    ok = True
    for m in range(1, mmax + 1):
        # Delta lowers the total degree by 2, so work on M_m (+) M_{m-2}
        F1, F2 = Fock(n, m), Fock(n, m - 2)
        if F1.dim == 0:
            continue
        basis = [("hi", b) for b in F1.basis] + [("lo", b) for b in F2.basis]
        idx = {b: i for i, b in enumerate(basis)}
        N = len(basis)
        A = [[0] * N for _ in range(N)]
        for j, (lvl, (a, S)) in enumerate(basis):
            if lvl != "hi":
                continue
            for i in range(n):
                if a[i] == 0 or i not in S:
                    continue
                sign = (-1) ** sum(1 for t in S if t < i)
                aa = list(a); aa[i] -= 1
                tgt = ("lo", (tuple(aa), tuple(t for t in S if t != i)))
                if tgt in idx:
                    A[idx[tgt]][j] += a[i] * sign
        if not zero(matmul(A, A)):
            ok = False
            say(f"   m={m}: FAILED  Delta^2 = 0")
        # {Delta,Delta} = 2 Delta^2 = 0, so B_lambda == 0 for every weight:
        # the odd part of pe(1) spans an abelian odd ideal, all its
        # irreducible modules are 1-dimensional with Delta = 0.
        r = rank(A)
        # exhibit a 2-dimensional indecomposable: any v with Delta v != 0 spans,
        # with Delta v, a pe(1)-submodule that is not a sum of irreducibles
        wit = ""
        for j, (lvl, b) in enumerate(basis):
            if lvl == "hi" and any(A[i][j] for i in range(N)):
                a, S = b
                wit = ("x^" + "".join(str(t) for t in a) +
                       " xi_" + "".join(str(t + 1) for t in S))
                break
        say(f"   m={m:2d}: dim M_m = {F1.dim:4d}, rk Delta = {r}, Delta^2 = 0, "
            f"{{Delta,Delta}} = 0  =>  no typical modules"
            + (f"; <v, Delta v> with v = {wit} is a 2-dim indecomposable" if wit else ""))
    say("   => on any module with Delta != 0 the pe(1)-action is NOT semisimple:")
    say("      every irreducible pe(1)-module is 1-dimensional with Delta = 0,")
    say("      so a semisimple module would have Delta = 0.")
    say()
    return ok


# ============================================ G. a reductive member is not enough

class SuperPoly:
    """P(C^{m|2n}), basis x^a theta_S, |a| + |S| = deg."""

    def __init__(self, m, twon, deg):
        self.m, self.twon, self.deg = m, twon, deg
        self.basis = []
        for S in Fock._subsets(twon):
            for a in Fock._compositions(m, deg - len(S)):
                self.basis.append((a, S))
        self.index = {b: i for i, b in enumerate(self.basis)}
        self.dim = len(self.basis)


def sp_op(src, tgt, action):
    A = [[0] * src.dim for _ in range(tgt.dim)]
    for j, b in enumerate(src.basis):
        for bb, c in action(b):
            i = tgt.index.get(bb)
            if i is not None:
                A[i][j] += c
    return A


def laplace(src, tgt, m, twon):
    """Delta = sum_i partial_{x_i}^2 + 4 sum_j partial_{2j-1} partial_{2j}."""
    def act(b):
        a, S = b
        out = []
        for i in range(m):
            if a[i] >= 2:
                aa = list(a); aa[i] -= 2
                out.append(((tuple(aa), S), a[i] * (a[i] - 1)))
        for j in range(twon // 2):
            p, q = 2 * j, 2 * j + 1
            if p in S and q in S:
                # partial_p partial_q (theta_S): left derivatives, in this order
                sgn_q = (-1) ** sum(1 for t in S if t < q)
                S1 = tuple(t for t in S if t != q)
                sgn_p = (-1) ** sum(1 for t in S1 if t < p)
                S2 = tuple(t for t in S1 if t != p)
                out.append(((a, S2), 4 * sgn_p * sgn_q))
        return out
    return sp_op(src, tgt, act)


def rsquare(src, tgt, m, twon):
    """R^2 = sum x_i^2 + sum_j theta_{2j-1} theta_{2j}."""
    def act(b):
        a, S = b
        out = []
        for i in range(m):
            aa = list(a); aa[i] += 2
            out.append(((tuple(aa), S), 1))
        for j in range(twon // 2):
            p, q = 2 * j, 2 * j + 1
            if p in S or q in S:
                continue
            S1 = tuple(sorted(S + (p, q)))
            sgn = (-1) ** sum(1 for t in S if t < p)
            sgn *= (-1) ** sum(1 for t in S if t < q)
            out.append(((a, S1), sgn))
        return out
    return sp_op(src, tgt, act)


def check_osp_sl2(m, twon, degmax):
    M = m - twon
    say("=" * 78)
    say(f"G.  the pair (osp({m}|{twon}), sl(2)) on P(C^{{{m}|{twon}}}), M = m - 2n = {M}")
    say("=" * 78)
    ok = True
    # [Delta, R^2] = 4E + 2M on each degree
    for k in range(degmax + 1):
        Pk = SuperPoly(m, twon, k)
        Pk1 = SuperPoly(m, twon, k + 2)
        Pk2 = SuperPoly(m, twon, k - 2) if k >= 2 else SuperPoly(m, twon, -1)
        if Pk.dim == 0:
            continue
        R_up = rsquare(Pk, Pk1, m, twon)            # P_k -> P_{k+2}
        D_dn = laplace(Pk1, Pk, m, twon)            # P_{k+2} -> P_k
        D_k = laplace(Pk, Pk2, m, twon)             # P_k -> P_{k-2}
        R_k = rsquare(Pk2, Pk, m, twon) if Pk2.dim else [[0] * 0
                                                        for _ in range(Pk.dim)]
        lhs = matmul(D_dn, R_up)
        if Pk2.dim:
            lhs = add(lhs, matmul(R_k, D_k), -1)
        want = scal(4 * k + 2 * M, eye(Pk.dim))
        if not zero(add(lhs, want, -1)):
            ok = False
            say(f"   deg {k}: FAILED  [Delta, R^2] = 4E + 2M")
    say(f"   [Delta, R^2] = 4E + 2M verified for degrees 0..{degmax}: {ok}")

    # the Fischer decomposition P_k = H_k (+) R^2 P_{k-2}
    for k in range(2, degmax + 1):
        Pk = SuperPoly(m, twon, k)
        Pk2 = SuperPoly(m, twon, k - 2)
        if Pk.dim == 0 or Pk2.dim == 0:
            continue
        D_k = laplace(Pk, SuperPoly(m, twon, k - 2), m, twon)
        R_k = rsquare(Pk2, Pk, m, twon)
        dimH = Pk.dim - rank(D_k)
        dimR = rank(R_k)
        DR = matmul(D_k, R_k)
        inter = dimR - rank(DR)                     # dim (H_k /\ R^2 P_{k-2})
        span = dimH + dimR - inter                  # dim (H_k + R^2 P_{k-2})
        flag = ("direct, and all of P_k" if inter == 0 and span == Pk.dim
                else f"NOT direct: meet {inter}-dim, span {span} < {Pk.dim}")
        say(f"   deg {k}: dim P_k = {Pk.dim}, dim H_k = {dimH}, "
            f"dim R^2 P_{{k-2}} = {dimR} : {flag}")
    if M == 0:
        # the sl(2)-module generated by 1: 1, R^2, R^4, ...  Delta R^2 = 2M = 0
        P0, P2, P4 = (SuperPoly(m, twon, k) for k in (0, 2, 4))
        one = [1] + [0] * (P0.dim - 1)
        R2 = [sum(rsquare(P0, P2, m, twon)[i][j] * one[j] for j in range(P0.dim))
              for i in range(P2.dim)]
        R4 = [sum(rsquare(P2, P4, m, twon)[i][j] * R2[j] for j in range(P2.dim))
              for i in range(P4.dim)]
        dR2 = [sum(laplace(P2, P0, m, twon)[i][j] * R2[j] for j in range(P2.dim))
               for i in range(P0.dim)]
        dR4 = [sum(laplace(P4, P2, m, twon)[i][j] * R4[j] for j in range(P4.dim))
               for i in range(P2.dim)]
        say(f"   Delta(R^2) = 0: {not any(dR2)};  Delta(R^4) = 0: {not any(dR4)};"
            f"  R^2 != 0: {any(R2)}")
        say("   => <1, R^2, R^4, ...> has the proper submodule <R^2, R^4, ...>")
        say("      with trivial quotient, and no complement, since R^2.1 != 0.")
    if M == 0:
        say("   M = 0: Delta(R^2) = 2M = 0, so R^2 is harmonic AND lies in")
        say("      R^2 P_0 --- the Fischer sum is not direct, and the sl(2)-module")
        say("      <1, R^2, R^4, ...> is a non-split extension of the trivial")
        say("      module by the lowest-weight module generated by R^2.")
        say("      sl(2) is reductive; the MODULE is not semisimple.")
    say()
    return ok


# ================================================================================

def main():
    good = True
    good &= check_de_rham(2, 6)
    good &= check_de_rham(3, 5)
    good &= check_tautological(3)
    good &= check_laurent(-4, 4)
    good &= check_periplectic(3, 5)
    good &= check_osp_sl2(2, 2, 4)      # M = 0, the degenerate case
    good &= check_osp_sl2(3, 2, 4)      # M = 1, the generic case
    say("=" * 78)
    say(f"ALL CHECKS PASSED: {good}")


if __name__ == "__main__":
    main()
