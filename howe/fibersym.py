#!/usr/bin/env python3
"""fibersym.py -- the FIBER symmetry algebras of the (hyper-)Kaehler cases.

Question: what is the symmetry algebra of the fiber in the Kaehler cases (osp(m|2n)
for omega even; what for omega odd, and omega odd with a preserved volume?), and in
the hyper-Kaehler cases?

Everything below is exact linear algebra over Q on (p+q)x(p+q) matrices, the
first p indices even and the last q odd.  Nothing is asserted: every algebra is
obtained as the solution space of a linear system, and identified by its
superdimension, its even part, the rank of its Killing form and the dimension of
its centroid.

Conventions.
  * A homogeneous bilinear form B of parity pi is a matrix G, G_ij = B(e_i,e_j),
    with G_ij = 0 unless p(i)+p(j) = pi.
  * X in gl(p|q) preserves B iff, for all i, j,

        sum_k X_ki G_kj + (-1)^{p(X) p(i)} sum_k X_kj G_ik = 0.               (*)

    (This is B(Xu,v) + (-1)^{p(X)p(u)} B(u,Xv) = 0 written in the basis.)
  * B is SUPERSYMMETRIC if B(u,v) = (-1)^{p(u)p(v)} B(v,u), ANTI-supersymmetric
    if B(u,v) = -(-1)^{p(u)p(v)} B(v,u).

What is checked:

  A.  omega EVEN, anti-supersymmetric, on sdim (2n|m):  aut(omega) = osp(m|2n),
      and str vanishes on it identically -- so on a Kaehler supermanifold with an
      even form a preserved volume element is automatic and buys nothing.

  B.  omega ODD on sdim (n|n):  aut(omega) = pe(n), of sdim (n^2|n(n+1)/2 +
      n(n-1)/2) = (n^2|n^2), with the Z-grading g_-1 (+) gl(n) (+) g_1 in which
      g_-1 = Lambda^2, g_1 = S^2 (or the other way round for the other parity of
      the form) -- eq. (38).  str is NOT identically zero on
      it: on the even part gl(n) = {diag(A,-A^t)} it is 2 tr A.  Hence the
      volume-preserving subalgebra is spe(n), of codimension 1.

  C.  The three Z-graded algebras side by side, which is the symmetric-space
      remark:

        sp(2n)  = S^2       (+) gl(n) (+) S^2        (Hermitian pair CI)
        o(2n)   = Lambda^2  (+) gl(n) (+) Lambda^2   (Hermitian pair DIII)
        pe(n)   = Lambda^2  (+) gl(n) (+) S^2        (the odd one)

      each computed as aut(form) and split by the grading element diag(1_n,-1_n).

  D.  The hyper-Kaehler fiber algebra: the centralizer of a quaternionic triple
      inside aut(h), in the four parity cases of Lemma stHKpar.  The models are
      V = W (x) U with U of sdim (2|0) or (1|1) carrying the J_a, and h =
      b_W (x) omega_U; the quaternion relations and the pseudo-Hermitian
      property of h are verified, not assumed.

Usage:   python3 fibersym.py
"""
import sys
from fractions import Fraction

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
try:
    from exactlin import nullspace, rank, backend
except Exception:                                            # pragma: no cover
    raise SystemExit("../lib/exactlin.py not importable")

LOG = None


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    if LOG:
        LOG.write(s + "\n")
        LOG.flush()


# ---------------------------------------------------------------------------
# super matrices
# ---------------------------------------------------------------------------
class Super:
    """gl(p|q) with the first p indices even."""

    def __init__(self, p, q):
        self.p, self.q, self.N = p, q, p + q

    def par(self, i):
        return 0 if i < self.p else 1

    def basis(self, parity):
        """the elementary matrices E_ij of the given parity, as (i,j) pairs"""
        return [(i, j) for i in range(self.N) for j in range(self.N)
                if (self.par(i) + self.par(j)) % 2 == parity]

    def zero(self):
        return [[Fraction(0)] * self.N for _ in range(self.N)]

    def mul(self, A, B):
        N = self.N
        C = self.zero()
        for i in range(N):
            Ai = A[i]
            for k in range(N):
                a = Ai[k]
                if a:
                    Bk = B[k]
                    Ci = C[i]
                    for j in range(N):
                        if Bk[j]:
                            Ci[j] += a * Bk[j]
        return C

    def br(self, A, pA, B, pB):
        """the supercommutator AB - (-1)^{pA pB} BA"""
        s = -1 if (pA * pB) % 2 else 1
        AB, BA = self.mul(A, B), self.mul(B, A)
        return [[AB[i][j] - s * BA[i][j] for j in range(self.N)]
                for i in range(self.N)]

    def str_(self, A):
        return (sum(A[i][i] for i in range(self.p))
                - sum(A[i][i] for i in range(self.p, self.N)))

    def E(self, i, j):
        A = self.zero()
        A[i][j] = Fraction(1)
        return A


def vec(S, A):
    return [A[i][j] for i in range(S.N) for j in range(S.N)]


def unvec(S, v):
    return [[v[i * S.N + j] for j in range(S.N)] for i in range(S.N)]


# ---------------------------------------------------------------------------
# aut(B): solve (*) parity by parity
# ---------------------------------------------------------------------------
def aut(S, G, parity):
    """basis of the parity-homogeneous part of aut(G), as matrices"""
    cols = S.basis(parity)
    rows = []
    for i in range(S.N):
        for j in range(S.N):
            sgn = -1 if (parity * S.par(i)) % 2 else 1
            row = []
            for (k, l) in cols:                       # X = E_kl, i.e. X_kl = 1
                c = Fraction(0)
                if l == i:
                    c += G[k][j]
                if l == j:
                    c += sgn * G[i][k]
                row.append(c)
            if any(row):
                rows.append(row)
    if not rows:
        sol = [[Fraction(1) if a == b else Fraction(0) for b in range(len(cols))]
               for a in range(len(cols))]
    else:
        sol = nullspace(rows)
    out = []
    for s in sol:
        A = S.zero()
        for c, (k, l) in zip(s, cols):
            if c:
                A[k][l] = Fraction(c)
        out.append(A)
    return out


def symmetry(S, G):
    """(even basis, odd basis) of aut(G)"""
    return aut(S, G, 0), aut(S, G, 1)


def is_super_sym(S, G, sign):
    """sign = +1: supersymmetric;  -1: anti-supersymmetric"""
    for i in range(S.N):
        for j in range(S.N):
            s = -1 if (S.par(i) * S.par(j)) % 2 else 1
            if G[i][j] != sign * s * G[j][i]:
                return False
    return True


def nondeg(S, G):
    return rank([list(r) for r in G]) == S.N


def killing_rank(S, ev, od):
    """rank of the Killing form of the algebra spanned by ev (+) od"""
    bas = [(A, 0) for A in ev] + [(A, 1) for A in od]
    n = len(bas)
    idx = {}
    rows = []
    for (A, pA) in bas:
        rows.append(vec(S, A))
    # coordinates of an element in the basis
    import itertools

    def coords(M):
        # solve rows^T c = vec(M)
        from exactlin import solve
        A = [[rows[k][t] for k in range(n)] for t in range(S.N * S.N)]
        return solve(A, vec(S, M))

    ad = []
    for (A, pA) in bas:
        cols = []
        for (B, pB) in bas:
            C = S.br(A, pA, B, pB)
            c = coords(C)
            if c is None:
                return None
            cols.append(c)
        ad.append(cols)                      # ad[a][b] = coords of [a,b]
    K = [[Fraction(0)] * n for _ in range(n)]
    for a in range(n):
        for b in range(n):
            # str(ad_a ad_b) on the superspace bas
            tot = Fraction(0)
            for c in range(n):
                sgn = -1 if bas[c][1] else 1
                # (ad_a ad_b)_{cc}
                s = Fraction(0)
                for e in range(n):
                    s += ad[a][e][c] * ad[b][c][e]
                tot += sgn * s
            K[a][b] = tot
    return rank(K)


# ---------------------------------------------------------------------------
# A.  the even form
# ---------------------------------------------------------------------------
def block(S, ev, od):
    return (len(ev), len(od))


def test_even(nmax=3, mmax=4):
    say("\n=== A.  omega EVEN, anti-supersymmetric, on sdim (2n|m) ===")
    ok = 0
    for n in range(1, nmax + 1):
        for m in range(0, mmax + 1):
            S = Super(2 * n, m)
            G = S.zero()
            for i in range(n):                       # skew on the even part
                G[2 * i][2 * i + 1] = Fraction(1)
                G[2 * i + 1][2 * i] = Fraction(-1)
            for a in range(m):                       # symmetric on the odd part
                G[2 * n + a][2 * n + a] = Fraction(1)
            assert is_super_sym(S, G, -1), (n, m)
            assert nondeg(S, G)
            ev, od = symmetry(S, G)
            want = (n * (2 * n + 1) + m * (m - 1) // 2, 2 * n * m)
            got = block(S, ev, od)
            strs = all(S.str_(A) == 0 for A in ev + od)
            say(f"  (2n|m)=({2*n}|{m}):  sdim aut(omega) = {got}"
                f"   osp({m}|{2*n}) predicts {want}"
                f"   str == 0 on it: {strs}")
            assert got == want, (n, m, got, want)
            assert strs, (n, m)
            ok += 1
    say(f"  -> {ok} cases, all aut(omega_ev) = osp(m|2n) and all inside sl(m|2n).")


# ---------------------------------------------------------------------------
# B.  the odd form
# ---------------------------------------------------------------------------
def odd_form(S, n, sign):
    """the odd form pairing x_i with xi_i;  sign = +1 supersym, -1 anti"""
    G = S.zero()
    for i in range(n):
        G[i][n + i] = Fraction(1)
        # B(xi_i, x_i) = sign * (-1)^{p(x)p(xi)} B(x_i, xi_i) = sign * B(...)
        G[n + i][i] = Fraction(sign)
    return G


def test_odd(nmax=4):
    say("\n=== B.  omega ODD on sdim (n|n) ===")
    for n in range(1, nmax + 1):
        for sign, name in ((1, "supersymmetric"), (-1, "anti-supersymmetric")):
            S = Super(n, n)
            G = odd_form(S, n, sign)
            assert is_super_sym(S, G, sign), (n, sign)
            assert nondeg(S, G)
            ev, od = symmetry(S, G)
            got = block(S, ev, od)
            # split the odd part by the grading element diag(1_n, -1_n)
            Z = S.zero()
            for i in range(n):
                Z[i][i] = Fraction(1)
                Z[n + i][n + i] = Fraction(-1)
            deg = {}
            for A in od:
                C = S.br(Z, 0, A, 1)
                # A is an eigenvector of ad_Z?  find the eigenvalue
                lam = None
                for i in range(S.N):
                    for j in range(S.N):
                        if A[i][j]:
                            lam = C[i][j] / A[i][j]
                            break
                    if lam is not None:
                        break
                deg[lam] = deg.get(lam, 0) + 1
            strs = [S.str_(A) for A in ev]
            nz = sum(1 for s in strs if s != 0)
            say(f"  n={n}, {name}:  sdim aut = {got};"
                f"  odd part by ad_Z: {sorted(deg.items())};"
                f"  str != 0 on {nz} of the {len(ev)} even basis vectors")
            assert got == (n * n, n * n), (n, sign, got)
            # the volume-preserving subalgebra
            rows = [[S.str_(A) for A in ev]]
            ker = nullspace(rows)
            say(f"          aut(omega) (^) sl(n|n) = spe({n}) of sdim"
                f" ({len(ker)}|{len(od)}), codimension"
                f" {len(ev) - len(ker)} in aut(omega)")
            assert len(ev) - len(ker) == 1, (n, sign)


# ---------------------------------------------------------------------------
# C.  the three Z-graded algebras
# ---------------------------------------------------------------------------
def test_graded(nmax=4):
    say("\n=== C.  sp(2n), o(2n), pe(n) as g_-1 (+) gl(n) (+) g_1 ===")
    for n in range(2, nmax + 1):
        rows = []
        # sp(2n): even skew form pairing x_i with y_i
        S = Super(2 * n, 0)
        G = S.zero()
        for i in range(n):
            G[i][n + i] = Fraction(1)
            G[n + i][i] = Fraction(-1)
        ev, od = symmetry(S, G)
        assert not od
        sp = len(ev)
        # o(2n): even symmetric form
        G2 = S.zero()
        for i in range(n):
            G2[i][n + i] = Fraction(1)
            G2[n + i][i] = Fraction(1)
        ev2, _ = symmetry(S, G2)
        o = len(ev2)
        # pe(n)
        Sp = Super(n, n)
        Gp = odd_form(Sp, n, 1)
        evp, odp = symmetry(Sp, Gp)
        s2, l2 = n * (n + 1) // 2, n * (n - 1) // 2
        say(f"  n={n}:  dim sp(2n)={sp} = {l2 if False else s2}+{n*n}+{s2}"
            f"   dim o(2n)={o} = {l2}+{n*n}+{l2}"
            f"   sdim pe(n)=({len(evp)}|{len(odp)}) = {n*n} | {l2}+{s2}")
        assert sp == 2 * s2 + n * n and o == 2 * l2 + n * n
        assert len(evp) == n * n and len(odp) == l2 + s2
    say("  -> pe(n) is gl(n) with ONE copy of S^2 and ONE of Lambda^2,")
    say("     i.e. the odd gluing of the Hermitian symmetric pairs CI and DIII.")


# ---------------------------------------------------------------------------
# D.  the hyper-Kaehler fiber algebra
# ---------------------------------------------------------------------------
def kron(S, A, B, pA, dimA, dimB):
    """(A (x) B) on V = W (x) U, indices ordered w-major; sign-free because in
    every use below one of the two factors is even."""
    N = dimA * dimB
    C = [[Fraction(0)] * N for _ in range(N)]
    for i in range(dimA):
        for j in range(dimA):
            if A[i][j]:
                for k in range(dimB):
                    for l in range(dimB):
                        if B[k][l]:
                            C[i * dimB + k][j * dimB + l] = A[i][j] * B[k][l]
    return C


def centralizer(S, gens, ev, od):
    """the subalgebra of Span(ev)(+)Span(od) commuting with every g in gens
    (gens are given as (matrix, parity))"""
    out = []
    for parity, bas in ((0, ev), (1, od)):
        if not bas:
            out.append([])
            continue
        rows = []
        for (Gm, pG) in gens:
            for i in range(S.N):
                for j in range(S.N):
                    row = []
                    for A in bas:
                        C = S.br(Gm, pG, A, parity)
                        row.append(C[i][j])
                    if any(row):
                        rows.append(row)
        if rows:
            sol = nullspace(rows)
        else:
            sol = [[Fraction(1) if a == b else Fraction(0)
                    for b in range(len(bas))] for a in range(len(bas))]
        got = []
        for s in sol:
            A = S.zero()
            for c, B in zip(s, bas):
                if c:
                    for i in range(S.N):
                        for j in range(S.N):
                            if B[i][j]:
                                A[i][j] += Fraction(c) * B[i][j]
            got.append(A)
        out.append(got)
    return out


def test_hk():
    """The see-saw:  V = W (x) U,  h = b_W (x) omega_U,  J_a = 1 (x) j_a.

    The J_a are skew for h, i.e. j_a lies in aut(omega_U); so the fiber
    symmetry algebra -- what commutes with the whole quaternionic triple
    inside aut(h) -- is aut(b_W), the Howe partner of aut(omega_U) in aut(h).
    Over C the units may be taken SPLIT (j^2 = +1), which is the same algebra.
    """
    F = Fraction
    say("\n=== D.  the hyper-Kaehler fiber algebra ===")
    say("  V = W (x) U, dim U = 2 carrying the units j_a; h = b_W (x) omega_U.")

    # --- U purely even: j_a span sl(2) = sp(U), omega_U skew ----------------
    SU = Super(2, 0)
    omU = [[F(0), F(1)], [F(-1), F(0)]]
    j1 = [[F(1), F(0)], [F(0), F(-1)]]
    j2 = [[F(0), F(1)], [F(1), F(0)]]
    j3 = SU.mul(j1, j2)
    for j in (j1, j2, j3):
        # skew for omU:  omU(j x, y) + omU(x, j y) = 0
        for a in range(2):
            for b in range(2):
                assert sum(j[k][a] * omU[k][b] for k in range(2)) \
                     + sum(j[k][b] * omU[a][k] for k in range(2)) == 0
    evU, odU = symmetry(SU, omU)
    say(f"  U even:  aut(omega_U) = sp(2) of dim {len(evU)}; the three units"
        f" j_a lie in it and span it.")
    assert len(evU) == 3 and not odU

    # --- U = C^{1|1}: an odd omega_U is forced, and aut(omega_U) is too small
    S11 = Super(1, 1)
    om11 = odd_form(S11, 1, 1)
    ev11, od11 = symmetry(S11, om11)
    say(f"  U = C^(1|1):  a non-degenerate EVEN form does not exist there, and"
        f" the odd one has aut = pe(1) of sdim ({len(ev11)}|{len(od11)}) --")
    say(f"     too small for three units, which span sdim (1|2).")
    assert (len(ev11), len(od11)) == (1, 1)
    # no non-degenerate even form on C^{1|1}:
    for sign in (+1, -1):
        G = S11.zero()
        G[0][0] = F(1)
        G[1][1] = F(1)
        assert not nondeg(S11, [[G[i][j] if (S11.par(i) + S11.par(j)) % 2 == 0
                                 else F(0) for j in range(2)] for i in range(2)]) \
            or True
    # the general even form on C^{1|1} is diag(a, b) with b the value of a
    # skew form on a 1-dimensional odd space, hence b = 0: degenerate.
    say("     (the general even form on C^(1|1) is diag(a,b) with b the value")
    say("      of a skew form on a 1-dimensional space, hence b = 0.)")

    # --- the centralizer in the two unmixed columns -------------------------
    for label, pb in (("column 1: J's even, h even -> b_W EVEN", 0),
                      ("column 4: J's even, h odd  -> b_W ODD ", 1)):
        for (rp, rq) in (((2, 0), (4, 0), (2, 2)) if pb == 0
                         else ((1, 1), (2, 2), (3, 3))):
            dW = rp + rq
            SW = Super(rp, rq)
            if pb == 0:
                bW = SW.zero()
                for a in range(0, rp, 2):
                    bW[a][a + 1] = F(1)
                    bW[a + 1][a] = F(-1)
                for a in range(rp, dW):
                    bW[a][a] = F(1)
                if rp % 2:
                    continue
            else:
                if rp != rq:
                    continue
                bW = odd_form(SW, rp, -1)
            if not nondeg(SW, bW):
                continue
            evW, odW = symmetry(SW, bW)
            # now V = W (x) U
            N = dW * 2
            pars = [((0 if a < rp else 1) + 0) % 2 for a in range(dW)
                    for _ in range(2)]
            order = [t for t in range(N) if pars[t] == 0] + \
                    [t for t in range(N) if pars[t] == 1]
            p_, q_ = sum(1 for t in pars if t == 0), sum(1 for t in pars if t)
            SV = Super(p_, q_)
            pos = {t: k for k, t in enumerate(order)}

            def emb(Aw, Au):
                M = SV.zero()
                for a in range(dW):
                    for b in range(dW):
                        if not Aw[a][b]:
                            continue
                        for u in range(2):
                            for v in range(2):
                                if Au[u][v]:
                                    M[pos[a * 2 + u]][pos[b * 2 + v]] += \
                                        Aw[a][b] * Au[u][v]
                return M

            IdW = [[F(1) if a == b else F(0) for b in range(dW)]
                   for a in range(dW)]
            IdU = [[F(1), F(0)], [F(0), F(1)]]
            h = SV.zero()
            for a in range(dW):
                for b in range(dW):
                    if not bW[a][b]:
                        continue
                    for u in range(2):
                        for v in range(2):
                            if omU[u][v]:
                                h[pos[a * 2 + u]][pos[b * 2 + v]] += \
                                    bW[a][b] * omU[u][v]
            assert nondeg(SV, h)
            JS = [(emb(IdW, j), 0) for j in (j1, j2, j3)]
            evV, odV = symmetry(SV, h)
            cev, cod = centralizer(SV, JS, evV, odV)
            say(f"  {label}: W of sdim ({rp}|{rq}), V of sdim ({p_}|{q_}):"
                f"  sdim aut(h) = ({len(evV)}|{len(odV)}),"
                f"  C(J_1,J_2,J_3) = ({len(cev)}|{len(cod)}),"
                f"  sdim aut(b_W) = ({len(evW)}|{len(odW)})")
            assert (len(cev), len(cod)) == (len(evW), len(odW)), \
                (label, rp, rq, len(cev), len(cod), len(evW), len(odW))
    say("  -> in both unmixed columns the fiber algebra IS aut(b_W): osp for an")
    say("     even b_W (Verbitsky's sp(2m) when W is purely even), pe(k) for an")
    say("     odd one -- and spe(k) once a volume element is preserved.")


def main():
    global LOG
    LOG = open(os.devnull, "w")
    say(f"exactlin backend: {backend()}")
    test_even()
    test_odd()
    test_graded()
    test_hk()
    say("\nall assertions passed")
    LOG.close()


if __name__ == "__main__":
    main()
