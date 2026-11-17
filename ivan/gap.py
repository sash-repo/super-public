#!/usr/bin/env python3
"""
The gap of Subsec. 1.4.2 of the text: anti-automorphisms with
*^2 = id (superinvolutions, condition (2) = GASInv) versus those with only
*^4 = id (condition (1) = antiInv).

For every map printed in Subsec. 1.4 we check, by exact arithmetic over Q(i)
and on the nose (all pairs of basis elements):

  1) is it an anti-automorphism in the super sense
        phi(uv) = (-1)^{p(u)p(v)} phi(v) phi(u) ?
  2) is it an automorphism  phi(uv) = phi(u)phi(v) ?
  3) what is phi^2 -- id, delta_{-1}: X -> (-1)^{p(X)}X, or neither?
  4) what is K = {a | a^* = -a}: its sdim, whether K_odd = 0,
     whether K is a Lie sub-superalgebra, its derived algebra, its center,
     and -- the point -- whether SL(K) = K'/(Z cap K') is simple.

The conclusion the run prints is the answer to the question of Subsec. 1.4.2:
the extension of Theorem 1.4.1 from *^2 = id to *^4 = id is FALSE, and the
counterexample is the plain supertransposition st on Mat(a|b).

Exact symbolic arithmetic throughout (sympy, ground field Q(i)); no numerics.
"""
import sys
import itertools
import sympy as sp

# ----------------------------------------------------------------- superspace


class Sup:
    """Mat(a|b): basis E_ij, parity p(i)+p(j), p(i)=0 for i<a."""

    def __init__(self, a, b):
        self.a, self.b, self.n = a, b, a + b
        self.par = [0] * a + [1] * b

    def p_index(self, i):
        return self.par[i]

    def parity(self, M):
        """parity of a homogeneous matrix, or None if not homogeneous."""
        ev = od = False
        for i in range(self.n):
            for j in range(self.n):
                if M[i, j] != 0:
                    if (self.par[i] + self.par[j]) % 2 == 0:
                        ev = True
                    else:
                        od = True
        if ev and od:
            return None
        return 1 if od else 0

    def basis(self):
        out = []
        for i in range(self.n):
            for j in range(self.n):
                M = sp.zeros(self.n, self.n)
                M[i, j] = 1
                out.append((M, (self.par[i] + self.par[j]) % 2))
        return out

    def blocks(self, X):
        a = self.a
        return X[:a, :a], X[:a, a:], X[a:, :a], X[a:, a:]

    def assemble(self, A, B, C, D):
        a, b = self.a, self.b
        M = sp.zeros(self.n, self.n)
        M[:a, :a] = A
        M[:a, a:] = B
        M[a:, :a] = C
        M[a:, a:] = D
        return M


def delta_m1(S, X):
    A, B, C, D = S.blocks(X)
    return S.assemble(A, -B, -C, D)


# ------------------------------------------------------------------- the maps


def st_printed(S, X):
    """the supertransposition as displayed in Subsec. 1.4:
         (A B / C D) -> (A^t  -C^t / B^t  D^t)"""
    A, B, C, D = S.blocks(X)
    return S.assemble(A.T, -C.T, B.T, D.T)


def Pi_map(S, X):
    """(A B / C D) -> (D C / B A); needs a = b."""
    A, B, C, D = S.blocks(X)
    return S.assemble(D, C, B, A)


def minus_Pi(S, X):
    return -Pi_map(S, X)


def minus_delta(S, X):
    return -delta_m1(S, X)


def star_form(S, G):
    """X -> G^{-1} X^{st} G : the supertransposition conjugated by the even
    non-degenerate supersymmetric form G.  This is the map that really
    singles out osp."""
    Gi = G.inv()

    def phi(S_, X):
        return Gi * st_printed(S_, X) * G
    return phi


def osp_form(a, b):
    """G = diag(1_a, J_b) with J_b the standard symplectic matrix (b even)."""
    assert b % 2 == 0
    n = a + b
    G = sp.zeros(n, n)
    for i in range(a):
        G[i, i] = 1
    for k in range(b // 2):
        G[a + 2 * k, a + 2 * k + 1] = 1
        G[a + 2 * k + 1, a + 2 * k] = -1
    return G


# --------------------------------------------------------------------- Q(n)
# Q(n) = {A + B j} inside Mat(n|n), j = (0 1 / 1 0) odd, j^2 = 1.
# (The J of the text has J^2 = -1: J = i*j.  Both are used below where needed.)


def Q_basis(n):
    """basis of Q(n) inside Mat(n|n) as (matrix, parity) pairs."""
    out = []
    for i in range(n):
        for j in range(n):
            A = sp.zeros(n, n)
            A[i, j] = 1
            Z = sp.zeros(2 * n, 2 * n)
            Z[:n, :n] = A
            Z[n:, n:] = A
            out.append((Z, 0))
    for i in range(n):
        for j in range(n):
            B = sp.zeros(n, n)
            B[i, j] = 1
            Z = sp.zeros(2 * n, 2 * n)
            Z[:n, n:] = B
            Z[n:, :n] = B
            out.append((Z, 1))
    return out


def minus_q(S, X):
    """the anti-automorphism -q of eq. (5):
         (A B / B A) -> (A^t  -i B^t / -i B^t  A^t)."""
    A, B, C, D = S.blocks(X)
    return S.assemble(A.T, -sp.I * B.T, -sp.I * B.T, A.T)


# -------------------------------------------------------------- the checkers


def is_anti(S, phi, basis):
    for (u, pu), (v, pv) in itertools.product(basis, repeat=2):
        lhs = phi(S, u * v)
        rhs = sp.Integer(-1) ** (pu * pv) * (phi(S, v) * phi(S, u))
        if sp.simplify(sp.Matrix(lhs - rhs)) != sp.zeros(S.n, S.n):
            return False
    return True


def is_auto(S, phi, basis):
    for (u, pu), (v, pv) in itertools.product(basis, repeat=2):
        if sp.simplify(sp.Matrix(phi(S, u * v) - phi(S, u) * phi(S, v))) != \
                sp.zeros(S.n, S.n):
            return False
    return True


def square_of(S, phi, basis):
    """identify phi^2 as id / delta_{-1} / neither."""
    isid = isdelta = True
    for (u, pu) in basis:
        sq = phi(S, phi(S, u))
        if sp.simplify(sp.Matrix(sq - u)) != sp.zeros(S.n, S.n):
            isid = False
        if sp.simplify(sp.Matrix(sq - delta_m1(S, u))) != sp.zeros(S.n, S.n):
            isdelta = False
    if isid:
        return "id"
    if isdelta:
        return "delta_{-1}"
    return "neither"


def anti_symmetric(S, phi, basis):
    """K = {a | phi(a) = -a}, returned as a list of (matrix, parity)
    spanning the even and the odd part separately."""
    out = []
    for par in (0, 1):
        vecs = [b for b, p in basis if p == par]
        if not vecs:
            continue
        cols = []
        for M in vecs:
            im = phi(S, M) + M
            cols.append([sp.expand(im[i, j])
                         for i in range(S.n) for j in range(S.n)])
        A = sp.Matrix(cols).T                      # columns = images
        ns = A.nullspace()
        for v in ns:
            K = sp.zeros(S.n, S.n)
            for c, M in zip(v, vecs):
                K = K + sp.expand(c) * M
            out.append((sp.simplify(K), par))
    return out


def sbracket(X, Y, px, py):
    return X * Y - sp.Integer(-1) ** (px * py) * Y * X


def span_dim(mats, n):
    if not mats:
        return 0
    rows = [[sp.expand(M[i, j]) for i in range(n) for j in range(n)]
            for M in mats]
    return sp.Matrix(rows).rank()


def in_span(M, mats, n):
    if not mats:
        return all(sp.simplify(M[i, j]) == 0 for i in range(n)
                   for j in range(n))
    return span_dim(mats + [M], n) == span_dim(mats, n)


def derived(S, K):
    br = []
    for (X, px), (Y, py) in itertools.product(K, repeat=2):
        br.append(sbracket(X, Y, px, py))
    return br


def reduce_basis(mats, n):
    """extract a basis of the span, keeping the matrices themselves."""
    keep = []
    for M in mats:
        if sp.simplify(sp.Matrix(M)) == sp.zeros(n, n):
            continue
        if not in_span(M, keep, n):
            keep.append(M)
    return keep


def generated_ideal(S, K, x):
    """the ideal of the Lie superalgebra K generated by x."""
    n = S.n
    cur = [x]
    changed = True
    while changed:
        changed = False
        for (Y, py) in K:
            for M in list(cur):
                pm = S.parity(M)
                pm = 0 if pm is None else pm
                for pM in ((0, 1) if S.parity(M) is None else (pm,)):
                    br = sbracket(Y, M, py, pM)
                    if sp.simplify(sp.Matrix(br)) == sp.zeros(n, n):
                        continue
                    if not in_span(br, cur, n):
                        cur.append(br)
                        changed = True
    return reduce_basis(cur, n)


def sdim(S, K):
    ev = [M for M, p in K if p == 0]
    od = [M for M, p in K if p == 1]
    return span_dim(ev, S.n), span_dim(od, S.n)


def report(name, S, phi, basis, note=""):
    anti = is_anti(S, phi, basis)
    auto = is_auto(S, phi, basis)
    sq = square_of(S, phi, basis)
    K = anti_symmetric(S, phi, basis)
    de, do = sdim(S, K)
    print(f"  {name:<28} anti-auto={str(anti):<5} auto={str(auto):<5} "
          f"*^2={sq:<11} sdim K=({de}|{do}) {note}")
    return anti, auto, sq, K


def main():
    print("=" * 78)
    print("1.  The five maps printed in Subsec. 1.4, on Mat(a|b) and Q(n)")
    print("=" * 78)

    for (a, b) in [(2, 2), (3, 3), (3, 2)]:
        S = Sup(a, b)
        bas = S.basis()
        print(f"\nR = Mat({a}|{b}),  dim_Z R = {(a+b)**2}"
              f"   (Z = C, R is central simple)")
        report("st  (as printed)", S, st_printed, bas)
        if a == b:
            report("-Pi (as printed)", S, minus_Pi, bas)
        report("-delta_{-1}", S, minus_delta, bas)
        if b % 2 == 0:
            phi = star_form(S, osp_form(a, b))
            report("G^{-1}(.)^{st}G  [osp]", S, phi, bas,
                   note="<- the map that really gives osp")

    for n in (2, 3):
        S = Sup(n, n)
        qb = Q_basis(n)
        print(f"\nR = Q({n}),  dim(Q({n})) = {2*n*n}")
        report("-q  (eq. (5))", S, minus_q, qb)
        report("-delta_{-1}", S, minus_delta, qb)

    print()
    print("=" * 78)
    print("2.  K_odd = 0 whenever *^2 = delta_{-1}   (the one-line proof:")
    print("    u^*=-u  ==>  u^{**}=u,  while *^2=delta_{-1} gives u^{**}=-u)")
    print("=" * 78)
    for (a, b) in [(2, 2), (3, 3), (3, 2), (4, 3)]:
        S = Sup(a, b)
        bas = S.basis()
        K = anti_symmetric(S, st_printed, bas)
        de, do = sdim(S, K)
        ok = (do == 0)
        print(f"  Mat({a}|{b}): st,  sdim K = ({de}|{do})"
              f"   K_odd = 0 ? {ok}"
              f"   dim o({a})+o({b}) = {a*(a-1)//2 + b*(b-1)//2}")
    for n in (2, 3, 4):
        S = Sup(n, n)
        qb = Q_basis(n)
        K = anti_symmetric(S, minus_q, qb)
        de, do = sdim(S, K)
        print(f"  Q({n}):    -q,  sdim K = ({de}|{do})"
              f"   K_odd = 0 ? {do == 0}"
              f"   dim o({n}) = {n*(n-1)//2}")

    print()
    print("=" * 78)
    print("3.  THE COUNTEREXAMPLE: R = Mat(3|3), * = st.")
    print("    st satisfies (1) [antiInv] but not (2) [GASInv];")
    print("    dim_Z R = 36 > 16; yet SL(K) is NOT simple.")
    print("=" * 78)
    S = Sup(3, 3)
    bas = S.basis()
    anti = is_anti(S, st_printed, bas)
    sq = square_of(S, st_printed, bas)
    K = anti_symmetric(S, st_printed, bas)
    de, do = sdim(S, K)
    print(f"  st is a super anti-automorphism : {anti}")
    print(f"  (st|_even)^2 = id               : "
          f"{square_of(S, st_printed, [(M,p) for M,p in bas if p==0]) == 'id'}")
    print(f"  st^2                            : {sq}   (so st^4 = id)")
    print(f"  sdim K                          : ({de}|{do})")
    Kb = [(M, p) for M, p in K]
    Kp = reduce_basis(derived(S, Kb), S.n)
    print(f"  dim K                           : {de + do}")
    print(f"  dim K' = [K,K]                  : {span_dim(Kp, S.n)}"
          f"   (so K' = K, K is perfect)")
    # the center of R meets K' ?
    Zc = sp.eye(S.n)
    print(f"  1_6 in K' ?                     : "
          f"{in_span(Zc, Kp, S.n)}   (so SL(K) = K' = K)")
    # exhibit the proper ideal:  o(3) sitting in the upper-left block
    x = sp.zeros(S.n, S.n)
    x[0, 1], x[1, 0] = 1, -1
    idl = generated_ideal(S, Kb, x)
    print(f"  ideal generated by E_12-E_21    : dim {span_dim(idl, S.n)}"
          f"  of dim K = {de+do}  ==>  PROPER ideal, K is NOT simple")
    print("  K = o(3) + o(3) (a direct sum of two ideals), not simple.")
    print()
    print("  ==> Theorem 1.4.1 with (3) replaced by (1) is FALSE.")

    print()
    print("=" * 78)
    print("4.  What is true instead: in the gap case K is the anti-symmetric part of")
    print("    R_even under the involution *|_{R_even} (Baxter-Herstein).")
    print("=" * 78)
    for (a, b) in [(3, 3), (3, 2), (4, 3)]:
        S = Sup(a, b)
        bas = S.basis()
        K = anti_symmetric(S, st_printed, bas)
        evbas = [(M, p) for M, p in bas if p == 0]
        Kev = anti_symmetric(S, st_printed, evbas)
        same = span_dim([M for M, _ in K], S.n) == \
            span_dim([M for M, _ in Kev], S.n)
        print(f"  Mat({a}|{b}): K = anti-symmetric part of R_even ? {same}"
              f"   (R_even = Mat({a}) + Mat({b}), simple iff one of a,b is 0)")
    for n in (3, 4, 5):
        S = Sup(n, n)
        qb = Q_basis(n)
        K = anti_symmetric(S, minus_q, qb)
        de, do = sdim(S, K)
        print(f"  Q({n}):    K = o({n}), dim {de+do} = {n*(n-1)//2};"
              f"  R_even = Mat({n}) IS simple,"
              f"  dim_Z R_even = {n*n} > 16 iff n >= 5")

    print()
    print("=" * 78)
    print("5.  Pi and delta_{-1} are AUTOMORPHISMS of the associative R;")
    print("    -Pi and -delta_{-1} are anti-automorphisms of the LIE")
    print("    superalgebra R^{SL} only (the observation of Subsec. 1.5).")
    print("=" * 78)

    def lie_anti(S, phi, basis):
        """sigma([x,y]) = -[sigma x, sigma y] for all basis pairs."""
        for (u, pu), (v, pv) in itertools.product(basis, repeat=2):
            lhs = phi(S, sbracket(u, v, pu, pv))
            rhs = -sbracket(phi(S, u), phi(S, v), pu, pv)
            if sp.simplify(sp.Matrix(lhs - rhs)) != sp.zeros(S.n, S.n):
                return False
        return True

    for (a, b) in [(2, 2), (3, 3)]:
        S = Sup(a, b)
        bas = S.basis()
        print(f"  Mat({a}|{b}):  Pi is an assoc. automorphism      : "
              f"{is_auto(S, Pi_map, bas)}")
        print(f"            Pi is an assoc. anti-automorphism : "
              f"{is_anti(S, Pi_map, bas)}")
        print(f"            -Pi is a Lie anti-automorphism    : "
              f"{lie_anti(S, minus_Pi, bas)}")
        print(f"            delta_-1 is an assoc. automorphism: "
              f"{is_auto(S, delta_m1, bas)}")
        print(f"            -delta_-1 is a Lie anti-autom.    : "
              f"{lie_anti(S, minus_delta, bas)}")
    for n in (2, 3):
        S = Sup(n, n)
        qb = Q_basis(n)
        print(f"  Q({n}):      delta_-1 is an assoc. automorphism: "
              f"{is_auto(S, delta_m1, qb)}")
        print(f"            -delta_-1 is a Lie anti-autom.    : "
              f"{lie_anti(S, minus_delta, qb)}")


if __name__ == "__main__":
    main()
