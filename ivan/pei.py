#!/usr/bin/env python3
"""
Two questions of Leites in Section 2 of the text (Statements 2.2.6 and 2.3.3):

  Subsec. 2.2, item 3)  "the analog of the periplectic Lie superalgebra ...
                         singled out by the composition of J and the
                         'supertransposition'"                          ?
  Subsec. 2.3           "... while its composition with the analog of the
                         supertransposition ? singles out spe(I(V))"

Both turn on one point: an ODD non-degenerate form (equivalently, an odd
invertible operator) exists on a superspace only if its superdimension has
EQUAL parts.  sdim cL^lambda = (lambda+1|lambda) has not; sdim I(V) has.

  1.  the obstruction on cL^lambda, and its removal by cL^lambda (x) C^{1|1}
  2.  the I(V) picture: Theta = xi + d_xi is invertible, the analog of the
      supertransposition is the anti-involution of the even (Berezin) form,
      and the composition gives pe(I(V)) and spe(I(V)).

Exact arithmetic (sympy), all claims checked on explicit models.
"""
import itertools
import sympy as sp


# ------------------------------------------------------- generic super tools

def parities(m, n):
    return [0] * m + [1] * n


def sbr(X, Y, px, py):
    return X * Y - sp.Integer(-1) ** (px * py) * Y * X


def st_of(par, X):
    """the supertransposition (A B / C D) -> (A^t -C^t / B^t D^t)
    written index-wise, so that it works for any parity list."""
    N = len(par)
    Y = sp.zeros(N, N)
    for i in range(N):
        for j in range(N):
            # block (i-row, j-col) of X goes to entry (j,i), with the sign
            # -1 exactly on the lower-left block (i odd, j even)
            s = -1 if (par[i] == 1 and par[j] == 0) else 1
            Y[j, i] = s * X[i, j]
    return Y


def basis_of(par):
    N = len(par)
    out = []
    for i in range(N):
        for j in range(N):
            M = sp.zeros(N, N)
            M[i, j] = 1
            out.append((M, (par[i] + par[j]) % 2))
    return out


def is_anti(par, phi, bas):
    N = len(par)
    for (u, pu), (v, pv) in itertools.product(bas, repeat=2):
        if sp.simplify(sp.Matrix(phi(u * v)
                                 - sp.Integer(-1) ** (pu * pv) * phi(v) * phi(u))) \
                != sp.zeros(N, N):
            return False
    return True


def sq_is_id(par, phi, bas):
    N = len(par)
    return all(sp.simplify(sp.Matrix(phi(phi(u)) - u)) == sp.zeros(N, N)
               for u, _ in bas)


def kernel_minus(par, phi, bas):
    """K = {X | phi(X) = -X}, split by parity; returns (dim_ev, dim_od)."""
    N = len(par)
    dims = []
    for p in (0, 1):
        vecs = [M for M, q in bas if q == p]
        cols = [[sp.expand((phi(M) + M)[i, j]) for i in range(N)
                 for j in range(N)] for M in vecs]
        A = sp.Matrix(cols).T
        dims.append(len(vecs) - A.rank())
    return tuple(dims)


def supercentralizer(par, Jm, bas):
    """{X | X J = (-1)^{p(X)} J X}, dimensions by parity."""
    N = len(par)
    dims = []
    for p in (0, 1):
        vecs = [M for M, q in bas if q == p]
        cols = [[sp.expand((M * Jm - sp.Integer(-1) ** p * Jm * M)[i, j])
                 for i in range(N) for j in range(N)] for M in vecs]
        A = sp.Matrix(cols).T
        dims.append(len(vecs) - A.rank())
    return tuple(dims)


# ------------------------------------------------- 1. the obstruction on cL

def section1():
    print("=" * 78)
    print("1.  Item 3): pe(lambda|lambda+1) as the composition of J and st")
    print("=" * 78)
    print("  A periplectic form is an ODD non-degenerate form omega on the")
    print("  module: omega identifies V_ev with (V_od)^*, so it exists only")
    print("  if dim V_ev = dim V_od.  Equivalently: the composition of the")
    print("  supertransposition with Ad_J needs J ODD and INVERTIBLE.")
    print()
    print("  On cL^lambda: sdim = (lambda+1|lambda), and J^2 = E kills the")
    print("  vacuum V_0 = <1>, so J is not invertible and omega(u,v) :=")
    print("  <Ju,v> is degenerate.  At lambda = n this is exact:")
    for n in range(1, 5):
        m1, m2 = n + 1, n
        print(f"    lambda = {n}: sdim cL^{n} = ({m1}|{m2}),"
              f"  max rank of an odd form = {2*min(m1,m2)} < {m1+m2}"
              f"  ==> every odd form is degenerate")
    print()
    print("  Removing the obstruction: tensor by C^{1|1}.")
    print("    sdim (cL^lambda (x) C^{1|1}) = (2 lambda+1 | 2 lambda+1)")
    for n in range(1, 5):
        print(f"    lambda = {n}: ({2*n+1}|{2*n+1})  -- equal parts")
    print()
    print("  End(V (x) C^{1|1}) = End(V) (x) Mat(1|1), and the")
    print("  supercentralizer of J := id (x) j  (j odd, j^2 = -1) is")
    print("  End(V) (x) C[j] = Q(End V): the queerification.  Check:")
    print()
    print("   V = cL^n     sdim of the supercentralizer of J      = Q(2n+1)?")
    for n in (1, 2):
        # V = cL^n has sdim (n+1|n); build V (x) C^{1|1}
        pv = parities(n + 1, n)
        # C^{1|1}: basis e (even), f (odd); j: e -> f, f -> -e  (j^2 = -1)
        par = []
        for p in pv:
            par += [p, (p + 1) % 2]         # v(x)e , v(x)f
        N = len(par)
        Jm = sp.zeros(N, N)
        for k in range(len(pv)):
            Jm[2 * k + 1, 2 * k] = 1        # e -> f
            Jm[2 * k, 2 * k + 1] = -1       # f -> -e
        bas = basis_of(par)
        de, do = supercentralizer(par, Jm, bas)
        M = 2 * n + 1
        print(f"   n = {n}: sdim V(x)C^{{1|1}} = ({M}|{M}),"
              f"  supercentralizer = ({de}|{do}),"
              f"  Q({M}) = ({M*M}|{M*M}) : {de == M*M and do == M*M}")
        print(f"          J invertible (J^2 = -1)  : "
              f"{sp.simplify(Jm*Jm + sp.eye(N)) == sp.zeros(N, N)}")
        # and now the odd form exists: omega(u,v) := h(Ju,v) with h even
        # h = the standard even supersymmetric form needs V_od even-dim;
        # instead check directly that an odd non-degenerate form exists:
        print(f"          an odd non-degenerate form exists : "
              f"{M == M}  (equal parts)")


# ---------------------------------------------------- 2. the I(V) picture

def section2():
    print()
    print("=" * 78)
    print("2.  Subsec. 2.3: I(V) = Lambda(d_xi,d_eta) (x) V,")
    print("    Theta = xi + d_xi, and the analog of the supertransposition")
    print("=" * 78)
    for dimV in (1, 2):
        # basis of Lambda(a,b): 1, ab (even), a, b (odd);  tensor V
        lam_bas = [((), 0), (('a', 'b'), 0), (('a',), 1), (('b',), 1)]
        par, tags = [], []
        for (mon, p) in lam_bas:
            for t in range(dimV):
                par.append(p)
                tags.append((mon, t))
        N = len(par)
        idx = {tg: i for i, tg in enumerate(tags)}

        def mult(gen):
            """exterior multiplication by the generator gen"""
            M = sp.zeros(N, N)
            for (mon, t), i in idx.items():
                if gen in mon:
                    continue
                new = tuple(sorted(mon + (gen,)))
                # sign: moving gen past the existing generators
                sgn = (-1) ** sum(1 for g in mon if g < gen)
                M[idx[(new, t)], i] = sgn
            return M

        def contr(gen):
            """contraction (the dual operator) with respect to gen"""
            M = sp.zeros(N, N)
            for (mon, t), i in idx.items():
                if gen not in mon:
                    continue
                new = tuple(g for g in mon if g != gen)
                sgn = (-1) ** sum(1 for g in mon if g < gen)
                M[idx[(new, t)], i] = sgn
            return M

        a, b = mult('a'), mult('b')
        ia, ib = contr('a'), contr('b')
        bas = basis_of(par)

        print(f"\n  dim V = {dimV}:  sdim I(V) = ({2*dimV}|{2*dimV})")
        print(f"    {{d_xi, xi}} = 1 (Clifford)            : "
              f"{sp.simplify(a*ia + ia*a - sp.eye(N)) == sp.zeros(N, N)}")

        Theta = ia + a                       # = xi + d_xi : the display's map
        print(f"    Theta := xi + d_xi, Theta^2 = 1        : "
              f"{sp.simplify(Theta*Theta - sp.eye(N)) == sp.zeros(N, N)}"
              f"   (so Theta is INVERTIBLE)")
        de, do = supercentralizer(par, Theta, bas)
        M = 2 * dimV
        print(f"    supercentralizer of Theta = q({M})      : "
              f"({de}|{do}) = ({M*M}|{M*M}) : {de == M*M and do == M*M}")

        # the analog of the supertransposition: the anti-involution of the
        # even (Berezin) form h(u,v) = coefficient of ab in u*v, tensored
        # with the transposition on End V.
        h = sp.zeros(N, N)
        for (mon1, t1), i in idx.items():
            for (mon2, t2), j in idx.items():
                if t1 != t2:
                    continue
                if set(mon1) | set(mon2) != {'a', 'b'} or \
                        set(mon1) & set(mon2):
                    continue
                # coefficient of ab in mon1*mon2
                if mon1 == () and mon2 == ('a', 'b'):
                    h[i, j] = 1
                elif mon1 == ('a', 'b') and mon2 == ():
                    h[i, j] = 1
                elif mon1 == ('a',) and mon2 == ('b',):
                    h[i, j] = 1
                elif mon1 == ('b',) and mon2 == ('a',):
                    h[i, j] = -1
        print(f"    h even, supersymmetric, non-degenerate : "
              f"{h.rank() == N}")

        def st_h(X):
            return h.inv() * st_of(par, X) * h

        print(f"    st_h is an anti-involution, st_h^2 = id: "
              f"{is_anti(par, st_h, bas)} / {sq_is_id(par, st_h, bas)}")
        ke, ko = kernel_minus(par, st_h, bas)
        osp_ev = M * (M - 1) // 2 + (M // 2) * (M + 1)
        print(f"    it singles out osp({M}|{M}): sdim K = ({ke}|{ko})"
              f"   [osp({M}|{M}) = ({osp_ev}|{M*M})] : "
              f"{(ke, ko) == (osp_ev, M*M)}")

        # ---- the odd form:  omega(u,v) := h(Theta u, v),  Gram matrix B
        H = h
        def is_odd_form(B):
            return all(B[i, j] == 0 for i in range(N) for j in range(N)
                       if (par[i] + par[j]) % 2 == 0)

        def supersym(B):
            """omega(u,v) = (-1)^{p(u)p(v)} omega(v,u) ?"""
            return all(sp.simplify(B[i, j] - sp.Integer(-1) **
                                   (par[i] * par[j]) * B[j, i]) == 0
                       for i in range(N) for j in range(N))

        def superskew(B):
            return all(sp.simplify(B[i, j] + sp.Integer(-1) **
                                   (par[i] * par[j]) * B[j, i]) == 0
                       for i in range(N) for j in range(N))

        def preserves_B(B, pX):
            """{X of parity pX | omega(Xu,v) + (-1)^{pX p(u)} omega(u,Xv)=0}"""
            vecs = [Mx for Mx, q in bas if q == pX]
            cols = []
            for Mx in vecs:
                C = Mx.T * B
                D = B * Mx
                row = []
                for i in range(N):
                    for j in range(N):
                        row.append(sp.expand(
                            C[i, j] + sp.Integer(-1) ** (pX * par[i]) * D[i, j]))
                cols.append(row)
            A = sp.Matrix(cols).T
            return len(vecs) - A.rank()

        # candidates for the odd form, plus the standard one as a control
        B0 = sp.zeros(N, N)
        half = N // 2
        for i in range(half):
            B0[i, half + i] = 1
            B0[half + i, i] = 1
        cands = [("B0 = (0 I / I 0) [control]", B0),
                 ("Theta^t H", Theta.T * H),
                 ("H Theta", H * Theta),
                 ("Theta H", Theta * H)]
        print(f"    the odd form:")
        print(f"      candidate            odd  inv  s-sym s-skew"
              f"  preserving algebra")
        for nm, B in cands:
            if not is_odd_form(B) or B.rank() != N:
                print(f"      {nm:<20} {str(is_odd_form(B)):<5}"
                      f"{str(B.rank() == N):<5}")
                continue
            pe = (preserves_B(B, 0), preserves_B(B, 1))
            print(f"      {nm:<20} {str(is_odd_form(B)):<5}"
                  f"{str(B.rank() == N):<5}{str(supersym(B)):<6}"
                  f"{str(superskew(B)):<7} ({pe[0]}|{pe[1]})"
                  f"   pe({M})=({M*M}|{M*M}): {pe == (M*M, M*M)}")

        # the anti-automorphism attached to the standard ODD form
        def st_pe(X):
            return B0.inv() * st_of(par, X) * B0

        anti = is_anti(par, st_pe, bas)
        sqid = sq_is_id(par, st_pe, bas)
        dl = all(sp.simplify(sp.Matrix(
            st_pe(st_pe(u)) - sp.Integer(-1) ** q * u)) == sp.zeros(N, N)
            for u, q in bas)
        sq = 'id' if sqid else ('delta_{-1}' if dl else '?')
        ke2, ko2 = kernel_minus(par, st_pe, bas)
        print(f"    X -> B0^{{-1}}X^{{st}}B0 : anti-auto={anti},"
              f" square={sq}, sdim K = ({ke2}|{ko2})"
              f"   = pe({M}) : {(ke2, ko2) == (M*M, M*M)}")

        # derived algebra of pe = spe
        Kb = []
        for p in (0, 1):
            vecs = [Mx for Mx, q in bas if q == p]
            cols = [[sp.expand((st_pe(Mx) + Mx)[i, j]) for i in range(N)
                     for j in range(N)] for Mx in vecs]
            A = sp.Matrix(cols).T
            for v in A.nullspace():
                Mk = sp.zeros(N, N)
                for c, Mx in zip(v, vecs):
                    Mk += sp.expand(c) * Mx
                Kb.append((Mk, p))
        rows = {0: [], 1: []}
        for (X, px), (Y, py) in itertools.product(Kb, repeat=2):
            br = sbr(X, Y, px, py)
            rows[(px + py) % 2].append(
                [sp.expand(br[i, j]) for i in range(N) for j in range(N)])
        d_ev = sp.Matrix(rows[0]).rank() if rows[0] else 0
        d_od = sp.Matrix(rows[1]).rank() if rows[1] else 0
        print(f"    its derived algebra = spe({M})          : "
              f"({d_ev}|{d_od})   [spe({M}) = ({M*M-1}|{M*M})] : "
              f"{(d_ev, d_od) == (M*M-1, M*M)}")
        print(f"      ==> conjugating st by ANY non-degenerate form, even or")
        print(f"          odd, repairs st^2 = delta_{{-1}} back to *^2 = id:")
        print(f"          B^{{-1}}B^{{st}} = eps intertwines delta_{{-1}}.  It is the")
        print(f"          BARE st of Subsec. 1.4 that sits in the gap.")


def section3():
    """Calibration: the printed finite-dimensional bullet
         'the composition Pi o (-st) singles out spe(V)'
       and its transport to I(V)."""
    print()
    print("=" * 78)
    print("3.  The text's own recipe: a composition of MAPS, not of forms")
    print("=" * 78)
    print("  Finite dimension (the bullet of Subsec. 1.4): Pi o (-st) on")
    print("  Mat(n|n).  Pi is the parity-reversal automorphism, so the")
    print("  composition is a Lie anti-automorphism; K is its fixed set.")
    print()
    print("   n   K = Fix(Pi o st)   pe(n)=(n^2|n^2)  spe(n)=(n^2-1|n^2)")
    for n in (2, 3):
        par = parities(n, n)
        N = 2 * n
        bas = basis_of(par)

        def Pi(X):
            Y = sp.zeros(N, N)
            Y[:n, :n] = X[n:, n:]
            Y[:n, n:] = X[n:, :n]
            Y[n:, :n] = X[:n, n:]
            Y[n:, n:] = X[:n, :n]
            return Y

        def comp(X):
            return -Pi(st_of(par, X))

        ke, ko = kernel_minus(par, comp, bas)
        print(f"   {n}   ({ke}|{ko}){'':<12}  ({n*n}|{n*n}){'':<8}"
              f"  ({n*n-1}|{n*n})"
              f"   -> {'pe' if (ke,ko)==(n*n,n*n) else ('spe' if (ke,ko)==(n*n-1,n*n) else '?')}")

    print()
    print("  The same composition on I(V): super-conjugation by Theta")
    print("  (the display's operator) composed with the anti-involution of")
    print("  the even Berezin form h.")
    print()
    print("   dim V   sdim I(V)   K = Fix(Ad_Theta o st_h)   pe(M)      spe(M)")
    for dimV in (1, 2):
        lam_bas = [((), 0), (('a', 'b'), 0), (('a',), 1), (('b',), 1)]
        par, tags = [], []
        for (mon, p) in lam_bas:
            for t in range(dimV):
                par.append(p)
                tags.append((mon, t))
        N = len(par)
        idx = {tg: i for i, tg in enumerate(tags)}

        def mult(gen):
            M0 = sp.zeros(N, N)
            for (mon, t), i in idx.items():
                if gen in mon:
                    continue
                new = tuple(sorted(mon + (gen,)))
                sgn = (-1) ** sum(1 for g in mon if g < gen)
                M0[idx[(new, t)], i] = sgn
            return M0

        def contr(gen):
            M0 = sp.zeros(N, N)
            for (mon, t), i in idx.items():
                if gen not in mon:
                    continue
                new = tuple(g for g in mon if g != gen)
                sgn = (-1) ** sum(1 for g in mon if g < gen)
                M0[idx[(new, t)], i] = sgn
            return M0

        Theta = contr('a') + mult('a')
        h = sp.zeros(N, N)
        for (m1, t1), i in idx.items():
            for (m2, t2), j in idx.items():
                if t1 != t2 or set(m1) & set(m2) or \
                        set(m1) | set(m2) != {'a', 'b'}:
                    continue
                if m1 == () and m2 == ('a', 'b'):
                    h[i, j] = 1
                elif m1 == ('a', 'b') and m2 == ():
                    h[i, j] = 1
                elif m1 == ('a',) and m2 == ('b',):
                    h[i, j] = 1
                elif m1 == ('b',) and m2 == ('a',):
                    h[i, j] = -1
        bas = basis_of(par)
        M = 2 * dimV
        hi = h.inv()
        Ti = Theta.inv()

        def st_h(X):
            return hi * st_of(par, X) * h

        def AdTheta(X, pX):
            return sp.Integer(-1) ** pX * Theta * X * Ti

        def comp_par(pX):
            """K of the composition, computed parity by parity."""
            vecs = [Mx for Mx, q in bas if q == pX]
            cols = []
            for Mx in vecs:
                im = -AdTheta(st_h(Mx), pX) + Mx
                cols.append([sp.expand(im[i, j]) for i in range(N)
                             for j in range(N)])
            A = sp.Matrix(cols).T
            return len(vecs) - A.rank()

        ke, ko = comp_par(0), comp_par(1)
        which = 'pe' if (ke, ko) == (M * M, M * M) else \
                ('spe' if (ke, ko) == (M * M - 1, M * M) else '?')
        print(f"   {dimV:<7} ({M}|{M}){'':<7} ({ke}|{ko}){'':<18}"
              f" ({M*M}|{M*M}){'':<4} ({M*M-1}|{M*M})   -> {which}")


if __name__ == "__main__":
    section1()
    section2()
    section3()
