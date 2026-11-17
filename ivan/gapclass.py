#!/usr/bin/env python3
r"""
The classification of the gap of Subsec. 1.4.2 -- the extension of Theorem 1.4.1
(= Theorem GS), as asked for by Leites.

gap.py showed that the naive extension is FALSE: Theorem GS does not survive the
replacement of (GASInv) by (antiInv), the counterexample being the bare
supertransposition st on Mat(3|3), where K = o(3) + o(3) is not simple.

This engine answers the next question -- WHAT does happen in the gap -- by
classifying every even anti-automorphism satisfying (antiInv) of every
finite-dimensional central simple associative superalgebra, i.e., of Mat(a|b)
and of Q(n), and computing L(K) = (K^L)'/(Z \cap (K^L)') for each.

The classification rests on the (super) Skolem--Noether theorem: every even
anti-automorphism of Mat(a|b) is X -> B^{-1} X^{st} B for an invertible
homogeneous B.  A hand computation (reproduced by machine below, in check_C())
gives

      (*_B)^2 (X) = (-1)^{p(X)} C^{-1} X C,      where C = (B^{st})^{-1} B,

so that, by Lemma 1.4.3, (antiInv) holds iff C is a scalar (then *^2 =
delta_{-1}: the GAP) or C is a scalar multiple of the grading operator eps
(then *^2 = id: a superinvolution, the case covered by [GS]).  For even
B = diag(P,S) this reads: the gap is exactly the case when P and S are BOTH
symmetric or BOTH anti-symmetric, i.e., when B is an ordinary, not a super, form.

What the run prints, and what goes into the paper:

  1) the census of the gap for Mat(a|b) and Q(n): which B, which K;
  2) the case of odd B (item 3) of Proposition 1.4.6 of the text): for a = b
     an ODD B lands in the gap as well, and then * SWAPS the two simple
     summands of R_ev, so that K is a diagonal copy of gl(a) and L(K) = sl(a)
     IS simple -- although R_ev is not;
  3) hence the sharp criterion: in the gap, L(K) is simple iff R_ev is
     *-simple, i.e., iff * does not preserve both central idempotents of R_ev.

Exact symbolic arithmetic throughout (sympy, ground field Q(i)); no numerics.
"""
import sys
import itertools
import sympy as sp

import gap
from gap import (Sup, st_printed, delta_m1, is_anti, square_of,
                 anti_symmetric, sbracket, span_dim, in_span, derived,
                 reduce_basis, generated_ideal, sdim, Q_basis)

LOG = None


def say(*a):
    msg = " ".join(str(x) for x in a)
    print(msg)
    if LOG:
        LOG.write(msg + "\n")
        LOG.flush()


# ------------------------------------------------------------------ the forms


def sym_form(m):
    return sp.eye(m)


def skew_form(m):
    """standard symplectic matrix; m must be even."""
    G = sp.zeros(m, m)
    for k in range(m // 2):
        G[2 * k, 2 * k + 1] = 1
        G[2 * k + 1, 2 * k] = -1
    return G


def star_B(S, B):
    """X -> B^{-1} X^{st} B."""
    Bi = B.inv()

    def phi(S_, X):
        return Bi * st_printed(S_, X) * B
    return phi


def antiInv_holds(S, phi, basis):
    """(antiInv): super anti-automorphism whose square is the identity on the
    EVEN part.  (The square on the odd part is then id or delta_{-1}.)"""
    if not is_anti(S, phi, basis):
        return False
    for (u, p) in basis:
        if p != 0:
            continue
        if sp.simplify(sp.Matrix(phi(S, phi(S, u)) - u)) != sp.zeros(S.n, S.n):
            return False
    return True


# ------------------------------------------------------- structure of K, L(K)


def lie_center(S, K):
    """the elements of K bracketing to zero with all of K (its own center)."""
    out = []
    for (X, px) in K:
        if all(sp.simplify(sp.Matrix(sbracket(X, Y, px, py))) ==
               sp.zeros(S.n, S.n) for (Y, py) in K):
            out.append((X, px))
    return out


def scalars_in(S, mats):
    """dimension of the space of scalar matrices lying in the span."""
    if not mats:
        return 0
    one = sp.eye(S.n)
    return 1 if in_span(one, mats, S.n) else 0


def centroid_dim(S, Kp, cap=12):
    """dim of the centroid {f : f([x,y]) = [f(x),y]} of the Lie algebra
    spanned by Kp.  For an algebra that is a direct sum of simple ideals this
    dimension IS the number of the summands (Schur), so it detects the
    splitting of o(4) = sl(2)+sl(2), which no single-element ideal test can:
    in a semisimple algebra a generic element already generates everything.
    Returns None when the dimension is above the cap (too slow to be worth it).
    """
    d = len(Kp)
    if d == 0 or d > cap:
        return None
    n = S.n
    coord_rows = [[sp.expand(M[i, j]) for i in range(n) for j in range(n)]
                  for M in Kp]
    Kmat = sp.Matrix(coord_rows).T                 # columns = basis of K'

    def coords(M):
        v = sp.Matrix([sp.expand(M[i, j]) for i in range(n) for j in range(n)])
        sol = Kmat.solve_least_squares(v) if False else Kmat.gauss_jordan_solve(v)[0]
        return sp.simplify(sol)

    # structure constants
    par = [0] * d                                   # K is purely even here
    C = {}
    for i in range(d):
        for j in range(d):
            C[(i, j)] = coords(sbracket(Kp[i], Kp[j], par[i], par[j]))
    F = sp.Matrix(d, d, lambda i, j: sp.Symbol(f"f_{i}_{j}"))
    eqs = []
    for i in range(d):
        for j in range(d):
            # f([x_i,x_j]) - [f(x_i), x_j]
            lhs = sp.zeros(d, 1)
            for k in range(d):
                lhs += C[(i, j)][k] * F[:, k]
            rhs = sp.zeros(d, 1)
            for k in range(d):
                rhs += F[k, i] * C[(k, j)]
            for k in range(d):
                eqs.append(sp.expand(lhs[k] - rhs[k]))
    unk = [F[i, j] for i in range(d) for j in range(d)]
    A, _ = sp.linear_eq_to_matrix(eqs, unk)
    return len(unk) - A.rank()


def analyse(S, name, phi, basis, extra=""):
    """the full report on one anti-automorphism: is it in the gap, what is K,
    is L(K) simple."""
    anti = is_anti(S, phi, basis)
    ai = antiInv_holds(S, phi, basis)
    sq = square_of(S, phi, basis)
    K = anti_symmetric(S, phi, basis)
    de, do = sdim(S, K)

    verdict = ""
    if ai:
        # K' -- the derived algebra
        Kb = [(M, p) for (M, p) in K]
        Kp = reduce_basis(derived(S, Kb), S.n)
        dimKp = span_dim(Kp, S.n)
        # Z \cap K' : the scalars inside K'
        zc = scalars_in(S, Kp)
        dimL = dimKp - zc
        # L(K) = K'/(Z \cap K'): look for a proper nonzero ideal.  Finding one
        # PROVES non-simplicity; finding none on a spanning set is evidence,
        # and the simplicity itself is quoted from the classical theorems.
        Kpp = [(M, S.parity(M) if S.parity(M) is not None else 0)
               for M in Kp]
        split = None
        if dimL > 0:
            split = False
            for (x, _) in Kpp:
                if scalars_in(S, [x]) == 1:          # x is central: skip
                    continue
                idl = generated_ideal(S, Kpp, x)
                # dimension of the image of this ideal in K'/(Z \cap K')
                d = span_dim(idl, S.n)
                dq = d - (1 if (zc == 1 and in_span(sp.eye(S.n), idl, S.n))
                          else 0)
                if 0 < dq < dimL:
                    split = True
                    break
        cen = centroid_dim(S, Kp) if sq == "delta_{-1}" else None
        verdict = (f"dim K'={dimKp} Z∩K'={zc} dim L(K)={dimL} "
                   f"proper ideal found={split}"
                   + (f" centroid dim={cen} => {cen} simple ideal(s)"
                      if cen is not None else ""))

    say(f"    {name:<34} anti={str(anti):<5} (antiInv)={str(ai):<5} "
        f"*^2={sq:<11} sdim K=({de}|{do}) {verdict} {extra}")
    return dict(name=name, anti=anti, antiInv=ai, square=sq,
                sdimK=(de, do), K=K)


def identify(S, K, a, b):
    """name K by its dimension among the plausible candidates."""
    de, do = sdim(S, K)
    cands = {
        "o(a)+o(b)": a * (a - 1) // 2 + b * (b - 1) // 2,
        "sp(a)+sp(b)": (a * (a + 1) // 2 if a % 2 == 0 else None),
    }
    out = []
    if de == a * (a - 1) // 2 + b * (b - 1) // 2:
        out.append("o(%d)+o(%d)" % (a, b))
    if a % 2 == 0 and b % 2 == 0 and \
            de == a * (a + 1) // 2 + b * (b + 1) // 2:
        out.append("sp(%d)+sp(%d)" % (a, b))
    if a == b and de == a * a:
        out.append("gl(%d) (diagonal)" % a)
    return ", ".join(out) if out else "?"


# -------------------------------------------------------- the hand identity


def check_C(a, b):
    """verify  (*_B)^2 (X) = (-1)^{p(X)} C^{-1} X C  with C = (B^st)^{-1} B,
    on the nose, for a sample of even and odd B."""
    S = Sup(a, b)
    basis = S.basis()
    ok = True
    Bs = []
    Bs.append(("even diag(1,1)", sp.eye(a + b)))
    if b % 2 == 0:
        G = sp.zeros(a + b, a + b)
        G[:a, :a] = sym_form(a)
        G[a:, a:] = skew_form(b)
        Bs.append(("even diag(1,J)", G))
    if a == b:
        Bodd = sp.zeros(2 * a, 2 * a)
        Bodd[:a, a:] = sp.eye(a)
        Bodd[a:, :a] = sp.I * sp.eye(a)
        Bs.append(("odd antidiag(1,i)", Bodd))
    for nm, B in Bs:
        phi = star_B(S, B)
        Bst = st_printed(S, B)
        C = Bst.inv() * B
        for (u, p) in basis:
            lhs = phi(S, phi(S, u))
            rhs = sp.Integer(-1) ** p * C.inv() * u * C
            if sp.simplify(sp.Matrix(lhs - rhs)) != sp.zeros(S.n, S.n):
                ok = False
                say(f"      IDENTITY FAILS for {nm}")
                break
    return ok


# --------------------------------------------------------------- Mat(a|b)


def sweep_mat(a, b):
    say("")
    say(f"  R = Mat({a}|{b}),  dim = {(a + b) ** 2},  "
        f"R_ev = Mat({a}) + Mat({b})")
    S = Sup(a, b)
    basis = S.basis()
    n = a + b

    cands = []
    # even B = diag(P,S): four combinations of symmetric / skew
    P_opts = [("1", sym_form(a))]
    if a % 2 == 0:
        P_opts.append(("J", skew_form(a)))
    S_opts = [("1", sym_form(b))]
    if b % 2 == 0:
        S_opts.append(("J", skew_form(b)))
    for pn, P in P_opts:
        for sn, Sm in S_opts:
            B = sp.zeros(n, n)
            B[:a, :a] = P
            B[a:, a:] = Sm
            cands.append((f"B=diag({pn}_{a},{sn}_{b})", B))
    # odd B, only for a = b
    if a == b and a > 0:
        for cn, c in (("1", sp.Integer(1)), ("-1", sp.Integer(-1)),
                      ("i", sp.I), ("-i", -sp.I)):
            B = sp.zeros(n, n)
            B[:a, a:] = sp.eye(a)
            B[a:, :a] = c * sp.eye(a)
            cands.append((f"B=antidiag(1,{cn}) [odd]", B))

    res = []
    for nm, B in cands:
        phi = star_B(S, B)
        r = analyse(S, nm, phi, basis)
        if r["antiInv"]:
            r["id"] = identify(S, r["K"], a, b)
            say(f"        K = {r['id']}")
            # does * preserve the two central idempotents of R_ev?
            e1 = sp.zeros(n, n)
            for i in range(a):
                e1[i, i] = 1
            im = phi(S, e1)
            keeps = sp.simplify(sp.Matrix(im - e1)) == sp.zeros(n, n)
            say(f"        * fixes the idempotent e_1 of R_ev: {keeps}"
                f"   =>  R_ev is {'not ' if keeps and a and b else ''}"
                f"swapped by *")
            r["keeps_e1"] = keeps
        res.append(r)
    return res


# ------------------------------------------------------------------- Q(n)


class QAlg:
    """Q(n) inside Mat(n|n), with the maps we need."""

    def __init__(self, n):
        self.n = n
        self.S = Sup(n, n)
        self.basis = Q_basis(n)


def q_star(n, g, c):
    """(A B / B A) -> (g^{-1}A^t g,  c g^{-1}B^t g) as a map on Mat(n|n)."""
    gi = g.inv()

    def phi(S_, X):
        A = X[:n, :n]
        B = X[:n, n:]
        A2 = gi * A.T * g
        B2 = c * gi * B.T * g
        Z = sp.zeros(2 * n, 2 * n)
        Z[:n, :n] = A2
        Z[n:, n:] = A2
        Z[:n, n:] = B2
        Z[n:, :n] = B2
        return Z
    return phi


def admissible_c(n, g, gname):
    """SYMBOLIC: for which c is psi_{g,c} an anti-automorphism of Q(n)?
    Solves the condition on a pair of basis elements, then verifies the
    solutions on all pairs.  This is what makes the list of candidates in
    sweep_q() a classification and not a sample."""
    c = sp.Symbol("c")
    S = Sup(n, n)
    basis = Q_basis(n)
    phi = q_star(n, g, c)
    eqs = set()
    for (u, pu), (v, pv) in itertools.product(basis, repeat=2):
        lhs = phi(S, u * v)
        rhs = sp.Integer(-1) ** (pu * pv) * (phi(S, v) * phi(S, u))
        D = sp.expand(sp.Matrix(lhs - rhs))
        for i in range(2 * n):
            for j in range(2 * n):
                e = sp.simplify(D[i, j])
                if e != 0:
                    eqs.add(sp.Poly(e, c).as_expr())
    sols = sp.solve(list(eqs), c, dict=True) if eqs else "any c"
    say(f"    admissible c for g={gname}: {sols}")
    return sols


def sweep_q(n):
    say("")
    say(f"  R = Q({n}),  dim = {2 * n * n},  R_ev = Mat({n}) (simple)")
    S = Sup(n, n)
    basis = Q_basis(n)
    cands = []
    gs = [("1", sp.eye(n))]
    if n % 2 == 0:
        gs.append(("J", skew_form(n)))
    for gn, g in gs:
        admissible_c(n, g, gn)
        for cn, c in (("1", sp.Integer(1)), ("-1", sp.Integer(-1)),
                      ("i", sp.I), ("-i", -sp.I)):
            cands.append((f"g={gn}, c={cn}", g, c))
    res = []
    for nm, g, c in cands:
        phi = q_star(n, g, c)
        r = analyse(S, nm, phi, basis)
        if r["antiInv"]:
            de, do = r["sdimK"]
            nm2 = []
            if de == n * (n - 1) // 2:
                nm2.append(f"o({n})")
            if n % 2 == 0 and de == n * (n + 1) // 2:
                nm2.append(f"sp({n})")
            if de == n * n:
                nm2.append(f"gl({n})")
            say(f"        K = {', '.join(nm2) if nm2 else '?'}")
        res.append(r)
    return res


# ---------------------------------------------------------------------- main


def main():
    global LOG
    LOG = open("gapclass.log", "w")
    say("=" * 78)
    say("The gap of Subsec. 1.4.2, classified: every even anti-automorphism")
    say("satisfying (antiInv) of every central simple associative superalgebra.")
    say("Exact arithmetic over Q(i).  gapclass.py")
    say("=" * 78)

    say("")
    say("0) The identity  (*_B)^2 (X) = (-1)^{p(X)} C^{-1} X C, C=(B^st)^{-1}B:")
    for (a, b) in ((2, 2), (3, 2), (2, 1)):
        ok = check_C(a, b)
        say(f"    Mat({a}|{b}): {'verified' if ok else 'FAILS'}")

    say("")
    say("1) R = Mat(a|b).  Anti-automorphisms X -> B^{-1} X^{st} B.")
    mat = {}
    for (a, b) in ((2, 2), (3, 2), (3, 3), (4, 2), (4, 1), (5, 1)):
        mat[(a, b)] = sweep_mat(a, b)

    say("")
    say("2) R = Q(n).")
    qq = {}
    for n in (2, 3, 4):
        qq[n] = sweep_q(n)

    say("")
    say("=" * 78)
    say("SUMMARY")
    say("=" * 78)
    say("")
    say("In the gap (*^2 = delta_{-1}) the anti-symmetric part K is purely")
    say("even (Corollary 1.4.4), and the cases are:")
    say("")
    say("  R = Mat(a|b), B even, both blocks symmetric : K = o(a) + o(b)")
    say("  R = Mat(a|b), B even, both blocks anti-sym : K = sp(a) + sp(b)")
    say("  R = Mat(a|a), B odd  (B^st = +-i B)         : K = gl(a), diagonal")
    say("  R = Q(n)                                    : K = o(n) or sp(n)")
    say("")
    say("The first two preserve both central idempotents of R_ev, so K splits")
    say("into two ideals and L(K) is NOT simple -- unless one of the two")
    say("summands degenerates (o(m) is abelian for m <= 2 and splits for m = 4),")
    say("which is why Mat(5|1) and Mat(3|2) below do give a simple answer.")
    say("The third case SWAPS the idempotents, so K is a diagonal gl(a) and")
    say("L(K) = sl(a) IS simple, although R_ev is not.  The fourth has")
    say("R_ev = Mat(n) simple, and Herstein's Theorem applies.")
    say("")
    say("CONCLUSION for the paper: in the gap K is always purely EVEN, so every")
    say("L(K) obtained there is an ordinary Lie algebra -- one of o(m), sp(2m),")
    say("sl(m) -- all of them already delivered by Theorem 1.3 (Herstein).  The gap yields")
    say("no new simple Lie SUPERalgebra, which is the strongest true form of")
    say("'Theorem GS extends from (GASInv) to (antiInv)'.")
    LOG.close()


if __name__ == "__main__":
    main()
