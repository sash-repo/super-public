"""Statements 11.1-11.3: the odd layer against the criterion of [BO, Prop. 14] = [Vi1, Thm. 2].

Over Spec Cee, a supermanifold with retract (M, Lambda E) given by a class
gamma in H^1(M; T_2) is even-homogeneous if and only if some copy a ~ sl(2)
inside H^0(M; T_0) = aut(E) + sl(2) annihilates gamma; the copies are, up to
conjugation and up to renumbering the k_i,

    s   = < d/dx,                        d/dy,                    [.,.] >
    s'  = < d/dx + xi_2 d/dxi_1,         d/dy + eta_1 d/deta_2,   [.,.] >   (k_1 = k_2)
    s'' = < d/dx + xi_2 d/dxi_1 + xi_3 d/dxi_2,
            d/dy + 2eta_1 d/deta_2 + 2eta_2 d/deta_3, [.,.] >               (k_1 = k_2 = k_3)

In the chart U one has d/dy = -x^2 d/dx - x sum_r k_r xi_r d/dxi_r and
eta_i d/deta_j = x^{k_j - k_i} xi_i d/dxi_j.

The question: is the criterion still the right one in ODD degree,
i.e., over a base with one odd parameter?  Here the classes live in
H^1(M; T_1) (odd dimension 2) and in H^1(M; T_3) (the retract CP^{1|4}), and
the answer is read off the invariants of s, s' and s''.

Part A   the retract (3,3): the invariants of s and of s' in H^1(M; T_1),
         and the rank of an invariant class as a 2 x 2 matrix.
Part B   every retract with k_1 <= 5: dim H^1(T_1)^s, dim H^1(T_1)^{s'} against
         the homogeneous families of Statements 10.1 and 11.2.
Part C   the retract CP^{1|4} in degree 3: the invariants of s, s', s'', s'''.
"""

from fractions import Fraction
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "lib"))

import ns                                                          # noqa: E402
import sl2inv as S                                                 # noqa: E402
import oddpar as O                                                 # noqa: E402


LOG = None


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    if LOG:
        LOG.write(s + "\n")
        LOG.flush()


# ------------------------------------------------------------------ the copies

def xi_d_xi(m, j, i, p):
    """The field x^p xi_j d/dxi_i."""
    Bs = [ns.g_zero() for _ in range(m)]
    Bs[i - 1] = ns.g_from((j,), ns.lp((p, 1)))
    return (ns.g_zero(), Bs)


def copy_sl2(ks, kind):
    """(e, f) of the copy `kind` of sl(2) inside v_0, in the chart U."""
    m = len(ks)
    e, h, f = O.sl2(ks)
    if kind == "s":
        return e, f
    if kind == "s'":
        assert ks[0] == ks[1]
        ee = S.fld_add(e, xi_d_xi(m, 2, 1, 0))
        ff = S.fld_add(f, xi_d_xi(m, 1, 2, 0))       # k_1 = k_2, so p = 0
        return ee, ff
    if kind == "s''":
        assert ks[0] == ks[1] == ks[2]
        ee = S.fld_add(S.fld_add(e, xi_d_xi(m, 2, 1, 0)), xi_d_xi(m, 3, 2, 0))
        ff = S.fld_add(S.fld_add(f, S.fld_scal(xi_d_xi(m, 1, 2, 0), 2)),
                       S.fld_scal(xi_d_xi(m, 2, 3, 0), 2))
        return ee, ff
    if kind == "s'''":                # the principal copy in aut(E), m = 4
        assert ks[0] == ks[1] == ks[2] == ks[3]
        ee = e
        ff = f
        for r in (1, 2, 3):
            ee = S.fld_add(ee, xi_d_xi(m, r + 1, r, 0))
            ff = S.fld_add(ff, S.fld_scal(xi_d_xi(m, r, r + 1, 0),
                                          r * (4 - r)))
        return ee, ff
    raise ValueError(kind)


def fld_eq(X, Y):
    D = S.fld_add(X, S.fld_scal(Y, -1))
    return not D[0] and not any(D[1])


def check_copy(ks, kind):
    """The copy really is an sl(2): [e,f] = h, [h,e] = 2e, [h,f] = -2f in v_0."""
    m = len(ks)
    e, f = copy_sl2(ks, kind)
    h = S.bracket_even(m, e, f)
    he = S.bracket_even(m, h, e)
    hf = S.bracket_even(m, h, f)
    return (fld_eq(he, S.fld_scal(e, 2)) and
            fld_eq(hf, S.fld_scal(f, -2)))


def invariants(ks, q, kind):
    """A basis of the a-invariants of H^1(M; T_q), a the copy `kind`."""
    L = O.Layer(ks, q)
    e, f = copy_sl2(ks, kind)
    return L, L.Q.invariants(e, f)


# ------------------------------------------------------------------- Part A

def partA():
    say("A. The retract (3,3) = (O(-3) + O(-3)), the odd layer H^1(M; T_1).")
    ks = [3, 3]
    L = O.Layer(ks, 1)
    say("   dim H^1(T_1) =", L.dim, "  basis:", L.Q.basis)

    # the basis v_0, .., v_3 of formula (16), as coordinate vectors
    named = {
        "v0 = x^-1 xi_1 d/dx": {(("u", (1,)), -1): Fraction(1)},
        "v1 = x^-1 xi_2 d/dx": {(("u", (2,)), -1): Fraction(1)},
        "v2 = x^-1 xi_1xi_2 d/dxi_1": {(("w", (1, 2), 1), -1): Fraction(1)},
        "v3 = x^-1 xi_1xi_2 d/dxi_2": {(("w", (1, 2), 2), -1): Fraction(1)},
    }
    coord = {}
    for nm, coc in named.items():
        coord[nm] = L.Q.coords(coc)
        say("  ", nm, "->", [str(c) for c in coord[nm]])

    for kind in ("s", "s'"):
        ok = check_copy(ks, kind)
        _, inv = invariants(ks, 1, kind)
        say(f"   {kind}: is an sl(2) in v_0: {ok};  "
            f"dim H^1(T_1)^{kind} = {len(inv)}")
        for v in inv:
            say(f"      invariant class: {[str(c) for c in v]}")
            c = solve_in_named(L, coord, v)
            if c is None:
                say("      (not expressible in v_0..v_3 -- look at the basis)")
            else:
                Q = c[0] * c[2] + c[1] * c[3]
                say(f"      in the basis v_0..v_3: {[str(x) for x in c]},"
                    f"  Q = {Q},  rank = {2 if Q else 1}")


def solve_in_named(L, coord, v):
    from exactlin import solve
    names = list(coord)
    A = [[coord[nm][i] for nm in names] for i in range(L.dim)]
    return solve(A, list(v))


# ------------------------------------------------------------------- Part B

def partB():
    say("")
    say("B. Every retract (a,b) with a <= 5: the invariants of the two copies")
    say("   in the odd layer H^1(M; T_1), against the homogeneous families")
    say("   of Statements 10.1 and 11.2 (they are over (3,3), (4,0), (4,2),")
    say("   (4,3), (4,4), (5,4), (5,5); over (2,0), (3,0), (3,1), (3,2), (4,1)")
    say("   there is none).")
    say("")
    say("   (a,b)   dim H^1(T_1)   dim ^s   dim ^s'   homogeneous family known")
    known = {(3, 3), (4, 0), (4, 2), (4, 3), (4, 4), (5, 4), (5, 5)}
    none = {(2, 0), (3, 0), (3, 1), (3, 2), (4, 1)}
    for a in range(0, 6):
        for b in range(0, a + 1):
            ks = [a, b]
            L = O.Layer(ks, 1)
            if L.dim == 0:
                continue
            _, invs = invariants(ks, 1, "s")
            ds = len(invs)
            if a == b:
                _, invs2 = invariants(ks, 1, "s'")
                ds2 = str(len(invs2))
            else:
                ds2 = "-"
            tag = ("yes" if (a, b) in known else
                   ("no" if (a, b) in none else ""))
            say(f"   ({a},{b})      {L.dim:3d}        {ds:3d}     {ds2:>3}"
                f"       {tag}")


# ------------------------------------------------------------------- Part C

def partC():
    say("")
    say("C. The retract CP^{1|4} = (1,1,1,1) in degree 3: the invariants of")
    say("   the copies of sl(2) in H^1(M; T_3), of dimension 8.")
    ks = [1, 1, 1, 1]
    L = O.Layer(ks, 3)
    say("   dim H^1(T_3) =", L.dim)
    for kind in ("s", "s'", "s''", "s'''"):
        ok = check_copy(ks, kind)
        _, inv = invariants(ks, 3, kind)
        say(f"   {kind}: sl(2) in v_0: {ok};  dim H^1(T_3)^{kind} = {len(inv)}")
        for v in inv:
            say(f"      invariant class: {[str(c) for c in v]}")
            say(f"      orbit dim (projective) = {L.orbit_dim_proj(list(v))}")


def partD():
    say("")
    say("D. The families carried by the invariant classes: even-homogeneity")
    say("   (the even part of the evaluation onto at every point of M) and")
    say("   homogeneity (the odd part as well).  dim_C v over Lambda(tau).")
    say("")
    import famtools as F
    cases = [([2, 0], "s"), ([4, 0], "s"), ([4, 1], "s"), ([4, 2], "s"),
             ([4, 3], "s"), ([4, 4], "s"), ([5, 4], "s"),
             ([3, 3], "s'"), ([5, 5], "s'")]
    for ks, kind in cases:
        L, inv = invariants(ks, 1, kind)
        if not inv:
            say(f"   {tuple(ks)} {kind}: no invariant class")
            continue
        vs = [list(v) for v in inv]
        if len(vs) > 1:                       # a generic member of the space
            gen = [sum(Fraction(j + 1) * vs[j][i] for j in range(len(vs)))
                   for i in range(L.dim)]
            vs = vs + [gen]
        for v in vs:
            coc = F.vec_to_coc(L.Q, v)
            M = F.cocycle_to_family(ks, coc, npar=1)
            deg = 12
            ev = M.evaluations(deg)
            drops = {}
            for chart in ("U", "V"):
                drops[chart] = ns._drop_locus(ev[chart], M.m)
            def bad(tag):
                for chart in ("U", "V"):
                    g = drops[chart][tag]
                    if g is not None and ns._has_affine_root(g):
                        return True
                return False
            evh = not bad("even")
            hom = evh and not bad("odd")
            sd = M.sdim_vect(deg)
            spl = ns.SM.split(ks, npar=1).sdim_vect(deg) \
                if hasattr(ns.SM, "split") else None
            say(f"   {tuple(ks)} {kind}: class {F.show(L.Q, v)}")
            say(f"      dim_C v = {sd}"
                + (f" against {spl} for the split family" if spl else "")
                + f";  even-homogeneous: {evh};  homogeneous: {hom}")


def partE():
    say("")
    say("E. The converse, over the retracts where the classification is")
    say("   complete: no invariant class, and indeed no even-homogeneous")
    say("   family.  H^1(T_1) is one orbit there, so one representative")
    say("   settles the retract.")
    import famtools as F
    for ks in ([3, 0], [3, 1], [3, 2]):
        L = O.Layer(ks, 1)
        _, inv = invariants(ks, 1, "s")
        say(f"   {tuple(ks)}: dim H^1(T_1) = {L.dim}, dim ^s = {len(inv)}")
        for j in range(L.dim):
            v = [Fraction(1) if i == j else Fraction(0) for i in range(L.dim)]
            coc = F.vec_to_coc(L.Q, v)
            M = F.cocycle_to_family(ks, coc, npar=1)
            ev = M.evaluations(12)
            def bad(tag):
                for chart in ("U", "V"):
                    g = ns._drop_locus(ev[chart], M.m)[tag]
                    if g is not None and ns._has_affine_root(g):
                        return True
                return False
            say(f"      class {F.show(L.Q, v)}: orbit dim (proj) ="
                f" {L.orbit_dim_proj(v)}, even-homogeneous: {not bad('even')},"
                f" homogeneous: {not bad('even') and not bad('odd')}")


def main():
    global LOG
    LOG = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "evenhom.log"), "w")
    say(__doc__)
    partA()
    partB()
    partC()
    partD()
    partE()
    LOG.close()


if __name__ == "__main__":
    main()
