#!/usr/bin/env python3
"""p = 2:  Skryabin's twist by an exponential of an out-of-range divided power,
done in the graded setting -- and what it reaches.

Skryabin's forms of the 2nd type are

    omega_{2,j} = d( exp(eps x_j^{(p^{u_j})}) alpha_0 ),   alpha_0 a primitive,

in which the exponential lives in the algebra whose j-th height is raised: the
divided power x_j^{(p^{u_j})} is out of range in O(m;u) itself.  At p = 2 with
u = 1 the factor is 1 + x_j^{(2)}, so

    omega_t = omega + d( x_j^{(2)} alpha ),      d alpha = omega,

and the question this engine answers is which classes of the census that
construction reaches over a SUPERalgebra.

Two things are peculiar to the graded case:

  * the twist is available along the EVEN indeterminates only.  An odd
    indeterminate has no divided powers to go out of range: xi^2 = 0 holds by
    supercommutativity, and raising the height of xi would make it even.  So
    over O(0|n;1) -- all indeterminates odd -- there is no twist at all;

  * a primitive of omega need not exist in O(m|n;1); it exists in the algebra
    with the raised height, which is where the twist is computed.  The twisted
    form is then asked to have its coefficients back in O(m|n;1).

Output: stwist.log.  Exact over F_2.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "lib"))
import sform as S
from exactlin import solve, backend
from scensus import (build, elementary_autos, gen_matrix, apply_map, show,
                     name)
from shomog import homogeneous_nondeg, block_types, h2_classes
from ssplit import orbit_of

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = open(os.path.join(HERE, "stwist.log"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


def embed(om):
    """a form of A is a form of A' -- same exponent tuples, wider caps."""
    return dict(om)


def in_range(A, X):
    """does every monomial of X lie in the smaller algebra A?"""
    for mm in X:
        for i, e in enumerate(mm):
            if e >= A.cap[i]:
                return False
    return True


def primitive(Ap, om):
    """alpha with d alpha = omega, in the bigger algebra; None if omega is not
    exact there."""
    B1 = Ap.basis(1)
    B2 = Ap.basis(2)
    i2 = {mm: i for i, mm in enumerate(B2)}
    d = Ap.d()
    cols = []
    for mm in B1:
        cols.append(S.vec(Ap, d({mm: 1}), i2))
    M = [[cols[k][r] for k in range(len(B1))] for r in range(len(B2))]
    b = S.vec(Ap, om, i2)
    try:
        x = solve(M, b, 2)
    except Exception:
        return None
    if x is None:
        return None
    al = {}
    for k, c in enumerate(x):
        if int(c) % 2:
            al[B1[k]] = 1
    # control
    chk = Ap.add(d(al), om)
    assert not chk, "the primitive found does not differentiate to omega"
    return al


def twist(Ap, al, j):
    """d( x_j^{(2)} alpha ) -- the twist along the j-th even indeterminate."""
    u = Ap.x(j, 2)                      # the out-of-range divided power
    return Ap.d()(Ap.mul(u, al))


def preserving_in(Ap, A, om):
    """{D in vect(A) | L_D omega = 0} for a form omega living in the bigger
    algebra Ap -- Skryabin's convention: the fields stay in the small algebra."""
    dims = []
    deg2 = Ap.basis(2)
    i2 = {mm: i for i, mm in enumerate(deg2)}
    for pr in (0, 1):
        B = S.vfield_basis(A, pr)
        rows = [S.vec(Ap, Ap.lie(D)(om), i2) for D in B]
        T = [[rows[k][i] for k in range(len(rows))] for i in range(len(deg2))]
        ns = S.nullspace_p(T, 2) if T else []
        dims.append(len(ns))
    return tuple(dims)


def diagonal(A, om):
    """the coefficients of the divided squares (dg_a)^{(2)} -- the Gram
    diagonal.  It is a vector of CONSTANTS for a closed form."""
    out = []
    for a in range(A.m + A.n):
        g = (A.idx + a) if a < A.m else (A.idxi + a - A.m)
        e = [0] * A.N
        e[g] = 2
        out.append(om.get(tuple(e), 0) % 2)
    return tuple(out)


def diagonal_split(A):
    """Theorem: Z^2 = (closed alternating forms) + K^M, the second summand being
    the constant diagonals.  Returns (dim Z^2, dim of the alternating part, M)."""
    B2 = A.basis(2)
    i3 = {mm: i for i, mm in enumerate(A.basis(3))}
    d = A.d()
    M = S.matrix_of(A, d, B2, i3)
    T = [[M[r][c] for r in range(len(B2))] for c in range(len(i3))]
    ns = S.nullspace_p(T, 2) if T else []
    alt = 0
    for v in ns:
        om = {B2[k]: 1 for k in range(len(B2)) if int(v[k]) % 2}
        if not any(diagonal(A, om)):
            alt += 1
    # the dimension of the alternating part of Z^2, by echelon on the diagonals
    rows = []
    for v in ns:
        om = {B2[k]: 1 for k in range(len(B2)) if int(v[k]) % 2}
        rows.append(list(diagonal(A, om)))
    r = S.rank_p(rows, 2) if rows else 0
    return len(ns), len(ns) - r, A.m + A.n, r


def exact_implies_alternating(A):
    """control: d(Omega^1) has no divided-square component, in every case."""
    B1 = A.basis(1)
    d = A.d()
    bad = 0
    for mm in B1:
        if any(diagonal(A, d({mm: 1}))):
            bad += 1
    return len(B1), bad


def run(m, n):
    A = S.Om(m, n, 2, (1,) * m, dh=2, dxh=2)
    B2, idx, zbas, gram_bits, G = build(A)
    gens = elementary_autos(A, graded=True)
    maps = [gen_matrix(A, phi, B2, idx) for phi in gens]
    red = h2_classes(A, B2, idx, None, maps)

    say("")
    say("=" * 78)
    nb1, bad = exact_implies_alternating(A)
    dz, dalt, MM, rk = diagonal_split(A)
    say(f"  O({m}|{n};1), p = 2   [d of each of the {nb1} basis 1-forms: "
        f"{bad} with a divided square in it]")
    say(f"    dim Z^2 = {dz} = {dalt} (alternating) + {rk} (diagonals), "
        f"M = {MM}: {'ok' if rk == MM else 'MISMATCH'}")
    for par, pname in ((0, "even"), (1, "odd")):
        el, dz = homogeneous_nondeg(A, zbas, gram_bits, G, par)
        if not el:
            continue
        left, classes = set(el), []
        while left:
            s = next(iter(left))
            orb = orbit_of(s, maps)
            best = min(orb, key=lambda v: (bin(v).count("1"), v))
            classes.append((best, orb))
            left -= orb
        classes.sort(key=lambda t: (bin(t[0]).count("1"), t[0]))
        label = {}
        for k, (rep, orb) in enumerate(classes):
            for v in orb:
                label[v] = k
        say(f"    {len(classes)} class(es) of {pname} forms")
        for k, (rep, orb) in enumerate(classes):
            om = show(A, rep, B2)
            say(f"      class {k}:  |orbit| = {len(orb):6d}  "
                f"diagonal = {diagonal(A, om)}  "
                f"sdim h = {S.preserving(A, om)[:2]}   "
                f"omega = {name(A, om)}")
        if m == 0:
            say("      NO twist here: every indeterminate is odd, and an odd")
            say("      indeterminate has no divided power to take out of range.")
            continue
        for j in range(m):
            u = [1] * m
            u[j] = 2
            Ap = S.Om(m, n, 2, tuple(u), dh=2, dxh=2)
            say(f"      twisting along x{j+1}, u = {tuple(u)}:")
            for k, (rep, orb) in enumerate(classes):
                om = show(A, rep, B2)
                al = primitive(Ap, embed(om))
                if al is None:
                    say(f"        class {k}: not exact in O({m}|{n};{tuple(u)})"
                        f" -- no primitive, no twist"
                        + ("  [diagonal non-zero]" if any(diagonal(A, om))
                           else ""))
                    continue
                omt = twist(Ap, al, j)
                tot = Ap.add(embed(om), omt)
                if not in_range(A, tot):
                    sd = preserving_in(Ap, A, tot)
                    sd0 = S.preserving(A, om)[:2]
                    say(f"        class {k}: the twisted form leaves "
                        f"O({m}|{n};1); its preserving algebra inside "
                        f"vect({m}|{n};1) has sdim {sd} against {sd0}")
                    continue
                v = 0
                for mm, c in tot.items():
                    if c % 2:
                        v |= 1 << idx[mm]
                if v not in label:
                    nd, _ = S.nondegenerate(A, show(A, v, B2))
                    say(f"        class {k} --> "
                        f"{'degenerate' if not nd else 'outside the census'}")
                    continue
                say(f"        class {k} --> class {label[v]}"
                    + ("  (unchanged)" if label[v] == k else "  *** MOVED ***"))


def dimchecks():
    say("")
    say("=" * 78)
    say("  Theorem: a closed form is (closed alternating) + (constant diagonal),")
    say("  and d of a 1-form never has a divided square.")
    for (m, n, u) in [(1, 0, (1,)), (0, 1, ()), (2, 0, (1, 1)), (1, 1, (1,)),
                      (0, 2, ()), (3, 0, (1, 1, 1)), (2, 1, (1, 1)),
                      (1, 2, (1,)), (0, 3, ()), (2, 0, (2, 1)), (1, 1, (2,)),
                      (4, 0, (1, 1, 1, 1)), (2, 2, (1, 1)), (1, 3, (1,))]:
        A = S.Om(m, n, 2, u, dh=2, dxh=2)
        dz, dalt, MM, rk = diagonal_split(A)
        nb, bad = exact_implies_alternating(A)
        say(f"    O({m}|{n};{u if u else '-'}): dim Z^2 = {dz} = {dalt} + {rk}, "
            f"M = {MM} {'ok' if rk == MM else 'MISMATCH'};  "
            f"{bad} of {nb} basis 1-forms have a divided square in d")
        assert rk == MM and bad == 0


if __name__ == "__main__":
    say(f"Skryabin's twist in the graded setting, p = 2  [exactlin: {backend()}]")
    t0 = time.time()
    dimchecks()
    for M in (1, 2, 3):
        for n in range(0, M + 1):
            run(M - n, n)
    say(f"\ntotal {time.time()-t0:.1f} s")
    LOG.close()
