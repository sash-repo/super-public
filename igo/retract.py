#!/usr/bin/env python3
"""The obstruction to splitness for an arbitrary completely reducible retract.

bott.py answers the question for a retract that is a sum of CHARACTERS (there
Lambda^2(E) is again a line bundle, so Bott's theorem applies to it directly),
and cot.py answers it for the one retract E = Omega^1.  This module does it for
an arbitrary sum of irreducible homogeneous bundles, which is what Remark 4.3.6
of the text needs: below the bound D_m the representation phi of P is
completely reducible, so E is such a sum, but Lambda^p(E) is then NOT
irreducible and Bott's theorem applies to it only after a decomposition.

The chain is

    phi = sum of irreducible l-modules      (given by highest weights)
      -> the weight multiset of E           (Freudenthal, over the Levi l)
      -> the weight multiset of Lambda^p(E)
      -> + theta, i.e., tensor with Theta_X
      -> Klimyk: decompose into irreducible l-modules
      -> Bott, constituent by constituent   (R.cohomology)

and the outcome is H^q(X; Theta_X (x) Lambda^p(E)), the cohomology of the
quotient in the extension (11) of the text,

    0 -> E^* (x) Lambda^{p+1}(E) -> T_p -> Theta_X (x) Lambda^p(E) -> 0.

By Green's theory the supermanifolds with retract (X, Lambda(E)) are classified
by the non-abelian H^1 of the sheaf of automorphisms identical modulo I^2, which
is filtered by the T_p with p >= 2 even.  The quotient halves carry the
obstructions to a RETRACTION: if H^1(X; Theta_X (x) Lambda^p(E)) = 0 for every
even p >= 2, every supermanifold with that retract admits a retraction onto X.
Splitness needs the sub halves as well; they are computed in sub.py.

Conventions are bott.py's: a weight is written in omega-coordinates, an
irreducible homogeneous bundle E_lam is the one Bott's theorem is applied to by
R.cohomology(lam), the character d^k is k*omega_node, and Theta_X = E_theta with
theta the highest root.  They are calibrated here by two independent tests: on a
retract of rank 2 by characters this module must reproduce bott.py's
tangent_twist, and on E = Omega^1 it must reproduce cot.py's Omega^p (x) Theta,
which is built the other way round, out of Lambda^p(n_-) (x) n_+.

Exact arithmetic throughout (Fraction); the inverse Cartan matrix, needed for
the inner product on the weight lattice, is taken from ../lib/exactlin.py.
"""

import os
import sys
from collections import defaultdict
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

from exactlin import solve                                   # noqa: E402

from bott import CHSS, chss_list, omega                      # noqa: E402
from cot import decompose, levi_weyl                         # noqa: E402


# --------------------------------------------------- the form on the weights

_GRAM = {}


def gram(R):
    """G[i][j] = (omega_i, omega_j), exactly.

    (alpha_i, alpha_j) = d_i A_ij, and omega_i is defined by
    2(omega_i, alpha_j)/(alpha_j, alpha_j) = delta_ij, i.e., (omega_i, alpha_j)
    = delta_ij d_j.  Writing omega_i = sum_k c_ik alpha_k gives C(DA) = D, so
    C = D A^{-1} D^{-1} and (omega_i, omega_j) = c_ji d_i = d_j (A^{-1})_ji.
    """
    key = (R.typ, R.rank)
    if key in _GRAM:
        return _GRAM[key]
    n = R.rank
    A = [[Fraction(R.A[i][j]) for j in range(n)] for i in range(n)]
    # A^{-1} column by column
    inv = [[None] * n for _ in range(n)]
    for j in range(n):
        e = [Fraction(1) if i == j else Fraction(0) for i in range(n)]
        x = solve(A, e)
        assert x is not None, "the Cartan matrix is invertible"
        for i in range(n):
            inv[i][j] = Fraction(x[i])
    G = [[R.d[j] * inv[j][i] for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            assert G[i][j] == G[j][i], "the Gram matrix is symmetric"
    _GRAM[key] = G
    return G


def ip(R, mu, nu):
    """(mu, nu) for two weights in omega-coordinates."""
    G = gram(R)
    n = R.rank
    return sum(Fraction(mu[i]) * Fraction(nu[j]) * G[i][j]
               for i in range(n) for j in range(n)
               if mu[i] and nu[j])


# ------------------------------------------------------------- the Levi data

def levi_pos(X):
    """The positive roots of l, as (coefficient vector, omega-coordinates)."""
    out = []
    for r in X.R.pos:
        if r[X.node - 1] == 0:
            out.append((r, tuple(X.R.to_omega(r))))
    return out


def levi_simple(X):
    """The simple roots of l, in omega-coordinates."""
    out = []
    for i in range(X.R.rank):
        if i != X.node - 1:
            e = tuple(1 if j == i else 0 for j in range(X.R.rank))
            out.append(tuple(X.R.to_omega(e)))
    return out


def levi_dim(X, lam):
    """Weyl's dimension formula for the irreducible l-module of h.w. lam.

    rho of g may be used in place of rho_l: the two pair alike with every root
    of l, since <rho, alpha_i^vee> = 1 = <rho_l, alpha_i^vee> for i not the node
    and every root of l is a combination of those simple roots.
    """
    R = X.R
    lr = [lam[i] + 1 for i in range(R.rank)]
    num = den = Fraction(1)
    for coeff, _ in levi_pos(X):
        num *= R.pair(lr, coeff)
        den *= R.pair(R.rho, coeff)
    r = num / den
    assert r.denominator == 1, (lam, r)
    return int(r)


# ------------------------------------------------------------- Freudenthal

_WEIGHTS = {}


def weights(X, lam):
    """The weight multiset {weight : multiplicity} of the l-module V_lam.

    Freudenthal's recursion, level by level below lam:
      (|lam+rho|^2 - |mu+rho|^2) m_mu = 2 sum_{alpha>0} sum_{k>=1}
                                           m_{mu+k alpha} (mu + k alpha, alpha).
    A weight mu != lam of an irreducible module always has mu + alpha_i a weight
    for some simple alpha_i, so generating each level from the one above it
    reaches every weight; Freudenthal returns 0 on a non-weight, which prunes.
    """
    key = (X.name, tuple(lam))
    if key in _WEIGHTS:
        return _WEIGHTS[key]
    R = X.R
    lam = tuple(lam)
    assert all(lam[i] >= 0 for i in range(R.rank) if i != X.node - 1), \
        "lam must be l-dominant"
    rho = tuple(R.rho)
    lr = tuple(lam[i] + rho[i] for i in range(R.rank))
    norm_lam = ip(R, lr, lr)

    pos = levi_pos(X)
    simple = levi_simple(X)

    mult = {lam: 1}
    level = [lam]
    while level:
        cand = set()
        for mu in level:
            for a in simple:
                cand.add(tuple(mu[j] - a[j] for j in range(R.rank)))
        cand -= set(mult)
        nxt = []
        for mu in sorted(cand):
            mr = tuple(mu[i] + rho[i] for i in range(R.rank))
            denom = norm_lam - ip(R, mr, mr)
            if denom == 0:
                continue
            tot = Fraction(0)
            for coeff, a in pos:
                k = 1
                while True:
                    up = tuple(mu[j] + k * a[j] for j in range(R.rank))
                    m = mult.get(up)
                    if m is None:
                        # not yet computed => above this level, so absent
                        break
                    if m:
                        tot += m * ip(R, up, a)
                    k += 1
            if tot == 0:
                continue
            m = 2 * tot / denom
            assert m.denominator == 1 and m > 0, (mu, m)
            mult[mu] = int(m)
            nxt.append(mu)
        level = nxt

    assert sum(mult.values()) == levi_dim(X, lam), \
        ("Freudenthal disagrees with Weyl", lam)
    _WEIGHTS[key] = mult
    return mult


# ------------------------------------------------------- multiset arithmetic

def add_weight(m, shift):
    """Tensor a weight multiset with a character / a 1-dimensional module."""
    out = defaultdict(int)
    for mu, c in m.items():
        out[tuple(mu[j] + shift[j] for j in range(len(shift)))] += c
    return dict(out)


def direct_sum(*ms):
    out = defaultdict(int)
    for m in ms:
        for mu, c in m.items():
            out[mu] += c
    return dict(out)


def wedge(m, p):
    """The weight multiset of Lambda^p(V) from that of V.

    A weight of multiplicity c contributes a c-dimensional space, whose
    exterior powers are binom(c, i)-dimensional; the weights add.
    """
    if p == 0:
        return {tuple(): 1} if False else {}
    items = sorted(m.items())
    n = len(items)
    out = defaultdict(int)

    def rec(i, left, wt, coef):
        if left == 0:
            out[tuple(wt)] += coef
            return
        if i == n:
            return
        mu, c = items[i]
        take = min(left, c)
        for t in range(take + 1):
            if t and t > c:
                break
            b = _binom(c, t)
            if b:
                rec(i + 1, left - t,
                    [wt[j] + t * mu[j] for j in range(len(mu))], coef * b)

    rank = sum(m.values())
    if p > rank:
        return {}
    first = next(iter(m))
    rec(0, p, [0] * len(first), 1)
    return {k: v for k, v in out.items() if v}


def _binom(n, k):
    if k < 0 or k > n:
        return 0
    r = 1
    for i in range(k):
        r = r * (n - i) // (i + 1)
    return r


# ----------------------------------------------------------- the obstruction

def cohomology_of(X, m, W=None):
    """H^*(X; E) for the homogeneous bundle with weight multiset m.

    -> {degree : [(dim, highest weight, multiplicity)]}
    """
    parts = decompose(X, m, W)
    out = defaultdict(list)
    for lam, mult in parts.items():
        c = X.R.cohomology(list(lam))
        if c is None:
            continue
        ell, hw, dd = c
        out[ell].append((dd, tuple(hw), mult))
    return dict(out)


def obstruction(X, E, p, W=None):
    """H^*(X; Theta_X (x) Lambda^p(E)) for the retract with weight multiset E."""
    lp = wedge(E, p)
    if not lp:
        return {}
    theta = list(X.R.theta)
    return cohomology_of(X, add_weight(lp, theta), W)


def dims_of(coh):
    return {q: sum(dd * mult for dd, _, mult in v) for q, v in coh.items()}


def retract(X, parts):
    """The weight multiset of E = sum of (highest weight, twist) pieces.

    parts: a list of (lam, k), meaning the irreducible l-module of highest
    weight lam twisted by the character d^k.  lam is given in omega-coordinates
    with 0 at the node, the twist carrying the whole of the central character.
    """
    ms = []
    for lam, k in parts:
        base = weights(X, lam)
        ms.append(add_weight(base, omega(X.R.rank, X.node, k)))
    return direct_sum(*ms)


def splits(X, E, W=None, pmax=None):
    """Do the quotient halves Theta_X (x) Lambda^p(E), p >= 2, have H^1 = 0?

    These carry the obstructions to a retraction (see the module docstring).
    Returns (True, {}) or (False, {p : dim H^1}).
    """
    rank = sum(E.values())
    if pmax is None:
        pmax = rank
    bad = {}
    for p in range(2, pmax + 1):
        d = dims_of(obstruction(X, E, p, W)).get(1, 0)
        if d:
            bad[p] = d
    return (not bad), bad


# ------------------------------------------------------------------- testing

def tests():
    ok = True

    def check(what, got, want):
        nonlocal ok
        good = got == want
        ok = ok and good
        print("  %-58s %s   %s" % (what, "ok " if good else "FAIL", got))

    print("Calibration against bott.py: a retract of rank 2 by characters.")
    # Lambda^2(d^k + d^k') = d^{k+k'}, so the answer must be tangent_twist(k+k').
    for name in ("Q_3", "Q_5", "Gr_2^4", "Gr_2^5", "LGr_4", "OGr_5"):
        X = named(name)
        W = levi_weyl(X)
        for k, kp in ((0, -2), (1, -3), (-1, -1), (2, -5), (0, -3)):
            E = retract(X, [([0] * X.R.rank, k), ([0] * X.R.rank, kp)])
            got = dims_of(obstruction(X, E, 2, W))
            c = X.tangent_twist(k + kp)
            want = {} if c is None else {c[0]: c[2]}
            check("%s, E = O(%d)+O(%d)" % (name, k, kp), got, want)

    print("\nCalibration against cot.py: E = Omega^1, built the other way.")
    import cot
    for name in ("CP^2", "Q_3", "Q_5", "Gr_2^4", "Gr_2^5", "LGr_3", "OGr_5"):
        X = named(name)
        W = levi_weyl(X)
        # Omega^1 = E_{-alpha_0}: the h.w. is minus the node's simple root
        a0 = tuple(X.R.to_omega(tuple(1 if j == X.node - 1 else 0
                                      for j in range(X.R.rank))))
        lam = [-a0[j] for j in range(X.R.rank)]
        E = weights(X, canonical(X, lam))
        check("%s, dim Omega^1 = dim X" % name, sum(E.values()), X.dimX)
        for p in (1, 2, 3):
            got = dims_of(obstruction(X, E, p, W))
            want = cot.dims(X, p, W)
            check("%s, H^*(Theta (x) Omega^%d)" % (name, p), got, want)

    print("\nFreudenthal against Weyl's dimension formula.")
    X = named("LGr_4")
    for lam, want in ((fund(X, 1), 4), (fund(X, 2), 6), (fund(X, 3), 4)):
        check("LGr_4, dim of the l-module %s" % (lam,),
              sum(weights(X, lam).values()), want)
    X = named("Gr_3^7")
    for lam, want in ((fund(X, 1), 3), (fund(X, 2), 3), (fund(X, 4), 4),
                      (fund(X, 5), 6)):
        check("Gr_3^7, dim of the l-module %s" % (lam,),
              sum(weights(X, lam).values()), want)

    print("\nLambda^p of a weight multiset.")
    X = named("LGr_4")
    id4 = weights(X, fund(X, 1))
    check("LGr_4, dim Lambda^2(id) = 6", sum(wedge(id4, 2).values()), 6)
    check("LGr_4, dim Lambda^3(id) = 4", sum(wedge(id4, 3).values()), 4)
    check("LGr_4, dim Lambda^4(id) = 1", sum(wedge(id4, 4).values()), 1)
    check("LGr_4, Lambda^5(id) = 0", sum(wedge(id4, 5).values()), 0)
    # Lambda^2 of a sum of 3 characters is 3-dimensional
    E = retract(X, [([0] * X.R.rank, 0)] * 3)
    check("LGr_4, dim Lambda^2(O+O+O) = 3", sum(wedge(E, 2).values()), 3)
    # Lambda^2(id) as a bundle must be the fundamental one, of dim 6
    got = decompose(X, wedge(id4, 2))
    check("LGr_4, Lambda^2(id) is irreducible of h.w. omega_2",
          got, {tuple(fund(X, 2)): 1})

    print()
    return ok


def fund(X, i):
    """The i-th fundamental weight of l' (i indexes the nodes of l)."""
    return omega(X.R.rank, i)


def canonical(X, lam):
    """Make lam l-dominant by moving the node's coordinate out of the way.

    The node's coordinate is the central character and is free; the others must
    be >= 0 for lam to be l-dominant.
    """
    return list(lam)


_BY_NAME = None


def named(name):
    global _BY_NAME
    if _BY_NAME is None:
        _BY_NAME = {X.name: X for X in chss_list()}
        _BY_NAME["CP^2"] = _BY_NAME["Gr_1^3"]
        _BY_NAME["CP^1"] = _BY_NAME["Gr_1^2"]
    return _BY_NAME[name]


# ---------------------------------------------------------------- the sweep

def levi_modules_up_to(X, mmax):
    """EVERY non-trivial irreducible l'-module of dimension <= mmax.

    Not merely the fundamental ones: the census below has to be exhaustive, so
    all l-dominant weights are enumerated.  The search is cut off coordinate by
    coordinate, Weyl's dimension formula being strictly increasing in each
    coordinate of a dominant weight, so a coordinate whose value already
    overshoots mmax can only overshoot further.
    """
    R = X.R
    idx = [i for i in range(R.rank) if i != X.node - 1]
    out = []

    def rec(pos, lam):
        if pos == len(idx):
            if any(lam):
                d = levi_dim(X, lam)
                if d <= mmax:
                    out.append((_name_of(X, lam), list(lam), d))
            return
        i = idx[pos]
        c = 0
        while True:
            lam[i] = c
            if levi_dim(X, lam) > mmax:
                lam[i] = 0
                return
            rec(pos + 1, lam)
            lam[i] = 0
            c += 1

    rec(0, [0] * R.rank)
    return sorted(out, key=lambda t: (t[2], t[1]))


def _name_of(X, lam):
    nz = [(i, lam[i]) for i in range(X.R.rank) if lam[i]]
    if len(nz) == 1:
        i, c = nz[0]
        if c == 1:
            return "id" if i == 0 else "L^%d(id)" % (i + 1)
        return "%d*omega_%d" % (c, i + 1)
    return "+".join("%d*omega_%d" % (c, i + 1) for i, c in nz)


def layer(X, lam, a, j, W=None):
    """H^*(X; Theta_X (x) Lambda^a(V_lam) (x) d^j), the building block.

    Every retract that is a sum of irreducible bundles reduces to these: for
    E = V (x) d^k + sum_i d^{k_i} one has

        Lambda^p(E) = sum_{a+b=p} Lambda^a(V) (x) d^{ak} (x)
                          sum_{i_1<...<i_b} d^{k_{i_1}+...+k_{i_b}},

    so H^*(X; Theta (x) Lambda^p(E)) is a sum of the layers below, and it has
    H^1 =/= 0 if and only if one of the layers entering it has.
    """
    if a == 0:
        base = {tuple([0] * X.R.rank): 1}
    else:
        base = wedge(weights(X, lam), a)
    if not base:
        return {}
    shift = [X.R.theta[t] + omega(X.R.rank, X.node, j)[t]
             for t in range(X.R.rank)]
    return cohomology_of(X, add_weight(base, shift), W)


def family_sweep(name, jrange=range(-30, 31), out=sys.stdout, mmax=None):
    """Every completely reducible retract over X of odd dimension <= mmax.

    Over LGr_n, mmax = 2n-1 is the range of Igonin's announcement in [Ig2].
    Below D_m the representation phi is completely reducible, so E is a sum of
    irreducible homogeneous bundles, and by the factorization in `layer` the
    whole question is whether some H^1(X; Theta (x) Lambda^a(V) (x) d^j) is
    non-zero for an irreducible constituent V that fits under the bound.  That
    is a finite check for each V, swept here over the twists j.
    """
    X = named(name)
    W = levi_weyl(X)
    Dm = min(X.min_irrep()[0], X.dimX + 1)
    if mmax is None:
        mmax = Dm - 1
    zero = [0] * X.R.rank

    small = levi_modules_up_to(X, mmax)

    print("%s: dim X = %d, D_m = %d; retracts of odd dimension <= %d."
          % (name, X.dimX, Dm, mmax), file=out)
    print("Non-character constituents that fit under the bound: %s"
          % (", ".join("%s (dim %d)" % (nm, d) for nm, _, d in small) or "none"),
          file=out)
    print("Twists swept: %d <= j <= %d.\n" % (jrange.start, jrange.stop - 1),
          file=out)

    hits = []
    # a = 0 is the pure-character layer, which is bott.py's obstruction_census
    print("  a = 0 (the characters alone: Lambda^2 of a sum of characters):",
          file=out)
    got = [j for j in jrange if dims_of(layer(X, zero, 0, j, W)).get(1, 0)]
    print("    non-zero H^1 for j in %s" % (got or "no j"), file=out)
    if got:
        hits.append(("characters", 0, got))

    for nm, lam, d in small:
        rank = d
        for a in range(1, min(rank, mmax) + 1):
            nz = []
            for j in jrange:
                v = dims_of(layer(X, lam, a, j, W)).get(1, 0)
                if v:
                    nz.append((j, v))
            if nz:
                hits.append((nm, a, nz))
                print("  %s, a = %d: non-zero H^1 for %s"
                      % (nm, a, ", ".join("j=%d (dim %d)" % t for t in nz)),
                      file=out)
            else:
                print("  %s, a = %d: H^1 = 0 for every j in the range"
                      % (nm, a), file=out)
    print(file=out)
    if hits:
        print("  => some layer Theta (x) Lambda^p(E) has H^1 =/= 0; see above.",
              file=out)
    else:
        print("  => H^1(X; Theta (x) Lambda^p(E)) = 0 for every such retract"
              " and every p >= 2: every supermanifold admits a retraction.",
              file=out)
    return hits


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


if __name__ == "__main__":
    if "--test" in sys.argv or len(sys.argv) == 1:
        if not tests():
            sys.exit("TESTS FAILED")
    if "--lgr" in sys.argv:
        # one writer, opened here: a run killed halfway still leaves the log
        with open("lgr.log", "w") as fh:
            tee = _Tee(sys.stdout, fh)
            for n in (2, 3, 4, 5, 6, 7):
                family_sweep("LGr_%d" % n, mmax=2 * n - 1, out=tee)
                print("-" * 70, file=tee)
