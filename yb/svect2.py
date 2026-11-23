"""
Problem 5.22.1 for g = svect^L_lambda(1|2): Proposition 5.22.9 and Remark 5.22.10.

The torus t = <w_lambda, h> of Lemma 5.22.3 4) is two-dimensional.  The degree-0 component g_0
of Lemma 5.22.3 2), which for lambda =/= 0 is sl(2) + C w_lambda, is the zero component of the
Z-grading deg = 2m + n, not the centralizer of t: of the sl(2) = <theta_1 d_2, theta_2 d_1, h>,
the two root vectors have h-weight +-2 and so do NOT lie in the t-weight-0 component.

vect^L(1|2) is Z^3-graded by the weight

       alpha = (m; n_1, n_2)  of  t^a theta_S d_mu,
       m = a - [mu = t],  n_i = [i in S] - [mu = theta_i],

div_lambda preserves that weight (it is a map to functions of the same weight), so
g = Ker div_lambda is Z^3-graded as well, with the SAME weights.  The torus t = <w_lambda, h>
sees only the projection

       alpha  |-->  ( 2m + (1+lambda)(n_1+n_2),  n_1 - n_2 ),

two conditions instead of three, and yet the fibre of that projection over 0 meets the weights
of g in ONE point as soon as lambda is not an integer:

   *  n_1 = n_2 = 0, m = 0 : the space <t d_t, th_1 d_1, th_2 d_2> cut by div_lambda, i.e., t
      itself, of dim 2;
   *  n_1 = n_2 = 1 : forces mu = t, S = {1,2}, and 2m + 2(1+lambda) = 0, i.e.,
      m = -(1+lambda), an integer if and only if lambda is an integer; the element is then
      t^{-lambda} theta_1 theta_2 d_t, which is div_lambda-free exactly for that exponent;
   *  n_1 = n_2 = -1 is carried by no monomial at all.

So, for lambda not an integer, t is SELF-CENTRALIZING in g and the argument of Proposition
5.22.5 3) goes through verbatim:

       H^1(g; M) = H^1(g, t; M)   for every t-semisimple M with M^g = 0,

which is formula (42).  For lambda in Z the centralizer is one dimension bigger,
t + C t^{-lambda} theta_1 theta_2 d_t, and g is not perfect: its derived algebra
{svect^L}'(1|2) is the member of the list (15).

Printed: parts 1)-4) and 7), the structure (the torus, the weights, the centralizer, the
derived algebra) for several lambda; part 5) the invariants of g (x) g of weight 0; part 6)
dim Z^1 and dim B^1 of the relative complex over the window |a| <= N, for lambda = 1/3
(N = 1, 2, 3), 1/2 and -1/3 (N = 1, 2), and, for comparison, lambda = 1 (N = 1, 2), where t is
not self-centralizing.

Everything below is exact rational arithmetic.

Run:  python3 svect2.py
"""

import os
import sys
from fractions import Fraction as Q
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "lib"), HERE]

from yb import Vect, SVect, _kernel_basis
from exactlin import rank, backend

FAILED = []

V = Vect(2)
ODD = (1, 2)


def say(s=""):
    print(s, flush=True)


def hdr(s):
    say("\n" + "=" * 74)
    say(s)
    say("=" * 74)


def check(name, ok):
    say("  %-64s %s" % (name, "ok" if ok else "FAILED"))
    if not ok:
        FAILED.append(name)
    return ok


# --------------------------------------------------------------- the Z^3-weight
def wt3(key):
    """alpha = (m; n_1, n_2) of the monomial t^a theta_S d_mu, eq. (wt)."""
    mu, a, S = key
    m = a - (1 if mu == 't' else 0)
    n = tuple((1 if i in S else 0) - (1 if mu == i else 0) for i in ODD)
    return (m,) + n


def mons(alpha):
    """The monomials of vect^L(1|2) of weight alpha (at most three of them)."""
    m, n1, n2 = alpha
    out = []
    n = (n1, n2)
    if all(x in (0, 1) for x in n):                     # mu = t
        S = tuple(i for i, x in zip(ODD, n) if x == 1)
        out.append(('t', m + 1, S))
    for i in ODD:                                       # mu = theta_i
        j = 3 - i
        ni, nj = n[i - 1], n[j - 1]
        if ni + 1 in (0, 1) and nj in (0, 1):
            S = tuple(sorted([k for k in ODD
                              if (k == i and ni + 1 == 1) or (k == j and nj == 1)]))
            out.append((i, m, S))
    return out


def parity(alpha):
    return (alpha[1] + alpha[2]) % 2


# ------------------------------------------------------------- the algebra g
class SV:
    """svect^L_lambda(1|2) with a basis organised by the Z^3-weight.

    For each weight the basis is the kernel basis of div_lambda on the (at most
    three) monomials of that weight; `_kernel_basis` returns it in the free-
    variable normal form, so every basis vector is 1 on its own free monomial
    and 0 on the other free ones -- which is what makes `coords` a read-off.
    """

    def __init__(self, lam, N):
        self.lam = lam
        self.N = N
        self.g = SVect(lam)
        self.bas = {}        # alpha -> list of dicts
        self.free = {}       # alpha -> list of the free monomials
        for alpha in self.weights():
            ks = mons(alpha)
            b = _kernel_basis(ks, self.g.div)
            if not b:
                continue
            # the free monomials: those on which exactly one basis vector is 1
            fr = []
            for k in ks:
                col = [v.get(k, Q(0)) for v in b]
                if col.count(Q(1)) == 1 and all(x in (Q(0), Q(1)) for x in col) \
                        and sum(1 for x in col if x) == 1:
                    fr.append(k)
            self.bas[alpha] = b
            self.free[alpha] = fr

    def weights(self):
        """Every weight carried by a monomial with |a| <= N."""
        out = set()
        for k in V.keys(range(-self.N, self.N + 1)):
            out.add(wt3(k))
        return sorted(out)

    def index(self):
        """A flat basis: list of (alpha, i), plus the lookup."""
        flat, idx = [], {}
        for alpha in sorted(self.bas):
            for i in range(len(self.bas[alpha])):
                idx[(alpha, i)] = len(flat)
                flat.append((alpha, i))
        return flat, idx

    def vec(self, alpha, i):
        return self.bas[alpha][i]

    def coords(self, X):
        """Write X (a dict over monomials, lying in g) in the basis above.

        Returns None if some component falls outside the window.
        """
        out = {}
        bywt = {}
        for k, c in X.items():
            if not c:
                continue
            bywt.setdefault(wt3(k), {})[k] = c
        for alpha, part in bywt.items():
            if alpha not in self.bas:
                return None
            fr = self.free[alpha]
            if len(fr) != len(self.bas[alpha]):
                return None                      # normal form not available
            cs = [part.get(k, Q(0)) for k in fr]
            # rebuild and compare -- proves the read-off is right
            rec = {}
            for c, v in zip(cs, self.bas[alpha]):
                if not c:
                    continue
                for k, e in v.items():
                    rec[k] = rec.get(k, Q(0)) + c * e
            rec = {k: c for k, c in rec.items() if c}
            if rec != {k: c for k, c in part.items() if c}:
                raise AssertionError("basis read-off failed at %s" % (alpha,))
            for i, c in enumerate(cs):
                if c:
                    out[(alpha, i)] = out.get((alpha, i), Q(0)) + c
        return out


def wlam(lam):
    return {('t', 1, ()): Q(2), (1, 0, (1,)): Q(1) + lam, (2, 0, (2,)): Q(1) + lam}


HELT = {(1, 0, (1,)): Q(1), (2, 0, (2,)): Q(-1)}


def tweight(alpha, lam):
    """The weight of alpha for the torus t = <w_lambda, h>."""
    m, n1, n2 = alpha
    return (Q(2) * m + (Q(1) + lam) * (n1 + n2), n1 - n2)


# =========================================================== 1) the torus is in g
def part1(lam, N=4):
    hdr("1) lambda = %s: t = <w_lambda, h> lies in g, is abelian, and acts on\n"
        "   the weight-alpha component by (2m+(1+lambda)(n_1+n_2), n_1-n_2)"
        % lam)
    sv = SV(lam, N)
    g = sv.g
    w, h = wlam(lam), HELT
    ok = check("div_lambda(w_lambda) = 0", not g.div(w))
    ok &= check("div_lambda(h) = 0", not g.div(h))
    ok &= check("[w_lambda, h] = 0", not g.brk(w, h))
    bad = 0
    total = 0
    for alpha in sorted(sv.bas):
        for X in sv.bas[alpha]:
            total += 1
            want = tweight(alpha, lam)
            for u, e in ((w, want[0]), (h, want[1])):
                br = g.brk(u, X)
                tgt = {k: e * c for k, c in X.items() if e * c}
                if {k: c for k, c in br.items() if c} != tgt:
                    bad += 1
    say("  window |a| <= %d: %d weights carry a component, dim g = %d"
        % (N, len(sv.bas), total))
    ok &= check("ad(w) and ad(h) diagonal with the stated eigenvalues", bad == 0)
    return ok


# ================================================== 2) g is Z^3-graded, dim check
def part2(lam, N=3):
    hdr("2) lambda = %s: div_lambda preserves the Z^3-weight, so g is Z^3-graded"
        % lam)
    g = SVect(lam)
    keys = V.keys(range(-N, N + 1))
    bad = 0
    for k in keys:
        a = wt3(k)
        for (b, S), c in g.div({k: Q(1)}).items():
            # the weight of the FUNCTION t^b theta_S is (b, [1 in S], [2 in S])
            if (b, 1 if 1 in S else 0, 1 if 2 in S else 0) != a:
                bad += 1
    ok = check("div_lambda is weight-preserving on every monomial", bad == 0)
    # compare on a weight-saturated set of monomials: every weight taken whole
    sv = SV(lam, N)
    inside = sorted(a for a in sv.weights() if all(abs(k[1]) <= N for k in mons(a)))
    sat = [k for a in inside for k in mons(a)]
    whole = len(_kernel_basis(sat, g.div))
    parts_in = sum(len(sv.bas.get(a, [])) for a in inside)
    say("  weight-saturated window: %d monomials in %d weights"
        % (len(sat), len(inside)))
    say("  dim Ker div_lambda on it = %d;  sum of the dims of its weight"
        " components = %d" % (whole, parts_in))
    ok &= check("the kernel splits into its weight components", whole == parts_in)
    return ok


# ============================================ 3) the centralizer of t (the point)
def part3(lam, N=6):
    hdr("3) lambda = %s: the t-weight-0 component of g, i.e. the centralizer"
        " of t" % lam)
    sv = SV(lam, N)
    zero = []
    for alpha in sorted(sv.bas):
        if tweight(alpha, lam) == (Q(0), 0):
            zero.append((alpha, len(sv.bas[alpha]), sv.bas[alpha]))
    dim = sum(d for _, d, _ in zero)
    say("  weights with t-weight 0 inside the window:")
    for alpha, d, b in zero:
        say("      alpha = %-14s dim = %d   %s"
            % (str(alpha), d, [fmt(x) for x in b]))
    say("  total dimension of the centralizer inside the window = %d" % dim)
    isint = (lam.denominator == 1)
    if isint:
        want = 3
        extra = ('t', -int(lam), (1, 2))           # t^{-lambda} th_1 th_2 d_t
        has = any(alpha == wt3(extra) for alpha, _, _ in zero)
        check("lambda integer: the extra element is t^{%s} th_1 th_2 d_t"
              % (-int(lam)), has)
        # it is even, it squares to 0, and ad of it shifts the Z^3-weight by
        # a non-zero vector -- so it is NOT semisimple and t stays maximal
        # TORAL; what fails at an integer lambda is self-centralization alone.
        E = {extra: Q(1)}
        check("the extra element is even", parity(wt3(extra)) == 0)
        check("[E, E] = 0", not sv.g.brk(E, E))
        check("ad(E) shifts the weight by %s =/= 0" % str(wt3(extra)),
              wt3(extra) != (0, 0, 0))
    else:
        want = 2
    ok = check("the centralizer is of dim %d (%s)"
               % (want, "t + one even element" if isint else "t itself"),
               dim == want)
    return ok


def fmt(X):
    out = []
    for (mu, a, S), c in sorted(X.items(), key=str):
        s = "t^%s" % a if a else ""
        s += "".join("th%d" % i for i in S)
        s += "d_%s" % (mu if mu == 't' else "th%d" % mu)
        out.append("%s*%s" % (c, s) if c != 1 else s)
    return "+".join(out)


# ======================================================== 4) the root spaces
def part4(lam, N=6):
    hdr("4) lambda = %s: the dimensions of the root spaces of g" % lam)
    sv = SV(lam, N)
    inside = [a for a in sv.weights() if all(abs(k[1]) <= N for k in mons(a))]
    dims = {}
    for alpha in inside:
        if tweight(alpha, lam) == (Q(0), 0):
            continue
        d = len(sv.bas.get(alpha, []))
        dims[d] = dims.get(d, 0) + 1
    for d in sorted(dims):
        say("      dim %d : %d roots" % (d, dims[d]))
    # the shape: dim g_alpha = dim vect_alpha - 1 if the weight carries
    # a function, and = dim vect_alpha otherwise (which never happens here)
    bad = 0
    for alpha in inside:
        m, n1, n2 = alpha
        carries = n1 in (0, 1) and n2 in (0, 1)
        want = len(mons(alpha)) - (1 if carries else 0)
        if len(sv.bas.get(alpha, [])) != want:
            bad += 1
    ok = check("dim g_alpha = dim vect_alpha - [the weight carries a function]",
               bad == 0)
    return ok and check("every root space is of dim <= 2", max(dims) <= 2)


# =========================================== 5) no invariants of t-weight 0
def act_pair(sv, x, p):
    """x . (u (x) v) for basis indices p = (ia, ib); returns a dict on pairs."""
    (aa, ia), (ab, ib) = p
    u, v = sv.vec(aa, ia), sv.vec(ab, ib)
    out = {}
    cu = sv.coords(sv.g.brk(x, u))
    if cu is None:
        return None
    for k, c in cu.items():
        out[(k, (ab, ib))] = out.get((k, (ab, ib)), Q(0)) + c
    sgn = (-1) ** (parity(wt3(next(iter(x)))) * parity(aa))
    cv = sv.coords(sv.g.brk(x, v))
    if cv is None:
        return None
    for k, c in cv.items():
        out[((aa, ia), k)] = out.get(((aa, ia), k), Q(0)) + sgn * c
    return {k: c for k, c in out.items() if c}


def part5(lam, N=2):
    hdr("5) lambda = %s: (g (x) g)^g has no element of t-weight 0 (window"
        " |a| <= %d)" % (lam, N))
    sv = SV(lam, N)
    flat, idx = sv.index()
    pairs = [(a, b) for a in flat for b in flat
             if tzero(tweight(a[0], lam), tweight(b[0], lam))]
    pidx = {p: i for i, p in enumerate(pairs)}
    say("  dim g (window) = %d, dim (g (x) g) of t-weight 0 = %d"
        % (len(flat), len(pairs)))
    # the TARGET of r |--> (x.r)_x is not the weight-0 space but the whole of
    # g (x) g, one weight space per root -- taking the weight-0 space for the
    # target makes the matrix empty and the test vacuous (inv2.py says the
    # same thing for vect^L(1|2)).
    inwin = set(flat)
    rows = {}
    for alpha, i in flat:
        x = sv.vec(alpha, i)
        for p in pairs:
            r = act_pair(sv, x, p)
            if r is None:
                continue
            for k, c in r.items():
                if k[0] not in inwin or k[1] not in inwin:
                    continue
                rows.setdefault(((alpha, i), k), {})[pidx[p]] = \
                    rows.setdefault(((alpha, i), k), {}).get(pidx[p], Q(0)) + c
    mat = [[r.get(j, Q(0)) for j in range(len(pairs))] for r in rows.values()]
    rk = rank(mat) if mat else 0
    say("  rank of the action = %d, so the invariants inside the window"
        " are of dim %d" % (rk, len(pairs) - rk))
    return check("no invariant of t-weight 0 inside the window",
                 len(pairs) - rk == 0)


def tzero(a, b):
    return a[0] + b[0] == 0 and a[1] + b[1] == 0


# ============================== 6) H^1 of the relative complex over a window
def part6(lam, N=2, pad=2):
    hdr("6) lambda = %s: Z^1 and B^1 of the relative complex, window |a| <= %d"
        % (lam, N))
    # the basis is built on a WIDER window (N+pad) so that a bracket leaving
    # the small window is still expressible; the cochain unknowns, and the
    # arguments of the cochains, live in the small one.
    sv = SV(lam, N + pad)
    flat_all, idx = sv.index()
    small_w = {a for a in sv.weights() if all(abs(k[1]) <= N for k in mons(a))}
    flat = [b for b in flat_all if b[0] in small_w]
    roots = [b for b in flat if tweight(b[0], lam) != (Q(0), 0)]
    bywt = {}
    for a in flat:
        for b in flat:
            w = (tweight(a[0], lam)[0] + tweight(b[0], lam)[0],
                 tweight(a[0], lam)[1] + tweight(b[0], lam)[1])
            bywt.setdefault(w, []).append((a, b))
    insmall = set(flat)

    cols, colof = [], {}
    for x in roots:
        for p in bywt.get(tweight(x[0], lam), []):
            colof[(x, p)] = len(cols)
            cols.append((x, p))
    say("  dim g = %d, root vectors = %d, unknowns = %d"
        % (len(flat), len(roots), len(cols)))

    def phi_of(x):
        if tweight(x[0], lam) == (Q(0), 0):
            return {}
        return {p: colof[(x, p)] for p in bywt.get(tweight(x[0], lam), [])}

    eqs, nrow, skipped = {}, 0, 0
    for i, x in enumerate(flat):
        for y in flat[i:]:
            br = sv.coords(sv.g.brk(sv.vec(*x), sv.vec(*y)))
            if br is None or any(k not in insmall for k in br):
                skipped += 1                 # phi([x,y]) is not among the unknowns
                continue
            acc, bad = {}, False
            for p, col in phi_of(y).items():
                r = act_pair(sv, sv.vec(*x), p)
                if r is None:
                    bad = True
                    break
                for k, c in r.items():
                    acc[(k, col)] = acc.get((k, col), Q(0)) + c
            if bad:
                skipped += 1
                continue
            sgn = -((-1) ** (parity(x[0]) * parity(y[0])))
            for p, col in phi_of(x).items():
                r = act_pair(sv, sv.vec(*y), p)
                if r is None:
                    bad = True
                    break
                for k, c in r.items():
                    acc[(k, col)] = acc.get((k, col), Q(0)) + sgn * c
            if bad:
                skipped += 1
                continue
            for k, c in br.items():
                for p, col in phi_of(k).items():
                    acc[(p, col)] = acc.get((p, col), Q(0)) - c
            byt = {}
            for (tgt, col), c in acc.items():
                if c:
                    byt.setdefault(tgt, {})[col] = \
                        byt.setdefault(tgt, {}).get(col, Q(0)) + c
            for tgt, row in byt.items():
                if row:
                    eqs[nrow] = row
                    nrow += 1
    say("  cocycle equations kept = %d  (pairs dropped by the window: %d)"
        % (nrow, skipped))
    mat = [[eqs[r].get(j, Q(0)) for j in range(len(cols))] for r in sorted(eqs)]
    rk = rank(mat) if mat else 0
    z1 = len(cols) - rk
    say("  rank = %d, so dim Z^1 (truncated conditions) = %d" % (rk, z1))

    M0 = bywt.get((Q(0), 0), [])
    say("  dim M of t-weight 0 inside the window = %d" % len(M0))
    brows = []
    for p0 in M0:
        vec = [Q(0)] * len(cols)
        skip = False
        for x in roots:
            r = act_pair(sv, sv.vec(*x), p0)
            if r is None:
                skip = True
                break
            for k, c in r.items():
                if (x, k) not in colof:
                    skip = True
                    break
                vec[colof[(x, k)]] += c
            if skip:
                break
        if not skip:
            brows.append(vec)
    b1 = rank(brows) if brows else 0
    say("  coboundaries d r with r and d r inside the window: %d of %d,"
        " their rank = %d" % (len(brows), len(M0), b1))
    say("  dim Z^1 - dim B^1 (window) = %d" % (z1 - b1))
    say("  NB: the truncation makes Z^1 an OVER-estimate and B^1 an UNDER-")
    say("      estimate, so a difference 0 means Z^1 = B^1 over this window;")
    say("      it does not prove H^1 = 0 for g.")
    return z1, b1


# ================= 7) the integer locus is where g is not the simple algebra
def part7(lam, N=3):
    hdr("7) lambda = %s: the derived algebra, and ad of the extra element" % lam)
    sv = SV(lam, N)
    flat, idx = sv.index()
    inside = {a for a in sv.weights() if all(abs(k[1]) <= N for k in mons(a))}
    small = [b for b in flat if b[0] in inside]
    # the span of the brackets, in the coordinates of the small basis
    col = {b: j for j, b in enumerate(small)}
    rows = []
    for i, x in enumerate(small):
        for y in small[i:]:
            c = sv.coords(sv.g.brk(sv.vec(*x), sv.vec(*y)))
            if c is None or any(k not in col for k in c):
                continue
            v = [Q(0)] * len(small)
            for k, e in c.items():
                v[col[k]] += e
            rows.append(v)
    rk = rank(rows) if rows else 0
    say("  dim g (weights entirely inside |a| <= %d) = %d,"
        " dim of the span of the brackets landing there = %d"
        % (N, len(small), rk))
    isint = (lam.denominator == 1)
    ok = check("g' is of codimension %s" % ("> 0" if isint else "0 (g is perfect)"),
               (rk < len(small)) if isint else (rk == len(small)))
    if isint:
        E = {('t', -int(lam), (1, 2)): Q(1)}
        nz = any(sv.g.brk(E, sv.vec(*b)) for b in small)
        ok &= check("ad(E) =/= 0, so E is not central and ad(E) is not"
                    " semisimple", nz)
    return ok


def main():
    say("exact-linear-algebra backend: %s" % backend())
    say("g = svect^L_lambda(1|2) = Ker div_lambda inside vect^L(1|2)")

    for lam in (Q(1, 3), Q(1, 2), Q(3, 8), Q(-1, 3)):
        part1(lam, 4)
        part2(lam, 3)
        part3(lam, 6)
        part4(lam, 6)
    for lam in (Q(0), Q(1), Q(2), Q(-1)):
        hdr("the integer locus: lambda = %s" % lam)
        part1(lam, 4)
        part3(lam, 6)
        part7(lam, 3)
    for lam in (Q(1, 3), Q(1, 2)):
        part7(lam, 3)

    for lam in (Q(1, 3), Q(1, 2), Q(3, 8)):
        part5(lam, 2)
    for N in (1, 2, 3):
        part6(Q(1, 3), N)
    for N in (1, 2):
        part6(Q(1, 2), N)
        part6(Q(-1, 3), N)
    hdr("the integer locus again: the same window computation at lambda = 1,\n"
        "   where t is NOT self-centralizing, so the relative complex of\n"
        "   Proposition 5.22.5 3) is not the right one")
    for N in (1, 2):
        part6(Q(1), N)

    hdr("SUMMARY")
    if FAILED:
        say("FAILED: %s" % FAILED)
    else:
        say("every structural check passed")


if __name__ == "__main__":
    main()
