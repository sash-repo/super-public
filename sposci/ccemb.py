#!/usr/bin/env python3
r"""
ccemb.py -- The question:

  "when you will have classified all, or some portion of, embeddings up to an
   equivalence, it would be a good idea to write what are the values of the
   central charge on differently embedded subalgebras.  Or this is not a 'good'
   idea, but a silly one?"

It is a good idea, and here it is carried out.  For every family of graded
embeddings i: witt --> g classified in the note on Conjecture (233)
(engine wittemb.py) we restrict the cocycle of g to i(witt) and read off the
central charge.

Normalisation.  witt = span(L_m), [L_m, L_n] = (n-m) L_{m+n} (the convention of
eq. equation 148), and the Gelfand-Fuchs cocycle is normalised there as
c(L_m, L_n) = ((m^3-m)/12) delta_{m,-n}.  Hence if the restriction is
    c(i(L_m), i(L_n)) = (A m^3 + B m) delta_{m,-n},
then, m delta_{m,-n} being a coboundary, the class of the restriction is
A (m^3 - m) and the central charge is
    c = 12 A.
The overall scale of the cocycle on g is a free normalisation (the parameters
alpha, beta, gamma of eq. (73)); what is invariant is the ratio of c
to its value on the reference embedding, so both are printed.

Everything is exact rational arithmetic.  The k^L side uses the xi-eta engine
kL4.py (labels xi_1 = 1, xi_2 = 2, eta_1 = 3, eta_2 = 4) and its cocycle
(alpha, beta, gamma) of eq. equation 273; the k^M side uses kM4.py
(zeta = 0, xi = 1, eta = 2, theta = 3) and eq. equation 275.  The subalgebras
k^L(1|n) with n < 4 sit inside k^L(1|4) as the functions in a subset of the odd
coordinates; for odd n one of the coordinates is the self-paired
theta' := xi_2 + eta_2 with {theta', theta'} = 2 rather than 1, i.e. theta'
= sqrt2 . theta -- a rescaling of the coordinate, which is absorbed by the free
parameters of the embedding (we say so where it matters).

Run:  python3 ccemb.py
"""

from fractions import Fraction as Q
from itertools import product
import sys

from stringy import mono, add, smul
import kL4
import kM4
import witt


# --------------------------------------------------------------- helpers

def fadd(*fs):
    return add(*fs)


def fscal(c, f):
    return smul(Q(c), f)


def fit_poly(vals, deg=3):
    """fit a polynomial of degree <= deg through {m: value} and verify it on
    every point given beyond the first deg+1; returns the coefficient list
    [c_0, ..., c_deg] or None."""
    ms = sorted(vals)
    if len(ms) < deg + 1:
        return None
    # Vandermonde solve on the first deg+1 points, exact Gaussian elimination
    rows = [[Q(m) ** j for j in range(deg + 1)] + [Q(vals[m])]
            for m in ms[:deg + 1]]
    n = deg + 1
    for i in range(n):
        p = next((r for r in range(i, n) if rows[r][i] != 0), None)
        if p is None:
            return None
        rows[i], rows[p] = rows[p], rows[i]
        pv = rows[i][i]
        rows[i] = [x / pv for x in rows[i]]
        for r in range(n):
            if r != i and rows[r][i]:
                f = rows[r][i]
                rows[r] = [a - f * b for a, b in zip(rows[r], rows[i])]
    co = [rows[i][n] for i in range(n)]
    for m in ms:
        if sum(co[j] * Q(m) ** j for j in range(n)) != vals[m]:
            return None
    return co


def fit_cubic(vals):
    """vals = {m: value} for m >= 2; fit A m^3 + B m and verify on the rest.

    Returns (A, B) or None if no such fit exists."""
    ms = sorted(vals)
    if len(ms) < 2:
        return None
    m1, m2 = ms[0], ms[1]
    # A m^3 + B m = v
    det = m1 ** 3 * m2 - m2 ** 3 * m1
    if det == 0:
        return None
    A = Q(vals[m1] * m2 - vals[m2] * m1, det)
    B = Q(vals[m1] * m2 ** 3 - vals[m2] * m1 ** 3, m1 * m2 ** 3 - m2 * m1 ** 3)
    for m in ms:
        if A * m ** 3 + B * m != vals[m]:
            return None
    return A, B


# ------------------------------------------------------ the L (Neveu-Schwarz) side
#
# A graded embedding is L_m |--> K_{G_m} with
#
#   G_m = (1/2) t^{m+1} + sum_M gamma_M(m) t^{m+1-deg(M)/2} M,
#
# M running over the even monomials in the odd coordinates and gamma_M a
# polynomial in m (see wittemb.py).  We write G_m as a linear combination
# sum_k p_k g^k_m of "pieces", each piece carrying one parameter p_k, so that
# the cocycle value is the quadratic form sum_{k,l} p_k p_l c(g^k_m, g^l_n).

def piece_vir(m):
    """the reference piece (1/2) t^{m+1}; parameter p_0 = 1."""
    return mono(m + 1, (), Q(1, 2))


def piece_mono(M, s):
    """gamma_M(m) = m^s: the piece m^s t^{m+1-deg/2} M."""
    d = len(M)

    def g(m):
        return mono(m + 1 - d // 2, M, Q(m) ** s)
    return g


def cocycle_form_L(pieces, mrange=(2, 3, 4), which=('alpha', 'beta', 'gamma')):
    """for each pair of pieces and each cocycle component, the (A, B) of the
    restriction; returns {(k,l): {comp: (A,B)}}."""
    comp = {'alpha': 0, 'beta': 1, 'gamma': 2}
    out = {}
    for k, gk in enumerate(pieces):
        for l, gl in enumerate(pieces):
            per = {}
            for name in which:
                j = comp[name]
                vals = {}
                ok = True
                for m in mrange:
                    v = kL4.c_fun(gk(m), gl(-m))[j]
                    vals[m] = v
                f = fit_cubic(vals)
                per[name] = f
            out[(k, l)] = per
    return out


def check_hom_L(Gfun, window=4, kappa=Q(1)):
    """{G_m, G_n} = (n-m) G_{m+n} for |m|,|n| <= window."""
    bad = 0
    for m in range(-window, window + 1):
        for n in range(-window, window + 1):
            lhs = kL4.bracket(Gfun(m), Gfun(n), kappa)
            rhs = fscal(n - m, Gfun(m + n))
            if fadd(lhs, fscal(-1, rhs)):
                bad += 1
    return bad


def restriction_L(Gfun, mrange=(2, 3, 4, 5)):
    """the (A,B) of c(G_m, G_{-m}) for each of alpha, beta, gamma."""
    out = {}
    for j, name in ((0, 'alpha'), (1, 'beta'), (2, 'gamma')):
        vals = {m: kL4.c_fun(Gfun(m), Gfun(-m))[j] for m in mrange}
        out[name] = (fit_cubic(vals), vals)
    return out


# ------------------------------------------------------------- the M (Ramond) side

def check_hom_M(Gfun, window=4, **kw):
    bad = 0
    for m in range(-window, window + 1):
        for n in range(-window, window + 1):
            lhs = kM4.bracket(Gfun(m), Gfun(n), **kw)
            rhs = fscal(n - m, Gfun(m + n))
            if fadd(lhs, fscal(-1, rhs)):
                bad += 1
    return bad


def restriction_M(Gfun, mrange=(2, 3, 4, 5)):
    vals = {m: kM4.c_fun(Gfun(m), Gfun(-m)) for m in mrange}
    return fit_cubic(vals), vals


# ------------------------------------------------------------ quadratic forms
#
# Writing G_m = sum_k p_k g^k_m (p_0 = 1 for the Virasoro piece), the value
# c(G_m, G_{-m}) is the quadratic form sum_{k,l} p_k p_l c(g^k_m, g^l_{-m}).
# We compute each c(g^k_m, g^l_{-m}) as a polynomial in m and keep the
# coefficient of m^3: that is the number A whose 12-fold is the central charge.

def m3_matrix(pieces, cfun, ncomp, mrange=range(2, 8), deg=4):
    """M[j][k][l] = coefficient of m^3 in c(g^k_m, g^l_{-m}), component j."""
    N = len(pieces)
    M = [[[Q(0)] * N for _ in range(N)] for _ in range(ncomp)]
    for k in range(N):
        for l in range(N):
            vals_all = {m: cfun(pieces[k][1](m), pieces[l][1](-m))
                        for m in mrange}
            for j in range(ncomp):
                vals = {m: (v[j] if ncomp > 1 else v)
                        for m, v in vals_all.items()}
                co = fit_poly(vals, deg)
                if co is None:
                    raise AssertionError('no polynomial fit for pair (%d,%d)' % (k, l))
                M[j][k][l] = co[3]
    return M


def form_str(pieces, Mj):
    """the symmetrised quadratic form as a string."""
    N = len(pieces)
    terms = []
    for k in range(N):
        for l in range(k, N):
            c = Mj[k][l] + (Mj[l][k] if l != k else 0)
            if c == 0:
                continue
            nk, nl = pieces[k][0], pieces[l][0]
            v = ('' if nk == '1' else nk) + ('' if nl == '1' else nl)
            v = v if v else '1'
            terms.append('%+s*%s' % (c, v))
    return ' '.join(terms) if terms else '0'


def evaluate(Mj, p):
    return sum(p[k] * p[l] * Mj[k][l] for k in range(len(p)) for l in range(len(p)))


def report_family(title, pieces, cfun, ncomp, compnames, homcheck, samples):
    print('=' * 74)
    print(title)
    print('  pieces: ' + ',  '.join('%s -> %s' % (n, 'the reference (1/2)t^{m+1}'
                                                  if n == '1' else 'parameter')
                                    for n, _ in pieces))
    M = m3_matrix(pieces, cfun, ncomp)
    for j in range(ncomp):
        print('  A(%s) = %s        [central charge c = 12 A]'
              % (compnames[j], form_str(pieces, M[j])))
    # cross-checks at sample parameter values
    for p in samples:
        G = lambda m, p=p: fadd(*[fscal(p[k], pieces[k][1](m))
                                  for k in range(len(pieces)) if p[k]])
        bad = homcheck(G)
        got = []
        for j in range(ncomp):
            vals = {m: (cfun(G(m), G(-m))[j] if ncomp > 1 else cfun(G(m), G(-m)))
                    for m in range(2, 7)}
            co = fit_poly(vals, 4)
            got.append(co[3] if co else None)
        pred = [evaluate(M[j], p) for j in range(ncomp)]
        ok = 'OK' if got == pred else 'MISMATCH %s vs %s' % (got, pred)
        if bad:
            ok += ' -- NOT a homomorphism, so not an embedding: A is meaningless'
        print('    p = %-28s hom failures %d;  A = %s;  c = %s   [%s]'
              % (tuple(str(x) for x in p), bad,
                 tuple(str(x) for x in pred),
                 tuple(str(12 * x) for x in pred), ok))
    return M


def main():
    XI1, XI2, ETA1, ETA2 = 1, 2, 3, 4
    Z, TH = 0, 3          # kM4 labels: zeta = 0, theta = 3

    print(__doc__.split('Run:')[0])

    # ------------------------------------------------------------- k^L(1|1)
    print('=' * 74)
    print("k^L(1|1): there is no even monomial in the one odd coordinate, so the")
    print("only graded embedding is the standard one, L_m |--> K_{t^{m+1}/2},")
    print("and the central charge of the restriction is the standard 3*alpha.")

    # ------------------------------------------------------------- k^L(1|2)
    pieces2 = [('1', piece_vir),
               ('a', piece_mono((XI1, ETA1), 0)),
               ('b', piece_mono((XI1, ETA1), 1))]
    report_family(
        "k^L(1|2) = functions in t, xi_1, eta_1.  Complete family of graded\n"
        "embeddings (wittemb.py):  G_m = (1/2)t^{m+1} + (a + b m) t^m xi_1 eta_1",
        pieces2, kL4.c_fun, 3, ('alpha', 'beta', 'gamma'), check_hom_L,
        [(1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 0, Q(1, 2)), (1, 2, 3),
         (1, Q(1, 2), Q(-1, 3))])

    # ------------------------------------------------------------- k^L(1|4)
    # Names: for each quadratic monomial M, the f'-piece (coefficient m) and the
    # f/t-piece (coefficient 1); for the top monomial, m^2 and m.  The piece
    # m^0 t^{m-1} xi_1 xi_2 eta_1 eta_2 is NOT included: wittemb.py shows it is
    # not a homomorphism (a constant coefficient of the top monomial is
    # excluded), and indeed it would give c(S_m, S_{-m}) = alpha/m, which no
    # 2-cocycle of witt can have.  This is checked below.
    quad = [((XI1, ETA1), 'b1', 'a1'), ((XI2, ETA2), 'b2', 'a2'),
            ((XI1, ETA2), 'c12', 'g12'), ((XI2, ETA1), 'c21', 'g21'),
            ((XI1, XI2), 'u', 'u0'), ((ETA1, ETA2), 'v', 'v0')]
    pieces4 = [('1', piece_vir)]
    for M, nb, na in quad:
        pieces4.append((nb, piece_mono(M, 1)))
        pieces4.append((na, piece_mono(M, 0)))
    TOP = (XI1, XI2, ETA1, ETA2)
    pieces4 += [('w2', piece_mono(TOP, 2)), ('w1', piece_mono(TOP, 1))]
    N4 = len(pieces4)
    s0 = [Q(0)] * N4
    s0[0] = Q(1)
    names4 = [n for n, _ in pieces4]

    def samp(**kw):
        p = list(s0)
        for k, v in kw.items():
            p[names4.index(k)] = Q(v)
        return tuple(p)

    report_family(
        "k^L(1|4) = functions in t, xi_1, xi_2, eta_1, eta_2.  The family of\n"
        "eq. (233), completed by the f/t-terms and by the two monomials\n"
        "xi_1 xi_2, eta_1 eta_2 found in the note:\n"
        "  G_m = (1/2)t^{m+1} + sum_M (a_M + b_M m) t^m M\n"
        "        + (w1 m + w2 m^2) t^{m-1} xi_1 xi_2 eta_1 eta_2",
        pieces4, kL4.c_fun, 3, ('alpha', 'beta', 'gamma'), check_hom_L,
        [tuple(s0), samp(b1=1), samp(b1=1, b2=1), samp(c12=1, c21=1),
         samp(u=1), samp(u=1, v=1), samp(w2=1), samp(w1=1),
         samp(b1=1, w2=1), samp(b1=2, b2=3, c12=5, c21=7, w2=11)])

    # the excluded piece: a constant coefficient of the top monomial
    Gbad = lambda m: fadd(piece_vir(m), piece_mono(TOP, 0)(m))
    print('  control: the excluded piece m^0 t^{m-1} xi_1 xi_2 eta_1 eta_2 gives')
    print('    %d homomorphism failures on |m|,|n| <= 4, and c(G_m,G_{-m}) = %s'
          % (check_hom_L(Gbad),
             {m: str(kL4.c_fun(Gbad(m), Gbad(-m))[0]) for m in (2, 3, 4)}))

    # ------------------------------------------------------------- k^M(1|2)
    def piece_virM(m):
        return mono(m + 1, (), Q(1, 2))

    def piece_monoM(M, s, shift=0):
        def g(m):
            return mono(m + 1 + shift, M, Q(m) ** s)
        return g

    piecesM2 = [('1', piece_virM),
                ('a', piece_monoM((Z, TH), 0)),
                ('b', piece_monoM((Z, TH), 1))]
    report_family(
        "k^M(1|2) = functions in t, theta, zeta (zeta the twisted coordinate).\n"
        "Complete family of graded embeddings (wittemb.py):\n"
        "  G_m = (1/2)t^{m+1} + (a + b m) t^{m+1} theta zeta",
        piecesM2, kM4.c_fun, 1, ('the cocycle equation 275',), check_hom_M,
        [(1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 2, 0), (1, 0, Q(1, 2)),
         (1, 2, 3)])

    # --------------------------------------------- vect^L(1|2), svect^L_lambda
    print('=' * 74)
    print("vect^L(1|2): the family D^a_m = t^{m+1}d_t + a(m+1)t^m(theta_1 d_1 +")
    print("theta_2 d_2) of the note; cocycle equation 76; engine witt.py.")
    for a in (0, 1, 2, Q(1, 2), Q(3, 7)):
        ok, vals = witt.check_witt_in_vect(a, 5)
        v = {m: vals[m] for m in (1, 2, 3, 4, 5)}
        co = fit_poly(v, 3)
        A = co[3] if co else None
        print('  a = %-6s witt relations hold: %-5s A = %-8s c = 12A = %-9s'
              ' (closed form -24a^2 = %s)'
              % (a, ok, A, 12 * A, -24 * Q(a) ** 2))

    print('=' * 74)
    print("svect^L_lambda(1|2): the family D_m = LL_m + (alpha + beta m) T_m of")
    print("the note; cocycle equation 78; engine witt.py.")
    for lam in (0, Q(1, 3), Q(1, 2)):
        for (al, be) in ((0, 0), (1, 0), (0, 1), (0, Q(1, 2)), (2, Q(1, 3))):
            ok, div_ok, vals = witt.check_DD(lam, al, be, 4)
            co = fit_poly({m: vals[m] for m in (1, 2, 3, 4)}, 3)
            A = co[3]
            print('  lambda = %-5s (alpha,beta) = (%-3s,%-4s) witt? %-5s in svect? %-5s'
                  '  A = %-8s c = 12A = %-8s (closed form 6-24beta^2 = %s)'
                  % (lam, al, be, ok, div_ok, A, 12 * A,
                     6 - 24 * Q(be) ** 2))


if __name__ == '__main__':
    main()
