r"""The two even parameters of the twisted Dolbeault algebra (Theorem 1.8.2):
osp(2|2), and no C Delta beside it.

The setting (even omega, even J, even parameters hbar_a in C^*):

    omega  = sum hat q_i hat p_i,          X^+ = mult by omega, X^- = its dual
    d      = sum hat z_a partial_{z_a},    d_c = J d J^{-1}, eq. (d_c)
    alpha  = (1/2) sum (q_i hat p_i - p_i hat q_i),  [d, alpha]     = omega
    alpha_c= -(1/2) sum z_a hat z_a,                  [d_c, alpha_c] = omega

    D^+   := d   + (hbar_1/2) alpha,      D^-   := [X^-, D^+]
    D_c^+ := d_c + (hbar_2/2) alpha_c,    D_c^- := [X^-, D_c^+]

Expected (C^2, C^4, C^6; hbar_a = 1, 2, 3, -5, 1/2, 7/3):

  hbar = 0, {d, delta, d_c, d_c^*, X^+, X^-}   sdim (4|4), centre (1|0)
        = hei(0|4) + sl(2) of eq. (ADolb), the centre being Delta
  hbar_1 only, {D^+, D^-, X^+, X^-}            sdim (3|2) = osp(1|2), eq. (Dpm)
  hbar_1, hbar_2, all six                      sdim (4|4), centre (0|0),
        perfect, g_ev = sl(2) + C, g_od four-dimensional and moved by g_ev
        ==> osp(2|2), and not osp(2|2) + C Delta; [Delta, alpha] != 0.

Run (from this folder):  ../venv/bin/python twohbareven.py
"""
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
import sys
from fractions import Fraction


from exactlin import rank                                       # noqa: E402
from flatmodel import Model, Echelon, closure, d_operator       # noqa: E402

half = Fraction(1, 2)


# ---------------------------------------------------------------------------
# the flat almost Kaehler model C^{2m|2k}
# ---------------------------------------------------------------------------
def kaehler(m, k=0):
    """q_1..q_m, p_1..p_m even; then xi_1..xi_k, eta_1..eta_k odd."""
    return Model([0] * (2 * m) + [1] * (2 * k))


def omega_pairs(m, k):
    """omega = sum \\hat q_i \\hat p_i + sum \\hat xi_j \\hat eta_j, as index pairs."""
    q = list(range(m))
    p = list(range(m, 2 * m))
    xi = list(range(2 * m, 2 * m + k))
    eta = list(range(2 * m + k, 2 * m + 2 * k))
    return list(zip(q, p)) + list(zip(xi, eta))


def Xplus(M, m, k):
    return M.add(*[M.mul(M.gen('c', a), M.gen('c', b))
                   for a, b in omega_pairs(m, k)])


def Xminus(M, m, k):
    return M.add(*[M.mul(M.gen('b', b), M.gen('b', a))
                   for a, b in omega_pairs(m, k)])


def J_standard(m, k):
    """J: \\hat q_i -> \\hat p_i -> -\\hat q_i, likewise on the odd pairs; J^2 = -id."""
    J = {}
    for a, b in omega_pairs(m, k):
        J[a] = [(Fraction(1), b)]
        J[b] = [(Fraction(-1), a)]
    return J


def d_c(M, J):
    """d_c = sum_a (J \\hat z)_a \\partial_{z_a}, the flat reading of J d J^{-1}."""
    terms = []
    for a, image in J.items():
        for coeff, b in image:
            terms.append(M.smul(coeff, M.mul(M.gen('c', b), M.gen('t', a))))
    return M.add(*terms)


def alpha(M, m, k):
    """The d-primitive of omega: [d, alpha] = omega.

    On an EVEN pair, \\hat z is odd and the primitive is the antisymmetric
    combination (1/2)(z_a \\hat z_b - z_b \\hat z_a), as classically.  On an ODD
    pair, \\hat z is EVEN, so \\hat z_a \\hat z_b is symmetric and the antisymmetric
    combination is d-closed; there the primitive is the symmetric one.
    """
    terms = []
    for a, b in omega_pairs(m, k):
        sign = -1 if M.p[a] == 0 else 1
        terms.append(M.smul(half, M.mul(M.gen('x', a), M.gen('c', b))))
        terms.append(M.smul(half * sign, M.mul(M.gen('x', b), M.gen('c', a))))
    return M.add(*terms)


def alpha_c(M, m, k):
    """The d_c-primitive of omega: [d_c, alpha_c] = omega (the Kaehler potential).

    On the EVEN pairs, -(1/2) sum z_a \\hat z_a, the classical formula; on the ODD
    pairs that diagonal expression is d_c-closed and the primitive is
    (1/2)(z_a \\hat z_a - z_b \\hat z_b) -- the roles of the symmetric and the
    antisymmetric combination are exchanged with those in `alpha`.
    """
    terms = []
    for a, b in omega_pairs(m, k):
        s = -half if M.p[a] == 0 else half
        terms.append(M.smul(s, M.mul(M.gen('x', a), M.gen('c', a))))
        terms.append(M.smul(-half, M.mul(M.gen('x', b), M.gen('c', b))))
    return M.add(*terms)


# ---------------------------------------------------------------------------
# linear algebra helpers
# ---------------------------------------------------------------------------
def span(M, elts):
    ech = {0: Echelon(), 1: Echelon()}
    for e in elts:
        if e:
            ech[M.parity(e)].add(e)
    return ech


def sdim_of(M, elts):
    ech = span(M, elts)
    return (len(ech[0].rows), len(ech[1].rows))


def contains(M, ech, e):
    if not e:
        return True
    return not ech[M.parity(e)].reduce(e)[0]


def _centre_dim(M, vecs, against):
    """dim of {u in span(vecs) | [u, v] = 0 for all v in `against`}."""
    if not vecs:
        return 0
    mat = [[] for _ in vecs]
    for v in against:
        brs = [M.br(u, v) for u in vecs]
        mons = sorted({mn for e2 in brs for mn in e2}, key=lambda mn: (len(mn), mn))
        for i in range(len(vecs)):
            mat[i] += [brs[i].get(mn, Fraction(0)) for mn in mons]
    return len(vecs) - (rank(mat) if any(any(r) for r in mat) else 0)


def centre(M, basis):
    """sdim of the centre of the span of `basis`, exactly (as a kernel of ad)."""
    ech = span(M, basis)
    elts = ech[0].elements + ech[1].elements
    return tuple(_centre_dim(M, [e for e in elts if M.parity(e) == par], elts)
                 for par in (0, 1))


def derived(M, basis):
    brs = [M.br(u, v) for u in basis for v in basis]
    return sdim_of(M, [b for b in brs if b])


# ---------------------------------------------------------------------------
def run(m, k, h1, h2, gens_wanted, tag):
    """gens_wanted: 'all' = the six operators, 'd' = Bernstein's four."""
    M = kaehler(m, k)
    Xp, Xm = Xplus(M, m, k), Xminus(M, m, k)
    d = d_operator(M)
    dc = d_c(M, J_standard(m, k))
    al, alc = alpha(M, m, k), alpha_c(M, m, k)

    assert M.br(d, al) == Xp, '[d, alpha] != omega'
    assert M.br(dc, alc) == Xp, '[d_c, alpha_c] != omega'

    Dp = M.add(d, M.smul(h1 * half, al))
    Dcp = M.add(dc, M.smul(h2 * half, alc))
    Dm, Dcm = M.br(Xm, Dp), M.br(Xm, Dcp)

    gens = [Dp, Dm, Xp, Xm] if gens_wanted == 'd' else [Dp, Dm, Dcp, Dcm, Xp, Xm]
    g, sd = closure(M, gens, limit=600)
    ech = span(M, g)
    ev = ech[0].elements
    od = ech[1].elements

    # the flat-model Laplacian: [d, delta] vanishes here (cf. eq. (Svanishes)),
    # the surviving second-order operator is [d_c, delta] = sum \partial_{z_a}^2
    delta = M.br(Xm, d)
    Delta = M.br(dc, delta)

    ev_der = [b for b in (M.br(u, v) for u in ev for v in ev) if b]
    moved = [b for b in (M.br(u, v) for u in ev for v in od) if b]

    print('  %-26s sdim %-8s centre %-8s [g,g] %-8s  g_ev %d ([g_ev,g_ev] %d, '
          'centre %d)  g_od %d (moved: %d)  Delta: in g %-5s central %-5s'
          % (tag, '(%d|%d)' % sd, '(%d|%d)' % centre(M, g),
             '(%d|%d)' % derived(M, g), len(ev), sdim_of(M, ev_der)[0],
             _centre_dim(M, ev, ev), len(od), sdim_of(M, moved)[1],
             contains(M, ech, Delta),
             all(not M.br(Delta, v) for v in g)))
    return sd


if __name__ == '__main__':
    print(__doc__.split('Run:')[0].strip())
    print()

    print('=' * 118)
    print('VALIDATION 1 -- hbar = 0, all six: must be hei(0|4) + sl(2), '
          'sdim (4|4), centre (1|0) = C Delta, see eq. (ADolb)')
    print('=' * 118)
    v1 = [run(m, 0, Fraction(0), Fraction(0), 'all', 'C^%d, untwisted' % (2 * m))
          for m in (1, 2, 3)]

    print('=' * 118)
    print("VALIDATION 2 -- one parameter, {D^+, D^-, X^+, X^-}: must be osp(1|2), "
          "sdim (3|2), Bernstein's line, eq. (Dpm)")
    print('=' * 118)
    v2 = [run(m, 0, Fraction(1), Fraction(0), 'd', 'C^%d, one hbar' % (2 * m))
          for m in (1, 2, 3)]

    print('=' * 118)
    print('THE CASE IN QUESTION -- two parameters, all six operators')
    print('=' * 118)
    two = []
    for m in (1, 2, 3):
        for h1, h2 in ((Fraction(1), Fraction(1)), (Fraction(3), Fraction(-5)),
                       (Fraction(1, 2), Fraction(7, 3))):
            two.append(run(m, 0, h1, h2, 'all',
                           'C^%d, hbar=(%s,%s)' % (2 * m, h1, h2)))

    print()
    print('validation 1 (want (4,4) with centre (1|0)) :', set(v1))
    print('validation 2 (want (3,2) = osp(1|2))        :', set(v2))
    print('two parameters, m = 1,2,3, six hbar-pairs   :', set(two),
          ' -- one answer :', len(set(two)) == 1)
    print()
    print('for reference: sdim osp(2|2) = sdim sl(2|1) = (4|4), simple, '
          'g_ev = sl(2) + C;  sdim (osp(2|2) + C Delta) = (5|4);')
    print('               sdim (hei(0|4) + sl(2)) = (4|4) as well, but with a '
          '1-dimensional centre, see eq. (ADolb).')

    print()
    print('=' * 118)
    print('APPENDIX, NOT USED IN THE TEXT -- the same six operators on a genuine '
          'supermanifold C^{2m|2k}.')
    print('The answer changes ((7|8) untwisted, (8|8) twisted), and the two '
          'primitives cease to be adapted to')
    print('one differential each ([d, alpha_c] and [d_c, alpha] no longer '
          'vanish).  Which J and which pair of')
    print('connections superize the classical picture is a separate question, '
          'not settled here.')
    print('=' * 118)
    for m, k in ((1, 1), (2, 1), (2, 2)):
        run(m, k, Fraction(0), Fraction(0), 'all',
            'C^{%d|%d}, untwisted' % (2 * m, 2 * k))
        run(m, k, Fraction(1), Fraction(1), 'all',
            'C^{%d|%d}, two hbar' % (2 * m, 2 * k))
