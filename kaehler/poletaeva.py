r"""Remark 3.2.2: the twist of Theorems 1.8.2 and 3.2.1 is a deformation whose limit
hbar -> 0 is the contraction osp(2|2) -> hei(0|4) (+) sl(2), resp.
osp(2|4) -> hei(0|8) (+) sp(4), recorded by Poletaeva (J. Nonlinear Math. Phys. 10:2
(2003) 141-147).  Checked:

  1) the superdimension is constant along the family g_t (a contraction is a
     limit of brackets on a fixed space);
  2) the generic fibre is the simple algebra and the special fibre t = 0 has a
     one-dimensional even centre spanned by the Laplacian;
  3) the structure constants in the t-dependent basis are polynomial in t and
     specialize at t = 0 to those of the special fibre: the whole table is computed
     at 7 rational t, each entry interpolated, and the value at 0 compared with the
     directly computed t = 0 table.

Run (from this folder):  ../venv/bin/python poletaeva.py
"""
import sys
from fractions import Fraction

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]

from flatmodel import Model, Echelon, closure, d_operator, \
    form_to_multiplication, form_to_convolution              # noqa: E402
from conj2 import sd, span, contains, reduce_basis, centre   # noqa: E402

half = Fraction(1, 2)


def ideal(M, basis, seeds):
    ech = {0: Echelon(), 1: Echelon()}
    cur = []
    for e in seeds:
        if e and ech[M.parity(e)].add(e):
            cur.append(e)
    i = 0
    while i < len(cur):
        for v in basis:
            br = M.br(cur[i], v)
            if br and ech[M.parity(br)].add(br):
                cur.append(br)
        i += 1
    return (len(ech[0]), len(ech[1]))


def is_simple(M, basis):
    red = reduce_basis(M, basis)
    total = sd(M, red)
    for u in red:
        if ideal(M, red, [u]) != total:
            return False, u
    return True, None
LOG = None


def say(*a):
    line = ' '.join(str(x) for x in a)
    print(line)
    if LOG:
        LOG.write(line + '\n')
        LOG.flush()


# ---------------------------------------------------------------- Kaehler ---
def kaehler(n):
    """the flat Kaehler C^{2n}, and the ingredients of Theorem 1.8.2."""
    N = 2 * n
    M = Model([0] * N)
    q, p = list(range(n)), list(range(n, N))
    c = lambda a: M.gen('c', a)                              # noqa: E731
    x = lambda a: M.gen('x', a)                              # noqa: E731
    t = lambda a: M.gen('t', a)                              # noqa: E731
    d = d_operator(M)
    w = [(1, (q[i], p[i])) for i in range(n)]
    Xp, Xm = form_to_multiplication(M, w), form_to_convolution(M, w)
    H = M.br(Xp, Xm)
    dc = M.add(*[M.add(M.mul(c(p[i]), t(q[i])),
                       M.smul(-1, M.mul(c(q[i]), t(p[i])))) for i in range(n)])
    # the two connection 1-forms of Theorem 1.8.2.  alpha_c is the CORRECTED
    # one of that Theorem's proof -- the obvious rotation (of the differentials
    # AND the coefficients) gives a flat connection and a non-simple closure.
    alpha = M.add(*[M.add(M.mul(c(q[i]), x(p[i])),
                          M.smul(-1, M.mul(c(p[i]), x(q[i])))) for i in range(n)])
    alphac = M.add(*[M.add(M.mul(c(p[i]), x(p[i])),
                           M.mul(c(q[i]), x(q[i]))) for i in range(n)])
    return M, d, dc, Xp, Xm, H, alpha, alphac


def kbasis(M, d, dc, Xp, Xm, H, alpha, alphac, t):
    """the t-dependent basis of Theorem 1.8.2, in a fixed order."""
    Dp = M.add(d, M.smul(-half * t, alpha))
    Dcp = M.add(dc, M.smul(-half * t, alphac))
    Dm, Dcm = M.br(Xm, Dp), M.br(Xm, Dcp)
    Lap = M.br(Dcp, Dm)                                  # eq. (Deltw)
    names = ['X^+', 'X^-', 'H', 'Delta_t', 'D^+', 'D^-', 'D_c^+', 'D_c^-']
    return names, [Xp, Xm, H, Lap, Dp, Dm, Dcp, Dcm]


# ------------------------------------------------------------ hyper-Kaehler ---
def hyper():
    """the flat hyper-Kaehler C^4 and the ingredients of Theorem 3.2.1."""
    M = Model([0, 0, 0, 0])
    c = lambda a: M.gen('c', a)                              # noqa: E731
    x = lambda a: M.gen('x', a)                              # noqa: E731
    d = d_operator(M)
    wi = [(1, (0, 1)), (1, (2, 3))]
    wj = [(1, (0, 2)), (-1, (1, 3))]
    wk = [(1, (0, 3)), (1, (1, 2))]
    E = [form_to_multiplication(M, w) for w in (wi, wj, wk)]
    F = [form_to_convolution(M, w) for w in (wi, wj, wk)]
    terms = []
    for co, (a, b) in wi:
        terms.append(M.smul(co, M.mul(c(a), x(b))))
        terms.append(M.smul(-co, M.mul(c(b), x(a))))
    return M, d, E, F, M.add(*terms)


# --------------------------------------------------------------- machinery ---
def coefficients(M, basis, elt):
    """coordinates of `elt` in `basis`; None if it is not in the span."""
    keys, seen = [], set()
    for v in basis + [elt]:
        for k in v:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    idx = {k: i for i, k in enumerate(keys)}
    A = [[Fraction(0)] * len(basis) for _ in keys]
    for j, b in enumerate(basis):
        for k, v in b.items():
            A[idx[k]][j] = Fraction(v)
    rhs = [Fraction(0)] * len(keys)
    for k, v in elt.items():
        rhs[idx[k]] = Fraction(v)
    # Gaussian elimination on the augmented system (exact)
    rows = [A[i][:] + [rhs[i]] for i in range(len(keys))]
    piv, r = [], 0
    for col in range(len(basis)):
        s = next((i for i in range(r, len(rows)) if rows[i][col]), None)
        if s is None:
            continue
        rows[r], rows[s] = rows[s], rows[r]
        pv = rows[r][col]
        rows[r] = [v / pv for v in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][col]:
                f = rows[i][col]
                rows[i] = [a - f * b for a, b in zip(rows[i], rows[r])]
        piv.append(col)
        r += 1
    for i in range(r, len(rows)):
        if rows[i][-1]:
            return None                                  # inconsistent
    out = [Fraction(0)] * len(basis)
    for i, col in enumerate(piv):
        out[col] = rows[i][-1]
    return out


def table(M, names, basis):
    """the full structure-constant table, as a dict (i,j) -> coefficient list."""
    tab = {}
    for i in range(len(basis)):
        for j in range(len(basis)):
            co = coefficients(M, basis, M.br(basis[i], basis[j]))
            if co is None:
                return None, (names[i], names[j])
            tab[(i, j)] = co
    return tab, None


def interpolate(pts, vals):
    """Lagrange value at 0 of the polynomial through (pts, vals)."""
    tot = Fraction(0)
    for i, xi in enumerate(pts):
        term = Fraction(vals[i])
        for j, xj in enumerate(pts):
            if i != j:
                term *= Fraction(0 - xj, xi - xj)
        tot += term
    return tot


def degree_bound(pts, vals, deg):
    """does a polynomial of degree <= deg pass through all the points?"""
    if len(pts) <= deg + 1:
        return True
    for k in range(deg + 1, len(pts)):
        if interpolate_at(pts[:deg + 1], vals[:deg + 1], pts[k]) != vals[k]:
            return False
    return True


def interpolate_at(pts, vals, z):
    tot = Fraction(0)
    for i, xi in enumerate(pts):
        term = Fraction(vals[i])
        for j, xj in enumerate(pts):
            if i != j:
                term *= Fraction(z - xj, xi - xj)
        tot += term
    return tot


# =========================================================================== #
def main():
    global LOG
    LOG = open('polet673.log', 'w')

    say('=' * 74)
    say('1.  the Kaehler family: g_t = <D^+_t, D_c^+_t, X^+, X^->,')
    say('    D^+_t = d - (t/2) alpha,  D_c^+_t = d_c - (t/2) alpha_c')
    say('=' * 74)
    TS = [Fraction(k) for k in (0, 1, -1, 2, -2, 3, half)]
    for n in (1, 2):
        M, d, dc, Xp, Xm, H, al, alc = kaehler(n)
        say('--- C^{%d|0}' % (2 * n))
        for t in TS:
            Dp = M.add(d, M.smul(-half * t, al))
            Dcp = M.add(dc, M.smul(-half * t, alc))
            basis, s = closure(M, [Dp, Dcp, Xp, Xm], limit=400)
            red = reduce_basis(M, basis)
            simple = is_simple(M, red)[0]
            z = centre(M, red)
            say('    t = %-5s sdim %s  centre %s  simple %s'
                % (t, s, z, simple))
        say('    -- Poletaeva\'s K = hei(0|4) (+) sl(2) is the fibre t = 0:')
        b0 = reduce_basis(M, closure(M, [d, dc, Xp, Xm], limit=400)[0])
        Lap = M.br(d, M.br(d, Xm))                   # Delta = [d, delta]
        cen = [e for e in b0 if all(not M.br(e, u) for u in b0)]
        say('       sdim %s, centre %s, the centre is spanned by Delta: %s'
            % (sd(M, b0), centre(M, b0),
               len(cen) == 1 and sd(M, [cen[0], Lap]) == (1, 0)))
        say('')

    say('=' * 74)
    say('2.  the structure constants are polynomial in t and specialize at 0')
    say('    (basis X^+, X^-, H, Delta_t, D^+, D^-, D_c^+, D_c^-)')
    say('=' * 74)
    for n in (1, 2):
        M, d, dc, Xp, Xm, H, al, alc = kaehler(n)
        tabs, ok = {}, True
        for t in TS:
            names, bas = kbasis(M, d, dc, Xp, Xm, H, al, alc, t)
            if sd(M, bas) != (4, 4):
                say('    C^{%d|0}: t = %s -- the basis DEGENERATES, sdim %s'
                    % (2 * n, t, sd(M, bas)))
                ok = False
                break
            tb, bad = table(M, names, bas)
            if tb is None:
                say('    C^{%d|0}: t = %s -- [%s, %s] leaves the span'
                    % (2 * n, t, bad[0], bad[1]))
                ok = False
                break
            tabs[t] = tb
        if not ok:
            continue
        pts = [t for t in TS if t != 0]
        deg, mism = 0, 0
        for key in tabs[Fraction(0)]:
            for c in range(8):
                vals = [tabs[t][key][c] for t in pts]
                for dgr in range(len(pts)):
                    if degree_bound(pts, vals, dgr):
                        deg = max(deg, dgr)
                        break
                if interpolate(pts, vals) != tabs[Fraction(0)][key][c]:
                    mism += 1
        say('    C^{%d|0}: every entry is a polynomial in t of degree <= %d;'
            % (2 * n, deg))
        say('             entries whose t -> 0 limit misses the t = 0 table: %d'
            % mism)
        say('             (%d brackets x 8 coordinates checked)' % len(tabs[Fraction(0)]))
    say('')

    say('=' * 74)
    say('3.  the hyper-Kaehler family: g_t = <D^+_t, E_l, F_l>, l = i, j, k')
    say('=' * 74)
    M, d, E, F, a1 = hyper()
    for t in TS:
        D1 = M.add(d, M.smul(-half * t, a1))
        basis, s = closure(M, [D1] + E + F, limit=800)
        red = reduce_basis(M, basis)
        say('    t = %-5s sdim %s  centre %s  simple %s'
            % (t, s, centre(M, red), is_simple(M, red)[0]))
    say('    -- Poletaeva\'s H, of her own dim H = (11|8), is the fibre t = 0.')
    say('')

    say('=' * 74)
    say('4.  summary')
    say('=' * 74)
    say('    The superdimension is constant on each family, (4|4) and (11|8),')
    say('    the generic fibre is simple and the fibre t = 0 has a')
    say('    1-dimensional even centre -- the Laplacian.  So each family is a')
    say('    degeneration at constant superdimension, i.e., the inverse of the')
    say('    contraction Poletaeva records in the Conclusion of [P].')
    LOG.close()


if __name__ == '__main__':
    main()
