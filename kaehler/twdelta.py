r"""The twisted Laplacian of eq. (Deltw) and its place in osp(2|2) (Theorem 1.8.2).

Same flat model and operators as in twohbareven.py:

    delta      := [X^-, d]                        (the flat-model codifferential)
    Delta      := [d_c, delta]  = sum partial_{z_a}^2         eq. (Del)
    D^+        := d   + (hbar_1/2) alpha,   D^-   := [X^-, D^+]
    D_c^+      := d_c + (hbar_2/2) alpha_c, D_c^- := [X^-, D_c^+]
    Delta_tw   := [D_c^+, D^-]                    eq. (Deltw)
    Delta_tw'  := [D^+, D_c^-]

and asks of the closure g = osp(2|2) of {D^+, D^-, D_c^+, D_c^-, X^+, X^-}:
  1) is Delta_tw in g?
  2) is Delta_tw outside g_ev' = [g_ev, g_ev] = sl(2)?  Then g_ev = sl(2) + C Delta_tw;
  3) is Delta_tw proportional to the central element z of g_ev exactly?
  4) does hbar -> 0 return the Delta of eq. (Del)?
Ranges: C^2, C^4, C^6 and six pairs (hbar_1, hbar_2).

Run (from this folder):  ../venv/bin/python twdelta.py
"""
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
import sys
from fractions import Fraction


from flatmodel import Model, Echelon, closure, d_operator          # noqa: E402
from twohbareven import (kaehler, Xplus, Xminus, J_standard, d_c,  # noqa: E402
                         alpha, alpha_c, span, sdim_of, contains,
                         centre, _centre_dim, half)


def central_elements(M, ev):
    """A basis of {u in span(ev) | [u, v] = 0 for all v in ev}, exactly."""
    if not ev:
        return []
    # solve the linear system directly: coordinates of sum c_i ev_i bracketing to 0
    from exactlin import nullspace
    rows = []
    for v in ev:
        brs = [M.br(u, v) for u in ev]
        mons = sorted({mn for e in brs for mn in e}, key=lambda mn: (len(mn), mn))
        for mn in mons:
            rows.append([brs[i].get(mn, Fraction(0)) for i in range(len(ev))])
    if not rows:
        return list(ev)
    ns = nullspace(rows)
    out = []
    for vec in ns:
        e = M.add(*[M.smul(c, ev[i]) for i, c in enumerate(vec) if c])
        if e:
            out.append(e)
    return out


def proportional(M, a, b):
    """Is a = lambda b for a non-zero scalar lambda?"""
    if not a or not b:
        return False
    mn = next(iter(b))
    if mn not in a:
        return False
    lam = a[mn] / b[mn]
    return M.add(a, M.smul(-lam, b)) == {}


def run(m, k, h1, h2, tag):
    M = kaehler(m, k)
    Xp, Xm = Xplus(M, m, k), Xminus(M, m, k)
    d = d_operator(M)
    dc = d_c(M, J_standard(m, k))
    al, alc = alpha(M, m, k), alpha_c(M, m, k)

    assert M.br(d, al) == Xp, '[d, alpha] != omega'
    assert M.br(dc, alc) == Xp, '[d_c, alpha_c] != omega'

    delta = M.br(Xm, d)
    Delta = M.br(dc, delta)                       # eq. (Del), untwisted

    Dp = M.add(d, M.smul(h1 * half, al))
    Dcp = M.add(dc, M.smul(h2 * half, alc))
    Dm, Dcm = M.br(Xm, Dp), M.br(Xm, Dcp)

    Dtw = M.br(Dcp, Dm)                           # the twisted Laplacian
    Dtw2 = M.br(Dp, Dcm)                          # its partner, cf. eq. (Del)

    g, sd = closure(M, [Dp, Dm, Dcp, Dcm, Xp, Xm], limit=600)
    ech = span(M, g)
    ev = ech[0].elements

    ev_der = [b for b in (M.br(u, v) for u in ev for v in ev) if b]
    ech_der = span(M, ev_der)                     # sl(2) = [g_ev, g_ev]

    z = central_elements(M, ev)
    z_ok = len(z) == 1

    # 4) the degeneration hbar -> 0
    M0 = kaehler(m, k)
    d0 = d_operator(M0)
    dc0 = d_c(M0, J_standard(m, k))
    Delta0 = M0.br(dc0, M0.br(Xminus(M0, m, k), d0))

    print('  %-24s sdim %-7s | Delta_tw: in g %-5s in sl(2) %-5s prop. to z %-5s '
          '| Delta_tw = Delta_tw'' %-5s | dim[g_ev,g_ev] %d, dim g_ev %d, '
          'centre of g_ev %d | untwisted Delta in g %-5s'
          % (tag, '(%d|%d)' % sd,
             contains(M, ech, Dtw),
             contains(M, ech_der, Dtw),
             (proportional(M, Dtw, z[0]) if z_ok else 'n/a'),
             proportional(M, Dtw, Dtw2) or (Dtw == Dtw2),
             len(ech_der.get(0).rows) if hasattr(ech_der, 'get')
             else len(ech_der[0].rows),
             len(ev), _centre_dim(M, ev, ev),
             contains(M, ech, Delta)))

    return dict(sdim=sd,
                in_g=contains(M, ech, Dtw),
                in_sl2=contains(M, ech_der, Dtw),
                prop_z=(proportional(M, Dtw, z[0]) if z_ok else None),
                twins=proportional(M, Dtw, Dtw2) or (Dtw == Dtw2),
                untw_in_g=contains(M, ech, Delta),
                degen=(Delta0 == Delta))


if __name__ == '__main__':
    print(__doc__.split('Run:')[0].strip())
    print()

    print('=' * 130)
    print('THE TWISTED LAPLACIAN Delta_tw := [D_c^+, D^-] inside the closure '
          'osp(2|2) of {D^+, D^-, D_c^+, D_c^-, X^+, X^-}')
    print('=' * 130)

    res = []
    for m in (1, 2, 3):
        for h1, h2 in ((Fraction(1), Fraction(1)), (Fraction(3), Fraction(-5)),
                       (Fraction(1, 2), Fraction(7, 3)),
                       (Fraction(2), Fraction(1)), (Fraction(-1), Fraction(4)),
                       (Fraction(7, 3), Fraction(-1, 2))):
            res.append(run(m, 0, h1, h2,
                           'C^%d, hbar=(%s,%s)' % (2 * m, h1, h2)))

    print()
    print('=' * 130)
    print('VALIDATION -- hbar = 0: the same formula must return the Delta of '
          'eq. (Del), central in hei(0|4) + sl(2)')
    print('=' * 130)
    val = [run(m, 0, Fraction(0), Fraction(0), 'C^%d, untwisted' % (2 * m))
           for m in (1, 2, 3)]

    print()
    print('twisted, %d cases:' % len(res))
    print('  sdim of the closure                    :',
          set(r['sdim'] for r in res))
    print('  Delta_tw lies in it                    :',
          set(r['in_g'] for r in res))
    print('  Delta_tw lies in [g_ev, g_ev] = sl(2)  :',
          set(r['in_sl2'] for r in res), ' <== must be False')
    print('  Delta_tw proportional to the centre z  :',
          set(r['prop_z'] for r in res))
    print('  [D_c^+, D^-] = [D^+, D_c^-] up to scalar:',
          set(r['twins'] for r in res))
    print('  the UNtwisted Delta of eq. (Del) in it :',
          set(r['untw_in_g'] for r in res), ' <== must be False')
    print('untwisted validation, %d cases:' % len(val))
    print('  sdim                                   :',
          set(r['sdim'] for r in val))
    print('  the same formula returns eq. (Del)     :',
          set(r['degen'] for r in val))
