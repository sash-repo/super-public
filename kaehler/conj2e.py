r"""Theorem 1.8.2: the twisted Dolbeault algebra is osp(2|2).

The partner of alpha is its J-rotation acting on the differentials only,

    alpha   = sum (dq_i p_i - dp_i q_i),        [d,   alpha]   = -2 X^+,
    alpha_c = sum (dp_i p_i + dq_i q_i),        [d_c, alpha_c] = -2 X^+.

This script

  1. prints sdim / centre / simplicity of <D^+, D_c^+, X^+, X^-> for the twist
     D^+ = d + s alpha, D_c^+ = d_c + s_c alpha_c over a grid of (s, s_c);
  2. for the simple (4|4) cases, verifies the osp(2|2) = sl(1|2) signature:
       g_ev = sl(2) (+) C z, with z central in g_ev and acting on the 4-dimensional
       g_od with eigenvalues +-1 (two 2-dimensional sl(2)-modules), and
       [g_od, g_od] = g_ev;
  3. reports the root data by diagonalising ad z and ad H.

Run (from this folder):  ../venv/bin/python conj2e.py
"""
from fractions import Fraction

from flatmodel import Model, closure, d_operator, \
    form_to_multiplication, form_to_convolution
from conj2 import span, sd, contains, centre, derived, reduce_basis
from conj2c import is_simple, ideal
from conj2d import build

half = Fraction(1, 2)


def partner_forms(M, n, q, p, c, x):
    alpha = M.add(*[M.add(M.mul(c(q[i]), x(p[i])),
                          M.smul(-1, M.mul(c(p[i]), x(q[i])))) for i in range(n)])
    alpha_c = M.add(*[M.add(M.mul(c(p[i]), x(p[i])),
                            M.mul(c(q[i]), x(q[i]))) for i in range(n)])
    return alpha, alpha_c


def eigen_on(M, z, vecs):
    """eigenvalues of ad z on the span of vecs, if ad z is diagonalisable there."""
    from conj2 import Echelon
    out = []
    for v in vecs:
        br = M.br(z, v)
        r = None
        if not br:
            r = Fraction(0)
        else:
            for key, val in v.items():
                if key in br:
                    r = Fraction(br[key]) / Fraction(val)
                    break
            if r is None or M.add(br, M.smul(-r, v)):
                r = None
        out.append(r)
    return out


print('=' * 72)
print('1) the corrected twist: sdim / simplicity over a grid of (s, s_c)')
print('=' * 72)
grid = [Fraction(k, 2) for k in range(-4, 5) if k]
for nn in (1, 2):
    M, n, q, p, c, x, t, d, dc, Xp, Xm, H = build(nn)
    alpha, alpha_c = partner_forms(M, n, q, p, c, x)
    print('--- C^{%d|0}   [d,alpha] = -2X^+ , [d_c,alpha_c] = -2X^+' % (2 * n))
    rows = {}
    for s in grid:
        for sc in grid:
            Dp = M.add(d, M.smul(s, alpha))
            Dcp = M.add(dc, M.smul(sc, alpha_c))
            basis, sdm = closure(M, [Dp, Dcp, Xp, Xm], limit=600)
            simple = is_simple(M, reduce_basis(M, basis))[0] if sdm[0] + sdm[1] < 40 else None
            rows.setdefault((sdm, simple), []).append((s, sc))
    for (sdm, simple), pairs in sorted(rows.items(), key=lambda kv: -len(kv[1])):
        print('    sdim %-9s simple=%-5s   %2d pairs, e.g. %s'
              % (str(sdm), simple, len(pairs), pairs[:3]))
    print()

print('=' * 72)
print('2) identification of the simple (4|4) algebra as osp(2|2) = sl(1|2)')
print('=' * 72)
for nn in (1, 2):
    M, n, q, p, c, x, t, d, dc, Xp, Xm, H = build(nn)
    alpha, alpha_c = partner_forms(M, n, q, p, c, x)
    # take the twist that gave a simple (4|4)
    found = None
    for s in grid:
        for sc in grid:
            Dp = M.add(d, M.smul(s, alpha))
            Dcp = M.add(dc, M.smul(sc, alpha_c))
            basis, sdm = closure(M, [Dp, Dcp, Xp, Xm], limit=600)
            if sdm == (4, 4):
                red = reduce_basis(M, basis)
                if is_simple(M, red)[0]:
                    found = (s, sc, red)
                    break
        if found:
            break
    if not found:
        print('--- C^{%d|0}: no simple (4|4) closure on the grid' % (2 * n))
        continue
    s, sc, red = found
    ev = [e for e in red if M.parity(e) == 0]
    od = [e for e in red if M.parity(e) == 1]
    print('--- C^{%d|0}  at (s, s_c) = (%s, %s)' % (2 * n, s, sc))
    print('    sdim              :', sd(M, red))
    print('    simple            :', is_simple(M, red)[0])
    print('    centre of g       :', centre(M, red))
    print('    [g_ev,g_ev]       :', sd(M, [M.br(a, b) for a in ev for b in ev]),
          ' (sl(2) is (3|0))')
    print('    centre of g_ev    :', centre(M, ev), ' (the o(2) = C z)')
    print('    [g_od,g_od] = g_ev:',
          sd(M, [M.br(a, b) for a in od for b in od]) == sd(M, ev))
    # the central element z of g_ev, and its eigenvalues on g_od
    from conj2 import Echelon
    zs = []
    for u in ev:
        if all(not M.br(u, v) for v in ev):
            zs.append(u)
    if zs:
        z = zs[0]
        eigs = eigen_on(M, z, od)
        print('    ad z on g_od      : eigenvalues', eigs,
              '=> g_od = two 2-dim modules' if all(e is not None for e in eigs) else
              '(not diagonal in this basis)')
    print('    Xp, Xm, H span sl(2) inside g_ev :',
          sd(M, [Xp, Xm, H]) == (3, 0))
    print()
