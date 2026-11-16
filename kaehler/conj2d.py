r"""Theorem 1.8.2, the choice of the partner alpha_c.

With D^+ = d - (1/2) alpha, D_c^+ = d_c - (1/2) alpha_c, where alpha_c rotates both
the differentials and the coefficients, the closure is (4|4), perfect and centreless
but not simple, because [d_c, alpha_c] is not a multiple of X^+.  This script

  1. computes [d,alpha], [d,alpha_c], [d_c,alpha], [d_c,alpha_c] for the two candidate
     J-rotations of alpha, and finds which 1-form d_c must be twisted by so that its
     curvature is proportional to X^+;
  2. scans the twist coefficients (s, s_c) over a grid and, for each, computes the
     closure, its sdim, centre and simplicity;
  3. checks the osp(2|2) signature for the surviving candidates:
     simple, sdim (4|4), g_ev = sl(2) (+) C, [g_od,g_od] = g_ev.

Run (from this folder):  ../venv/bin/python conj2d.py
"""
from fractions import Fraction

from flatmodel import Model, closure, d_operator, \
    form_to_multiplication, form_to_convolution
from conj2 import span, sd, contains, centre, derived, reduce_basis
from conj2c import is_simple, ideal

half = Fraction(1, 2)


def build(n):
    """The flat Kaehler model on C^{2n}: coordinates q_i, p_i, omega = sum dq^dp."""
    N = 2 * n
    M = Model([0] * N)
    q = list(range(n))
    p = list(range(n, N))
    c = lambda a: M.gen('c', a)      # dq_a  (the "hat" generators)
    x = lambda a: M.gen('x', a)      # q_a
    t = lambda a: M.gen('t', a)      # partial_{dq_a}
    d = d_operator(M)
    Xp = form_to_multiplication(M, [(1, (q[i], p[i])) for i in range(n)])
    Xm = form_to_convolution(M, [(1, (q[i], p[i])) for i in range(n)])
    H = M.br(Xp, Xm)
    # J : dq_i -> dp_i, dp_i -> -dq_i, and the same on q_i, p_i
    dc = M.add(*[M.add(M.mul(c(p[i]), t(q[i])),
                       M.smul(-1, M.mul(c(q[i]), t(p[i])))) for i in range(n)])
    return M, n, q, p, c, x, t, d, dc, Xp, Xm, H


def one_forms(M, n, q, p, c, x):
    """alpha and the two candidate J-rotations of it."""
    # alpha = sum (dq_i p_i - dp_i q_i);  d alpha = -2 omega
    alpha = M.add(*[M.add(M.mul(c(q[i]), x(p[i])),
                          M.smul(-1, M.mul(c(p[i]), x(q[i])))) for i in range(n)])
    # (i) conj2.py's alphac
    ac_old = M.add(*[M.add(M.smul(-1, M.mul(c(p[i]), x(q[i]))),
                           M.smul(-1, M.mul(c(q[i]), x(p[i])))) for i in range(n)])
    # (ii) J acting on the dq's only:  dq->dp, dp->-dq, coefficients kept
    ac_hat = M.add(*[M.add(M.mul(c(p[i]), x(p[i])),
                           M.mul(c(q[i]), x(q[i]))) for i in range(n)])
    # (iii) J acting on the coefficients only: q->p, p->-q
    ac_co = M.add(*[M.add(M.smul(-1, M.mul(c(q[i]), x(q[i]))),
                          M.smul(-1, M.mul(c(p[i]), x(p[i])))) for i in range(n)])
    return alpha, {'conj2 alpha_c': ac_old, 'J on dq only': ac_hat,
                   'J on coeffs only': ac_co, 'alpha itself': alpha}


def prop_to(M, e, Xp):
    """if e is a rational multiple of Xp, return it, else None."""
    if not e:
        return Fraction(0)
    for key, v in e.items():
        if key in Xp:
            r = Fraction(v) / Fraction(Xp[key])
            if not M.add(e, M.smul(-r, Xp)):
                return r
            return None
    return None


def structure(M, basis):
    red = reduce_basis(M, basis)
    s = sd(M, red)
    ev = [e for e in red if M.parity(e) == 0]
    od = [e for e in red if M.parity(e) == 1]
    simple, wit = is_simple(M, red)
    ev_der = sd(M, [M.br(a, b) for a in ev for b in ev])
    od_sq = sd(M, [M.br(a, b) for a in od for b in od])
    return dict(sdim=s, centre=centre(M, red), simple=simple,
                ev_derived=ev_der, od_squared=od_sq,
                ev_centre=centre(M, ev))


print('=' * 70)
print('1) which 1-form is the right partner of d_c ?')
print('=' * 70)
for n in (1, 2):
    M, n, q, p, c, x, t, d, dc, Xp, Xm, H = build(n)
    alpha, cands = one_forms(M, n, q, p, c, x)
    print('--- C^{%d|0}' % (2 * n))
    print('   [d, alpha]   = %s * X^+' % prop_to(M, M.br(d, alpha), Xp))
    for name, ac in cands.items():
        r_dc = prop_to(M, M.br(dc, ac), Xp)
        r_d = prop_to(M, M.br(d, ac), Xp)
        print('   %-18s : [d_c, .] = %-6s * X^+ ; [d, .] = %-6s * X^+'
              % (name, r_dc, r_d))
    print()

print('=' * 70)
print('2) scan of the twist coefficients, D^+ = d + s*alpha, D_c^+ = d_c + s_c*ac')
print('=' * 70)
grid = [Fraction(k, 2) for k in range(-4, 5) if k]
for n in (1, 2):
    M, n, q, p, c, x, t, d, dc, Xp, Xm, H = build(n)
    alpha, cands = one_forms(M, n, q, p, c, x)
    print('--- C^{%d|0}' % (2 * n))
    seen = {}
    for name, ac in cands.items():
        for s in grid:
            for sc in grid:
                Dp = M.add(d, M.smul(s, alpha))
                Dcp = M.add(dc, M.smul(sc, ac))
                basis, sdm = closure(M, [Dp, Dcp, Xp, Xm], limit=400)
                if sdm == (4, 4):
                    st = structure(M, basis)
                    key = (name, st['simple'], st['sdim'], st['centre'])
                    if st['simple']:
                        seen.setdefault(key, []).append((s, sc))
    if seen:
        for key, pairs in seen.items():
            print('   SIMPLE (4|4) found: partner=%s  at (s,s_c) = %s'
                  % (key[0], pairs[:6]))
    else:
        print('   no choice of (s, s_c) on the grid gives a SIMPLE (4|4) algebra')
    print()

print('=' * 70)
print('3) the (4|4) algebra of conj2.py: what is its proper ideal?')
print('=' * 70)
for n in (1, 2):
    M, n, q, p, c, x, t, d, dc, Xp, Xm, H = build(n)
    alpha, cands = one_forms(M, n, q, p, c, x)
    ac = cands['conj2 alpha_c']
    Dp = M.add(d, M.smul(-half, alpha))
    Dcp = M.add(dc, M.smul(-half, ac))
    basis, sdm = closure(M, [Dp, Dcp, Xp, Xm], limit=400)
    red = reduce_basis(M, basis)
    print('--- C^{%d|0}  sdim %s' % (2 * n, sdm))
    for u in red:
        i = ideal(M, red, [u])
        if i != sdm:
            print('     a proper ideal of sdim %s (generated by one element)' % (i,))
            break
    # the ideals generated by each element, as a multiset
    sizes = sorted({ideal(M, red, [u]) for u in red})
    print('     sdims of the 1-generated ideals:', sizes)
    print()
