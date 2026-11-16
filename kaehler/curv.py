r"""The curvature forms of the two connections of Theorem 1.8.2 and Subsec. 1.9.

In the Kaehler case there is one form, the even omega, and both curvatures are
proportional to it: [d, alpha] = [d_c, alpha_c] = -2 X^+, [d, alpha_c] = [d_c, alpha] = 0;
the twisting constants of D^+ = d + s alpha, D_c^+ = d_c + s_c alpha_c are s = hbar_1,
s_c = hbar_2.  In the periplectic case the single odd omega serves both connections.

Run (from this folder):  ../venv/bin/python curv.py
"""
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
import sys
from fractions import Fraction


from flatmodel import Model, d_operator, \
    form_to_multiplication, form_to_convolution              # noqa: E402

half = Fraction(1, 2)


# The flat Kaehler model and the two connection 1-forms, copied verbatim from
# conj2d.build / conj2e.partner_forms so that importing them (and re-running
# their whole scripts) is not needed.
def build(n):
    """the flat Kaehler C^{2n}: coordinates q_i, p_i, omega = sum dq_i dp_i."""
    N = 2 * n
    M = Model([0] * N)
    q = list(range(n))
    p = list(range(n, N))
    c = lambda a: M.gen('c', a)                              # noqa: E731
    x = lambda a: M.gen('x', a)                              # noqa: E731
    t = lambda a: M.gen('t', a)                              # noqa: E731
    d = d_operator(M)
    Xp = form_to_multiplication(M, [(1, (q[i], p[i])) for i in range(n)])
    Xm = form_to_convolution(M, [(1, (q[i], p[i])) for i in range(n)])
    H = M.br(Xp, Xm)
    dc = M.add(*[M.add(M.mul(c(p[i]), t(q[i])),
                       M.smul(-1, M.mul(c(q[i]), t(p[i])))) for i in range(n)])
    return M, n, q, p, c, x, t, d, dc, Xp, Xm, H


def partner_forms(M, n, q, p, c, x):
    alpha = M.add(*[M.add(M.mul(c(q[i]), x(p[i])),
                          M.smul(-1, M.mul(c(p[i]), x(q[i])))) for i in range(n)])
    alpha_c = M.add(*[M.add(M.mul(c(p[i]), x(p[i])),
                            M.mul(c(q[i]), x(q[i]))) for i in range(n)])
    return alpha, alpha_c


def cmp_to(M, e, Xp, name):
    """is e proportional to X^+ ?  print the ratio."""
    if not e:
        return '0'
    for mono, co in Xp.items():
        if mono in e:
            r = Fraction(e[mono]) / Fraction(co)
            if not M.add(e, M.smul(-r, Xp)):
                return '%s * %s' % (r, name)
            break
    return 'NOT proportional to %s' % name


print('=' * 74)
print('1.  the Kaehler case 2a): the flat C^{2n}, omega = sum dq_i dp_i')
print('=' * 74)
for nn in (1, 2, 3):
    M, n, q, p, c, x, t, d, dc, Xp, Xm, H = build(nn)
    alpha, alpha_c = partner_forms(M, n, q, p, c, x)
    print('--- C^{%d|0}' % (2 * n))
    print('    [d,   alpha  ] =', cmp_to(M, M.br(d, alpha), Xp, 'X^+'))
    print('    [d_c, alpha_c] =', cmp_to(M, M.br(dc, alpha_c), Xp, 'X^+'))
    print('    [d,   alpha_c] =', cmp_to(M, M.br(d, alpha_c), Xp, 'X^+'))
    print('    [d_c, alpha  ] =', cmp_to(M, M.br(dc, alpha), Xp, 'X^+'))

print()
print('=' * 74)
print('2.  the twisting constants: D^+ = d + s alpha, D_c^+ = d_c + s_c alpha_c')
print('    so the curvature of the first connection is  2s . omega,')
print('    that of the second  2 s_c . omega:  s = hbar_1, s_c = hbar_2.')
print('=' * 74)
M, n, q, p, c, x, t, d, dc, Xp, Xm, H = build(1)
alpha, alpha_c = partner_forms(M, n, q, p, c, x)
for s in (Fraction(1), Fraction(-half), Fraction(3, 2)):
    Dp = M.add(d, M.smul(s, alpha))
    Dcp = M.add(dc, M.smul(s, alpha_c))
    print('    s = %-5s [D^+,  D^+ ]/2 = %s' % (s, cmp_to(M, M.smul(half, M.br(Dp, Dp)), Xp, 'X^+')))
    print('            [D_c^+,D_c^+]/2 = %s' % cmp_to(M, M.smul(half, M.br(Dcp, Dcp)), Xp, 'X^+'))

print()
print('=' * 74)
print('3.  the periplectic case (Subsec. 1.9.4): d alpha_(hbar) = hbar omega_od')
print('    -- one odd form omega_od, both connections, cf. eq. (alphatwo)')
print('=' * 74)
for nn in (1, 2):
    P = Model([0] * nn + [1] * nn)                 # q_i even, pi_i odd
    cg = lambda a: P.gen('c', a)                   # noqa: E731
    xg = lambda a: P.gen('x', a)                   # noqa: E731
    dP = d_operator(P)
    # omega_od = sum dq_i dpi_i  (odd), X^+ its multiplication
    XpP = form_to_multiplication(P, [(1, (i, nn + i)) for i in range(nn)])
    XmP = form_to_convolution(P, [(1, (i, nn + i)) for i in range(nn)])
    # alpha_(hbar) = (1/2) sum (pi_i dq_i - q_i dpi_i), the odd analogue
    alphaP = P.add(*[P.add(P.mul(cg(i), xg(nn + i)),
                           P.smul(-1, P.mul(cg(nn + i), xg(i))))
                     for i in range(nn)])
    print('--- C^{%d|%d}: [d, alpha] = %s'
          % (nn, nn, cmp_to(P, P.br(dP, alphaP), XpP, 'X^+ (= omega_od)')))
