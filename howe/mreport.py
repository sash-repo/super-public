"""
mreport.py -- render singular vectors of I(V) over M(n) in the [GLS] format, using
Kochetkov's own names for the generators of U(M(n)_{<0}):

    'p_i = 'partial_i = '(-xi_i)   (even),   'd_i = 'delta_i = 'x_i  (odd),   D
and, inside V, words in the lowering operators of pe(n) applied to m1 (see report.py):
    a_{ij} = D_{x_i xi_j} (i<j),   b_{ij} = D_{xi_i xi_j} (i<j).
"""

from fractions import Fraction as F
from report import wordname


def fmtc(c):
    c = F(c)
    return str(c.numerator) if c.denominator == 1 else "%d/%d" % (c.numerator, c.denominator)


def umono_name(n, u):
    a, S, t = u
    s = ''
    for k in range(n):
        for _ in range(a[k]):
            s += "'p%d" % (k + 1)
    for k in S:
        s += "'d%d" % (k + 1)
    if t:
        s += 'D'
    return s or '1'


def render(I, sol, names=None):
    n = I.n
    if names is None:
        names = {}
    parts = []
    key = lambda kv: (kv[0][0][2], kv[0][0][0], kv[0][0][1], kv[0][2])
    for (u, mu, i), c in sorted(sol.items(), key=key):
        w = I.V.wtbasis(mu)[i]
        nm = names.setdefault((mu, w), wordname(I.pe, w))
        cc = fmtc(c)
        cs = '' if cc == '1' else ('-' if cc == '-1' else cc + ' ')
        parts.append('%s%s (x) %s' % (cs, umono_name(n, u), nm))
    return ' + '.join(parts).replace('+ -', '- ')
