"""
report.py -- render the singular vectors of I(V(lambda)) over le(n) in the format of
[GLS] (Grozman-Leites-Shchepochkina, math.RT/0202193, sections 3-13): every vector is
written as a sum  u (x) m_j,  where u is a monomial in the basis of L_{-1} and the m_j
are named elements of V obtained from the highest weight vector m_1 by the lowering
operators of L_0 = pe(n).

Names of the lowering operators (Kochetkov's Borel: n_+ = n_+^{gl} + <D_{x_i x_j}>):

    a_{ij} = D_{x_i xi_j}   (i < j)        even,  weight eps_j - eps_i
    b_{ij} = D_{xi_i xi_j}  (i < j)        odd,   weight eps_i + eps_j

Usage:  python3 report.py <n> <lambda_1> ... <lambda_n> [maxdeg] [ht]
"""

import sys
from fractions import Fraction as F
from imod import Le, Induced
from scan import scan, fmtwt, fmtc


def opname(pe, k):
    """name of the k-th lowering operator"""
    n = pe.n
    f = pe.nneg[k]
    a, S = next(iter(f))
    if S and sum(a) == 1:
        i = a.index(1)
        return 'a%d%d' % (i + 1, S[0] + 1)
    if len(S) == 2:
        return 'b%d%d' % (S[0] + 1, S[1] + 1)
    return 'c%d%d' % (S[0] + 1, S[1] + 1) if S else '?'


def wordname(pe, w):
    if not w:
        return 'm1'
    return '.'.join(opname(pe, k) for k in w) + '.m1'


def umono_name(n, u):
    a, S = u
    s = ''
    for k in range(n):
        for _ in range(a[k]):
            s += 'p%d' % (k + 1)
    for k in S:
        s += 'd%d' % (k + 1)
    return s or '1'


def render(I, sol, names):
    n = I.n
    parts = []
    for (u, mu, i), c in sorted(sol.items(), key=lambda kv: (sum(kv[0][0][0]) + len(kv[0][0][1]),
                                                             kv[0][0], kv[0][2])):
        w = I.V.wtbasis(mu)[i]
        nm = names.setdefault(w, wordname(I.pe, w))
        cc = fmtc(c)
        cs = '' if cc == '1' else ('-' if cc == '-1' else cc + ' ')
        parts.append('%s%s (x) %s' % (cs, umono_name(n, u), nm))
    return ' + '.join(parts).replace('+ -', '- ')


def run(n, lam, maxdeg=3, ht=6):
    le = Le(n, odd_up=False)
    I = Induced(le, lam)
    res = scan(n, lam, maxdeg, ht, odd_up=False)
    names = {}
    out = []
    for m, nu, sols in res:
        for s in sols:
            out.append('  d=%d  V%s --> wt %s :  %s'
                       % (m, fmtwt(I.lam), fmtwt(nu), render(I, s, names)))
    return out


if __name__ == '__main__':
    n = int(sys.argv[1])
    lam = tuple(int(x) for x in sys.argv[2:2 + n])
    maxdeg = int(sys.argv[2 + n]) if len(sys.argv) > 2 + n else 3
    ht = int(sys.argv[3 + n]) if len(sys.argv) > 3 + n else 6
    for line in run(n, lam, maxdeg, ht):
        print(line)
