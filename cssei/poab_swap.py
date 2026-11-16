#!/usr/bin/env python3
"""Is the swap sigma: tau -> tau + sum q_i xi_i, q_i <-> xi_i (an automorphism of the
contact structure of m_- once parity is forgotten) carrying the hyperplane <c1> of m_0
onto <c1+c2>?  If so, F(b_1(n;1)) = F(b_oo(n;1)) at N~ = 1 up to isomorphism."""
import sys
import series as S
from f2vf import Space, parse_field, find_weights
from prolong import prolong, Graded

N = int(sys.argv[1])
X = ['tau'] + ['q%d' % i for i in range(1, N + 1)] + ['xi%d' % i for i in range(1, N + 1)]
m = len(X)
neg = ['dtau'] + ['dq%d + xi%d dtau' % (i, i) for i in range(1, N + 1)] + ['dxi%d' % i for i in range(1, N + 1)]
nm = S.names(X)
sp0 = Space([1] * m, [2] + [1] * (2 * N))
W = find_weights(sp0, [parse_field(sp0, s, nm) for s in neg])
sp = Space([1] * m, [2] + [1] * (2 * N), None, W)
negf = [parse_field(sp, s, nm) for s in neg]
Q = [1 + i for i in range(N)]
XI = [1 + N + i for i in range(N)]


def mono(i):
    return sp.one[i]


def fmul(f, g):
    out = set()
    for a in f:
        for b in g:
            if not (a & b):
                out ^= {a | b}
    return out


# new -> old and old -> new substitutions (sigma is an involution)
def subst_img(i):
    """phi^* of coordinate i, as a function of the other side's coordinates"""
    if i == 0:
        f = {mono(0)}
        for a, b in zip(Q, XI):
            f ^= {mono(a) | mono(b)}
        return f
    if i in Q:
        return {mono(XI[Q.index(i)])}
    return {mono(Q[XI.index(i)])}


SUB = [subst_img(i) for i in range(m)]


def rewrite(f):
    """a function of the old coordinates written in the new ones"""
    out = set()
    for A in f:
        g = {0}
        for j in range(m):
            if A & sp.mask[j]:
                g = fmul(g, SUB[j])
        out ^= g
    return out


def push(Xf):
    res = set()
    for k in range(m):
        val = sp.apply(Xf, SUB[k])          # X(phi^* x'_k), old coordinates
        for Amon in rewrite(val):
            res ^= {(Amon << 5) | k}
    return res


# sanity: sigma preserves m_-
Gneg = Graded(sp)
for v in negf:
    Gneg.add(v)
assert all(Gneg.contains(push(v)) for v in negf), 'sigma does not preserve m_-'
Gm = prolong(sp, negf, None, verbose=False, maxdeg=0)
ch = S.poab_g0_choices(sp, Gm)
H = {}
for lab, g0b in ch.items():
    Hh = Graded(sp)
    for v in g0b:
        Hh.add(set(v))
    H[lab] = Hh
for lab, g0b in ch.items():
    imgs = [push(set(v)) for v in g0b]
    where = [l2 for l2 in ch if all(H[l2].contains(w) for w in imgs)]
    print('sigma(%s) lies in %s' % (lab, where))
