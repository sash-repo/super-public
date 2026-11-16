"""Explicit conditions on the labels (Serganova's [S2] pp. 119-121 and ours), tested against
the ground truth of dom.py.  Label convention (the one that reproduces [S2]'s formulas):
  even node delta_q - delta_r : mu_q - mu_r;  eps_i - s eps_j : nu_i - s nu_j;  2 delta : mu;
  last eps node x+y (fork)    : nu_x + nu_y;
  odd node joining delta_q and s*eps_i (either order) : s nu_i + mu_q.
"""
from fractions import Fraction as F
import random, sys
from dom import *

def sval(lam, n, letter):
    if letter[0] == 'd': return lam[letter[1]]
    return letter[2]*lam[n+letter[1]]

def labels(n, m, word, lam):
    """list of (kind, label) along the chain; kind 'o' odd, 'e' even; the last entry is the
    end node (2delta: kind 'e'; fork x+y: 'e'; triangle delta+eps: 'o')."""
    out = []
    for j in range(len(word)-1):
        x, y = word[j], word[j+1]
        if x[0] == y[0] == 'd': out.append(('e', sval(lam, n, x)-sval(lam, n, y)))
        elif x[0] == y[0] == 'e': out.append(('e', sval(lam, n, x)-sval(lam, n, y)))
        else:
            d = x if x[0] == 'd' else y; e = y if x[0] == 'd' else x
            out.append(('o', sval(lam, n, e)+sval(lam, n, d)))
    x, y = word[-2], word[-1]
    if y[0] == 'd': out.append(('e', sval(lam, n, y)))
    elif x[0] == 'e': out.append(('e', sval(lam, n, x)+sval(lam, n, y)))
    else: out.append(('o', sval(lam, n, x)-sval(lam, n, y)))   # delta + eps: mu - nu
    return out

def nz(x): return 1 if x != 0 else 0
def zplus(x): return x.denominator == 1 and x >= 0

def cross(c, evens):
    """[S2]'s recursion: c_0 = c, c_l = c_{l-1} - e_l - [c_{l-1} != 0]."""
    for e in evens: c = c - e - nz(c)
    return c

def segment_conditions(labs, oddpos):
    """middle sl-type conditions for consecutive odd nodes i<j (only even nodes between)."""
    res = []
    for i, j in zip(oddpos, oddpos[1:]):
        evens = [labs[k][1] for k in range(i+1, j)]
        c = cross(labs[i][1], evens)
        res.append(c - labs[j][1] - nz(c))
    return res

def criterion(n, m, word, lam, variant='ours'):
    labs = labels(n, m, word, lam)
    N = len(labs)
    if any(k == 'e' and not zplus(v) for k, v in labs): return False
    t = 'II' if word[-1][0] == 'd' else ('I' if word[-2][0] == 'e' else 'III')
    odd = [k for k in range(N) if labs[k][0] == 'o']
    conds = []
    if t == 'I':
        conds += segment_conditions(labs, odd)
        if odd:      # end diagram I: a_1 = last odd node, then evens, then the fork
            i = odd[-1]
            c = cross(labs[i][1], [labs[k][1] for k in range(i+1, N-2)])
            A1, A = labs[N-2][1], labs[N-1][1]
            if variant == 'S2':
                if A1*A*c != 0: b = c - (A1+A)/2 - 2
                elif (A1*c == 0 and A != 0) or (A*c == 0 and A1 != 0): b = c - (A1+A)/2 - 1
                elif A1 == A == c == 0: b = F(0)
                else: b = None
                if b is None: return None
            else:
                # delta crosses the eps_{m-1} (label c) and then eps_{m} (two ways: +-eps_m)
                nu1 = (A1+A)/2
                d1 = c - A1 - nz(c)     # label (delta, eps_m) after the first crossing  (?)
                d2 = c - A - nz(c)
                b1 = c - nu1 - nz(c) - nz(d1)
                b2 = c - nu1 - nz(c) - nz(d2)
                conds += [b1, b2]; b = b1
            conds.append(b)
    elif t == 'III':
        tri = odd[-2:]
        conds += segment_conditions(labs, odd[:-2])
        A1, A = labs[N-2][1], labs[N-1][1]
        b = (A1+A)/2 - 1 if (A1 != 0 or A != 0) else F(0)
        conds.append(b)
        if len(odd) > 2:
            i = odd[-3]
            c = cross(labs[i][1], [labs[k][1] for k in range(i+1, N-2)])
            conds += [c - A1 - nz(c), c - A - nz(c)]
    return all(zplus(x) for x in conds)
