"""Dominance (finite-dimensionality) of L(lambda,B) for osp(2m|2n), m>=2, any base B.
Ground truth = Serganova's Theorem ([S2] p. 118): lambda is B-dominant iff for every base C
obtained from B by odd reflections, the recalculated weight has a nonnegative integer label on
every even simple root of C.  Coordinates: lambda = sum mu_q delta_q + sum nu_i eps_i,
(delta,delta)=-1, (eps,eps)=1.  Exact arithmetic (Fraction)."""
from fractions import Fraction as F
from itertools import combinations
import random

def form(n, m):
    return [-1]*n + [1]*m

def ip(x, y, g):
    return sum(a*b*c for a, b, c in zip(x, y, g))

def vec(n, m, letter, sign=1):
    v = [0]*(n+m)
    if letter[0] == 'd': v[letter[1]] = sign
    else: v[n+letter[1]] = sign*letter[2]
    return tuple(v)

def base_of_word(n, m, word):
    """word: list of ('d',q) or ('e',i,s); simple roots x_j - x_{j+1}, last 2x_N or x_{N-1}+x_N."""
    xs = [vec(n, m, w) for w in word]
    B = [tuple(a-b for a, b in zip(xs[j], xs[j+1])) for j in range(len(xs)-1)]
    if word[-1][0] == 'd': B.append(tuple(2*a for a in xs[-1]))
    else: B.append(tuple(a+b for a, b in zip(xs[-2], xs[-1])))
    return B

def is_even(r, n):
    # odd roots are +-delta +- eps (one delta coordinate and one eps coordinate nonzero)
    return not (any(r[:n]) and any(r[n:]))

def ground_truth(n, m, B, lam):
    g = form(n, m)
    seen = {}
    stack = [(frozenset(B), tuple(lam))]
    while stack:
        C, l = stack.pop()
        if C in seen: continue
        seen[C] = l
        for a in C:
            if is_even(a, n):
                lab = F(2*ip(l, a, g), ip(a, a, g))
                if lab.denominator != 1 or lab < 0: return False
        for a in C:
            if not is_even(a, n) and ip(a, a, g) == 0:
                C2 = set()
                for b in C:
                    if b == a: C2.add(tuple(-x for x in a))
                    elif ip(a, b, g) != 0: C2.add(tuple(x+y for x, y in zip(a, b)))
                    else: C2.add(b)
                l2 = l if ip(l, a, g) == 0 else tuple(x-y for x, y in zip(l, a))
                stack.append((frozenset(C2), l2))
    return True

def kac(n, m, lam):
    """Kac's conditions in the distinguished base delta_1..delta_n eps_1..eps_m:
    mu dominant for sp(2n), nu dominant for o(2m), and if mu_n < m then nu_{mu_n+1}=...=nu_m=0."""
    mu, nu = lam[:n], lam[n:]
    for q in range(n-1):
        d = mu[q]-mu[q+1]
        if d.denominator != 1 or d < 0: return False
    if mu[-1].denominator != 1 or mu[-1] < 0: return False
    for i in range(m-1):
        d = nu[i]-nu[i+1]
        if d.denominator != 1 or d < 0: return False
    d = nu[m-2]+nu[m-1]
    if d.denominator != 1 or d < 0: return False
    k = int(mu[-1])
    if k < m and any(nu[i] != 0 for i in range(k, m)): return False
    return True

def words(n, m, ending):
    """all words (shuffles, order preserved, eps_m with either sign) of given end type:
    'I' ..eps eps, 'II' ..delta (2 delta end), 'III' ..delta eps (triangle)."""
    N = n+m
    out = []
    for pos in combinations(range(N), n):
        w = []; qi = 0; ei = 0
        for j in range(N):
            if j in pos: w.append(('d', qi)); qi += 1
            else: w.append(('e', ei, 1)); ei += 1
        t = 'II' if w[-1][0] == 'd' else ('I' if w[-2][0] == 'e' else 'III')
        if t != ending: continue
        out.append(w)
        if t != 'III' or True:
            j = [k for k, x in enumerate(w) if x[0] == 'e' and x[1] == m-1][0]
            w2 = list(w); w2[j] = ('e', m-1, -1)
            if t != 'III': out.append(w2)   # for III the two signs give the same base
    return out
