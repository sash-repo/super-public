"""Proposition 8.2.1 and Remark 8.2.2 of the text: roots of the deform svect~(0|n) = {D | L_D((1+t xi_1...xi_n)vol)=0}
vs svect'(0|n): weights wrt the torus, weight-0 space, generation of n+ by the
indecomposable positive root spaces.  t=1 (n even) or t=0."""
import sys, itertools, random
from fractions import Fraction
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import rank, nullspace

def gmul(a, b):          # Grassmann monomials as sorted tuples; return (sign, mono) or None
    if set(a) & set(b): return None
    l = list(a) + list(b); s = 1
    for i in range(len(l)):
        for j in range(len(l) - 1 - i):
            if l[j] > l[j+1]: l[j], l[j+1] = l[j+1], l[j]; s = -s
    return s, tuple(l)
def padd(p, m, c):
    if c: p[m] = p.get(m, 0) + c
    if m in p and p[m] == 0: del p[m]
def pmul(f, g):
    r = {}
    for a, x in f.items():
        for b, y in g.items():
            t = gmul(a, b)
            if t: padd(r, t[1], t[0]*x*y)
    return r
def dpart(i, f):         # left derivative d/dxi_i
    r = {}
    for a, x in f.items():
        if i in a:
            k = a.index(i); padd(r, a[:k]+a[k+1:], (-1)**k * x)
    return r
def ppar(f):
    ps = {len(a) % 2 for a in f}; assert len(ps) <= 1; return ps.pop() if ps else 0
# vector field: dict i -> poly ; parity = p(f)+1
def vpar(D):
    for i, f in D.items():
        if f: return (ppar(f) + 1) % 2
    return 0
def vapply(D, g):
    r = {}
    for i, f in D.items():
        for m, c in pmul(f, dpart(i, g)).items(): padd(r, m, c)
    return r
def vbr(D1, D2, n):
    p1, p2 = vpar(D1), vpar(D2); r = {}
    for j in range(n):
        a = vapply(D1, D2.get(j, {})); b = vapply(D2, D1.get(j, {}))
        c = dict(a)
        for m, x in b.items(): padd(c, m, -(-1)**(p1*p2) * x)
        if c: r[j] = c
    return r
def div(D):
    r = {}
    for i, f in D.items():
        for m, c in dpart(i, f).items(): padd(r, m, (-1)**ppar(f) * c if f else 0)
    return r

def run(n, t):
    monos = [tuple(s) for k in range(n+1) for s in itertools.combinations(range(n), k)]
    basis = [(m, i) for m in monos for i in range(n)]
    top = tuple(range(n))
    def vec(b): return {b[1]: {b[0]: 1}}
    # the condition c(D) = div D + s t D(xi_top), s=+-1 chosen so that the result closes
    def cond(D, s):
        r = div(D)
        for m, c in vapply(D, {top: 1}).items(): padd(r, m, s*t*c)
        return r
    # weights: xi^S d_i has weight e_S - e_i, take coordinates mod (1,...,1)
    def wt(b):
        w = [0]*n
        for k in b[0]: w[k] += 1
        w[b[1]] -= 1
        return tuple(x - w[-1] for x in w[:-1])
    out = {}
    for s in (1, -1):
        # parity-homogeneous and weight-homogeneous pieces: solve per (weight, parity)
        blocks = {}
        for b in basis: blocks.setdefault((wt(b), (len(b[0])+1) % 2), []).append(b)
        alg = []   # list of (weight, parity, D)
        for (w, p), bs in blocks.items():
            rows = {}
            cols = []
            for b in bs:
                cols.append(cond(vec(b), s))
            ms = sorted({m for c in cols for m in c})
            M = [[c.get(m, 0) for c in cols] for m in ms]
            ns = nullspace(M) if ms else [[1 if k == j else 0 for k in range(len(bs))] for j in range(len(bs))]
            for v in ns:
                D = {}
                for coef, b in zip(v, bs):
                    if coef: D.setdefault(b[1], {}); padd(D[b[1]], b[0], coef)
                alg.append((w, p, {i: f for i, f in D.items() if f}))
        # closure test on a sample
        ok = True
        random.seed(1)
        for _ in range(300):
            (w1, p1, D1), (w2, p2, D2) = random.choice(alg), random.choice(alg)
            if cond(vbr(D1, D2, n), s): ok = False; break
        out[s] = (ok, alg)
    s = 1 if out[1][0] else -1
    assert out[s][0], "no sign closes"
    return s, out[s][1], wt, basis

def flat(D, basis):
    return [Fraction(D.get(i, {}).get(m, 0)) for (m, i) in basis]

def analyse(n, t):
    s, alg, wt, basis = run(n, t)
    if t == 0:    # pass to the derived algebra svect'
        brs = [vbr(a[2], b[2], n) for a in alg for b in alg]
        brs = [D for D in brs if D]
        R = rank([flat(D, basis) for D in brs])
    mult = {}
    for w, p, D in alg: mult.setdefault(w, [0, 0])[p] += 1
    dimE = sum(v[0] for v in mult.values()); dimO = sum(v[1] for v in mult.values())
    print(f"n={n} t={t} sign={s}  sdim=({dimE}|{dimO})", end="")
    if t == 0: print(f"  dim svect'={R}")
    else: print()
    print("   weight 0 space:", mult.get(tuple([0]*(n-1))))
    return alg, mult, basis

if __name__ == "__main__":
    for n in (3, 4, 5, 6):
        a0, m0, B = analyse(n, 0)
        a1, m1, B = analyse(n, 1)
        diff = {w: (m0.get(w), m1.get(w)) for w in set(m0) | set(m1) if m0.get(w) != m1.get(w)}
        print("   weights whose multiplicity differs (svect, svect~):", diff)
