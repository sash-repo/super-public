"""Section 5 of the text. ab(3)=F(4): Serganova's criterion of dominance in every base reached by odd reflections.
Coordinates (e1,e2,e3,d), Gram diag(1,1,1,-3).  Exact (Fraction)."""
from fractions import Fraction as F
from itertools import product
G = (1, 1, 1, -3)
def ip(u, v): return sum(g*a*b for g, a, b in zip(G, u, v))
def add(u, v): return tuple(a+b for a, b in zip(u, v))
def sub(u, v): return tuple(a-b for a, b in zip(u, v))
def neg(u): return tuple(-a for a in u)
def sc(c, u): return tuple(c*a for a in u)
h = F(1, 2)
EVEN = set()
for i in range(3):
    for s in (1, -1):
        v = [0]*4; v[i] = s; EVEN.add(tuple(map(F, v)))
        for j in range(i+1, 3):
            for t in (1, -1):
                v = [0]*4; v[i] = s; v[j] = t; EVEN.add(tuple(map(F, v)))
EVEN |= {(F(0),)*3+(F(1),), (F(0),)*3+(F(-1),)}
ODD = {tuple(h*s for s in sg) for sg in product((1, -1), repeat=4)}
ROOTS = EVEN | ODD
def reflect(B, lam, a):
    nb = tuple(neg(a) if b == a else (add(a, b) if ip(a, b) != 0 else b) for b in B)
    return nb, (sub(lam, a) if ip(lam, a) != 0 else lam)
def cartan(B):
    """rows normalized as hwmod: even rows 2(a,b)/(a,a); isotropic rows (a,b) scaled so that
    the first nonzero off-diagonal entry is an integer with gcd 1 of the row (sign kept)."""
    A = []
    for a in B:
        if ip(a, a) != 0: A.append([2*ip(a, b)/ip(a, a) for b in B])
        else: A.append([ip(a, b) for b in B])
    return A
def bases_from(B0):
    seen = {}; todo = [tuple(B0)]
    while todo:
        B = todo.pop()
        if frozenset(B) in seen: continue
        seen[frozenset(B)] = B
        for a in B:
            if a in ODD: todo.append(reflect(B, (F(0),)*4, a)[0])
    return list(seen.values())
def orbit(B, lam):
    seen = {}; todo = [(B, lam)]
    while todo:
        C, l = todo.pop()
        if frozenset(C) in seen: continue
        seen[frozenset(C)] = (C, l)
        for a in C:
            if a in ODD: todo.append(reflect(C, l, a))
    return list(seen.values())
def positive(B):
    """positive roots of B: roots that are nonneg combos of B (solve linear system)."""
    import sympy
    M = sympy.Matrix([list(b) for b in B]).T
    Mi = M.inv()
    pos = set()
    for r in ROOTS:
        c = Mi * sympy.Matrix(list(r))
        if all(x >= 0 for x in c): pos.add(r)
    return pos
def even_simple(B):
    pos = positive(B)
    ev = [r for r in pos if r in EVEN]
    simple = [r for r in ev if not any(sub(r, s) in pos and sub(r, s) in EVEN and s in ev for s in ev if s != r) and
              not any((add(s, t) == r) for s in ev for t in ev)]
    return simple
def dominant(B, lam):
    orb = orbit(B, lam)
    for g in even_simple(B):
        for C, l in orb:
            if g in C:
                lab = 2*ip(l, g)/ip(g, g)
                if lab.denominator != 1 or lab < 0: return False
                break
        else: raise ValueError("no base contains %s" % (g,))
    return True
def lam_from_labels(B, A, labels):
    """solve lambda(h_i)=labels: even rows 2(l,a)/(a,a); isotropic rows k*(l,a) with k s.t.
    row = k*(a,b)."""
    import sympy
    rows = []; rhs = []
    for i, a in enumerate(B):
        if ip(a, a) != 0:
            rows.append([sympy.Rational(2*G[j]*a[j], 1)/sympy.Rational(ip(a, a)) for j in range(4)])
        else:
            k = None
            for j, b in enumerate(B):
                if j != i and ip(a, b) != 0: k = sympy.Rational(A[i][j])/sympy.Rational(ip(a, b)); break
            rows.append([k*G[j]*a[j] for j in range(4)])
        rhs.append(labels[i])
    s = sympy.Matrix(rows).solve(sympy.Matrix(rhs))
    return tuple(F(int(x.p), int(x.q)) for x in s)
