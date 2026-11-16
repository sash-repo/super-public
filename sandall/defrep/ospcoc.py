"""p odd: the odd cocycles of osp(1|2) with coefficients in gl(Pi(L^{(p-1)/2})) of
weight -p and +p, in the basis m_i (i = (p-1)/2 ... (1-p)/2), m_{i-1} = y1 m_i:
which combinations of the entries printed in eq. (osp12def) are cocycles?"""
import sys, itertools
from hwmod import Fp
from prol import Rep, op_bracket, Span, nullspace_sparse
p = int(sys.argv[1]); K = Fp(p); m = (p - 1) // 2
r = Rep([[1]], [1], p, [m])
N = r.N
# m_i  <->  basis index d = m - i
def E(i, j):            # m_i (x) m_j^* : m_j -> m_i
    return {(m - i, m - j): 1}
x1 = {k: v for c, cv in r.E[0].items() for k, v in [((rr, c), vv) for rr, vv in cv.items()]}
y1 = {k: v for c, cv in r.F[0].items() for k, v in [((rr, c), vv) for rr, vv in cv.items()]}
g = {"x1": (x1, 1), "y1": (y1, 1)}
g["h1"] = (op_bracket(K, x1, 1, y1, 1), 0)
g["x2"] = (op_bracket(K, x1, 1, x1, 1), 0)
g["y2"] = (op_bracket(K, y1, 1, y1, 1), 0)
names = ["x1", "x2", "h1", "y1", "y2"]
def coords(Z):
    from hwmod import rref
    basis = [g[n][0] for n in names]
    keys = sorted(set(k for X in basis for k in X) | set(Z))
    rows = [[X.get(k, 0) for X in basis] + [Z.get(k, 0)] for k in keys]
    R, piv = rref(K, rows); n = len(names)
    assert n not in piv
    sol = [0] * n
    for q, c in enumerate(piv): sol[c] = R[q][n]
    return sol
sgn = lambda e: -1 if e % 2 else 1
def is_cocycle(c, pc=1):
    for a, b in itertools.combinations_with_replacement(names, 2):
        (X, px), (Y, py) = g[a], g[b]
        lhs = {}
        def acc(coef, Z):
            for k, v in Z.items(): lhs[k] = K.norm(lhs.get(k, 0) + coef * v)
        cy = c.get(b, {}); cx = c.get(a, {})
        opar = lambda A: ((r.parity[list(A)[0][0]] + r.parity[list(A)[0][1]]) % 2) if A else 0
        if cy: acc(sgn(px * pc), op_bracket(K, X, px, cy, opar(cy)))
        if cx: acc(-sgn(py * (px + pc)), op_bracket(K, Y, py, cx, opar(cx)))
        co = coords(op_bracket(K, X, px, Y, py))
        for nm, cf in zip(names, co):
            if cf: acc(-cf, c.get(nm, {}))
        if any(v % p for v in lhs.values()): return False
    return True
def lin(*terms):
    out = {}
    for cf, A in terms:
        for k, v in A.items(): out[k] = K.norm(out.get(k, 0) + cf * v)
    return {k: v for k, v in out.items() if v}
# weight -p: c(x2) = u E(m-1... ) ; search over coefficient triples
lo = -m
found = []
for u, v, w in itertools.product(range(p), repeat=3):
    if not (u or v or w): continue
    c = {"x2": lin((u, E(1 - m, m)), (v, E(-m, m - 1))), "x1": lin((w, E(-m, m)))}
    if is_cocycle(c): found.append((u, v, w))
print("p=%d weight -p: cocycles u*m_{%d}m*_{%d} x2^ + v*m_{%d}m*_{%d} x2^ + w*m_{%d}m*_{%d} x1^ :" % (p, 1 - m, m, -m, m - 1, -m, m), found[:6], len(found))
found = []
for u, v, w in itertools.product(range(p), repeat=3):
    if not (u or v or w): continue
    c = {"y2": lin((u, E(m, m - 1 - (m - 1) + (1 - m) if False else m - 1 - 0 if False else 1 - m + 0)))}
    # printed: ((p+1)/2 m_{m} m*_{1-m} + m_{m-1} m*_{-m}) y2^ + (p+1)/2 m_m m*_{-m} y1^
    c = {"y2": lin((u, E(m, 1 - m)), (v, E(m - 1, -m))), "y1": lin((w, E(m, -m)))}
    if is_cocycle(c): found.append((u, v, w))
print("p=%d weight +p: cocycles u*m_{%d}m*_{%d} y2^ + v*m_{%d}m*_{%d} y2^ + w*m_{%d}m*_{%d} y1^ :" % (p, m, 1 - m, m - 1, -m, m, -m), found[:6], len(found))
# the uniform formulas
cm = {"x1": lin((-1, E(-m, m))), "x2": lin((1, E(1 - m, m)), (1, E(-m, m - 1)))}
cp = {"y1": lin(((p + 1) // 2, E(m, -m))), "y2": lin((1, E(m, 1 - m)), (-1, E(m - 1, -m)))}
print("uniform formulas are cocycles:", is_cocycle(cm), is_cocycle(cp))
