import sys
import os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
src = open(os.path.join(_HERE, "irr.py")).read().split('print("--- closure')[0]
exec(src)
from exactlin import nullspace
def basis(h):
    hb = []
    for A, p in h:
        if rank([flat(B) for B, _ in hb]+[flat(A)]) > len(hb): hb.append((A, p))
    return hb
def prolong(hb, deg):
    """returns dims of degrees 1..deg; elements of degree i as tensors: list of (T, parity),
    T[a] = element of degree i-1 (as tensor) = image of w_a."""
    prev = [(A, p) for A, p in hb]   # degree 0 elements: matrices
    def ev(T, i):  # flatten degree-(i) element to vector
        if i == 0: return flat(T)
        return [x for a in range(N) for x in ev(T[a], i-1)]
    def app(T, i, b):  # apply degree-i element to w_b -> degree i-1 element (i>=1), as nested; i==0: column b vector
        return T[b] if i >= 1 else [T[r][b] for r in range(N)]
    dims = []
    for i in range(1, deg+1):
        new = []; tot = 0
        for par in (0, 1):
            unk = [(a, s) for a in range(N) for s in range(len(prev)) if prev[s][1] == (par+pW[a]) % 2]
            eqs = []
            for a in range(N):
                for b in range(a, N):
                    sign = (-1)**(pW[a]*pW[b])
                    # phi(a)(w_b) - sign phi(b)(w_a) = 0 as degree i-2 element (vector)
                    L = len(ev(app(prev[0][0], i-1, 0), i-2)) if i >= 2 else N
                    for r in range(L):
                        row = []
                        for (a2, s) in unk:
                            v = 0
                            T = prev[s][0]
                            if a2 == a: v += (ev(app(T, i-1, b), i-2)[r] if i >= 2 else T[r][b])
                            if a2 == b: v -= sign*(ev(app(T, i-1, a), i-2)[r] if i >= 2 else T[r][a])
                            row.append(v)
                        eqs.append(row)
            ns = nullspace(eqs)
            for v in ns:
                T = [None]*N
                for a in range(N):
                    acc = None
                    for (a2, s), c in zip(unk, v):
                        if a2 != a or c == 0: continue
                        X = prev[s][0]
                        if acc is None: acc = scale(X, c, i-1)
                        else: acc = addt(acc, scale(X, c, i-1), i-1)
                    T[a] = acc if acc is not None else zero(i-1)
                new.append((T, par))
            tot += len(ns)
        dims.append(tot); prev = new
        print("   degree", i, "dim", tot, flush=True)
    return dims
def zero(i):
    return [[0]*N for _ in range(N)] if i == 0 else [zero(i-1) for _ in range(N)]
def scale(X, c, i):
    return [[c*x for x in r] for r in X] if i == 0 else [scale(Y, c, i-1) for Y in X]
def addt(X, Y, i):
    return [[x+y for x, y in zip(r, s)] for r, s in zip(X, Y)] if i == 0 else [addt(a, b, i-1) for a, b in zip(X, Y)]
deg = int(sys.argv[4]) if len(sys.argv) > 4 else 2
print("u0 prolong:"); prolong(basis(U), deg)
for key, vs in blocks.items():
    S = genbasis(vs[0]); Sh = []
    for v in S:
        for p in (0, 1):
            w = [v[q] if reps[q][1] == p else 0 for q in range(nq)]
            if any(w) and rank(Sh+[w]) > rank(Sh): Sh.append(w)
    h = U + [(tomat(v), vpar(v)) for v in Sh]
    print("h0 from", sorted(key), "sdim", len(basis(h))); prolong(basis(h), deg)
