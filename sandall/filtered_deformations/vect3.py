#!/usr/bin/env python3
"""svect(3;1) over F_3 and H^2(g_-; g) -- the Claim on H^2 of svect(3;1).

Basis of vect(3;1): x^(a) d_i with a in {0,1,2}^3; grading deg(x^(a)d_i)=|a|-1;
the weight used to split the cochain complex is the Z^3-multidegree a - e_i.
"""
from itertools import combinations, product
from k3 import P, LieAlg, rref, rank, nullspace, binom
from coh import Complex

EXP = [(a, b, c) for a in range(P) for b in range(P) for c in range(P)]


def omul(m1, m2):
    a = tuple(x + y for x, y in zip(m1, m2))
    if any(e >= P for e in a):
        return None, 0
    co = 1
    for x, y in zip(m1, m2):
        co = co * binom(x + y, x) % P
    return a, co


def oder(m, i):
    if m[i] == 0:
        return None
    n = list(m)
    n[i] -= 1
    return tuple(n)


BASIS = [(m, i) for m in EXP for i in range(3)]
POS = {b: k for k, b in enumerate(BASIS)}


def vbracket(b1, b2):
    """[f d_i, g d_j] = f d_i(g) d_j - g d_j(f) d_i."""
    (f, i), (g, j) = b1, b2
    out = {}
    dg = oder(g, i)
    if dg is not None:
        m, co = omul(f, dg)
        if co:
            k = POS[(m, j)]
            out[k] = (out.get(k, 0) + co) % P
    df = oder(f, j)
    if df is not None:
        m, co = omul(g, df)
        if co:
            k = POS[(m, i)]
            out[k] = (out.get(k, 0) - co) % P
    return {k: v for k, v in out.items() if v}


def build_vect():
    def brk(a, b):
        return vbracket(BASIS[a], BASIS[b])
    wt = [tuple(m[t] - (1 if t == i else 0) for t in range(3)) for (m, i) in BASIS]
    names = []
    for (m, i) in BASIS:
        s = "".join(("x_%d" % (t + 1)) + ("^{(2)}" if m[t] == 2 else "")
                    for t in range(3) if m[t])
        names.append((s if s else "") + "\\partial_%d" % (i + 1))
    return LieAlg(list(BASIS), brk, wt, names)


def divergence(vec, v):
    """div(sum c_k x^(a) d_i) as a polynomial dict."""
    out = {}
    for k, co in enumerate(v):
        if not co:
            continue
        m, i = BASIS[k]
        d = oder(m, i)
        if d is not None:
            out[d] = (out.get(d, 0) + co) % P
    return {k: v for k, v in out.items() if v}


def build_svect():
    vect = build_vect()
    rows = []
    mons = {m: k for k, m in enumerate(EXP)}
    for k in range(vect.n):
        v = [0] * vect.n
        v[k] = 1
        d = divergence(vect, v)
        r = [0] * len(EXP)
        for m, co in d.items():
            r[mons[m]] = co
        rows.append(r)
    # kernel of the transpose: vectors with zero divergence
    M = [[rows[k][c] for k in range(vect.n)] for c in range(len(EXP))]
    ker = nullspace(M, vect.n)
    red = rref(ker, vect.n)[0]
    return vect, red


def subalg_from_rows(alg, red, wtfun=None):
    """LieAlg on the span of the rref rows `red` of alg."""
    piv = [next(i for i, x in enumerate(r) if x) for r in red]

    def coords(v):
        v = v[:]
        out = [0] * len(red)
        for i, pc in enumerate(piv):
            if v[pc]:
                co = v[pc]
                out[i] = co
                v = [(a - co * b) % P for a, b in zip(v, red[i])]
        assert not any(v), "not in the span"
        return out

    def brk(i, j):
        return {k: c for k, c in enumerate(coords(alg.bracket(red[i], red[j]))) if c}

    wt, names = [], []
    for r in red:
        ws = {alg.wt[k] for k, x in enumerate(r) if x}
        assert len(ws) == 1, "row is not a weight vector"
        wt.append(ws.pop())
        names.append(" + ".join(("" if x == 1 else str(x)) + alg.names[k]
                                for k, x in enumerate(r) if x))
    a = LieAlg(list(range(len(red))), brk, wt, names)
    a.rows = red
    return a


if __name__ == "__main__":
    from coh import invariants
    vect, ker = build_svect()
    print("dim vect(3;1) =", vect.n, " dim svect(3;1) =", len(ker))
    sv = subalg_from_rows(vect, ker)
    print("Jacobi:", sv.check_jacobi()[0])
    deg = [sum(w) for w in sv.wt]
    print("graded dims:", {d: deg.count(d) for d in sorted(set(deg))})
    gm = [i for i in range(sv.n) if deg[i] < 0]
    g0 = [i for i in range(sv.n) if deg[i] == 0]
    cx = Complex(sv, gm)
    H = cx.cohomology(2)
    bydeg = {}
    for w, (d, src, reps, B) in sorted(H.items()):
        bydeg[sum(w)] = bydeg.get(sum(w), 0) + d
    print("dim H^2(g_-;g) =", sum(d for d,*_ in H.values()), " by degree:", bydeg)
    inv, Z, B = invariants(cx, H, g0)
    print("dim (H^2(g_-;g))^{g_0} =", len(inv))
    src = cx.cochains(2)
    for v in inv:
        terms=[f"{'' if co==1 else co}{sv.names[m]}(x)" + "^".join(sv.names[cx.h[i]] for i in S)
               for co,(S,m) in zip(v,src) if co]
        w={cx.weight(*c) for co,c in zip(v,src) if co}
        print("   deg", {sum(x) for x in w}, ":", " + ".join(terms))
