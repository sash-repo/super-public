"""Statement 2.8.11: the weight-0 part of H^2(g; C) for g = k^L'(1|4), k^L(1|4) and k^M(1|4)
(realized with x^{1/2} on the monomials odd in theta_4), by the window method: the unknowns are
the values c(a, b) on pairs of monomials of total weight 0 in the window |power of x| <= M,
the equations are the cocycle identities for the triples in the window |power| <= N.  Also tested:
whether the functionals (1) (on k^L'(1|4) only), (2), (3) of Table (77) are cocycles, coboundaries, independent mod
coboundaries, identically zero.  Run:  python3 h2k14.py 3 7
"""
from fractions import Fraction as Q
from itertools import product
import sys, time
import os; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"), os.path.dirname(os.path.abspath(__file__))]
from exactlin import rank, nullspace, solve, backend
from k14 import monos, br, parity, deg, mono, TOP, c1, c2, c3, add

def key(f):
    (k, _), = f.items(); return k

def run(N, M, moebius=False, drop_z=False, test=()):
    W1 = monos(N, moebius); W2 = monos(M, moebius)
    z = (Q(-1), TOP)
    if drop_z:
        W1 = [f for f in W1 if key(f) != z]; W2 = [f for f in W2 if key(f) != z]
    idx = {key(f): i for i, f in enumerate(W2)}
    par = {key(f): parity(f) for f in W2}
    dg = {key(f): deg(f) for f in W2}
    unk = {}
    keys = list(idx)
    for i, a in enumerate(keys):
        for b in keys[i:]:
            if dg[a] + dg[b] != 4 or par[a] != par[b]: continue
            if a == b and par[a] == 0: continue
            unk[(a, b)] = len(unk)
    def cval(a, b):
        if (a, b) in unk: return {unk[(a, b)]: Q(1)}
        if (b, a) in unk: return {unk[(b, a)]: -Q((-1) ** (par[a] * par[b]))}
        return {}
    def clin(X, Y):  # X,Y dicts
        out = {}
        for ka, va in X.items():
            for kb, vb in Y.items():
                if ka not in idx or kb not in idx: return None
                for u, s in cval(ka, kb).items():
                    out[u] = out.get(u, 0) + s * va * vb
        return out
    rows = []
    for f, g, h in product(W1, repeat=3):
        if deg(f) + deg(g) + deg(h) != 6: continue
        sgn = (-1) ** (parity(f) * parity(g))
        t1 = clin(f, br(g, h)); t2 = clin(br(f, g), h); t3 = clin(g, br(f, h))
        if None in (t1, t2, t3): continue
        r = dict(t1)
        for u, s in t2.items(): r[u] = r.get(u, 0) - s
        for u, s in t3.items(): r[u] = r.get(u, 0) - sgn * s
        r = {u: s for u, s in r.items() if s}
        if r: rows.append(r)
    n = len(unk)
    # dedupe rows
    seen = set(); R = []
    for r in rows:
        t = tuple(sorted(r.items()))
        if t in seen: continue
        seen.add(t); R.append([r.get(j, Q(0)) for j in range(n)])
    Z = nullspace(R) if R else [[Q(int(i == j)) for j in range(n)] for i in range(n)]
    # coboundaries: beta on weight-0 elements (deg 2), (d beta)(a,b) = beta({a,b})
    w0 = [k for k in keys if dg[k] == 2]
    Bd = []
    for kw in w0:
        v = [Q(0)] * n
        for (a, b), j in unk.items():
            v[j] = br({a: Q(1)}, {b: Q(1)}).get(kw, Q(0))
        Bd.append(v)
    # restrict to inner pairs (both in window N0=1)
    inner = [j for (a, b), j in unk.items() if abs(a[0]) <= 1 and abs(b[0]) <= 1]
    proj = lambda vs: [[v[j] for j in inner] for v in vs]
    rz = rank(proj(Z)); rb = rank(proj(Bd)) if Bd else 0
    rzb = rank(proj(Z) + proj(Bd))
    out = {"unknowns": n, "eqs": len(R), "dimZ": len(Z), "inner dim Z": rz,
           "inner dim B": rb, "H2(inner)": rzb - rb}
    for name, c in test:
        v = [Q(0)] * n
        for (a, b), j in unk.items():
            val = c({a: Q(1)}, {b: Q(1)}); v[j] = val
        isco = rank(Z + [v]) == len(Z) if Z else False
        # is it a coboundary on inner pairs?
        iscob = rank(proj(Bd) + proj([v])) == rb
        out[name] = ("cocycle" if isco else "NOT cocycle", "coboundary" if iscob else "nontrivial")
    if len(test) >= 2:
        vs = []
        for name, c in test:
            v = [Q(0)] * n
            for (a, b), j in unk.items(): v[j] = c({a: Q(1)}, {b: Q(1)})
            vs.append(v)
        out["rank of tests mod B (inner)"] = rank(proj(Bd) + proj(vs)) - rb
        out["tests identically zero on window"] = [all(x == 0 for x in v) for v in vs]
    return out

if __name__ == "__main__":
    N, M = int(sys.argv[1]), int(sys.argv[2])
    print("backend", backend())
    for label, mo, dz in (("k^L'(1|4)", False, True), ("k^L(1|4)", False, False), ("k^M(1|4)", True, False)):
        t = time.time()
        tests = (("(1)", c1), ("(2)", c2), ("(3)", c3)) if dz else (("(2)", c2), ("(3)", c3))
        print(label, run(N, M, mo, dz, test=tests), "%.0fs" % (time.time() - t), flush=True)
