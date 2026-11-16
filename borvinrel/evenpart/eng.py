"""g(A) for an arbitrary (non-symmetrizable allowed) Cartan matrix A over F_p,
generators of parities par.  n_+ built by multidegree; an element of height>=2 is
identified with its lowering vector (ad f_i u)_i (injective on g(A)).
General bracket of n_+ elements by the derivation rule for ad f_i.
Conventions: [h_i,e_j]=A_ij e_j, [e_i,f_j]=delta_ij h_i."""
import sys
sys.setrecursionlimit(10000)

def inv(x, p): return pow(x % p, p - 2, p)

class G:
    def __init__(s, A, par, p):
        s.A = [[x % p for x in r] for r in A]; s.par = par; s.p = p; s.n = len(A)
        s.dim = {}; s.low = {}; s.solver = {}; s.memo = {}
        for i in range(s.n):
            s.dim[s.unit(i)] = 1
    def unit(s, i): return tuple(1 if t == i else 0 for t in range(s.n))
    def ht(s, a): return sum(a)
    def pa(s, a): return sum(a[t] * s.par[t] for t in range(s.n)) % 2
    def ev(s, a, i): return sum(a[k] * s.A[i][k] for k in range(s.n)) % s.p
    def add(s, a, b): return tuple(x + y for x, y in zip(a, b))
    def sub(s, a, i): return tuple(a[t] - (t == i) for t in range(s.n))
    # ad f_i of basis element (a,b): returns ('h', c) meaning c*h_i, or vector in g_{a-e_i}, or None
    def fi(s, i, a, b):
        p = s.p
        if s.ht(a) == 1:
            if a[i] == 1:
                return ('h', (-1 if s.par[i] == 0 else 1) % p)
            return None
        return s.low[a][b].get(i)
    def lowvec(s, a, b, beta, w):
        """lowering vector (dict i->vec in g_{a+beta-e_i}) of [u_{a,b}, v_{beta,w}]"""
        p = s.p; gam = s.add(a, beta); out = {}
        for i in range(s.n):
            g = s.sub(gam, i)
            if min(g) < 0 or s.dim.get(g, 0) == 0:
                continue
            d = s.dim[g]; vec = [0] * d
            # [f_i u, v]
            fu = s.fi(i, a, b)
            if fu is not None:
                if isinstance(fu, tuple):
                    c = fu[1] * s.ev(beta, i) % p
                    if c: vec[w] = (vec[w] + c) % p   # g == beta here
                else:
                    a1 = s.sub(a, i)
                    for k, c in enumerate(fu):
                        if c:
                            r = s.br(a1, k, beta, w)
                            for t, x in enumerate(r): vec[t] = (vec[t] + c * x) % p
            sg = -1 if (s.par[i] and s.pa(a)) else 1
            fv = s.fi(i, beta, w)
            if fv is not None:
                if isinstance(fv, tuple):
                    # [u, c h_i] = -c [h_i,u] = -c a(h_i) u
                    c = (-fv[1] * s.ev(a, i)) % p
                    if c: vec[b] = (vec[b] + sg * c) % p
                else:
                    b1 = s.sub(beta, i)
                    for k, c in enumerate(fv):
                        if c:
                            r = s.br(a, b, b1, k)
                            for t, x in enumerate(r): vec[t] = (vec[t] + sg * c * x) % p
            out[i] = vec
        return out
    def flat(s, gam, L):
        v = []
        for i in range(s.n):
            g = s.sub(gam, i)
            if min(g) < 0 or s.dim.get(g, 0) == 0: continue
            v += L.get(i, [0] * s.dim[g])
        return v
    def br(s, a, b, beta, w):
        key = (a, b, beta, w)
        if key in s.memo: return s.memo[key]
        gam = s.add(a, beta)
        d = s.dim.get(gam, 0 if s.ht(gam) <= getattr(s, "H", 0) else None)
        if d is None: raise RuntimeError("degree not built %s" % (gam,))
        if d == 0:
            r = []
        else:
            r = s.coords(gam, s.flat(gam, s.lowvec(a, b, beta, w)))
        s.memo[key] = r
        return r
    def coords(s, gam, v):
        # solve sum x_k basisrow_k = v
        P, piv, Tm = s.solver[gam]
        p = s.p
        # reduce v by the stored echelon (rows in terms of basis combos)
        v = v[:]; comb = [0] * s.dim[gam]
        for (col, row, cb) in P:
            c = v[col]
            if c:
                for t in range(len(v)): v[t] = (v[t] - c * row[t]) % p
                for t in range(len(comb)): comb[t] = (comb[t] + c * cb[t]) % p
        assert not any(v), "vector not in span"
        return comb
    def build_height(s, h):
        p = s.p; cands = {}
        for a in [x for x in s.dim if s.ht(x) == h - 1 and s.dim[x] > 0]:
            for j in range(s.n):
                cands.setdefault(s.add(a, s.unit(j)), []).append((j, a))
        for gam in sorted(cands):
            rows = []; P = []; basis = []
            s.dim[gam] = 0
            # temporarily set dim so blocks are known; lowering lands in lower degrees only
            for (j, a) in cands[gam]:
                for b in range(s.dim[a]):
                    L = s.lowvec(s.unit(j), 0, a, b)
                    v = s.flat(gam, L)
                    # reduce
                    r = v[:]; cb = [0] * 0
                    cbv = {}
                    for (col, row, cbr) in P:
                        c = r[col]
                        if c:
                            for t in range(len(r)): r[t] = (r[t] - c * row[t]) % p
                    nz = [t for t, x in enumerate(r) if x]
                    if nz:
                        basis.append(L)
                        P.append((nz[0], r, None))
                        # normalize
                        k = len(basis) - 1
                        P[-1] = (nz[0], [x * inv(r[nz[0]], p) % p for x in r], k)
            d = len(basis)
            s.dim[gam] = d
            if d == 0: continue
            s.low[gam] = basis
            # build solver: echelon with combos
            S = []
            for k, L in enumerate(basis):
                v = s.flat(gam, L); cb = [0] * d; cb[k] = 1
                for (col, row, cbr) in S:
                    c = v[col]
                    if c:
                        v = [(x - c * y) % p for x, y in zip(v, row)]
                        cb = [(x - c * y) % p for x, y in zip(cb, cbr)]
                col = next(t for t, x in enumerate(v) if x)
                iv = inv(v[col], p)
                v = [x * iv % p for x in v]; cb = [x * iv % p for x in cb]
                # eliminate col from previous rows
                S2 = []
                for (c2, r2, b2) in S:
                    c = r2[col]
                    if c:
                        r2 = [(x - c * y) % p for x, y in zip(r2, v)]
                        b2 = [(x - c * y) % p for x, y in zip(b2, cb)]
                    S2.append((c2, r2, b2))
                S = S2 + [(col, v, cb)]
            s.solver[gam] = (S, None, None)
        # mark zero dims for all neighbors at this height
        for a in [x for x in list(s.dim) if s.ht(x) == h - 1 and s.dim[x] > 0]:
            for j in range(s.n):
                s.dim.setdefault(s.add(a, s.unit(j)), 0)
    def build(s, H):
        s.H = H
        for h in range(2, H + 1):
            s.build_height(h)
    def totals(s, H):
        return [sum(d for a, d in s.dim.items() if s.ht(a) == h) for h in range(1, H + 1)]
