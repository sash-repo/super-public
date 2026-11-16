"""Invariants of a Lie superalgebra over F_2 given by (tab, sq, par) -- numpy, exact mod 2."""
import numpy as np, itertools
from f2lin import rref, rank, nullspace

def span(vecs):
    vecs = [np.asarray(v, dtype=np.uint8) & 1 for v in vecs]
    if not vecs: return np.zeros((0, 0), dtype=np.uint8)
    return rref(np.array(vecs))[0]

def bracket(tab, x, y):
    return (np.einsum('i,j,ijk->k', x.astype(np.int64), y.astype(np.int64), tab.astype(np.int64)) & 1).astype(np.uint8)

def square(tab, sq, par, x):
    """x odd: x^2 = sum x_i^2 s(e_i) + sum_{i<j} x_i x_j [e_i,e_j]"""
    x = x.astype(np.int64); out = (x @ sq.astype(np.int64))
    idx = np.nonzero(x)[0]
    for a, b in itertools.combinations(idx, 2): out = out + tab[a, b]
    return (out & 1).astype(np.uint8)

def ideal(tab, sq, par, gens):
    d = len(par); basis = span(gens)
    while True:
        new = [bracket(tab, np.eye(d, dtype=np.uint8)[i], v) for v in basis for i in range(d)]
        new += [square(tab, sq, par, v) for v in basis if all(par[np.nonzero(v)[0]] == 1) and v.any()]
        nb = span(list(basis) + new)
        if len(nb) == len(basis): return basis
        basis = nb

def homog_basis(basis, par):
    """split a graded subspace given by a basis into its even and odd parts (dims)"""
    ev = rank(np.array([v * (par == 0) for v in basis])) if len(basis) else 0
    od = rank(np.array([v * (par == 1) for v in basis])) if len(basis) else 0
    return ev, od

def is_simple(tab, sq, par):
    d = len(par)
    for p in (0, 1):
        idx = np.nonzero(par == p)[0]
        for c in itertools.product([0, 1], repeat=len(idx)):
            if not any(c): continue
            x = np.zeros(d, dtype=np.uint8); x[idx] = c
            if len(ideal(tab, sq, par, [x])) < d: return False, x
    return True, None

def derived(tab, par):
    d = len(par)
    return span([tab[i, j] for i in range(d) for j in range(d)])

def center(tab):
    d = tab.shape[0]
    M = tab.transpose(1, 2, 0).reshape(d * d, d)     # rows (j,k), columns i : sum_i c_i tab[i,j,k]
    return nullspace(M, d)

def adm(tab, i):
    return tab[i].T.copy()    # column j = [e_i, e_j]

def assoc_dim(tab):
    d = tab.shape[0]; gens = [adm(tab, i) for i in range(d)]
    E = [np.eye(d, dtype=np.uint8).reshape(-1)]; B = rref(np.array(E))[0]; queue = [np.eye(d, dtype=np.uint8)]
    while queue:
        X = queue.pop()
        for M in gens:
            Y = (M.astype(np.int64) @ X) & 1
            t = rref(np.vstack([B, Y.reshape(1, -1).astype(np.uint8)]))[0]
            if len(t) > len(B): B = t; queue.append(Y.astype(np.uint8))
    return len(B)

def derivations(tab, sq, par):
    """even and odd derivations: D[a,b] = [Da,b] + [a,Db] (char 2 signs vanish); even D also
    D(s(a)) = [a, Da] for odd a; odd D: s(a) has no odd-derivation condition beyond D[a,a]? For odd D
    and odd a, D(a^2) = [Da, a]."""
    d = len(par); res = {}
    for pD in (0, 1):
        idx = [(l, k) for l in range(d) for k in range(d) if (par[l] + par[k]) % 2 == pD]
        cols = []
        for (l, k) in idx:
            P = np.zeros((d, d), dtype=np.int64); P[l, k] = 1
            T = tab.astype(np.int64)
            E = np.einsum('abl,lk->abk', T, P) + np.einsum('al,lbk->abk', P, T) + np.einsum('bl,alk->abk', P, T)
            S = sq.astype(np.int64) @ P + np.einsum('al,alk->ak', P, T)
            S = S[par == 1]
            cols.append(np.concatenate([E.reshape(-1), S.reshape(-1)]) & 1)
        M = np.array(cols, dtype=np.uint8).T
        res[pD] = len(idx) - rank(M)
    return res

def inner_sdim(tab, par):
    d = len(par)
    ev = rank(np.array([adm(tab, i).reshape(-1) for i in range(d) if par[i] == 0]))
    od = rank(np.array([adm(tab, i).reshape(-1) for i in range(d) if par[i] == 1]))
    return ev, od

def restricted(tab, par):
    """is ad(x)^2 in ad(s_ev) for every even basis x?"""
    d = len(par); ev = [i for i in range(d) if par[i] == 0]
    A = rref(np.array([adm(tab, i).reshape(-1) for i in ev]))[0]
    bad = 0
    for i in ev:
        M = adm(tab, i).astype(np.int64); M2 = ((M @ M) & 1).astype(np.uint8)
        if rank(np.vstack([A, M2.reshape(1, -1)])) > len(A): bad += 1
    return bad == 0, bad

def invariant_forms(tab, sq, par):
    """symmetric bilinear B (char 2) with B([x,y],z)=B(x,[y,z]) all basis triples, B(x^2,z)=B(x,[x,z])
    for odd basis x.  Returns dims (even forms, odd forms) and whether a nondegenerate one exists (F_2)."""
    d = len(par); out = {}
    for pB in (0, 1):
        idx = [(a, b) for a in range(d) for b in range(a, d) if (par[a] + par[b]) % 2 == pB]
        pos = {k: n for n, k in enumerate(idx)}
        def v(a, b):
            a, b = min(a, b), max(a, b); return pos.get((a, b))
        rows = []
        for x in range(d):
            for y in range(d):
                for z in range(d):
                    r = np.zeros(len(idx), dtype=np.uint8)
                    for l in np.nonzero(tab[x, y])[0]:
                        p = v(l, z)
                        if p is not None: r[p] ^= 1
                    for l in np.nonzero(tab[y, z])[0]:
                        p = v(x, l)
                        if p is not None: r[p] ^= 1
                    if r.any(): rows.append(r)
        for x in np.nonzero(par == 1)[0]:
            for z in range(d):
                r = np.zeros(len(idx), dtype=np.uint8)
                for l in np.nonzero(sq[x])[0]:
                    p = v(l, z)
                    if p is not None: r[p] ^= 1
                for l in np.nonzero(tab[x, z])[0]:
                    p = v(x, l)
                    if p is not None: r[p] ^= 1
                if r.any(): rows.append(r)
        N = nullspace(np.array(rows), len(idx)) if rows else np.eye(len(idx), dtype=np.uint8)
        mats = []
        for n in N:
            G = np.zeros((d, d), dtype=np.uint8)
            for (a, b), c in zip(idx, n):
                if c: G[a, b] = G[b, a] = 1
            mats.append(G)
        nondeg = None
        if len(mats) <= 16:
            nondeg = False
            for c in itertools.product([0, 1], repeat=len(mats)):
                if not any(c): continue
                G = sum(ci * M for ci, M in zip(c, mats)) & 1
                if rank(G) == d: nondeg = True; break
        maxrank = max([rank(M) for M in mats], default=0)
        out[pB] = (len(mats), nondeg, maxrank, mats)
    return out

def central_ext(tab, sq, par):
    """H^2(s; K) for K even (cochains (c,q): c symmetric on s_ev x s_ev (alternating) and on s_od x s_od,
    q quadratic on s_od polar to c) and K odd (c on s_ev x s_od, q = 0)."""
    d = len(par); res = {}
    for pK in (0, 1):
        var = []
        for a in range(d):
            for b in range(a, d):
                if (par[a] + par[b]) % 2 != pK: continue
                if a == b and (par[a] == 0 or pK == 1): continue
                var.append((a, b))
        pos = {k: n for n, k in enumerate(var)}
        def cv(a, b):
            if a == b: return None if par[a] == 0 else pos.get((a, a))   # (a,a) odd = q(e_a); c(a,a)=0 anyway
            return pos.get((min(a, b), max(a, b)))
        def cvec(a, w):   # c(e_a, w) as row
            r = np.zeros(len(var), dtype=np.uint8)
            for l in np.nonzero(w)[0]:
                if l == a: continue
                p = cv(a, l)
                if p is not None: r[p] ^= 1
            return r
        rows = []
        for x, y, z in itertools.combinations(range(d), 3):
            r = cvec(x, tab[y, z]) ^ cvec(y, tab[z, x]) ^ cvec(z, tab[x, y])
            if r.any(): rows.append(r)
        for x in np.nonzero(par == 1)[0]:
            for y in range(d):
                r = cvec(y, sq[x]) ^ cvec(x, tab[x, y])
                # c(s(x), y): cvec(y, sq[x]) gives c(y, l) for l in s(x), l != y; if y in s(x), c(y,y) term:
                if sq[x][y] and par[y] == 1 and pK == 0 and cv(y, y) is not None:
                    pass   # c(y,y) = 0 (bracket part), not q
                if r.any(): rows.append(r)
        Z = nullspace(np.array(rows), len(var)) if rows else np.eye(len(var), dtype=np.uint8)
        # coboundaries: f: s -> K of parity pK : c(a,b) = f([a,b]), q(a) = f(a^2)
        Bv = []
        for l in range(d):
            if par[l] != pK: continue
            r = np.zeros(len(var), dtype=np.uint8)
            for (a, b), n in pos.items():
                val = sq[a][l] if a == b else tab[a, b][l]
                r[n] = val
            Bv.append(r)
        rB = rank(np.array(Bv)) if Bv else 0
        res[pK] = len(Z) - rB
    return res

def odd_null_squares(tab, sq, par):
    d = len(par); idx = np.nonzero(par == 1)[0]; n = 0
    for c in itertools.product([0, 1], repeat=len(idx)):
        if not any(c): continue
        x = np.zeros(d, dtype=np.uint8); x[idx] = c
        if not square(tab, sq, par, x).any(): n += 1
    return n

def adjoint_submodules_desuper(tab, par):
    """ideals of the desuperization (Lie algebra: bracket only, all elements), minimal ones via
    ideals generated by single elements over F_2: returns sorted list of distinct dims"""
    d = len(par); dims = set()
    E = np.eye(d, dtype=np.uint8)
    for c in itertools.product([0, 1], repeat=d):
        if not any(c): continue
        x = np.array(c, dtype=np.uint8); B = span([x])
        while True:
            nb = span(list(B) + [bracket(tab, E[i], v) for v in B for i in range(d)])
            if len(nb) == len(B): break
            B = nb
        dims.add(len(B))
    return sorted(dims)
