#!/usr/bin/env python3
"""Invariant forms on the Buttin family b_{a,b}(n;1) over F_p.

Everything is graded and sparse:
  * elements are generating functions f (dicts on the monomials of O(n|n+1)),
    M_f the pericontact field; [M_f,M_g] is computed on fields and read back
    as a function through a sparse echelon of the M_u;
  * the kernel of Div_{a,b} is split into its (degree, parity) layers;
  * the perfect part is computed layer by layer, stopping a layer as soon as
    it is full;
  * a NIS on a simple graded algebra pairs L_i with L_{d-i} (Lemma 1.5 of
    KLLS), so the graded sdims are tested for symmetry first;
  * the invariant forms of a candidate degree d are solved from the
    invariance under a generating set only.
Usage: nisvect.py p n a b
msimple.py uses its sparse echelon SpEch and the helper lin.
"""
import sys, time
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from mvect import make, pericontact
from exactlin import nullspace


def log(*a):
    print(*a, flush=True)


class SpEch:
    """sparse reduced echelon mod p; rows are dicts, with optional payload."""

    def __init__(self, p):
        self.p, self.rows = p, {}            # pivot -> (row, payload)

    def reduce(self, v, pay=None):
        p = self.p
        v = dict(v)
        pay = dict(pay) if pay is not None else None
        changed = True
        while changed:
            changed = False
            for k in list(v):
                if k in v and k in self.rows:
                    c = v[k]
                    r, rp = self.rows[k]
                    for kk, cc in r.items():
                        w = (v.get(kk, 0) - c * cc) % p
                        if w:
                            v[kk] = w
                        else:
                            v.pop(kk, None)
                    if pay is not None:
                        for kk, cc in rp.items():
                            w = (pay.get(kk, 0) - c * cc) % p
                            if w:
                                pay[kk] = w
                            else:
                                pay.pop(kk, None)
                    changed = True
        return v, pay

    def add(self, v, pay=None):
        if pay is None:
            pay = {}
        v, pay = self.reduce(v, pay)
        if not v:
            return False
        p = self.p
        k = min(v)
        inv = pow(v[k], p - 2, p)
        v = {kk: c * inv % p for kk, c in v.items()}
        if pay is not None:
            pay = {kk: c * inv % p for kk, c in pay.items()}
        # keep it reduced against the new pivot
        for kk, (r, rp) in list(self.rows.items()):
            if k in r:
                c = r[k]
                for a, b in v.items():
                    w = (r.get(a, 0) - c * b) % p
                    if w:
                        r[a] = w
                    else:
                        r.pop(a, None)
                if rp is not None:
                    for a, b in pay.items():
                        w = (rp.get(a, 0) - c * b) % p
                        if w:
                            rp[a] = w
                        else:
                            rp.pop(a, None)
        self.rows[k] = (v, pay if pay is not None else {})
        return True

    @property
    def dim(self):
        return len(self.rows)


def lin(S, fields, f):
    X = {}
    for u, c in f.items():
        if u in fields:
            X = S.add(X, fields[u], c)
    return X


def main(p, n, a, b):
    t0 = time.time()
    S = make(p, n)
    _, fields = pericontact(S, n)
    mons = S.mons
    # field -> function
    F2f = SpEch(p)
    for u, X in fields.items():
        F2f.add(X, {u: 1})
    assert F2f.dim == len(fields), "f -> M_f not injective"

    def tofun(X):
        r, pay = F2f.reduce(X, {})
        assert not r, "bracket left the algebra"
        return {u: (-c) % p for u, c in pay.items()}

    def br(f, g):
        return tofun(S.bracket(lin(S, fields, f), lin(S, fields, g)))

    # Div_{a,b} on functions (mvect.buttin, eq. (2.7.3) of BKLS)
    idx = {u: i for i, u in enumerate(mons)}
    M = [[0] * len(mons) for _ in mons]
    for j, u in enumerate(mons):
        w, c = S.gen_act(('xi', 0), u)
        if c:
            aa, bb = w
            e = sum(aa) + sum(bb[1:])
            co = (b * n - a * e) * c % p
            if co:
                M[idx[w]][j] = (M[idx[w]][j] + co) % p
        for i in range(n):
            w1, c1 = S.gen_act(('x', i, 0), u)
            if not c1:
                continue
            w2, c2 = S.gen_act(('xi', 1 + i), w1)
            if c2:
                M[idx[w2]][j] = (M[idx[w2]][j] - a * c1 * c2) % p
    ker = nullspace(M, p=p) if any(any(r) for r in M) else \
        [[int(i == k) for i in range(len(mons))] for k in range(len(mons))]

    def key(u):          # (Z-degree of M_u, parity of M_u)
        return (S.deg_mon(u) - 2, (S.par_mon(u) + 1) % 2)

    layers = {}
    for v in ker:
        for j, c in enumerate(v):
            c = int(c) % p
            if c and mons[j] in fields:
                layers.setdefault(key(mons[j]), {})
        parts = {}
        for j, c in enumerate(v):
            c = int(c) % p
            if c and mons[j] in fields:
                parts.setdefault(key(mons[j]), {})[mons[j]] = c
        for k, f in parts.items():
            layers.setdefault(k, None)
            E = layers.get(('E',) + k)
            if E is None:
                E = layers[('E',) + k] = SpEch(p)
            E.add(f)
    L = {k: [r for r, _ in layers[('E',) + k].rows.values()]
         for k in layers if k[0] != 'E' and ('E',) + k in layers}
    L = {k: v for k, v in L.items() if v}
    tot = sum(len(v) for v in L.values())
    log(f"b_({a}:{b})({n};1) p={p}: kernel dim {tot}  [{time.time()-t0:.0f}s]")

    # perfect part
    it = 0
    while True:
        it += 1
        D = {k: SpEch(p) for k in L}
        keys = sorted(L)
        for i, k1 in enumerate(keys):
            for k2 in keys[i:]:
                t = (k1[0] + k2[0], (k1[1] + k2[1]) % 2)
                if t not in L or D[t].dim == len(L[t]):
                    continue
                for x in L[k1]:
                    if D[t].dim == len(L[t]):
                        break
                    for y in L[k2]:
                        if D[t].dim == len(L[t]):
                            break
                        z = br(x, y)
                        if z:
                            D[t].add(z)
        newL = {k: [r for r, _ in D[k].rows.values()] for k in L}
        newL = {k: v for k, v in newL.items() if v}
        nt = sum(len(v) for v in newL.values())
        log(f"  derived step {it}: dim {nt}  [{time.time()-t0:.0f}s]")
        if nt == tot:
            break
        L, tot = newL, nt
    degs = sorted({k[0] for k in L})
    sd = {d: (len(L.get((d, 0), [])), len(L.get((d, 1), []))) for d in degs}
    ev = sum(x[0] for x in sd.values()); od = sum(x[1] for x in sd.values())
    log(f"  perfect part sdim ({ev}|{od}); graded: " +
        " ".join(f"{d}:({sd[d][0]}|{sd[d][1]})" for d in degs))
    # symmetry test
    lo, hi = degs[0], degs[-1]
    d = lo + hi
    cand = []
    for par in (0, 1):
        ok = all(sd.get(i, (0, 0))[e] == sd.get(d - i, (0, 0))[(e + par) % 2]
                 for i in range(lo, hi + 1) for e in (0, 1))
        if ok:
            cand.append(par)
    log(f"  RESULT b_({a}:{b})({n};1) p={p}: sdim=({ev}|{od}) "
        f"symmetric-parities={[['even','odd'][c] for c in cand]}  [{time.time()-t0:.0f}s]")
    if not cand or '--forms' not in sys.argv:
        return
    forms(p, L, br, d, cand, t0)


def forms(p, L, br, d, cand, t0):
    """invariant forms of total degree d, from the invariance under generators."""
    keys = sorted(L)
    basis = [(k, j) for k in keys for j in range(len(L[k]))]
    E = {k: SpEch(p) for k in keys}
    for k in keys:
        for j, f in enumerate(L[k]):
            E[k].add(f, {j: 1})

    def coords(k, f):
        r, pay = E[k].reduce(f, {})
        assert not r
        return {j: (-c) % p for j, c in pay.items()}

    # generators: lowest layers up to degree 1, checked to generate
    gens = [kk for kk in keys if kk[0] <= 1]
    G = {k: SpEch(p) for k in keys}
    for k in gens:
        for f in L[k]:
            G[k].add(f)
    for dd in sorted({k[0] for k in keys}):
        for k in [kk for kk in keys if kk[0] == dd]:
            for k1 in keys:
                for k2 in keys:
                    if (k1[0] + k2[0], (k1[1] + k2[1]) % 2) != k or k1[0] >= dd or k2[0] >= dd:
                        continue
                    if G[k].dim == len(L[k]):
                        break
                    for x in [r for r, _ in G[k1].rows.values()]:
                        for y in [r for r, _ in G[k2].rows.values()]:
                            if G[k].dim == len(L[k]):
                                break
                            z = br(x, y)
                            if z:
                                G[k].add(z)
    assert all(G[k].dim == len(L[k]) for k in keys), "degrees <=1 do not generate"
    par = {k: k[1] for k in keys}
    for fp in cand:
        unk = [(k1, i, k2, j) for k1 in keys for k2 in keys
               if k1[0] + k2[0] == d and (k1[1] + k2[1]) % 2 == fp
               for i in range(len(L[k1])) for j in range(len(L[k2]))]
        uix = {u: n for n, u in enumerate(unk)}
        rows = []
        for kg in gens:
            for g in L[kg]:
                for ky in keys:
                    for kz in keys:
                        if kg[0] + ky[0] + kz[0] != d or (kg[1] + ky[1] + kz[1]) % 2 != fp:
                            continue
                        kgy = (kg[0] + ky[0], (kg[1] + ky[1]) % 2)
                        kgz = (kg[0] + kz[0], (kg[1] + kz[1]) % 2)
                        sg = (-1) ** (kg[1] * ky[1])
                        for i, y in enumerate(L[ky]):
                            cy = coords(kgy, br(g, y)) if kgy in L else {}
                            for j, z in enumerate(L[kz]):
                                cz = coords(kgz, br(g, z)) if kgz in L else {}
                                row = {}
                                for t, c in cy.items():
                                    n_ = uix[(kgy, t, kz, j)]
                                    row[n_] = (row.get(n_, 0) + c) % p
                                for t, c in cz.items():
                                    n_ = uix[(ky, i, kgz, t)]
                                    row[n_] = (row.get(n_, 0) + sg * c) % p
                                row = {a: c for a, c in row.items() if c}
                                if row:
                                    rows.append(row)
        Ech = SpEch(p)
        for r in rows:
            Ech.add(r)
        dimf = len(unk) - Ech.dim
        log(f"  invariant forms of degree {d}, parity {['even','odd'][fp]}: {dimf} "
            f"(unknowns {len(unk)})  [{time.time()-t0:.0f}s]")
        if dimf and '--rank' in sys.argv:
            from exactlin import rank
            free = [u for u in range(len(unk)) if u not in Ech.rows]
            pos = {bk: n_ for n_, bk in enumerate(basis)}
            Nn = len(basis)
            for fv in free:
                x = {fv: 1}
                for pv, (r, _) in Ech.rows.items():
                    if fv in r:
                        x[pv] = (-r[fv]) % p
                Gm = [[0] * Nn for _ in range(Nn)]
                for u, c in x.items():
                    k1, i, k2, j = unk[u]
                    Gm[pos[(k1, i)]][pos[(k2, j)]] = c % p
                log(f"    form from free unknown {fv}: rank {rank(Gm, p)} of {Nn}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    p, n, a, b = map(int, sys.argv[1:5])
    main(p, n, a, b)
