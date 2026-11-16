#!/usr/bin/env python3
"""The 15-dim factor (112 2-nilpotent, 41 toral): Jacobi, absolute simplicity via the
multiplication algebra (dim 225 = End F_2^15 => absolutely irreducible by Burnside),
brute-force recount, and comparison with h_I(2;N) = {d1 f d1 + d2 f d2} (non-alternating
Hamiltonian fields) and h_Pi(2;N) = {d2 f d1 + d1 f d2} inside W(2;N)."""
import glob, itertools, json, numpy as np
from lie import *
from gf2 import mm, rank, Echelon
import ambients as A
import ident

def jacobi(ad):
    d = ad.shape[0]
    for i in range(d):
        assert not ad[i][:, i].any()
        for j in range(d):
            assert (ad[i][:, j] == ad[j][:, i]).all()
    # ad[e_i] ad[e_j] - ad[e_j] ad[e_i] = ad([e_i,e_j])
    for i in range(d):
        for j in range(d):
            c = ad[i][:, j]
            lhs = mm(ad[i], ad[j]) ^ mm(ad[j], ad[i])
            rhs = (ad[np.nonzero(c)[0]].sum(0) & 1).astype(np.uint8) if c.any() else np.zeros_like(lhs)
            if (lhs ^ rhs).any(): return False
    return True

def multdim(ad):
    d = ad.shape[0]
    E = Echelon(d * d)
    E.add(np.eye(d, dtype=np.uint8).reshape(1, -1))
    gens = [ad[i] for i in range(d)]
    new = E.add(np.array([g.reshape(-1) for g in gens]))
    while len(new):
        prods = [mm(g, v.reshape(d, d)).reshape(-1) for v in new for g in gens]
        new = E.add(np.array(prods))
    return E.dim()

def brute(ad):
    d = ad.shape[0]; a = t = 0
    for m in range(1 << d):
        idx = [k for k in range(d) if m >> k & 1]
        X = (ad[idx].sum(0) & 1).astype(np.uint8) if idx else np.zeros((d, d), np.uint8)
        X2 = mm(X, X)
        a += not X2.any(); t += not (X2 ^ X).any()
    return a, t

def ham(Ns, kind):
    amb = A.wittN(Ns); m = 2; N = amb.N
    gens = []
    for f in range(N):
        v = np.zeros(N * N, np.uint8)
        for i in range(2):
            j = i if kind == 'I' else 1 - i
            b = A_d(Ns, f, j)
            if b is not None: v ^= amb.basis[b * m + i]
        if v.any(): gens.append(v)
    E = Echelon(amb.L); E.add(np.array(gens))   # the span {H_f}
    return amb, E

def A_d(Ns, a, i):
    off = sum(Ns[:i]); fld = (a >> off) & ((1 << Ns[i]) - 1)
    return None if fld == 0 else a - (1 << off)

if __name__ == '__main__':
    for p in glob.glob('found/*.npz'):
        z = np.load(p)
        if str(z['fp']).startswith('15:1:False:simple:14'):
            ad = z['ad']; print('file', p)
    print('Jacobi', jacobi(ad), ' dim [L,L]', len(derived(ad)), ' dim mult. algebra', multdim(ad), 'of', 15 * 15)
    print('brute (2-nilp, toral) incl. 0:', brute(ad), ' numba:', ident.inv(ad))
    rng = np.random.default_rng(1)
    for Ns in ((2, 1), (2, 2), (3, 1), (3, 2), (2, 3), (3, 3)):
        for kind in ('I', 'Pi'):
            amb, E = ham(Ns, kind)
            spn = E.dim()
            E2 = closure(amb, list(E.E), 10 ** 6)
            ad2 = table_of(amb, E2); out = []; ab = factors(ad2, rng, out)
            desc = []
            for o in out:
                d = o[0].shape[0]
                desc.append((d, o[1], restricted(o[0]), ident.inv(o[0]) if d <= 31 else None))
            print('h_%s(2;%s): span{H_f} %d, closed %s, generated dim %d, abelian factors %d, simple factors %s'
                  % (kind, Ns, spn, spn == E2.dim(), E2.dim(), ab, desc), flush=True)
