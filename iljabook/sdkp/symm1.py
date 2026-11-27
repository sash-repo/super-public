"""Symmetries of order <= 1 (generating functions affine in the first x- and
t-jets, polynomial in x,y,t and the components), even and odd, of the super
dKP system; blocks by the two scaling weights; parallel."""
import sys, os, itertools, pickle, time; sys.path.insert(0, '.'); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'lib'))
from fractions import Fraction as F
from multiprocessing import Pool
from symm import *

def jetvar(c, k): return (c, 1, 0) if k == 'x' else (c, 0, 1)

def basis1(sigma, cdeg, udeg_eta, udeg_jet):
    names, ns = weights(); idx = {n: i for i, n in enumerate(names)}
    def W(ce, u, extra):
        return tuple(sum(a * v[idx[k]] for k, a in ce.items()) + sum(v[idx[c]] for c in u) + extra(v) for v in ns)
    B = []
    cm = coordmonos(cdeg)
    for comp in COMPS:
        target = (P_[comp] + sigma) % 2
        for ce in cm:
            for u in umonos(udeg_eta):
                if sum(P_[c] for c in u) % 2 != target: continue
                B.append((('eta', comp, tuple(sorted(ce.items())), u, None),
                          {comp: qmono(ce, u)}, W(ce, u, lambda v: -v[idx[comp]])))
            for c2 in COMPS:
                for k in ('x', 't'):
                    for u in umonos(udeg_jet):
                        if (sum(P_[c] for c in u) + P_[c2]) % 2 != target: continue
                        q = qmono(ce, u) * Q.var(jetvar(c2, k))
                        B.append((('jet', comp, tuple(sorted(ce.items())), u, (c2, k)),
                                  {comp: q}, W(ce, u, lambda v: v[idx[c2]] - v[idx[k]] - v[idx[comp]])))
    # xi^y terms (the y-derivative replaced by the equation)
    for ce in cm:
        us = [()] if sigma == 0 else [(c,) for c in COMPS if P_[c]]
        for u in us:
            xi = qmono(ce, u)
            phi = {comp: -(xi * RHS[comp]) for comp in COMPS}
            B.append((('xiy', None, tuple(sorted(ce.items())), u, None), phi,
                      W(ce, u, lambda v: -v[idx['y']])))
    return B

SIG = None
def work(args):
    key, items, sigma = args
    conds = [condition(phi, sigma) for _, phi in items]
    rowkeys = sorted({k for c in conds for k in c}, key=str)
    if not rowkeys: return key, [[(lab, F(1))] for lab, _ in items]
    ri = {k: i for i, k in enumerate(rowkeys)}
    M = [[F(0)] * len(items) for _ in rowkeys]
    for j, c in enumerate(conds):
        for k, v in c.items(): M[ri[k]][j] = v
    return key, [[(items[j][0], v) for j, v in enumerate(vec) if v] for vec in nullspace(M)]

def main(sigma, cdeg, ue, uj):
    global SIG; SIG = sigma
    B = basis1(sigma, cdeg, ue, uj)
    blocks = {}
    for lab, phi, w in B: blocks.setdefault(w, []).append((lab, phi))
    print(f'sigma={sigma}: {len(B)} ansatz terms in {len(blocks)} blocks, largest {max(map(len, blocks.values()))}', flush=True)
    t0 = time.time()
    with Pool(9) as pool:
        res = pool.map(work, [(k, v, sigma) for k, v in sorted(blocks.items(), key=lambda kv: -len(kv[1]))], chunksize=1)
    sols = [(k, s) for k, ss in res for s in ss]
    print(f'  {len(sols)} solutions, {time.time()-t0:.0f}s', flush=True)
    pickle.dump(sols, open(f'sols_s{sigma}_c{cdeg}_{ue}{uj}.pkl', 'wb'))
    return sols

def pr(sol):
    out = []
    for lab, c in sol:
        kind, comp, ce, u, j = lab
        co = '*'.join(f'{k}^{a}' if a > 1 else k for k, a in ce if a)
        f = '*'.join([co] if co else [] + []) 
        mon = '*'.join(([co] if co else []) + list(u) + ([f'{j[0]}_{j[1]}'] if j else [])) or '1'
        out.append(f'{c}*{"xi^y" if kind=="xiy" else "phi["+comp+"]"}({mon})')
    return ' + '.join(out)

if __name__ == '__main__':
    sigma, cdeg, ue, uj = map(int, sys.argv[1:5])
    sols = main(sigma, cdeg, ue, uj)
    for k, s in sorted(sols, key=lambda z: z[0]): print(' ', [str(a) for a in k], pr(s))
