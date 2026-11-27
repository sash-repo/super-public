"""Point symmetries (even and odd) of the super dKP system, by an ansatz
polynomial in (x,y,t) and in the undifferentiated components, solved
block by block in the scaling weights.  Exact over Q."""
import sys, os, itertools; sys.path.insert(0, '.'); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'lib'))
from fractions import Fraction as F
from jets import *
from exactlin import nullspace, rank
import jets

RHS = build()
COMPS = sorted(RHS)                       # U10 U11 ... U41
P_ = {c: jets.PAR[c] for c in COMPS}

# ---- scaling weights: solve for weights making every equation homogeneous ----
def weights():
    # unknowns: wx, wy, wt (weights of the coordinates), w[c] for components
    names = ['x', 'y', 't'] + COMPS
    idx = {n: i for i, n in enumerate(names)}
    rows = []
    def wvec(m):
        v = [0] * len(names)
        ev, od = m
        for g, a in list(ev) + [(o, 1) for o in od]:
            if len(g) == 1: v[idx[g[0]]] += a
            else:
                v[idx[g[0]]] += a; v[idx['x']] -= a * g[1]; v[idx['t']] -= a * g[2]
        return v
    for c in COMPS:
        lhs = [0] * len(names); lhs[idx[c]] += 1; lhs[idx['y']] -= 1
        for m in RHS[c].t:
            rows.append([a - b for a, b in zip(wvec(m), lhs)])
    ns = nullspace(rows)
    return names, ns

def wt(vec, names, coordexp, umon, comp):
    # weight of the ansatz term coords^alpha * umon in eta_comp, as a generator
    idx = {n: i for i, n in enumerate(names)}
    w = -vec[idx[comp]] if comp else 0
    for k, a in coordexp.items(): w += a * vec[idx[k]]
    for c in umon: w += vec[idx[c]]
    return w

def umonos(maxdeg):
    out = [()]
    for d in range(1, maxdeg + 1):
        for combo in itertools.combinations_with_replacement(COMPS, d):
            odd = [c for c in combo if P_[c]]
            if len(odd) != len(set(odd)): continue
            out.append(combo)
    return out

def coordmonos(maxdeg):
    out = []
    for a in range(maxdeg + 1):
        for b in range(maxdeg + 1 - a):
            for c in range(maxdeg + 1 - a - b):
                out.append({'x': a, 'y': b, 't': c})
    return out

def qmono(coordexp, umon):
    q = Q.const(1)
    for k, a in coordexp.items():
        for _ in range(a): q = q * Q.var((k,))
    for c in umon: q = q * Q.var((c, 0, 0))
    return q

def X_apply(phi, q, sigma):
    """the evolutionary derivation with generating functions phi (dict comp->Q)"""
    cache = {}
    def act(v):
        if len(v) != 3: return None
        if v in cache: return cache[v]
        r = phi.get(v[0])
        if r is None: return None
        for _ in range(v[1]): r = Dx(r)
        for _ in range(v[2]): r = Dt(r)
        cache[v] = r; return r
    return derivation(q, act, sigma)

def condition(phi, sigma):
    out = {}
    for c in COMPS:
        e = Dy(phi.get(c, Q()), RHS) - X_apply(phi, RHS[c], sigma)
        for m, k in e.t.items(): out[(c, m)] = k
    return out

def basis(sigma, cdeg, udeg):
    """list of (label, phi) for the point ansatz of parity sigma"""
    B = []
    cm = coordmonos(cdeg); um = umonos(udeg)
    for comp in COMPS:
        target = (P_[comp] + sigma) % 2
        for ce in cm:
            for u in um:
                if sum(P_[c] for c in u) % 2 != target: continue
                B.append((('eta', comp, tuple(sorted(ce.items())), u), {comp: qmono(ce, u)}, (comp, ce, u)))
    # xi terms: xi^k(coords, odd-linear for sigma=1): phi_A -= xi^k * U_{A,k}
    for k in ('x', 'y', 't'):
        for ce in cm:
            us = [()] if sigma == 0 else [(c,) for c in COMPS if P_[c]]
            for u in us:
                xi = qmono(ce, u)
                phi = {}
                for comp in COMPS:
                    if k == 'x': d = Q.var((comp, 1, 0))
                    elif k == 't': d = Q.var((comp, 0, 1))
                    else: d = RHS[comp]
                    phi[comp] = -(xi * d)
                B.append((('xi', k, tuple(sorted(ce.items())), u), phi, ('xi', k, ce, u)))
    return B

def solve_blocks(sigma, cdeg, udeg, verbose=True):
    names, ns = weights()
    B = basis(sigma, cdeg, udeg)
    # weight of each basis element under every scaling in ns
    def wkey(meta):
        if meta[0] == 'xi':
            _, k, ce, u = meta
            # xi^k U_{A,k}: weight = w(xi) - w(k-coordinate) ... generator weight
            return tuple(wt(v, names, ce, u, None) - v[names.index(k)] for v in ns)
        comp, ce, u = meta
        return tuple(wt(v, names, ce, u, comp) for v in ns)
    blocks = {}
    for lab, phi, meta in B:
        blocks.setdefault(wkey(meta), []).append((lab, phi))
    sols = []
    for key, items in sorted(blocks.items()):
        conds = [condition(phi, sigma) for _, phi in items]
        rowkeys = sorted({k for c in conds for k in c}, key=str)
        if not rowkeys:
            for lab, phi in items: sols.append((key, [(lab, F(1))]))
            continue
        ri = {k: i for i, k in enumerate(rowkeys)}
        M = [[F(0)] * len(items) for _ in rowkeys]
        for j, c in enumerate(conds):
            for k, v in c.items(): M[ri[k]][j] = v
        for vec in nullspace(M):
            sols.append((key, [(items[j][0], v) for j, v in enumerate(vec) if v]))
    return names, ns, sols

def wt_xi_fix():
    pass

if __name__ == '__main__':
    names, ns = weights()
    print('scaling weights (rows):', names)
    for v in ns: print(' ', [str(a) for a in v])

def pretty(sol):
    out=[]
    for lab,c in sol:
        if lab[0]=='eta':
            _,comp,ce,u=lab; co=''.join(f'{k}^{a}' if a>1 else k for k,a in ce if a)
            out.append(f'{c}*eta[{comp}]({co or 1}*{"*".join(u) or 1})')
        else:
            _,k,ce,u=lab; co=''.join(f'{kk}^{a}' if a>1 else kk for kk,a in ce if a)
            out.append(f'{c}*xi^{k}({co or 1}*{"*".join(u) or 1})')
    return ' + '.join(out)
def run(sigma,cdeg,udeg):
    names,ns,sols=solve_blocks(sigma,cdeg,udeg)
    print(f'sigma={sigma} cdeg={cdeg} udeg={udeg}: {len(sols)} solutions')
    for key,s in sols: print(' ',key,':',pretty(s))
    return sols
if __name__=="__main__" and len(sys.argv)>1:
    run(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))
