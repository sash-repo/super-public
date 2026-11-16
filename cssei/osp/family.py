"""Try to integrate an infinitesimal deformation lam in H^2_ev into a polynomial family
mu_t = sum_n t^n (C_n, Q_n) over F_2[t]: at order n solve delta(C_n,Q_n) = sum_{j+k=n, 0<j,k<n} R(C_j,C_k,Q_k).
Returns the list of orders, or the order at which the obstruction is not in im(delta) for the choices made."""
import numpy as np, pickle, itertools, sys
from sconst import load, NAME
from deform import Deform
from f2lin import solve, Solver

def integrate(Df, c1, maxn=12):
    terms = [(Df.T0, Df.S0), Df.unpack(c1)]
    for n in range(2, maxn + 1):
        rhs = np.zeros(Df.D.shape[0], dtype=np.int64)
        for j in range(1, n):
            k = n - j
            if k >= len(terms) or j >= len(terms): continue
            rhs = rhs + Df.R(terms[j][0], terms[k][0], terms[k][1])
        rhs = (rhs & 1).astype(np.uint8)
        if not rhs.any():
            if all(not terms[m][0].any() and not terms[m][1].any() for m in range(max(2, n - 1), len(terms))) and n > len(terms):
                return terms, n
            terms.append((np.zeros_like(Df.T0), np.zeros_like(Df.S0)))
            # all later rhs vanish once the last nonzero order is < n/2
            last = max(m for m in range(len(terms)) if terms[m][0].any() or terms[m][1].any())
            if 2 * last < n + 1: return terms[:last + 1], None
            continue
        x = Df.solver.solve(rhs)
        if x is None: return terms, n            # obstructed at order n
        terms.append(Df.unpack(x))
        last = max(m for m in range(len(terms)) if terms[m][0].any() or terms[m][1].any())
        if 2 * last < n + 1: return terms[:last + 1], None
    return terms, 'maxn'

def evaluate(terms, tval=1):
    T = sum(t[0] for t in terms) & 1; S = sum(t[1] for t in terms) & 1
    return T.astype(np.uint8), S.astype(np.uint8)

if __name__ == '__main__':
    o = pickle.load(open('obstr.pkl', 'rb'))
    res = {}
    for t in ('inner', 'outer'):
        tab, sq, par, _ = load(t)
        Df = Deform(tab, sq, par); Df.cohomology(); Df.solver = Solver(Df.D)
        H = Df.H; sols = o[t]['sols']
        out = []
        import random; random.seed(1)
        if len(sols) > 400: sols = random.sample(sols, 400)
        for lam in sols:
            c1 = (np.array(lam, dtype=np.int64) @ H.astype(np.int64)) & 1
            terms, status = integrate(Df, c1.astype(np.uint8))
            out.append((lam, len(terms) - 1, status, terms))
        from collections import Counter
        print(NAME[t], Counter((deg, st) for _, deg, st, _ in out))
        res[t] = out
    pickle.dump(res, open('family.pkl', 'wb'))
