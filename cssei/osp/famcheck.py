"""For the polynomial families found by family.py, evaluate at t=1 over F_2 and compare invariants."""
import numpy as np, pickle, sys
from collections import Counter
from sconst import load, NAME
from deform import Deform
from invariants import *
res = pickle.load(open('family.pkl', 'rb'))
def invs(tab, sq, par):
    D = derived(tab, par)
    return (is_simple(tab, sq, par)[0], homog_basis(D, par), len(center(tab)), assoc_dim(tab),
            restricted(tab, par)[0], odd_null_squares(tab, sq, par))
for t in ('inner', 'outer'):
    tab, sq, par, _ = load(t)
    Df = Deform(tab, sq, par)
    base = invs(tab, sq, par)
    print('==', NAME[t], 'base', base)
    seen = Counter(); ex = {}
    for lam, deg, st, terms in res[t]:
        if st is not None: continue
        T = (sum(x[0] for x in terms) & 1).astype(np.int64); S = (sum(x[1] for x in terms) & 1).astype(np.int64)
        # identities must hold exactly
        r = Df.R(T, T, S)
        assert not r.any(), 'family member violates identities'
        iv = invs(T.astype(np.uint8), S.astype(np.uint8), par)
        seen[iv] += 1; ex.setdefault(iv, (lam, deg))
    for iv, n in seen.items(): print('  ', n, 'x', iv, 'e.g. lam', ex[iv])
    sys.stdout.flush()
