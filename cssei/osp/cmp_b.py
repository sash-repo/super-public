"""Invariants over GF(4) of b_w(2;1) (superization, (8|8)) vs s_a and the simple (8|8) members of its
polynomial families (evaluated at t = 1 and t = w)."""
import numpy as np, pickle, itertools, sys
from gf4 import SA, MUL
from sconst import load
def jacobi_ok(A):
    d = A.d; E = np.eye(d, dtype=np.uint8)
    for x, y, z in itertools.combinations(range(d), 3):
        v = A.br(E[x], A.tab[y, z]) ^ A.br(E[y], A.tab[z, x]) ^ A.br(E[z], A.tab[x, y])
        if v.any(): return False
    for i in np.nonzero(A.par == 1)[0]:
        for y in range(d):
            if (A.br(A.sq[i], E[y]) ^ A.br(E[i], A.tab[i, y])).any(): return False
    return True
def inv(A):
    D = A.derived()
    return dict(simple=A.is_simple_sample(), derived=A.sdim_of(D), center=len(A.center()), assoc=A.assoc_dim(),
                restricted=A.restricted(), der=A.der_sdim(), nullsq=A.odd_null_squares())
def show(name, A):
    print('%-34s' % name, 'Jacobi', jacobi_ok(A), inv(A)); sys.stdout.flush()
z = np.load('b_w.npz'); show('b_w(2;1) superization', SA(z['tab'], z['sq'], z['par']))
tab, sq, par, _ = load('inner'); show('s_a', SA(tab, sq, par))
res = pickle.load(open('family.pkl', 'rb'))
done = set()
for lam, deg, st, terms in res['inner']:
    if st is not None: continue
    T1 = (sum(x[0] for x in terms) & 1); S1 = (sum(x[1] for x in terms) & 1)
    A1 = SA(T1, S1, par)
    if not A1.is_simple_sample(tries=50): continue
    key = (len(A1.derived()),)
    # evaluate at t = w : coefficient w^n
    wp = [1, 2, 3]
    Tw = np.zeros_like(tab); Sw = np.zeros_like(sq)
    for n, (C, Q) in enumerate(terms):
        c = wp[n % 3]
        Tw ^= MUL[c, C.astype(np.uint8)] if n else C.astype(np.uint8)
        Sw ^= MUL[c, Q.astype(np.uint8)] if n else Q.astype(np.uint8)
    Aw = SA(Tw, Sw, par)
    sig = str(lam)
    show('s_a member t=1 lam=%s deg %d' % (''.join(map(str, lam)), deg), A1)
    show('s_a member t=w lam=%s' % ''.join(map(str, lam)), Aw)
    if len(done) > 6: break
    done.add(sig)
