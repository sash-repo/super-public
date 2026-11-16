"""Finer invariant over GF(4): histogram of rank(ad x) for odd x != 0 with x^2 = 0, and for
all odd x; projective counts.  b_w(2;1) vs the linear deformation mu + c of s_a (lam = e_4)."""
import numpy as np, pickle, itertools, sys
from collections import Counter
from gf4 import SA, MUL, rank
from sconst import load
def adx(A, x):
    M = np.zeros((A.d, A.d), dtype=np.uint8)
    for i in np.nonzero(x)[0]: M ^= MUL[x[i], A.ad(i)]
    return M
def prof(A):
    idx = np.nonzero(A.par == 1)[0]; h0 = Counter(); h1 = Counter()
    for c in itertools.product(range(4), repeat=len(idx)):
        if not any(c) or next(v for v in c if v) != 1: continue
        x = np.zeros(A.d, dtype=np.uint8); x[idx] = c
        r = rank(adx(A, x))
        (h0 if not A.square(x).any() else h1)[r] += 1
    return dict(sorted(h0.items())), dict(sorted(h1.items()))
z = np.load('b_w.npz'); print('b_w(2;1)       ', prof(SA(z['tab'], z['sq'], z['par']))); sys.stdout.flush()
tab, sq, par, _ = load('inner'); print('s_a            ', prof(SA(tab, sq, par))); sys.stdout.flush()
res = pickle.load(open('family.pkl', 'rb'))
for lam, deg, st, terms in res['inner']:
    if ''.join(map(str, lam)) in ('0000100000', '0000100100'):
        T1 = sum(x[0] for x in terms) & 1; S1 = sum(x[1] for x in terms) & 1
        print('mu+c lam=%s     ' % ''.join(map(str, lam)), prof(SA(T1, S1, par))); sys.stdout.flush()
