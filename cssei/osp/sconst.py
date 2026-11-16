"""Structure constants of the two method-2 superizations s_a (inner, (8|8)) and s_b (outer,
(9|8)) of gs(2) over F_2, as numpy arrays: tab[i,j,k] = coeff of e_k in [e_i,e_j], sq[i,k] =
coeff of e_k in e_i^2 (odd i), par[i]."""
import json, os, numpy as np
from galg import GFField
from msa import MSA
here = os.path.dirname(os.path.abspath(__file__))
_cache = {}
def load(t):
    if t in _cache: return _cache[t]
    F = GFField(2)
    S = json.load(open(os.path.join(here, 's_gs2.json')))
    A = MSA(F, S[t]['even'], S[t]['odd'])
    d = A.d
    tab = np.array([[[x & 1 for x in v] for v in row] for row in A.tab], dtype=np.uint8)
    sq = np.zeros((d, d), dtype=np.uint8)
    for i, s in enumerate(A.sq):
        if s is not None: sq[i] = [x & 1 for x in s]
    par = np.array(A.par)
    mats = [np.array(M, dtype=np.uint8) for M in S[t]['even'] + S[t]['odd']]
    _cache[t] = (tab, sq, par, mats)
    return _cache[t]
NAME = {'inner': 's_a', 'outer': 's_b'}
