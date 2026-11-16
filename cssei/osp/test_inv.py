"""calibration of invariants.py on gl(2), sl(2)+... over F_2 (all even)."""
import numpy as np, itertools
from invariants import *
def matalg(mats, par):
    d = len(mats); flat = np.array([m.reshape(-1) for m in mats]) & 1
    R, piv = rref(flat)
    def coords(X):
        # solve sum c_i mats_i = X
        from f2lin import solve
        return solve(flat.T, X.reshape(-1) & 1)
    tab = np.zeros((d, d, d), dtype=np.uint8)
    for i in range(d):
        for j in range(d):
            tab[i, j] = coords((mats[i] @ mats[j] + mats[j] @ mats[i]) & 1)
    sq = np.zeros((d, d), dtype=np.uint8)
    for i in range(d):
        if par[i]: sq[i] = coords((mats[i] @ mats[i]) & 1)
    return tab, sq, np.array(par)
E = lambda i, j, n: np.eye(n, dtype=np.int64)[:, [i]] @ np.eye(n, dtype=np.int64)[[j], :]
gl2 = [E(i, j, 2) for i in range(2) for j in range(2)]
tab, sq, par = matalg(gl2, [0] * 4)
f = invariant_forms(tab, sq, par)
print('gl(2): invariant forms dim', f[0][0], 'nondeg', f[0][1], '(expect 2: tr XY, trX trY)')
print('gl(2): center', len(center(tab)), '(1) ; Der', derivations(tab, sq, par), 'H^2(g;K)', central_ext(tab, sq, par))
# gl(1|1): even E11,E22, odd E12,E21 ; squares E12^2=0
gl11 = [E(0, 0, 2), E(1, 1, 2), E(0, 1, 2), E(1, 0, 2)]
tab, sq, par = matalg(gl11, [0, 0, 1, 1])
f = invariant_forms(tab, sq, par)
print('gl(1|1): even invariant forms', f[0][0], 'nondeg', f[0][1], '(str XY is invariant and nondegenerate)')
print('gl(1|1): odd x^2=0 count', odd_null_squares(tab, sq, par), '(E12, E21 : 2 ; E12+E21 squares to 1)')
