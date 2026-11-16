"""Check of Proposition 5.3 (osp(1|2)-invariant maps F_a -> F_b on O(1;N|1)) against the
closed form, and which of them are invariant under the whole k(1;N|1).
Usage: kfull.py p N.  Reuses osp1.setup/generators/rep/equivariant (sign s=-1: K_theta = theta d_t - d_theta)."""
import sys
import os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from osp1 import setup, generators, rep, equivariant
from math import comb

def mult_t(S, r):            # multiplication by t^{(r)} on O(1;N|1), basis index 2r'+e
    P, n, p = S['P'], S['n'], S['p']; M = S['zero']()
    for q in range(P):
        if q + r < P:
            for e in range(2):
                M[2*(q+r)+e][2*q+e] = comb(q+r, r) % p
    return M

def kgens(S):
    """all K_f, f = t^{(r)} (even) and t^{(r)} theta (odd): K_g = 2g d_t + g' theta d_theta, K_{g theta} = g K_theta."""
    mul, add, P = S['mul'], S['add'], S['P']
    Dt, Dth, TH = S['Dt'], S['Dth'], S['TH']
    E = mul(TH, Dth); Kth = add(mul(TH, Dt), Dth, coef=[1, -1])
    out = []
    for r in range(P):
        g = mult_t(S, r); g1 = mult_t(S, r-1) if r else S['zero']()
        out.append((add(mul(g, Dt), mul(g1, E), coef=[2, 1]), g1, 0))
        out.append((mul(g, Kth), mul(g1, TH), 1))
    return out

def predicted(p, N, a, b, par):
    P = p**N
    if par == 0: return p**(N-1) if a == b else 0       # d_t^{pj}; a = b scalar case included (j=0)
    return p**(N-1) if (a + b) % p == 1 else 0          # D_theta d_t^k, k = -a mod p

if __name__ == "__main__":
    p, N = int(sys.argv[1]), int(sys.argv[2])
    S = setup(p, N); osp = generators(S, -1); kk = kgens(S)
    bad = 0; cases = 0
    for a in range(p):
        for b in range(p):
            for par in (0, 1):
                cases += 1
                d = len(equivariant(S, rep(S, osp, a), rep(S, osp, b), par)[0])
                if d != predicted(p, N, a, b, par):
                    bad += 1; print("MISMATCH", a, b, par, d)
                dk = len(equivariant(S, rep(S, kk, a), rep(S, kk, b), par)[0])
                if dk and not (a == b and par == 0 and dk == 1):
                    print(f"p={p} N={N} k-invariant F_{a}->F_{b} {'odd' if par else 'even'} dim {dk}")
    print(f"p={p} N={N}: {cases} cases, {bad} mismatches with the closed form")
