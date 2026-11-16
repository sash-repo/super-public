# Odd-parameter deformations of the embedding L^{a,b} with a in 1/2+Z:
# H^1(witt; M_lam), M_lam: L_m v_n = (lam*m - n) v_{m+n} (n in Z after shifting by a),
# lam = 1/2+beta for the G-part (1/2-beta for the H-part). Window |m|,|n|<=K.
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from fractions import Fraction as F
from exactlin import rank
K=12
idx={m:i for i,m in enumerate(range(-K,K+1))}
for lam in [F(x,2) for x in range(-4,9)]:
    rows=[]
    for m in range(-K,K+1):
        for n in range(-K,K+1):
            if abs(m+n)>K: continue
            row=[F(0)]*len(idx)
            row[idx[n]]+=lam*m-n; row[idx[m]]-=lam*n-m; row[idx[m+n]]-=(m-n)
            rows.append(row)
    # unknowns phi(k) for |k|<=2K; restrict count to the |k|<=K ones that are determined
    rk=rank(rows); free=len(idx)-rk
    # unknowns phi(k), K<|k|<=2K appear only via m+n: subtract those not pinned (count separately)
    cob=1 if lam!=0 else 0
    print("lam",lam,"cocycles",free,"coboundaries",cob,"H^1",free-cob)
