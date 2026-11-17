"""The claim after Proposition pstrlin: for omega of the 1st type and N = One, the equation
(ad_f)^p = ad_g, f a monomial, has no solution g in O(F) exactly for the translations f = u_i
whose p-th power in Proposition pstrlin is non-zero, i.e. u_j (j <= k) with a non-zero j-th
column of alpha, and u_{~i} with a non-zero i-th row of alpha.  Checked on every monomial of
O(2k;One), every normal shape of the 1st type, for the listed (p, k)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skrpb import Ham
from pstrH import shapes

if __name__ == "__main__":
    cases = bad = nf = 0
    for p, k in ((3, 1), (5, 1), (7, 1), (3, 2), (5, 2), (3, 3)):
        for name, B in shapes(k, p):
            H = Ham(p, (1,)*(2*k), B, 1, off=1)
            found = sorted(H.is_p_closed())
            found = [e for e, obs in found]
            pred = []
            for j in range(k):            # u_j, j-th column of alpha
                if any(B[i][j] for i in range(k)):
                    pred.append(tuple(1 if t == j else 0 for t in range(2*k)))
            for i in range(k):            # u_{~i}, i-th row of alpha
                if any(B[i][j] for j in range(k)):
                    pred.append(tuple(1 if t == i + k else 0 for t in range(2*k)))
            cases += 1; nf += len(found)
            ok = sorted(found) == sorted(pred)
            if not ok:
                bad += 1
                print("MISMATCH", p, k, name, "found", found, "predicted", sorted(pred))
    print("shape-cases", cases, "monomials without a p-th power in O(F)", nf, "mismatches", bad)
