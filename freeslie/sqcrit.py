"""Is a squaring forced by pointwise data?  Test, on the census of
Statement 8.8, the claim (Proposition 8.9): g admits a squaring (strong axiom) iff for every odd v
(ad_v)^2 = ad_z for some even z."""
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from evenrules import GF, SAlg2, enumerate_brackets, all_squarings

def adm(A, u):
    return tuple(A.br(u, A.e(i)) for i in range(A.n))

def pointwise(A):
    ads = {adm(A, z) for z in A.even_elements()}
    for v in A.odd_elements():
        sq = tuple(A.br(v, A.br(v, A.e(i))) for i in range(A.n))
        if sq not in ads:
            return False
    return True

for F in (GF(1), GF(2)):
    for m, n in [(1, 2), (2, 2), (1, 3)]:
        if F.q == 4 and m + n > 3: continue
        names = ["u%d" % i for i in range(m)] + ["x%d" % i for i in range(n)]
        par = [0]*m + [1]*n
        good = nos = mism = 0
        for t in enumerate_brackets(m, n):
            A = SAlg2(names, par, t, F)
            if A.check_bracket(): continue
            good += 1
            g = bool(all_squarings(A))
            p = pointwise(A)
            nos += (not g)
            mism += (g != p)
        print(F.name, (m, n), "objects", good, "no squaring", nos, "mismatch", mism)
