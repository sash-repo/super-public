import sys
from rc import *
def model(n, kind):
    frame = []
    m = n if kind == 'NS' else n-1
    for i in range(1, m+1):
        L = vzero(n); L[i] = {(0, ()): Fr(1)}; L[0] = {(0, (i,)): Fr(-1)}; frame.append(L)
    if kind == 'R0':
        L = vzero(n); L[n] = {(0, ()): Fr(1)}; frame.append(L)
    def alpha(Y):
        a = dict(Y[0])
        for i in range(1, m+1):
            a = fadd(a, fmul(Y[i], {(0, (i,)): Fr(1)}))
        return a
    return Distr(n, frame, alpha)
n = int(sys.argv[1]); kind = sys.argv[2]; dmax = int(sys.argv[3])
D = model(n, kind)
print(kind, n, "sdim g_k:", [(k, sum(1 for b in D.comp(k) if vpar(b,n)==0), sum(1 for b in D.comp(k) if vpar(b,n)==1)) for k in range(-2, dmax+1)], flush=True)
for d in range(int(sys.argv[4]) if len(sys.argv)>4 else 1, dmax+1):
    r = cohom(D, d, check=True)
    print(f"d={d}: even H2={r[0]['H2']} (Z{r[0]['Z2']},B{r[0]['B2']})  odd H2={r[1]['H2']} (Z{r[1]['Z2']},B{r[1]['B2']})", flush=True)
