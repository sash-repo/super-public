# Statement 4.1.4 of the paper: the row-wise condition (no global sign of the form)
from fractions import Fraction as F
import bases as Bm
def row_ok(G,p,i,allow_imag=True):
    n=len(G); d=G[i][i]; off=[G[i][j] for j in range(n) if j!=i]
    if d==0:
        return True if p[i]==1 else (all(x<=0 for x in off) or all(x>=0 for x in off))
    rs=[F(-2*x,d) for x in off]
    real=all(r>=0 and r.denominator==1 and (p[i]==0 or r.numerator%2==0) for r in rs)
    imag=all(F(x,d)>=0 for x in off)
    return real or (allow_imag and imag)
def ok(G,p,allow_imag=True): return all(row_ok(G,p,i,allow_imag) for i in range(len(G)))
def glaff(seq):   # affine sl(m|n)^(1): cyclic chain, delta isotropic & orthogonal
    v=lambda c:1 if c=='e' else -1
    N=len(seq); G=[[0]*N for _ in range(N)]
    for i in range(N):
        a,b=seq[i],seq[(i+1)%N]; G[i][i]=v(a)+v(b)
        G[i][(i+1)%N]+= -v(b); G[(i+1)%N][i]+= -v(b)
    p=[int(seq[i]!=seq[(i+1)%N]) for i in range(N)]
    return G,p
if __name__=="__main__":
    cases=dict(Bm.cases)
    cases.update({'sl(2|2)^(1)':glaff('eedd'),'sl(3|3)^(1)':glaff('eeeddd'),'sl(2|3)^(1)':glaff('eeddd'),'sl(3|4)^(1)':glaff('eeedddd')})
    for name,(G,p) in cases.items():
        O=Bm.orbit(G,p)
        print(f"{name:12s} bases {len(O):4d}  row-wise, real nodes only: {sum(ok(a,b,False) for a,b in O):4d}  row-wise with imaginary rows: {sum(ok(a,b) for a,b in O):4d}")
