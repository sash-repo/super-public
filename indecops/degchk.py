"""Remark 4.7, items 2) and 3): for a few positive weight tuples, the dimension
of the invariant operators of each order k and the highest power of the last
variable in their symbols, printed as (k, dim, max degree in X_m)."""
import sys; sys.path.insert(0,'.')
from fractions import Fraction as F
from line import inv_space, midx
import random
random.seed(1)
for ws in [(1,1,1),(F(1,3),2,F(5,2)),(F(1,7),F(1,5),F(1,2)),(1,1,1,1),(F(1,3),F(1,2),2,F(1,5))]:
    m=len(ws); res=[]
    for k in range(0, m*(m-1)//2+3):
        B=inv_space(tuple(F(x) for x in ws),k)
        cols=midx(m,k)
        dmax=max((z[-1] for v in B for z,c in zip(cols,v) if c),default=None)
        res.append((k,len(B),dmax))
    print(ws,res)
