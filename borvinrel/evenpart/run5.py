import sys
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from eng import G, inv
from run1 import sqrtmod, prime_for, mat
import run3
H=int(sys.argv[1]); p=prime_for([21,15,13,6,5]); run3.p=p
a=(-11+sqrtmod(21,p))*inv(10,p)%p
g=G(mat('a',a,p),[1,1,1],p); g.build(H)
print(g.totals(H)); print(run3.mingens(g,H),flush=True)
