"""Is K_0 = sum_d [K_d,K_{-d}]?  Degree-0 parts of brackets, odd polys in h of degree <= 2M+1."""
import sympy as sp
from osh import *
for lam in (sp.Rational(7,3), sp.Rational(5,2), sp.Integer(3)):
    Ua = U(lam,'QQ'); Ka = K(Ua)
    M = 6
    rows = []
    for d in range(1,4):
        for j in range(4):
            for k in range(4):
                c = dict(Ka.brk((d,j),(-d,k)))
                if all(key[1] <= M for key in c):
                    rows.append([c.get((0,i),0) for i in range(M+1)])
    r = sp.Matrix(rows).rank(); r2 = sp.Matrix(rows + [[1]+[0]*M]).rank()
    print("lam",lam,": rank of brackets in odd polys deg<=",2*M+1,":",r,"/",M+1,"; h in span:", r2==r)
