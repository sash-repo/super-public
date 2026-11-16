"""Over Q, at a=b=-1, k=4: the 2-dimensional space of osp(1|2)-invariant maps of block S_4 (the shape of the
map Q of Statement 7.9) contains no nonzero map invariant under K_{t^2 theta}, hence none k(1|1)-invariant.
Usage: python3 binCm1.py      (prints the solutions (x,y); only x=y=0 is expected)"""
import sympy as sp
import binC
from binC import conditions, a, b
x, y = sp.symbols('x y')
def J(k):   # 2-dim osp space at a=b=-1, k=4: x*(c04=1,c13=2) + y*(c40=1,c31=-2)
    return {(0,4): x, (1,3): 2*x, (4,0): y, (3,1): -2*y}
binC.J = J
eqs = conditions(4)
eqs = [sp.expand(e.subs({a: -1, b: -1})) for e in eqs]
print(sp.solve([e for e in eqs if e != 0], [x, y], dict=True), [e for e in eqs if e!=0][:5])
