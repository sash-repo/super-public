"""Statement 5.10.3, last claim: d/dx : F_{-1/2} -> F_{1/2} is not k^L(1|3)-invariant (so that
St_3 = D_1^{-1} D_theta1 D_theta2 D_theta3 is not a cocycle), whereas for n = 0 the map
d/dx : F_0 -> F_1 is invariant (the control).  The cocycle property of D_theta1 D_theta2 D_theta3
with values in F_{1/2} is checked by nsgrass.py.  Run:  python3 miura3.py
"""
import sympy as sp
import os, sys; sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
from nsgrass import *
th3=['ta','tb','tc']
for oH in (0,1):
    H = rsf(th3, oH, 'e7'); H = H if oH==0 else H*G.g('e5')
    w = rsf(th3, 0, 'e8')
    d = Kact(H, w.dx(), sp.Rational(1,2), th3) - Kact(H, w, -sp.Rational(1,2), th3).dx()
    print('parity',oH,'d intertwines K_H on F_{-1/2}->F_{1/2}:', d.iszero())
# for comparison n=0: d: F_0 -> F_1 invariant
H=rsf([],0,'e7'); w=rsf([],0,'e8')
print('n=0, d: F_0->F_1:', (Kact(H,w.dx(),1,[])-Kact(H,w,0,[]).dx()).iszero())
