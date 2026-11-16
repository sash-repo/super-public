"""Proposition 8.3.9 and formula (31) of the text: is the shear delta->delta+2e of k^L(1|2) induced by an automorphism (spectral flow)?
phi: eta t^k -> eta t^{k+1}, xi t^k -> xi t^{k-1}, xi eta t^k -> xi eta t^k, t^{k+1} -> t^{k+1} + c xi eta t^k.
And for k^L(1|3): delta -> delta + e changes the parity of root spaces (checked below)."""
import kL, random
from kL import m, br, add, sm
from fractions import Fraction as F
kL.SIGN=-1
def phi(f,c):
    o={}
    for (a,S),v in f.items():
        if S==(1,): g=m(a+1,1)
        elif S==(0,): g=m(a-1,0)
        elif S==(0,1): g=m(a,0,1)
        else: g=add(m(a),m(a-1,0,1,c=c))
        o=add(o,sm(v,g))
    return o
def test(c):
    n=0
    for p in range(300):
        f=kL.rnd(2,random.randint(0,1)); g=kL.rnd(2,random.randint(0,1))
        n+=1
        if add(phi(br(f,g,2),c),sm(-1,br(phi(f,c),phi(g,c),2))): return False,n
    return True,n
for c in (F(-2),F(2),F(0),F(-1)): print("c=",c,test(c))
