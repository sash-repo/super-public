"""Which power of vol carries a modular form: w=+k/2 or w=-k/2?

The text states M_k = (F_{k/2})^Gamma, i.e. the exponent +k/2, not -k/2.
Checked here numerically with the
genuine weight-4 Eisenstein series E_4 under z -> -1/z (a=0,b=-1,c=1,d=0).
"""
import mpmath as mp
mp.mp.dps = 30

def E4(z):
    q = mp.e**(2j*mp.pi*z)
    return 1 + 240*mp.nsum(lambda n: n**3*q**n/(1-q**n), [1, mp.inf])

k = 4
z = mp.mpf('0.31') + 1j*mp.mpf('1.17')
a,b,c,d = 0,-1,1,0
zt = (a*z+b)/(c*z+d)
mult = (c*z+d)

print("modularity  E4(~z) == (cz+d)^k E4(z) :",
      mp.nstr(E4(zt) - mult**k*E4(z), 8))

# density f*vol^w transforms as f(~z)*(d~z)^w = f(~z)*(cz+d)^{-2w}*(dz)^w
for w, name in ((mp.mpf(k)/2, "+k/2"), (-mp.mpf(k)/2, "-k/2")):
    coeff = E4(zt)*mult**(-2*w)          # coefficient of (dz)^w after the change
    print(f"  w={name:5}: transformed coeff / E4(z) = {mp.nstr(coeff/E4(z), 10)}"
          f"   {'INVARIANT' if abs(coeff/E4(z)-1) < mp.mpf('1e-20') else 'not invariant'}")
