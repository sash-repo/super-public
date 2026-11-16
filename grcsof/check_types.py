from fractions import Fraction as Fr
import kasV
from kas6 import gens_kas
from superN import mono
F = gens_kas(); F['t2'] = mono(2,0)
sub = {k: F[k] for k in ['1','xi1','xi1eta1','xi1xi2','t','txi1','c0','t2']}
for nm in ['L3','L3p','L3m','Sp','Sm']:
    V = kasV.TYPES[nm]
    for lam in (Fr(0), Fr(3,7)):
        b = kasV.check_module(V, lam, sub, tmax=1)
        print(f"  {nm:4s} dim={V.dim:2d} lam={str(lam):4s}: "
              + ("module axiom OK" if not b else f"FAILS {b[:2]}"), flush=True)
    print("        weights:", V.wts, flush=True)
