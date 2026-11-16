from sconst import load, NAME
from deform import Deform
for t in ('inner', 'outer'):
    Df = Deform(*load(t)[:3])
    print(NAME[t], 'vars', Df.nv, 'eqs', Df.D.shape[0], 'Z,B,H =', Df.cohomology())
