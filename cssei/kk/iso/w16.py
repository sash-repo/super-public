"""W(1;6)' over F_2: basis x^(a)d, 0<=a<=63; [x^(a)d, x^(b)d] = (C(a+b-1,a)-C(a+b-1,b)) x^(a+b-1)d.
Export its derived algebra as a json table."""
import json
from math import comb
from f2 import *
n0 = 64
def c(a, b):
    if a + b - 1 > 63 or a + b == 0: return 0
    return (comb(a + b - 1, a) + comb(a + b - 1, b)) % 2   # C(.,a)-C(.,b) mod 2, x^(a)d(x^(b))=C(a+b-1,a)x^(a+b-1)
T = [[(1 << (a + b - 1)) if c(a, b) else 0 for b in range(n0)] for a in range(n0)]
W = Alg(T)
D = span(W.br(1 << a, 1 << b) for a in range(n0) for b in range(n0)).basis()
D2 = span(W.br(a, b) for a in D for b in D).basis()
print('dim W', n0, "W'", len(D), "W''", len(D2))
# coordinates of D
E = Ech()
for k, v in enumerate(D): E.add(v, 1 << k)
def coord(v):
    r, t = E.reduce(v); assert r == 0; return t
tab = [[coord(W.br(a, b)) for b in D] for a in D]
json.dump({'name': "W(1;6)'", 'dim': len(D), 'tab': tab}, open('W1_6d.json', 'w'))
