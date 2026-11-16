"""Derived superalgebras ([s,s] + squares of odd) of the char-2 orthosymplectic candidates, and
invariants to compare with the two superizations of gs(2)."""
import json, itertools
from galg import GFField, Ech, flat, br, mm
from msa import MSA, assoc_dim
from ospc import candidate, I, Pi
from torus import analyse
from h2 import h2
F = GFField(2)
def derived(ev, od):
    E0 = Ech(F); E1 = Ech(F); b0 = []; b1 = []
    B = [(X, 0) for X in ev] + [(X, 1) for X in od]
    for (X, p), (Y, q) in itertools.product(B, B):
        Z = br(F, X, Y)
        if (p + q) % 2:
            if E1.add(flat(Z)): b1.append(Z)
        elif E0.add(flat(Z)): b0.append(Z)
    for X in od:
        Z = mm(F, X, X)
        if E0.add(flat(Z)): b0.append(Z)
    return b0, b1
def inv(name, ev, od, deep=True):
    A = MSA(F, ev, od)
    z = A.center()
    ad = [A.admat(i) for i in range(A.d)]
    b0, b1 = derived(ev, od)
    hh = h2([[[x & 1 for x in v] for v in row] for row in A.tab], [None if s is None else [x & 1 for x in s] for s in A.sq], A.par, verbose=False)[0]
    print('%-34s sdim (%d|%d) center %d assoc(ad) %d derived (%d|%d) H2_ev %d' % (name, len(ev), len(od), len(z), assoc_dim(F, ad), len(b0), len(b1), hh))
    if deep: analyse(name, ev, od)
    return b0, b1
S = json.load(open('s_gs2.json'))
for t in ('inner', 'outer'):
    inv('gs2 ' + t, S[t]['even'], S[t]['odd'])
for nm, B0 in (('1_4', I(4)), ('Pi_4', Pi(4))):
    ev, od = candidate(B0, Pi(2), True, False)
    b0, b1 = inv('osp(%s|Pi_2) alt' % nm, ev, od)
    if (len(b0), len(b1)) != (len(ev), len(od)):
        b0b, b1b = inv('  its derived', b0, b1)
