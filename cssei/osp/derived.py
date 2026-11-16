import json
from galg import GFField, Ech, flat, br, mm
F = GFField(2)
S = json.load(open('s_gs2.json'))
for t in ('inner', 'outer'):
    B = S[t]['even'] + S[t]['odd']; ne = len(S[t]['even'])
    E = Ech(F); Ee = Ech(F); Eo = Ech(F)
    for i, X in enumerate(B):
        for j, Y in enumerate(B):
            Z = br(F, X, Y); E.add(flat(Z))
            (Eo if (i >= ne) != (j >= ne) else Ee).add(flat(Z))
    # squares of odd
    Esq = Ech(F)
    for X in S[t]['odd']:
        Esq.add(flat(mm(F, X, X)))
    print(t, 'dim [s,s] =', len(E), ' even part of [s,s]:', len(Ee), ' odd:', len(Eo))
