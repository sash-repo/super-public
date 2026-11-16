import json, sys
from galg import *
from msa import *
K = int(sys.argv[1]) if len(sys.argv) > 1 else 6
F = GFField(K)
S = json.load(open('s_gs2.json'))
def report(name, A):
    d = A.d
    ad = [A.admat(i) for i in range(d)]
    z = A.center()
    ev = [i for i in range(d) if A.par[i] == 0]
    # derived series of even part as a Lie algebra, and [odd,odd]
    print('%-28s sdim (%d|%d) center %d  assoc(ad) %d (d^2=%d)' % (name, len(ev), d - len(ev), len(z), assoc_dim(F, ad), d * d))
for t in ('inner', 'outer'):
    A = MSA(F, S[t]['even'], S[t]['odd']); report('gs2 ' + t, A)
alphas = [F.gen, F.exp[7]]
if K % 2 == 0: alphas.append(F.exp[(F.q - 1) // 3])
for a in alphas:
    Am = [[0, F.el(a), 0], [F.el(a), 0, 1], [0, 1, 0]]
    N, dims, el, central = generate(Am, [1, 0, 0], [0, 0, 1], F)
    par = [1, 0, 0]
    ev = [X for d_, L in el.items() for X in L if sum(x * y for x, y in zip(d_, par)) % 2 == 0]
    od = [X for d_, L in el.items() for X in L if sum(x * y for x, y in zip(d_, par)) % 2 == 1]
    A = MSA(F, ev, od); report('bgl1(3;%d) in L^(001)' % a, A)
from ospc import *
for nm, B0 in (('1_4', I(4)), ('Pi_4', Pi(4))):
    for nm1, B1 in (('Pi_2', Pi(2)), ('1_2', I(2))):
        for a0, a1 in ((True, False), (False, True), (True, True), (False, False)):
            ev, od = candidate(B0, B1, a0, a1)
            try:
                A = MSA(F, ev, od)
                # squares closed?
                report('osp B0=%s B1=%s alt%d%d' % (nm, nm1, a0, a1), A)
            except ValueError as e:
                print('osp B0=%s B1=%s alt%d%d (%d|%d) not closed' % (nm, nm1, a0, a1, len(ev), len(od)))
