"""Export the structure tables of the eleven algebras of dim <= 31 found by the random search as
{'dim','tab'} json (tab[i][j] = bitmask of [e_i,e_j]), for grad/css_grad.py."""
import json, glob, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.join(HERE, '..', 'blocks')
out = {}
for p in sorted(glob.glob(os.path.join(B, 'k*_*_*.npz'))):
    ad = np.load(p)['ad']; n = ad.shape[0]
    tab = [[int(sum(int(ad[i][k, j]) << k for k in range(n))) for j in range(n)] for i in range(n)]
    name = os.path.basename(p)[:-4]
    json.dump({'name': name, 'dim': n, 'tab': tab}, open(os.path.join(HERE, name + '.json'), 'w'))
    print(name, n)
for src, name in [('../cmp/r15.json', 'k15_112_41'), ('../cmp/sbt7.json', 'k28_27584_12545')]:
    T = json.load(open(os.path.join(HERE, src)))
    json.dump({'name': name, 'dim': T['dim'], 'tab': T['tab']}, open(os.path.join(HERE, name + '.json'), 'w'))
    print(name, T['dim'])
