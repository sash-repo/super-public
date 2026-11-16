"""supmin on one grading per class of the invariants of grad/css_grad.py (its 'examples')
plus EXTRA random gradings.  usage: sup31.py NAME [extra]"""
import sys, os, json, random, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from supsand import superization, analyse, bits
from supmin import smallest_super
name = sys.argv[1]; extra = int(sys.argv[2]) if len(sys.argv) > 2 else 0
R = os.path.join(HERE, '..', 'results', 'cssgrad')
tab = json.load(open(os.path.join(HERE, '..', 'new11', name + '.json')))['tab']
G = json.load(open(os.path.join(R, name + '.json')))
W = json.load(open(os.path.join(R, name + '_words.json')))
DER = W['der']; n = len(tab)
random.seed(2)
todo = [(k, w) for k, w in G['examples'].items()] + [('random', w) for w in random.sample(W['words'], extra)]
for cls, w in todo:
    t = time.time()
    M = [0] * n
    for k in bits(w):
        M = [M[j] ^ DER[k][j] for j in range(n)]
    D = tuple(M)
    T, par, sq, a, b = superization(tab, D)
    r = analyse(tab, D)
    if 'codim_n' in r:
        c, found, sd = smallest_super(T, par, sq, cmax=sum(r['codim_n']))
        r['min_codim'] = c; r['min_sdims'] = sorted(set(sd)); r['n_is_min'] = (c == sum(r['codim_n']))
    r['class'] = cls; r['sec'] = round(time.time() - t)
    print(name, json.dumps(r, sort_keys=True), flush=True)
