"""For each kind of superization with n(s) maximal, one smallest-codimension search.
usage: target.py NAME [tries]"""
import sys, os, json, random, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from supsand import superization, analyse, bits
from supmin import smallest_super
name = sys.argv[1]; tries = int(sys.argv[2]) if len(sys.argv) > 2 else 80
R = os.path.join(HERE, '..', 'results', 'cssgrad')
tab = json.load(open(os.path.join(HERE, '..', 'new11', name + '.json')))['tab']
W = json.load(open(os.path.join(R, name + '_words.json'))); DER = W['der']; n = len(tab)
random.seed(5); done = set()
for w in random.sample(W['words'], tries):
    M = [0] * n
    for k in bits(w):
        M = [M[j] ^ DER[k][j] for j in range(n)]
    D = tuple(M)
    r = analyse(tab, D)
    if not r.get('n_max'):
        continue
    key = (tuple(r['sdim']), tuple(r['sdim_s']), tuple(r['codim_n']))
    if key in done:
        continue
    done.add(key)
    t = time.time()
    T, par, sq, a, b = superization(tab, D)
    c, found, sd = smallest_super(T, par, sq, cmax=sum(r['codim_n']))
    r.update(word=w, min_codim=c, min_sdims=sorted(set(sd)), n_is_min=(c == sum(r['codim_n'])), sec=round(time.time() - t))
    print(name, json.dumps(r, sort_keys=True), flush=True)
