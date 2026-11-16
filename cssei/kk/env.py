import sys, json, glob
sys.path.insert(0, '..')
from abslie import Lie, two_envelope
for f in sys.argv[1:]:
    tab = json.load(open(f))['tab']; n = len(tab)
    L = Lie(n, lambda i, j: tab[i][j])
    print(f, n, 'envelope', two_envelope(L, [1 << i for i in range(n)]), flush=True)
