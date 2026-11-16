"""Export the two method-2 superizations s(gs(2),gr) (one grading of each kind) as
matrix superalgebras inside End(gs(2)) over F_2: lists of even and odd 14x14 matrices."""
import sys, os, json
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(here))
os.chdir(os.path.dirname(here))
__file__ = os.path.abspath('grad/shen_grad3.py')
exec(open('grad/shen_grad3.py').read().split("done = set()")[0])
out = {}
for D in res:
    g1 = colspace(D); g0 = nullspace(D)
    t = 'outer' if derived_series(g0) == [6] else 'inner'
    if t in out: continue
    M0 = [ad(x).astype(np.int64) for x in g0]
    M1 = [ad(x).astype(np.int64) for x in g1]
    sqs = [(A @ A) % 2 for A in M1]
    fl = lambda A: np.asarray(A, dtype=np.uint8).flatten()
    ev = span_basis([fl(A) for A in M0] + [fl(A) for A in sqs])
    od = span_basis([fl(A) for A in M1])
    out[t] = {'n': n, 'even': [e.reshape(n, n).tolist() for e in ev], 'odd': [o.reshape(n, n).tolist() for o in od],
              'D': D.tolist()}
    print(t, len(ev), len(od))
json.dump(out, open('osp/s_gs2.json', 'w'))
