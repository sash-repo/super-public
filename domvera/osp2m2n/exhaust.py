"""Theorem 6.4 of the text (osp(2m|2n), the three end types): the explicit conditions on the labels
(crit.py, typeII.py) against Serganova's criterion run by odd reflections (dom.py), exhaustively
over a box of labels, every base (word) of every end type, (n,m) small."""
from typeII import *
from itertools import product
import multiprocessing as mp

def job(args):
    n, m, t, w = args
    B = base_of_word(n, m, w)
    kinds = [k for k, _ in labels(n, m, w, tuple(F(0) for _ in range(n+m)))]
    ev = [F(x) for x in range(-1, 4)]
    od = [F(x) for x in range(-3, 5)] + [F(x, 2) for x in (-1, 1, 3)]
    res = [0, 0, 0]
    for labvals in product(*[ev if k == 'e' else od for k in kinds]):
        lam = lam_from_labels(n, m, w, labvals)
        g = ground_truth(n, m, B, lam)
        c = criterionII(n, m, w, lam) if t == 'II' else criterion(n, m, w, lam, 'ours')
        res[0] += 1; res[1] += g; res[2] += (g != c)
    return (n, m, t, ''.join('d' if x[0]=='d' else ('e' if x[2]==1 else 'E') for x in w), res)

if __name__ == '__main__':
    tasks = [(n, m, t, w) for n, m in ((1,2),(2,2),(1,3),(3,2),(2,3)) for t in ('I','II','III') for w in words(n, m, t)]
    p = mp.Pool(10)
    out = p.map(job, tasks)
    p.close(); p.join()
    tot = [0, 0, 0]
    for o in out:
        print(o); tot = [a+b for a, b in zip(tot, o[4])]
    print("TOTAL cases, dominant, mismatches:", tot)
