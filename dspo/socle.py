"""Statement 3.2.6: the terminal derived term P of s = g_x/r(g_x): sdim, centroid
bound, Killing rank of P_0.
Usage: python3 socle.py m:name [m:name ...], e.g. 6:xi1xi2xi3eta1eta2 7:xi1xi2eta1
"""
import sys, numpy as np
from radical2 import *
from probe import sub_alg, killing_rank
todo = sys.argv[1:]   # entries m:name
for item in todo:
    m, name = item.split(":"); m = int(m)
    po = Po(m)
    msk = next(mm for mm in range(1 << m) if po.name(mm) == name)
    q = Quot(po, {msk: 1})
    T = np.array([[[modp(c) for c in v] for v in row] for row in q.table()], dtype=np.int64)
    s, _ = radical_quotient(Alg(T, q.par))
    cur = np.eye(s.d, dtype=np.int64)
    while True:
        nxt = s.span_br(cur, cur)
        if len(nxt) == len(cur): break
        cur = nxt
    Pa = sub_alg(s, cur)
    ev = [i for i in range(Pa.d) if Pa.par[i] == 0]
    E = sub_alg(Pa, np.eye(Pa.d, dtype=np.int64)[ev])
    print(f"m={m} {name}: s={s.sdim()} P={Pa.sdim()} centroid(P)<={Pa.centroid_bound()} P_0: Killing rank {killing_rank(E, range(E.d))} toral {E.toral_rank()}", flush=True)
