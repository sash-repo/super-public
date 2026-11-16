import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eick166 import TAB, lebedev, lie_closure, conj_class_rep, matT
reps = {}
for row in TAB.splitlines():
    N, gens = lebedev(row)
    els, d = lie_closure(gens)
    reps[N] = (conj_class_rep(els), conj_class_rep([matT(M) for M in els]))
    print("N%d dim %d" % (N, d), flush=True)
Ns = sorted(reps)
for a in Ns:
    print("N%d conjugate to:" % a, [b for b in Ns if reps[b][0] == reps[a][0]],
          "; its dual (transpose) conjugate to:", [b for b in Ns if reps[b][0] == reps[a][1]], flush=True)
