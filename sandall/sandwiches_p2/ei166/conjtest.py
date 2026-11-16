import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eick166 import TAB, lebedev, lie_closure, conj_class_rep, matT, matmul, inverse, gl4, span_of
from inv166 import dens
N, gens = lebedev(TAB.splitlines()[0]); n1, _ = lie_closure(gens)
N, gens = lebedev(TAB.splitlines()[1]); n7, _ = lie_closure(gens)
f1, _ = lie_closure(dens(1)); f0, _ = lie_closure(dens(0))
c = {k: conj_class_rep(v) for k, v in dict(N1=n1, N7=n7, F1=f1, F0=f0).items()}
print({a: [b for b in c if c[b] == c[a]] for a in c})
# an explicit conjugating matrix N1 -> N7
t = span_of(n7)
for g in gl4():
    if span_of([matmul(matmul(g, M), inverse(g)) for M in n1]) == t:
        print("g =", g); break
