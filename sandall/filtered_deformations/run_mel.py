#!/usr/bin/env python3
"""Build the two prolongs (the two constructions of the Melikyan algebra) and report their dimensions."""
import sys
from mel5 import *

mats, forms = setup()
B = forms[0]
K = Contact(B, (5, 5, 5, 5, 5))
# Jacobi spot check
import random
random.seed(1)
ms = K.monos(3) + K.monos(4) + K.monos(2)
for _ in range(200):
    a, b, c = [{random.choice(ms): 1} for _ in range(3)]
    j = K.lin((1, K.br(a, K.br(b, c))), (1, K.br(b, K.br(c, a))), (1, K.br(c, K.br(a, b))))
    assert not j, "Jacobi fails"
print("Jacobi: 200 random triples OK")
gm2 = [mono(0, 0, 0, 0, 0)]
gm1 = [{lin_index(n): 1} for n in range(4)]
H = [quad_hamiltonian(K, A) for A in mats]
print("Hamiltonians exist:", [h is not None for h in H])
t = mono(1, 0, 0, 0, 0)
which = sys.argv[1:] or ["gl2", "cvect"]
for name in which:
    g0 = [t] + (H[:3] if name == "gl2" else H)
    comps = prolong(K, gm2, gm1, g0)
    dims = {d: len(v) for d, v in comps.items()}
    print(name, "dim g_0 =", len(g0), " total dim =", sum(dims.values()), dims)
