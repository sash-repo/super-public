"""
Parity split of H^2(k^L'(1|4); C) in weight 0 (theta-realisation), and the
independence of the three cochains alpha, beta, gamma of the corrected table
(70) modulo coboundaries.
"""
from fractions import Fraction as Q
from nsTheta import monomials, weight, name_of, table_kvdl, c_mono
from solveTheta import (Unknowns, by_weight, eqs, coboundaries, reduce_rows,
                        nullspace, rank, S0KEY)

B, kappa = 2, Q(1)
A = 2 * B + 1
keysA = [k for k in monomials(range(-A, A + 1)) if k != S0KEY]
keysB = [k for k in monomials(range(-B, B + 1)) if k != S0KEY]
U = Unknowns()
WA = by_weight(keysA)
for k1 in keysB:
    for k2 in WA.get(-weight(k1), ()):
        U.get(k1, k2)
nB = len(U.keys)
piv = reduce_rows(list(eqs(U, keysB, kappa)))
Z = [v[:nB] for v in nullspace(piv, len(U.keys))]
Bd = coboundaries(U, keysB, nB, kappa)
dZ, dB = rank(Z, nB), rank(Bd, nB)
print("dim Z = %d, dim B = %d, dim H^2 (weight 0) = %d" % (dZ, dB, dZ - dB))

# parity split
pz = reduce_rows([{i: x for i, x in enumerate(v) if x} for v in Z])
def par_of_unknown(i):
    k1, k2 = U.keys[i]
    return (len(k1[1]) + len(k2[1])) % 2
ev = [v for v in Z if all(par_of_unknown(i) == 0 for i in range(nB) if v[i])]
print("cocycles supported on even pairs: rank %d" % rank(ev, nB) if ev else
      "cocycles supported on even pairs: rank 0")
# the even part of Z: project every cocycle on its even and odd halves
Zev = [[v[i] if par_of_unknown(i) == 0 else Q(0) for i in range(nB)] for v in Z]
Zod = [[v[i] if par_of_unknown(i) == 1 else Q(0) for i in range(nB)] for v in Z]
Bev = [[v[i] if par_of_unknown(i) == 0 else Q(0) for i in range(nB)] for v in Bd]
Bod = [[v[i] if par_of_unknown(i) == 1 else Q(0) for i in range(nB)] for v in Bd]
print("  even part: dim Z_ev = %d, dim B_ev = %d, dim H_ev = %d"
      % (rank(Zev, nB), rank(Bev, nB), rank(Zev, nB) - rank(Bev, nB)))
print("  odd  part: dim Z_od = %d, dim B_od = %d, dim H_od = %d"
      % (rank(Zod, nB), rank(Bod, nB), rank(Zod, nB) - rank(Bod, nB)))

# the three cochains of the corrected table, as vectors on the window
def build(al, be, ga):
    v = [Q(0)] * nB
    for i in range(nB):
        k1, k2 = U.keys[i]
        c = c_mono(k1, k2, table_kvdl)
        v[i] = al * c[0] + be * c[1] + ga * c[2]
    return v

A_, Bt, G = build(1, 0, 0), build(0, 1, 0), build(0, 0, 1)
print()
print("the three cochains of the corrected table:")
print("   each is a cocycle:",
      all(rank(Z + [v], nB) == dZ for v in (A_, Bt, G)))
print("   independent modulo coboundaries:",
      rank(Bd + [A_, Bt, G], nB) - dB == 3)
print("   so they span H^2 (weight 0):",
      rank(Bd + [A_, Bt, G], nB) == dZ)
