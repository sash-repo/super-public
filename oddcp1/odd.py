"""The odd (degree-1) obstructions: the quotient against the whole sheaf.

Over a base Spec C with C_odd non-zero the automorphism supersheaf acquires its
odd-degree layers, and the degree-1 layer is H^1(M; T_1).  The sheaf T_1 is the
non-split extension

        0 --> Lambda^2 E (x) E^*  -->  T_1  -->  Theta (x) E --> 0,

and Theta (x) E = Theta (x) E^1(E) is the space in which [LT, eq. (9)] reads
the degree-1 obstructions.  For m = 1 the sub vanishes and nothing is lost --
which is the case treated in [L] = "On odd parameters in geometry".  For m = 2
it does not vanish as soon as one of the two degrees is at least 2.

The dimensions of the pieces of T_1 are read off B--O's Prop. 18 (confirmed
against the honest count of global vector fields in run.py, Part A).
"""

from ns import h0, h1

print("m = 1: E = O(-k)")
print(f"{'k':>4} {'dim H^1(Theta(x)E) = [LT] (8)':>30} {'dim H^1(T_1)':>14}")
for k in range(0, 9):
    print(f"{k:>4} {h1(2 - k):>30} {h1(2 - k):>14}")

print()
print("m = 2: E = O(-a) + O(-b),  T_1 = T_1^{(1)} + T_1^{(2)},")
print("       T_1^{(i)} = L_{-k_i} + L_{2-k_i} if k_j = 0, else 2 L_{1-k_i}")
print(f"{'(a,b)':>8} {'[LT] (9): H_a + H_b':>22} {'sub Lambda^2E(x)E^*':>22}"
      f" {'whole H^1(T_1)':>16} {'H^0(T_1)':>10}")
for a in range(0, 8):
    for b in range(0, a + 1):
        ks = [a, b]
        lt = h1(2 - a) + h1(2 - b)
        sub = h1(b - a - b) + h1(a - a - b)          # Lambda^2E (x) E_j^*
        whole1 = whole0 = 0
        for i in (0, 1):
            ki, kj = ks[i], ks[1 - i]
            degs = [-ki, 2 - ki] if kj == 0 else [1 - ki, 1 - ki]
            whole1 += sum(h1(d) for d in degs)
            whole0 += sum(h0(d) for d in degs)
        flag = "" if whole1 == lt else "   <-- the sub is seen"
        print(f"{str((a,b)):>8} {lt:>22} {sub:>22} {whole1:>16} {whole0:>10}"
              f"{flag}")
