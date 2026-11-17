"""The binary atoms over all weights: Grozman's theorem, as our calibration."""
import sys
from fractions import Fraction as F
from line import inv_dim, atoms

print("order-2 binary invariants live on a*b*(a+b+1)=0; which are indecomposable?")
for a, b in [(F(1), F(-2)), (F(2), F(-3)), (F(1,2), F(-3,2)), (F(0), F(3)),
             (F(3), F(0)), (F(0), F(0)), (F(-1,2), F(-1,2))]:
    d, dd, _ = atoms((a, b), 2)
    print(f"   ({a},{b}): dim {d}, decomposable {dd}, indecomposable {d-dd}")
print()
print("order 3 on the Grozman line and around it:")
for a, b in [(F(-2,3), F(-2,3)), (F(-2,3), F(-1,3)), (F(-1), F(-1)),
             (F(-1), F(0)), (F(-3,2), F(-3,2)), (F(-1,2), F(-1,2))]:
    d, dd, _ = atoms((a, b), 3)
    print(f"   ({a},{b}): dim {d}, decomposable {dd}, indecomposable {d-dd}")
print()
print("sweep: every (a,b) with a,b in (1/2)Z, -4<=a<=b<=4, order<=6, atoms only")
found = []
for ai in range(-8, 9):
    for bi in range(ai, 9):
        a, b = F(ai, 2), F(bi, 2)
        for s in range(0, 7):
            d, dd, _ = atoms((a, b), s)
            if d - dd > 0:
                found.append((a, b, s, d - dd))
for a, b, s, m in found:
    if s >= 2:
        print(f"   ({a},{b}) order {s}: {m} indecomposable")
print(f"   (orders 0 and 1 give an atom at almost every weight: "
      f"{sum(1 for x in found if x[2] < 2)} entries)")
