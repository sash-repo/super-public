"""The orbit census for the odd (degree-1) obstructions, m = 1 and m = 2."""
import orbits
print("m = 1:  E = O(-k),  T_1 = Theta (x) E = O(2-k),  H^1 = R_{k-4}")
for k in range(4, 12):
    orbits.report([k])
print()
print("m = 2:  E = O(-a) + O(-b),  T_1^{(i)} = V (x) O(1-k_i) for k_j =/= 0")
for a in range(1, 8):
    for b in range(0, a + 1):
        orbits.report([a, b])
