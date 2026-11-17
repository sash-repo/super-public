"""The complete classification for m = 2 over Spec Lambda(tau): the orbits in
H^1(T_1) + H^1(T_2), the odd and the even layer together."""
from fractions import Fraction
import oddpar as O


def samples(n):
    pr, k, out = [], 2, []
    while len(pr) < n:
        if all(k % p for p in pr):
            pr.append(k)
        k += 1
    out.append(pr)
    out.append([1] * n)
    out.append([(i * i + 3) % 7 + 1 for i in range(n)])
    out.append([1 if i % 2 else 2 for i in range(n)])
    return out


print(f"{'retract':>9} {'dim H^1(T_1)':>13} {'dim H^1(T_2)':>13} {'dim P':>6}"
      f" {'max orbit':>10} {'moduli <=':>10}   [LT]'s even moduli")
print("-" * 92)
for a in range(2, 7):
    for b in range(0, a + 1):
        S = O.Sum([a, b])
        if S.dim == 0:
            continue
        best = -1
        for s in samples(S.dim):
            v = [Fraction(c) for c in s]
            best = max(best, S.orbit_dim_proj(v))
        for i in range(S.dim):
            v = [Fraction(1 if j == i else 0) for j in range(S.dim)]
            best = max(best, S.orbit_dim_proj(v))
        P = S.dim - 1
        lt = a + b - 4
        print(f"{str((a,b)):>9} {S.dims[0]:>13} {S.dims[1]:>13} {P:>6}"
              f" {best:>10} {P - best:>10}   P^{lt}" if lt >= 0 else
              f"{str((a,b)):>9} {S.dims[0]:>13} {S.dims[1]:>13} {P:>6}"
              f" {best:>10} {P - best:>10}   empty", flush=True)
