"""Statement 3.11.4: for III_1(3,2) there is no semi-invariant of degree 2, 3 or 4,
of any weight, on any piece F^e of the filtration."""
from fractions import Fraction as F
import stab, orb, sv, t3, inv
from probe import TYPES

p, q, terms = TYPES["III_1"](3, 2)
res = stab.analyse("III_1(3,2)", p, q, terms, 10, verbose=False)
L, classes, cert, notes = t3.classify(res)
print(f"III_1(3,2): dim Nt_f = {L.n}, {len(classes)} classes, "
      f"{'CERTIFIED' if cert else 'NOT CERTIFIED'}, "
      f"{sum(1 for c in classes if c['family'])} families")
for e in L.degrees:
    idx = [i for i in range(L.n) if L.degs[i] >= e]
    sub = []
    for M in L.span:
        S = [[M[i][j] for j in idx] for i in idx]
        if any(any(r) for r in S):
            sub.append(S)
    if not sub:
        continue
    for d in (2, 3, 4):
        zero, N = inv.invariants_of_degree(sub, len(idx), d, cap=4000)
        nz = 0 if zero is None else len([P for P in zero if inv.verify_zero(P, sub)])
        semi = inv.semi_invariants_fast(sub, len(idx), d, cap=4000)
        semi = [(P, w) for P, w in semi if inv.verify_semi(P, sub) == w]
        print(f"  F^{e} (dim {len(idx)}), degree {d}: {N if zero is not None else '-'} monomials, "
              f"{nz} invariant(s) of weight zero, {len(semi)} semi-invariant(s) of nonzero weight")
        for P, w in semi[:3]:
            print("      ", inv.show_poly(P, L, idx), " weights", w)
