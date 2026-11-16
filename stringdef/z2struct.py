"""What a 2-cocycle in the truncation is determined by, and the lambda-scan."""
import witt2
from exactlin import rank, nullspace

m = witt2.make_module('adj')
print("Z^2(Q_N) in weight w: dimension, and whether the coordinates")
print("  c(1,2),...,c(1,N-1), c(2,3)   [that is N-1 of them]  form a basis of its dual")
for w in (-5, -1, 0, 1, 3, 7):
    line = []
    for N in (10, 14, 18, 22, 26):
        d2, s2, _ = witt2.differential(2, N, w, m)
        Z = nullspace(d2)
        idx = {I: k for k, I in enumerate(s2)}
        coords = [(1, j) for j in range(2, N)] + [(2, 3)]
        coords = [c for c in coords if c in idx]
        ev = [[v[idx[c]] for c in coords] for v in Z]
        line.append("N=%d: dimZ2=%-3d rk(ev)=%-3d %s" %
                    (N, len(Z), rank(ev) if ev and ev[0] else 0,
                     "iso" if (len(Z) == N - 1 and rank(ev) == len(Z)) else "NO"))
    print(" w=%3d  " % w + " | ".join(line))

print("\nlambda-scan: H^2 of witt_- with coefficients in the LAURENT module F_lambda")
print("(lambda = -1 is witt itself).  Weights with H^2 != 0, at N = 20 and N = 24.")
from fractions import Fraction
for num in range(-8, 9):
    lam = Fraction(num, 2)
    mod = witt2.make_module("F%s" % lam)
    hits = []
    for w in range(-12, 13):
        h20 = witt2.cohomology(2, 20, w, mod)[0]
        h24 = witt2.cohomology(2, 24, w, mod)[0]
        if h24:
            hits.append("w=%d:%d" % (w, h24))
    print("  lambda=%-5s  %s" % (lam, ", ".join(hits) if hits else "0 in every weight |w|<=12"))
