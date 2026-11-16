"""Subsec. 3.9.1: the filtration by degree is broken for III_1(5,4) and III_1(6,5),
and the quasi-homogeneous weights repair it."""
from fractions import Fraction as F
import stab, orb, sv, t3
from probe import TYPES

for (n, m) in [(5, 4), (6, 5), (5, 3), (3, 2)]:
    p, q, terms = TYPES["III_1"](n, m)
    res = stab.analyse(f"III_1({n},{m})", p, q, terms, 2*n+4, verbose=False)
    L = t3.Layer(res)
    degs = [stab.mdeg(b) for b in L.basis]
    bad = [(i, j) for M in L.mats for i in range(L.n) for j in range(L.n)
           if M[i][j] and degs[i] < degs[j]]
    uniq = sorted(set(bad))
    print(f"\nIII_1({n},{m}) = x^{n} + x.xi_1.xi_2 + x^{m}.xi_3.xi_4")
    print(f"    naive degree filtration: {len(bad)} offending matrix entries "
          f"({len(uniq)} distinct positions)")
    for i, j in uniq[:4]:
        print(f"      {sv.show_mon(L.basis[j],'x','xi')} (degree {degs[j]}) "
              f"--> {sv.show_mon(L.basis[i],'x','xi')} (degree {degs[i]})")
    w = t3.quasi_weights(res["f"], p, q)
    if w is None:
        print("    no quasi-homogeneous weights")
        continue
    wx, wxi = w
    print(f"    quasi-homogeneous weights: w(x) = {wx[0]}, "
          f"w(xi) = {wxi}   (all positive: {all(t > 0 for t in wxi)})")
    wdeg = [sum(e * wx[i] for i, e in enumerate(b[0])) + sum(wxi[j] for j in b[1])
            for b in L.basis]
    bad2 = [(i, j) for M in L.mats for i in range(L.n) for j in range(L.n)
            if M[i][j] and wdeg[i] < wdeg[j]]
    print(f"    weighted filtration: {len(bad2)} offending entries; "
          f"L.degs is the {'weighted' if L.weights else 'naive'} one")
    if uniq:
        i, j = uniq[0]
        print(f"      the pair above now has weights {wdeg[j]} --> {wdeg[i]}")
    # the weight of f itself
    wt = lambda mo: sum(e * wx[k] for k, e in enumerate(mo[0])) + sum(wxi[k] for k in mo[1])
    print(f"    weights of the monomials of f: {sorted({wt(mo) for mo in res['f']})}")
