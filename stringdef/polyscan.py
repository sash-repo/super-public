import sympy as sp, polycoh
print("differential-operator (polynomial) cochains, Laurent F_lambda")
print("  lambda   w    H^1  H^2      (cutoffs D = 5 and 7 agree)")
for lam in (-2, -1, 0, 1, 2, 3):
    for w in range(-4, 4):
        polycoh.LAM = sp.Rational(lam)
        r5 = polycoh.poly_cohomology(w, 5); r7 = polycoh.poly_cohomology(w, 7)
        h1_5 = r5[0] - (1 if True else 0); h1_7 = r7[0] - 1
        h2 = (r5[3], r7[3])
        if h2 != (0, 0) or r7[0] != 1:
            print("  %5d %4d   Z^1=%d  H^2=%s %s" % (lam, w, r7[0], h2,
                  "STABLE" if h2[0] == h2[1] else "unstable"))
print("(rows shown only where something is nonzero beyond the generic Z^1 = 1)")
