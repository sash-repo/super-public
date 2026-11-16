"""mcheck0.py -- self-tests of the M(n) algebra layer in mn.py."""
import sys
from fractions import Fraction as F
from le import padd, pscal
from mn import (mono, one, tau, mbr, Mvf, mdiv, mdeg, mwt, par_m, layer, Delta)


def eq(p, q):
    return not padd(p, q, -1)


def test(n, dmax=3):
    print("=== n = %d ===" % n)
    els = []
    for j in range(-2, dmax + 1):
        for f in layer(n, j):
            els.append((j, f))
    print("basis elements up to degree %d: %d" % (dmax, len(els)))

    # 1. the map f -> 'f is a homomorphism onto vector fields
    bad = 0
    tested = 0
    for j1, f in els:
        for j2, g in els:
            if j1 + j2 > dmax:
                continue
            lhs = Mvf(mbr(f, g, n), n)
            rhs = Mvf(f, n).bracket(Mvf(g, n))
            for c in range(2 * n + 1):
                if not eq(lhs.comp.get(c, {}), rhs.comp.get(c, {})):
                    bad += 1
                    if bad < 4:
                        print("  MISMATCH", f, g, "comp", c,
                              lhs.comp.get(c, {}), rhs.comp.get(c, {}))
                    break
            tested += 1
    print("  '[f,g] = ['f,'g] : %d pairs, %d bad" % (tested, bad))

    # 2. super anti-symmetry and Jacobi for mbr
    bad = 0
    for j1, f in els:
        for j2, g in els:
            if j1 + j2 > dmax:
                continue
            s = (-1) ** (par_m(f) * par_m(g))
            if not eq(mbr(f, g, n), pscal(mbr(g, f, n), -s)):
                bad += 1
    print("  antisymmetry: %d bad" % bad)

    bad = tested = 0
    for j1, f in els:
        for j2, g in els:
            for j3, h in els:
                if j1 + j2 + j3 > dmax:
                    continue
                pf, pg, ph = par_m(f), par_m(g), par_m(h)
                t1 = mbr(f, mbr(g, h, n), n)
                t2 = mbr(mbr(f, g, n), h, n)
                t3 = mbr(g, mbr(f, h, n), n)
                r = padd(padd(t1, t2, -1), t3, -(-1) ** (pf * pg))
                if r:
                    bad += 1
                    if bad < 4:
                        print("  JACOBI", f, g, h, r)
                tested += 1
    print("  Jacobi: %d triples, %d bad" % (tested, bad))

    # 3. gradedness
    bad = 0
    for j1, f in els:
        for j2, g in els:
            b = mbr(f, g, n)
            if b and mdeg(b, n) != j1 + j2:
                bad += 1
    print("  gradedness: %d bad" % bad)

    # 4. weights add up
    bad = 0
    for j1, f in els:
        for j2, g in els:
            b = mbr(f, g, n)
            if b:
                try:
                    w = mwt(b, n)
                except AssertionError:
                    bad += 1
                    continue
                if w != tuple(u + v for u, v in zip(mwt(f, n), mwt(g, n))):
                    bad += 1
    print("  weights additive: %d bad" % bad)

    # 5. the negative part
    D = one(n)
    parts = [pscal(mono(n, [0] * n, [i]), -1) for i in range(n)]   # 'partial_i = '(-xi_i)
    dels = [mono(n, [1 if k == i else 0 for k in range(n)], []) for i in range(n)]
    ok = True
    for i in range(n):
        for j in range(n):
            b = mbr(parts[i], dels[j], n)
            want = pscal(D, 2) if i == j else {}
            if not eq(b, want):
                ok = False
                print("  ['partial_%d,'delta_%d] = %s" % (i, j, b))
            if mbr(parts[i], parts[j], n) or mbr(dels[i], dels[j], n):
                ok = False
    print("  L_{-1} brackets ['p_i,'d_j] = 2 delta_ij D, rest 0 :", ok)
    print("  [D,D] =", mbr(D, D, n), "  [D,L_{-1}] =",
          [mbr(D, p, n) for p in parts + dels])

    # 6. div: z = 'tau, h = 'sum x_i xi_i, SM(n) = M_{a,b} at a + nb = 0
    z = tau(n)
    hh = {}
    for i in range(n):
        hh = padd(hh, mono(n, [1 if k == i else 0 for k in range(n)], [i]))
    print("  div z =", mdiv(z, n), "  div h =", mdiv(hh, n))
    comb = padd(pscal(z, n), hh, -1)          # a = n, b = -1  ->  a + nb = 0
    print("  div (n z - h) =", mdiv(comb, n), " (should be 0: SM(n))")

    # 7. z acts as the grading operator
    bad = 0
    for j, f in els:
        if not eq(mbr(z, f, n), pscal(f, j)):
            bad += 1
    print("  ad z = grading: %d bad" % bad)

    # 8. centre of the whole thing (degrees <= dmax)
    print("  dim L_j:", {j: len(layer(n, j)) for j in range(-2, dmax + 1)})


if __name__ == '__main__':
    for n in [int(a) for a in sys.argv[1:]] or [2, 3]:
        test(n, 3 if n <= 3 else 2)
