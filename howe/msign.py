"""
msign.py -- fix the signs in Kochetkov's bracket

    [f,g]_M = Delta f . dg/dtau - (-1)^k df/dtau . Delta g + [f,g]_BB,
    k = (p(f)+1)(p(g)+1)

As printed this is not super-anticommutative for the parity p(f)+1 (products of
polynomials are supercommutative, so moving Delta g past d f/d tau costs a sign the
formula does not carry).  We look for the unique corrected sign pattern

    [f,g] = s1(p_f,p_g) . Delta f . d_tau g  +  s2(p_f,p_g) . d_tau f . Delta g
            + s3(p_f,p_g) . [f,g]_BB

that is super-anticommutative and satisfies the graded Jacobi identity, normalised by
s3 = +1 on the tau-free part (there the bracket must be Kochetkov's Buttin antibracket,
i.e. le(n) sits inside M(n) as ML(n)).
"""
import itertools
import sys
from fractions import Fraction as F
from le import padd, pscal, pmul, abr, is_hom_par
from mn import mono, Delta, dtau, layer, par_m, mdeg


def make(s1, s2, s3, n):
    def br(f, g):
        pf, pg = is_hom_par(f), is_hom_par(g)
        out = pscal(pmul(Delta(f, n), dtau(g, n)), s1[(pf, pg)])
        out = padd(out, pmul(dtau(f, n), Delta(g, n)), s2[(pf, pg)])
        out = padd(out, abr(f, g, n), s3[(pf, pg)])
        return out
    return br


def signs_from(bits):
    return {k: (1 if b else -1) for k, b in zip([(0, 0), (0, 1), (1, 0), (1, 1)], bits)}


def search(n=2, dmax=1):
    els = [f for j in range(-2, dmax + 1) for f in layer(n, j)]
    els = [(mdeg(f, n), f) for f in els]
    print("n = %d, %d basis elements of degree <= %d" % (n, len(els), dmax))

    good = []
    for b1 in itertools.product([0, 1], repeat=4):
        s1 = signs_from(b1)
        # antisymmetry forces s2(pf,pg) = (-1)^{pf} s1(pg,pf)
        s2 = {(pf, pg): (-1) ** pf * s1[(pg, pf)] for pf in (0, 1) for pg in (0, 1)}
        for b3 in itertools.product([0, 1], repeat=4):
            s3 = signs_from(b3)
            if s3[(0, 0)] != 1:
                continue
            br = make(s1, s2, s3, n)
            ok = True
            for d1, f in els:
                for d2, g in els:
                    s = (-1) ** (par_m(f) * par_m(g))
                    if padd(br(f, g), pscal(br(g, f), -s), -1):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            for d1, f in els:
                for d2, g in els:
                    for d3, h in els:
                        pf, pg = par_m(f), par_m(g)
                        r = padd(padd(br(f, br(g, h)), br(br(f, g), h), -1),
                                 br(g, br(f, h)), -(-1) ** (pf * pg))
                        if r:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                good.append((b1, b3))
                print("  OK   s1 =", s1, "  s2 =", s2, "  s3 =", s3)
    print("%d sign patterns survive" % len(good))
    return good


if __name__ == '__main__':
    search(int(sys.argv[1]) if len(sys.argv) > 1 else 2,
           int(sys.argv[2]) if len(sys.argv) > 2 else 1)
