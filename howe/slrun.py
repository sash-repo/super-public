"""
slrun.py -- driver for slcheck.py.

Self-tests first (nothing is claimed unless all of them pass):

  (S1)  d^2 = 0;
  (S2)  L_D(omega) = 0 for every generating function f of degree <= dmax,
        i.e. the realisation really is le(n) and the Lie derivative is right;
  (S3)  [L_D, d] = 0;
  (S4)  Y_+^2 = Y_-^2 = 0, hence <Y_+, Y_-, E = [Y_+,Y_-]> is a quotient of
        sl(1|1) with E central;
  (S5)  [E, Y_+] = [E, Y_-] = 0 (E is central) and E is not 0.

Then the question the fibre cannot see:

  (Q)   for which D = D_f is [L_D, Y_-] = 0 ?
        [L2] says SL(n), i.e. Delta f = 0.  We compute the answer.

Usage:  python3 slrun.py [n] [cmax] [fmax] [dmax]
"""

import sys
from fractions import Fraction as F

import slcheck as S
from le import D as VF_of


_HOMTEST = 12      # how many generating functions enter the homomorphism test
_HOMSTEP = 7       # stride through the test space for that (quadratic) test


def nonzero(e):
    return {m: c for m, c in e.items() if c != 0}


def run(n, cmax, fmax, dmax):
    print("=" * 74)
    print("n = %d   coefficients of degree <= %d,  forms of degree <= %d,"
          "  f of degree <= %d" % (n, cmax, fmax, dmax))
    print("=" * 74)

    mons = S.monomials(n, cmax, fmax)
    tests = [{m: F(1)} for m in mons]
    print("test space: %d monomials of Omega" % len(mons))

    om = S.omega(n)
    fs = S.gen_functions(n, dmax)
    vfs = [(f, VF_of(f, n)) for f in fs]

    # ---------------------------------------------------------------- (S1)
    bad = [m for m, e in zip(mons, tests) if nonzero(S.d_ext(n, S.d_ext(n, e)))]
    assert not bad, "d^2 != 0 on %s" % (bad[:3],)
    print("(S1) d^2 = 0                                              OK")

    # ---------------------------------------------------------------- (S2)
    bad = [S.fname(n, f) for f, v in vfs if nonzero(S.lie(n, v, om))]
    assert not bad, "L_D(omega) != 0 for f = %s" % bad[:5]
    print("(S2) L_D(omega) = 0 for all %3d generating functions      OK" % len(fs))

    # ---------------------------------------------------------------- (S3)
    bad = []
    for f, v in vfs:
        for e in tests:
            # [L_D, d] = L_D d - (-1)^{p(D)} d L_D
            r = S.lie(n, v, S.d_ext(n, e))
            S.addall(r, S.d_ext(n, S.lie(n, v, e)), -((-1) ** v.par))
            if nonzero(r):
                bad.append(S.fname(n, f))
                break
    assert not bad, "[L_D, d] != 0 for f = %s" % bad[:5]
    print("(S3) [L_D, d] = 0                                         OK")

    # ---------------------------------------------------------------- (S6)
    # D -> L_D is a homomorphism of Lie superalgebras:
    #      L_{[D,D']} = [L_D, L_D'] = L_D L_D' - (-1)^{pp'} L_D' L_D.
    bad = []
    for f, v in vfs[:_HOMTEST]:
        for g, w in vfs[:_HOMTEST]:
            br = v.bracket(w)
            for e in tests[::_HOMSTEP]:
                r = S.lie(n, v, S.lie(n, w, e))
                S.addall(r, S.lie(n, w, S.lie(n, v, e)),
                         -((-1) ** (v.par * w.par)))
                S.addall(r, S.lie(n, br, e), -1)
                if nonzero(r):
                    bad.append((S.fname(n, f), S.fname(n, g)))
                    break
            if bad:
                break
        if bad:
            break
    assert not bad, "L is not a homomorphism: %s" % bad[:3]
    print("(S6) L_{[D,D']} = [L_D, L_D']                             OK")

    # ---------------------------------------------------------------- (S4)
    bad = [m for m, e in zip(mons, tests) if nonzero(S.Yplus(n, S.Yplus(n, e)))]
    assert not bad, "Y_+^2 != 0 on %s" % (bad[:3],)
    bad = [m for m, e in zip(mons, tests) if nonzero(S.Yminus(n, S.Yminus(n, e)))]
    assert not bad, "Y_-^2 != 0 on %s" % (bad[:3],)
    print("(S4) Y_+^2 = Y_-^2 = 0                                    OK")

    # ---------------------------------------------------------------- (S5)
    for U, nm in ((S.Yplus, "Y_+"), (S.Yminus, "Y_-")):
        bad = [m for m, e in zip(mons, tests)
               if nonzero(S.comm(n, S.Ecart, 0, U, 1, e))]
        assert not bad, "[E, %s] != 0 on %s" % (nm, bad[:3])
    nz = [m for m, e in zip(mons, tests) if nonzero(S.Ecart(n, e))]
    assert nz, "E = 0"
    print("(S5) [E, Y_+] = [E, Y_-] = 0,  E != 0  =>  sl(1|1)        OK")

    # -------------------------------------------------- Y_+ is le(n)-invariant
    bad = []
    for f, v in vfs:
        LD = (lambda vv: (lambda nn, ee: S.lie(nn, vv, ee)))(v)
        for e in tests:
            if nonzero(S.comm(n, LD, v.par, S.Yplus, 1, e)):
                bad.append(S.fname(n, f))
                break
    print("     [L_D, Y_+] != 0 for %d of %d f's%s"
          % (len(bad), len(fs), ("  -> " + " ".join(bad[:6])) if bad else "   (Y_+ is le(n)-invariant)"))

    # ------------------------------------------------------------------- (Q)
    print()
    print("  f                     Delta f = 0 ?   [L_D, Y_-] = 0 ?")
    print("  " + "-" * 54)
    rows = []
    for f, v in vfs:
        harmonic = not S.laplace(n, f)
        inv = True
        witness = None
        LD = (lambda vv: (lambda nn, ee: S.lie(nn, vv, ee)))(v)
        for m, e in zip(mons, tests):
            if nonzero(S.comm(n, LD, v.par, S.Yminus, 1, e)):
                inv = False
                witness = m
                break
        rows.append((S.fname(n, f), harmonic, inv, witness))
        print("  %-20s  %-13s   %s%s"
              % (S.fname(n, f), "yes" if harmonic else "no",
                 "yes" if inv else "NO",
                 "" if inv else "   (fails on %s)" % (witness,)))

    agree = all(h == i for _, h, i, _ in rows)
    print()
    print("  Delta f = 0  <=>  [L_D, Y_-] = 0 :  %s" % ("YES" if agree else "NO"))
    if not agree:
        mism = [(nm, h, i) for nm, h, i, _ in rows if h != i]
        print("  mismatches (%d): %s" % (len(mism), mism[:12]))
    inv_all = all(i for _, _, i, _ in rows)
    print("  Y_- invariant under all of le(n)  :  %s" % ("YES" if inv_all else "no"))
    print()
    return rows


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    cmax = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    fmax = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    dmax = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    run(n, cmax, fmax, dmax)
