"""The Mumford exponents: the second parameter of k^L(1|2) in the xi-eta frame."""
from fractions import Fraction as Q
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ghost import Densities, DensitiesK12, gamma, virasoro, check_rep_mod
from xieta import ContactXiEta

LOG = open(os.devnull, "w")
def say(s=""):
    print(s); LOG.write(s + "\n"); LOG.flush()

def charge(mod, alg, ms=(2, 3, 4)):
    g1 = gamma(mod, virasoro(alg, 1, 2), virasoro(alg, -1, 2), 0)
    vals = []
    for m in ms:
        gm = gamma(mod, virasoro(alg, m, 2), virasoro(alg, -m, 2), 0, pad=4*m+6)
        vals.append(Q(12, m**3 - m) * (gm - g1 * m))
    return (vals[0], all(v == vals[0] for v in vals))

say("k^L(1|2) in the hyperbolic frame: C(a,b)")
say("=" * 62)
for sg in (1, -1):
    alg = ContactXiEta(sg)
    bad, tot = alg.check_jacobi()
    say("\nsigma=%+d  Jacobi: %s" % (sg, "OK" if not bad else "FAILS %d" % bad))
    # b = 0 must reproduce the theta-frame answer, C = -6
    ok0 = [charge(Densities(alg, a, 2), alg) for a in (Q(-1), Q(0), Q(1), Q(2))]
    say("   b=0:  C(a) for a=-1,0,1,2 : %s   (theta frame gave -6)"
        % "  ".join("%s%s" % (c, "" if k else "!") for c, k in ok0))
    for s in (1, -1):
        bad, tot = check_rep_mod(DensitiesK12(alg, Q(1, 2), Q(1), 2, s), alg)
        say("   Div_2 sign s=%+d: rho_{a,b} a representation: %s (%d triples)"
            % (s, "OK" if not bad else "FAILS in %d" % bad, tot))
        if bad:
            continue
        for b in (Q(0), Q(1, 2), Q(1), Q(2), Q(-1), Q(-2)):
            row = []
            for a in (Q(-1), Q(0), Q(1, 2), Q(1)):
                c, k = charge(DensitiesK12(alg, a, b, 2, s), alg)
                row.append("%s%s" % (c, "" if k else "!"))
            say("      b=%-5s C for a=-1,0,1/2,1:  %s" % (b, "  ".join(row)))
LOG.close()
