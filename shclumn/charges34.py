"""The charges C_g(a) for k^M(1|3), k^M(1|4), k^L(1|4): all vanish (cf. ghost.py)."""
import sys, os
from fractions import Fraction as Q
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import yb
from ghost import check_virasoro, check_rep, central_charge
def main():
    LOG = open(os.devnull, "w")
    def say(s=""):
        print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
    W = [Q(-1), Q(0), Q(1, 2), Q(1), Q(3, 2)]
    for name, alg in [("k^M(1|3)", yb.Ramond(3)), ("k^M(1|4)", yb.Ramond(4)),
                      ("k^L(1|4)", yb.Contact(4))]:
        say("%s Virasoro: %s" % (name, "OK" if not check_virasoro(alg, 2) else "FAIL"))
        b, t = check_rep(alg, Q(1, 2), 2, span=range(-1, 2))
        say("   rho_{1/2} rep: %s (%d triples)" % ("OK" if not b else "FAIL %d" % b, t))
        for a in W:
            C, ok, vals = central_charge(alg, a, 2)
            say("   a=%-5s C=%s %s" % (a, C, "" if ok else "INCONSISTENT %s" % vals))
    LOG.close()
main()
