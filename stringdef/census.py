"""The census: H^2(g; g) in a window, Laurent against polynomial coefficients.

Run:  ../venv/bin/python census.py [names...]

(names select the algebras of sections 1, 2, 4; the name `svect` selects
section 3.)

Each line reports

    h(N, M) = dim Z|core - dim B|core

for a window of N steps and a core of M steps -- a decreasing-in-N upper bound
for the number of classes of H^2 visible on the core (see defco.py).
"""

from fractions import Fraction as Q
import sys, time
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]

from defco import Graded, h2_core, algebras
import yb

def say(s):
    print(s, flush=True)


def sweep(g, Ns, M, weight=0, cochain_parity=0):
    out = []
    for N in Ns:
        t0 = time.time()
        try:
            r = h2_core(g, N, M, weight, cochain_parity)
        except Exception as e:
            say("  %-18s N=%-2d  ERROR %s: %s" % (g.name, N, type(e).__name__, e))
            break
        out.append(r["h"])
        say("  %-18s N=%-2d M=%-2d  vec=%-4d unk=%-6d rows=%-7d "
            "Z|c=%-4d B|c=%-4d  h=%-3d  [%.1fs]"
            % (g.name, N, M, r["vectors"], r.get("unknowns", 0), r.get("rows", 0),
               r.get("Z", 0), r.get("B", 0), r["h"], time.time() - t0))
    return out


def main():
    want = sys.argv[1:]
    say("\n" + "=" * 78)
    say("H^2(g;g) census  %s" % time.strftime("%Y-%m-%d %H:%M"))
    say("=" * 78)

    say("\n--- 1. Laurent coefficients (stringy), weight 0, even cochains")
    for g in algebras(laurent=True):
        if want and g.name not in want:
            continue
        sweep(g, (3, 4, 5, 6), 2)

    say("\n--- 2. polynomial coefficients, the same computation")
    for g in algebras(laurent=False):
        if want and g.name not in want:
            continue
        sweep(g, (4, 5, 6, 7), 2)

    say("\n--- 3. svect^L_lambda(1|2): the one-parameter family")
    for lam in () if (want and "svect" not in want) else (Q(0), Q(1), Q(1, 3), Q(2, 5)):
        g = Graded(yb.SVect(lam), "svect^L_{%s}(1|2)" % lam)
        sweep(g, (3, 4, 5), 2)

    say("\n--- 4. odd cochains (deformations with odd parameters), Laurent")
    for g in algebras(laurent=True):
        if want and g.name not in want:
            continue
        sweep(g, (3, 4, 5), 2, 0, 1)

    say("\ndone")


if __name__ == "__main__":
    main()
