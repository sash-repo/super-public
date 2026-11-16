"""Which rows does no level-4 unknown reach, and how do they depend on lam?
Along a random line lam = t v (v random), interpolate each free row at
t = 0..6 and read off its degree (exact: degree <= 4 must come out)."""
import os, sys, random
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
import multiprocessing as mp
from fractions import Fraction as Q
import n4phi2 as P
from n4noeth import C
TAG = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "all"   # all | top
def lamvec(t):
    rng = random.Random(11)
    out = []
    for i, _ in enumerate(P._G["ker0"]):
        c = C(rng.randint(-5, 5), rng.randint(-5, 5))
        if MODE == "top" and i < 448 - 56:
            c = C()
        out.append(c * t)
    return out
def ev(t):
    ph, _ = P.phi_of_coeffs(P.coeffs_of_lam(lamvec(t)))
    return t, {k: v for k, v in ph.items() if k[0] == "free"}
if __name__ == "__main__":
    P._init(TAG)
    with mp.Pool(7, P._init, (TAG,)) as p:
        V = dict(p.map(ev, range(7)))
    keys = set().union(*V.values())
    # Newton forward differences: degree = last nonzero difference order
    from collections import Counter
    deg = Counter(); slots = Counter(); low = Counter()
    for k in keys:
        f = [V[t].get(k, C()) for t in range(7)]
        d = f[:]; dg = -1; lo = None
        diffs = [d[0]]
        for o in range(1, 7):
            d = [d[i + 1] - d[i] for i in range(len(d) - 1)]
            diffs.append(d[0])
        dg = max([o for o in range(7) if diffs[o]], default=-1)
        deg[dg] += 1
        slots[(k[1][0], k[1][1], len(k[1][2][3]))] += 1
    print("free rows:", len(keys), "degrees along the line:", sorted(deg.items()))
    print("by (C/E, slot, theta-level):", sorted(slots.items()))
