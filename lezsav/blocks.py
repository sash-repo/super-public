"""Which components of Phi depend on the top block lam_N alone (N = 4, the
top of the level-2 window), with every other block arbitrary?
lam = (other blocks random_a, top block v):  compare Phi at a = a1, a2, a3
with v fixed; a component equal in all three depends on v only.  Then its
degree in v along a line (t = 0..5)."""
import os, sys, random, time
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
import multiprocessing as mp
import n4phi2 as P
from n4noeth import C
TAG = sys.argv[1]; TOP = int(sys.argv[2]) if len(sys.argv) > 2 else 4
NLO = -3
def lam(seed_other, t):
    n = len(P._G["ker0"]); out = []
    ro = random.Random(seed_other); rv = random.Random(99)
    for i in range(n):
        blk = NLO + i // 56
        if blk == TOP:
            out.append(C(rv.randint(-5, 5), rv.randint(-5, 5)) * t)
        elif blk > TOP:
            out.append(C())
        else:
            out.append(C(ro.randint(-5, 5), ro.randint(-5, 5)) if seed_other else C())
    return out
def ev(a):
    s, t = a
    ph, _ = P.phi_of_coeffs(P.coeffs_of_lam(lam(s, t)))
    return (s, t), ph
if __name__ == "__main__":
    P._init(TAG)
    jobs = [(s, 1) for s in (1, 2, 3)] + [(0, t) for t in range(6)]
    with mp.Pool(8, P._init, (TAG,)) as p:
        V = dict(p.map(ev, jobs))
    keys = set().union(*V.values())
    only = [k for k in keys if V[(1, 1)].get(k, C()) == V[(2, 1)].get(k, C()) == V[(3, 1)].get(k, C())
            and V[(1, 1)].get(k, C()) == V[(0, 1)].get(k, C())]
    from collections import Counter
    deg = Counter(); nz = Counter(); nonzero = 0
    for k in only:
        f = [V[(0, t)].get(k, C()) for t in range(6)]
        diffs = [f[0]]; d = f
        for o in range(1, 6):
            d = [d[i + 1] - d[i] for i in range(len(d) - 1)]; diffs.append(d[0])
        dg = max([o for o in range(6) if diffs[o]], default=-1)
        deg[(k[0], dg)] += 1
    print("top block n=%d: %d components of Phi depend on it alone; (kind, degree):"
          % (TOP, len(only)), sorted(deg.items()))
