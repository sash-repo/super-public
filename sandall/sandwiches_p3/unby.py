"""Statement SSkrheight: by(7;un), sby(7;un) (z-height 2), smy(6;un) (one y-height 2): is (ad d_top)^3 nonzero?"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); import prolong as PR
import unskry
from dpvect import DPVect
LOG = os.path.join(HERE, "runs", "unby1766.log")
log = open(LOG, "w", buffering=1)
def say(m): print(m, flush=True); log.write(m + "\n")
dmax = int(sys.argv[2]) if len(sys.argv) > 2 else 8
for kind in sys.argv[1].split(","):
    V0, gminus, layers0 = PR.build(kind)
    n = V0.n
    tops = [i for i in range(n) if V0.degs[i] == max(V0.degs)]
    for top in tops[:1] if kind in ("by", "sby") else tops:
        h = [1]*n; h[top] = int(os.environ.get("HT", "2"))
        V = DPVect(V0.degs, V0.wts, h, V0.names)
        layers = {d: [dict(D) for D in L] for d, L in layers0.items()}
        unskry.prolong(V, layers, dmax, lambda m: None)
        dtop = {(tuple([0]*n), top): 1}
        nz = 0; ex = None; dimt = sum(len(L) for L in layers.values())
        for dg, fields in sorted(layers.items()):
            for D in fields:
                B = V.bracket(dtop, V.bracket(dtop, V.bracket(dtop, D)))
                if B:
                    nz += 1
                    if ex is None: ex = (dg, V.show(B))
        say("%s heights %s: dim through degree %d = %d; (ad d_%s)^3 nonzero on %d fields; first %s"
            % (kind, h, max(layers), dimt, V0.names[top], nz, ex))
