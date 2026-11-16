"""Statement SSkrheight 1): my(6;un) with one y-height raised, each of the three; dmax 5 (the witness is in g_4)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); import unskry
LOG = os.path.join(HERE, "runs", "unskry1766.log")
log = open(LOG, "w", buffering=1)
def say(m): print(m); log.write(m + "\n")
for k in range(3):
    h = [1]*6; h[3+k] = 2
    V, X, Y, g0 = unskry.build(tuple(h))
    layers = {-1: X, -2: Y, 0: g0}
    unskry.prolong(V, layers, 5, say)
    dy = Y[k]; nz = 0; ex = None
    for dg, fields in sorted(layers.items()):
        for D in fields:
            B = V.bracket(dy, V.bracket(dy, V.bracket(dy, D)))
            if B:
                nz += 1
                if ex is None: ex = (dg, V.show(B))
    say("heights %s: (ad d_y%d)^3 nonzero on %d fields of degree<=5; first %s" % (h, k+1, nz, ex))
