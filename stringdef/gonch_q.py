"""dim H^q(g; K) of a positive part, weight by weight.  usage: gonch_q.py kind W q
(kinds as in nsgonch.algebra: ns = NS_{>=1/2}, ram = R_{>=1}, nsd3 = NS_{>=3/2}, ...)."""
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from nsgonch import *
kind, W, q = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
B, br = algebra(kind, W)
for n in range(1, 2*W+1):
    w = F(n, 2)
    def rk(qq):
        if qq <= 0: return 0
        M, a, b = dmat(B, br, qq, w)
        return rank(M) if a and b else 0
    h = len(chains(B, q, w)) - rk(q) - rk(q+1)
    if h: print(kind, q, str(w), h, flush=True)
print("done", kind, W, q)
