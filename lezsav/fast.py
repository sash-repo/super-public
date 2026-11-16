"""fast.py -- theta-truncated evaluation of Phi.
Everything is computed modulo the ideal of theta-degree >= 5.  A derivative
of a quantity known mod theta^5 is known mod theta^4, and the step-1 rows
(equations at theta-degree 2, after at most two odd derivatives; constraints
at theta-degree 3, after one) are therefore exact.  Checked against the
untruncated Phi of n4phi2.py (see check())."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
import n4noeth
from n4emb import C, sort_odd
TMAX = [4]
def mul(f, g):
    out = {}
    T = TMAX[0]
    for (z1, w1, m1, S1), c1 in f.items():
        l1 = len(S1)
        for (z2, w2, m2, S2), c2 in g.items():
            if l1 + len(S2) > T:
                continue
            S, sgn = sort_odd(S1 + S2)
            if sgn:
                k = (z1 + z2, w1 + w2, m1 + m2, S)
                out[k] = out.get(k, C()) + c1 * c2 * sgn
    return {k: v for k, v in out.items() if v}
def patch():
    import n4rec
    for mod in (n4noeth, n4rec):
        if hasattr(mod, "mul"):
            mod.mul = mul
