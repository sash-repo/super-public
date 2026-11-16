"""S_1 and S_2 again, with G and H carrying distinct odd constants."""
import sympy as sp
import radul as R
from radul import t, e, d, Diffeo
from test_radul import gen
from run import check, S_radul, cocycle
import run as RUN

for n in (1, 2):
    R.setup(n); R.set_order(3)
    G = Diffeo(gen(n), e, n); H = Diffeo(gen(n, seed=3), d, n)
    S = lambda X: S_radul(X, n)
    w = sp.Rational(4 - n, 2)
    check("n=%d: S_n is a cocycle of the weight %s" % (n, w), cocycle(S, w, n, G, H))
    for bad in (w - 1, w + 1, sp.Integer(2)):
        if bad == w:
            continue
        check("n=%d:   weight %s fails" % (n, bad), not cocycle(S, bad, n, G, H))
print("\n%d ok, %d FAILED" % (RUN.OK, RUN.BAD))
