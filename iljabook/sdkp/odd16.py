"""The odd symmetries of order at most 1 found by the census (symm1.py, sigma = 1, coefficients
of degree <= 3 in x, y, t) span exactly the 16 fields Xi_h, Xi'_h, A_a, B_b with h, a in
{1, t, t^2, t^3}.  Needs sols_s1_c3_21.pkl and sols_s0_c3_21.pkl (run symm1.py first)."""
import sys; sys.path.insert(0, '.')
from brackets import O, express, Q
from symm import condition
from fields import Xi, Xip, A, B
G = [f(k) for f in (Xi, Xip, A, B) for k in range(4)]
print('census: odd solutions', len(O))
print('the 16 fields are symmetries:', all(len(condition(g, 1)) == 0 for g in G))
print('every census solution is a combination of the 16 fields:', all(express(o, G) is not None for o in O))
print('every one of the 16 fields is a combination of the census solutions:', all(express(g, O) is not None for g in G))
# control: a field outside the family is not in the span of the census
bad = dict(B(0)); bad['U10'] = Q.var(('t',))
print('control (B_1 plus t in U_1^0) is not in the span:', express(bad, O) is None)
