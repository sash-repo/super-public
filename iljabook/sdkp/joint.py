"""The joint invariants of Xi'_1 and an even symmetry: the brackets [E, Xi'_1] of the 17 even
fields of the census (symm1.py, sigma = 0), expressed in the basis Xi_h, Xi'_h, A_a, B_b
(h, a, b in {1, t, t^2, t^3}) of the odd ones.  Modulo K Xi'_1 they span the fields
Xi'_h + A_a + B_b with h in span(t, t^2), deg a <= 2, deg b <= 3 (dimension 9); the fields E
with [E, Xi'_1] in K Xi'_1 are spanned by T_f - S_{f'/3} (f = 1, t, t^2, t^3), d_y, d_t and
the two scalings (the census solutions of weight (0, 0)).
Needs sols_s0_c3_21.pkl and sols_s1_c3_21.pkl (run symm1.py first)."""
import sys; sys.path.insert(0, '.')
from fractions import Fraction as F
from brackets import E, S0, bracket, express, Q, RHS
from fields import Xi, Xip, A, B, Tf, Sf, lin, V, COMPS
from exactlin import rank
names = [f'{n}({k})' for n in ('Xi', "Xi'", 'A', 'B') for k in range(4)]
G = [f(k) for f in (Xi, Xip, A, B) for k in range(4)]
X1 = Xip(0)
iX1 = names.index("Xi'(0)")
coords = []
for e in E:
    c = express(bracket(e, 0, X1, 1), G)
    assert c is not None, 'a bracket is not in the span of the odd symmetries'
    coords.append(c)
print('every [E, Xi\'_1] is a combination of Xi_h, Xi\'_h, A_a, B_b:', True)
mod = [[c[j] for j in range(16) if j != iX1] for c in coords]
names_mod = [n for j, n in enumerate(names) if j != iX1]
used = sorted({names_mod[j] for row in mod for j, x in enumerate(row) if x})
print('basis fields occurring modulo Xi\'_1:', used)
expected = ["Xi'(1)", "Xi'(2)", 'A(0)', 'A(1)', 'A(2)', 'B(0)', 'B(1)', 'B(2)', 'B(3)']
r = rank(mod)
print('they are exactly Xi\'_t, Xi\'_{t^2}, A_1, A_t, A_{t^2}, B_1, ..., B_{t^3}:', used == sorted(expected))
print('rank of the image modulo Xi\'_1:', r, '(expected 9)')
# the kernel: 17 - 9 = 8 fields
dt = {c: V(c, 0, 1) for c in COMPS}
dy = {c: RHS[c] for c in COMPS}
scal = [e for (key, sol), e in zip(S0, E) if key == (0, 0)]
K = [lin((1, Tf(k)), (F(-k, 3), Sf(k - 1))) if k else Tf(0) for k in range(4)] + [dy, dt] + scal
def in_K_X1(f):
    c = express(bracket(f, 0, X1, 1), G)
    return c is not None and all(x == 0 for j, x in enumerate(c) if j != iX1)
print('the 8 fields listed:', len(K), '; each lies in the census span:', all(express(f, E) is not None for f in K))
print('each has [E, Xi\'_1] in K Xi\'_1:', all(in_K_X1(f) for f in K))
Kc = [express(f, E) for f in K]
print('they are linearly independent:', rank(Kc) == 8, '; 8 + 9 = 17 = number of even fields:', 8 + r == len(E))
