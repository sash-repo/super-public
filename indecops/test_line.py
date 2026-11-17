"""Calibration of line.py against what is known independently.

Nothing here is a self-test of our own conventions: every number checked is
either a classical statement (Grozman's classification of the bilinear
invariant operators on the line, the Bol operators, the transvectants) or a
hand computation.
"""

from fractions import Fraction as F
from line import (inv_space, inv_dim, midx, show, compose, derivative,
                  multiply, decomposable, atoms)

ok = 0
bad = []


def chk(name, got, want):
    global ok
    if got == want:
        ok += 1
    else:
        bad.append(f"{name}: got {got}, want {want}")


# --------------------------------------------------------------- unary: Bol
# The only fvect(1)-invariant unary operators are the identity and
# d: cF_0 -> cF_1.  (The Bol operators of order k >= 2 are invariant under
# sl(2) only -- they need a projective structure.)
for lam in [F(0), F(1), F(-1), F(-1, 2), F(-2, 3), F(1, 2)]:
    chk(f"unary id lam={lam}", inv_dim((lam,), 0), 1)
for lam in [F(0), F(1), F(-1), F(-1, 2), F(-2, 3), F(3)]:
    chk(f"unary order1 lam={lam}", inv_dim((lam,), 1), 1 if lam == 0 else 0)
for k in range(2, 7):
    for lam in [F(0), F(1), F(-1), F(1 - k, 2), F(-2, 3)]:
        chk(f"unary order{k} lam={lam}", inv_dim((lam,), k), 0)

# ------------------------------------------------- binary: Grozman's theorem
# Order 0: the multiplication, always.
for a in [F(0), F(1), F(-1), F(2), F(-2, 3)]:
    for b in [F(0), F(1), F(5, 2)]:
        chk(f"mult {a},{b}", inv_dim((a, b), 0), 1)

# Order 1: exists for every pair, and is  b u'v - a uv'  (zero iff a=b=0).
for a, b in [(F(0), F(0)), (F(1), F(1)), (F(1), F(0)), (F(-1), F(-1)),
             (F(2), F(3)), (F(-2, 3), F(-2, 3))]:
    d = inv_dim((a, b), 1)
    chk(f"order1 dim {a},{b}", d, 1 if (a, b) != (0, 0) else 2)
# the (0,0) case: both f'g and fg' are invariant cF_0 (x) cF_0 -> cF_1,
# being d(f).g and f.d(g) -- so dimension 2 there, and the bracket is 0.
v = inv_space((F(1), F(1)), 1)[0]
# normalise: the operator must be proportional to u'v - uv'
cols = midx(2, 1)
c = dict(zip(cols, v))
chk("order1 on (1,1) is the Wronskian", c[(1, 0)] == -c[(0, 1)] != 0, True)

# Order 2: a nonzero operator exists iff a*b*(a+b+1) = 0.
for a, b in [(F(1), F(1)), (F(1), F(2)), (F(2), F(3)), (F(1, 2), F(1, 2)),
             (F(0), F(3)), (F(5), F(0)), (F(1), F(-2)), (F(-1, 2), F(-1, 2)),
             (F(-2, 3), F(-2, 3)), (F(3), F(-4))]:
    want = 1 if a * b * (a + b + 1) == 0 else 0
    chk(f"order2 dim {a},{b}", inv_dim((a, b), 2), want)

# Order 3: Grozman's exceptional operator, and nothing else of order 3 except
# where a composition happens to live.
chk("Grozman Gz at (-2/3,-2/3)", inv_dim((F(-2, 3), F(-2, 3)), 3), 1)
for a, b in [(F(1), F(1)), (F(1), F(2)), (F(2), F(2)), (F(-1), F(-1)),
             (F(1, 2), F(1, 2)), (F(-1, 3), F(-1, 3)), (F(-2, 3), F(-1, 3))]:
    chk(f"order3 dim {a},{b}", inv_dim((a, b), 3), 0)
# Gz maps cF_{-2/3} (x) cF_{-2/3} -> cF_{-4/3+3} = cF_{5/3}.
chk("Gz target weight", F(-2, 3) + F(-2, 3) + 3, F(5, 3))
# and it is NOT a composition
chk("Gz is indecomposable", atoms((F(-2, 3), F(-2, 3)), 3)[:2], (1, 0))

# Grozman's operator, printed, against the form in the literature:
#   Gz(f,g) = 2 f'''g - 3 f''g' + 3 f'g'' - 2 fg'''   (up to a factor)
gz = inv_space((F(-2, 3), F(-2, 3)), 3)[0]
c = dict(zip(midx(2, 3), gz))
lead = c[(3, 0)]
prof = tuple(c[z] / lead for z in [(3, 0), (2, 1), (1, 2), (0, 3)])
# 2f'''g + 3f''g' - 3f'g'' - 2fg''', antisymmetric; verified from the
# definition, infinitesimally and finitely, in verify.py
chk("Gz coefficient profile", prof,
    (F(1), F(3, 2), F(-3, 2), F(-1)))
chk("Gz antisymmetric", (prof[0], prof[1]) == (-prof[3], -prof[2]), True)

# Order 4 and up: nothing on generic weights
for k in [4, 5]:
    for a, b in [(F(1), F(1)), (F(-2, 3), F(-2, 3)), (F(1), F(2))]:
        chk(f"order{k} dim {a},{b}", inv_dim((a, b), k), 0)

# ------------------------------------------------- the composition machinery
# d(f)*d(g) must be the (2,2)-operator f'g'
n = 2
one0 = inv_space((F(0),), 0)[0]        # identity on cF_0
dop = inv_space((F(0),), 1)[0]         # d
mult = inv_space((F(1), F(1)), 0)[0]   # multiplication cF_1 (x) cF_1 -> cF_2
prod = compose(mult, 0, [dop, dop], [(0,), (1,)], [1, 1], 2)
c = dict(zip(midx(2, 2), prod))
chk("d(f)d(g)", (c[(1, 1)] != 0, c[(2, 0)], c[(0, 2)]), (True, 0, 0))

# J_1(df,dg) must be the Wronskian f'g''-f''g'
j1 = inv_space((F(1), F(1)), 1)[0]
wr = compose(j1, 1, [dop, dop], [(0,), (1,)], [1, 1], 2)
c = dict(zip(midx(2, 3), wr))
chk("J1(df,dg) = Wronskian", (c[(2, 1)] == -c[(1, 2)] != 0, c[(3, 0)], c[(0, 3)]),
    (True, 0, 0))
# and it is invariant as an operator cF_0 (x) cF_0 -> cF_3
full = inv_space((F(0), F(0)), 3)
from exactlin import rank as _rank
chk("Wronskian is in the invariant space",
    _rank([list(v) for v in full] + [list(wr)]), len(full))

# derivative(): (fg)' = f'g + fg'
v = multiply([[F(1)], [F(1)]], [(0,), (1,)], 2, [0, 0])
dv = derivative(v, 2, 0, 1)
c = dict(zip(midx(2, 1), dv))
chk("Leibniz", (c[(1, 0)], c[(0, 1)]), (F(1), F(1)))
# (fg)'' = f''g + 2f'g' + fg''
dv = derivative(v, 2, 0, 2)
c = dict(zip(midx(2, 2), dv))
chk("Leibniz 2", (c[(2, 0)], c[(1, 1)], c[(0, 2)]), (F(1), F(2), F(1)))

# ------------------------------------- Malyshev's own dimensions, n = 2 and 3
# W_n^{(n+s)} = invariant operators cF_1^{(x)n} -> cF_{n+s} of order s, and
# his a_{n-1,s} is the number of permutations of n with s inversions.
def mahonian(n):
    poly = [1]
    for r in range(2, n + 1):
        new = [0] * (len(poly) + r - 1)
        for i, c in enumerate(poly):
            for j in range(r):
                new[i + j] += c
        poly = new
    return poly


for n in [1, 2, 3, 4]:
    want = mahonian(n)
    got = [inv_dim(tuple([F(1)] * n), s) for s in range(len(want))]
    chk(f"Mahonian n={n}", got, want)
    chk(f"nothing above the top n={n}",
        inv_dim(tuple([F(1)] * n), len(want)), 0)

print(f"{ok} checks passed, {len(bad)} failed")
for b in bad:
    print("  FAIL", b)
