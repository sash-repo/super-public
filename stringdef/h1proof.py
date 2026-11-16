"""Symbolic check of the hand proof that H^1(witt_-; witt)_w = 0."""
import sympy as sp
w, t = sp.symbols('w t')

# 1-cochain gamma_i, weight w.  Cocycle: (j+w-i)g_j - (i+w-j)g_i - (j-i)g_{i+j} = 0
def coc(g, i, j):
    return sp.expand((j + w - i) * g[j] - (i + w - j) * g[i] - (j - i) * g[i + j])

# Normalize g_1 = 0 (subtract the coboundary t*(w-i), whose value at i=1 is w-1;
# legitimate for w != 1).  Then the equations (1,j) determine g_{j+1} from g_j.
g = {1: sp.Integer(0), 2: sp.Symbol('g2')}
for j in range(2, 12):                       # g_{j+1} = (j+w-1) g_j / (j-1)
    unk = sp.Symbol('x')
    g[j + 1] = unk
    g[j + 1] = sp.simplify(sp.solve(coc(g, 1, j), unk)[0])
print("g_3..g_6 in terms of g_2 :")
for k in range(3, 7):
    print("   g_%d =" % k, sp.factor(g[k]))

obstruction = sp.factor(sp.simplify(coc(g, 2, 3) / sp.Symbol('g2')))
print("\nequation (2,3) forces g_2 * P(w) = 0 with P(w) =", obstruction)
print("integer roots of P :", [r for r in sp.solve(obstruction, w) if r.is_integer])
print("all roots           :", sp.solve(obstruction, w))

# The remaining equations at the exceptional weight w = 1, by machine.
import witt2
m = witt2.make_module('adj')
print("\nw = 1 handled separately:  dim Z^1, dim B^1, dim H^1 at truncation N")
for N in (8, 12, 16, 20, 24):
    d1, s1, _ = witt2.differential(1, N, 1, m)
    d0, s0, _ = witt2.differential(0, N, 1, m)
    z = len(s1) - witt2._rank(d1)
    b = witt2._rank(d0)
    print("   N=%2d  dim C^1=%2d  Z^1=%d  B^1=%d  H^1=%d" % (N, len(s1), z, b, z - b))
print("\ndim Z^1 (= ker d_1) per weight at N=20, module witt:")
for ww in range(-6, 7):
    d1, s1, _ = witt2.differential(1, 20, ww, m)
    print("   w=%3d  dim C^1=%d  dim Z^1=%d" % (ww, len(s1), len(s1) - witt2._rank(d1)))
