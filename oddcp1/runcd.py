"""Parts C and D of the census, re-run after the sign repair in the cocycle.

C. Dimension 1|3: B--O's Theorem 1, and the cases a naive reading of the
   tangent sheaf as a direct sum would add to it.
D. The degree-2 obstructions of [LT] against the homogeneous ones.
"""

from fractions import Fraction
from ns import SM, g_from, g_mul, g_add, g_zero, lp, h1

DEG = 10
LOG = open("runcd.log", "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


def xi(j):
    return g_from((j,), lp((0, 1)))


def coef(e, c=1):
    return g_from((), lp((e, c)))


def cocycle_pair(i, j, c, ks):
    """B--O's (41): c(x) xi_i xi_j d/dx + (1/2) k_r x^{-2} xi_i xi_j xi_r d/dxi_r
    -- the three odd factors IN THAT ORDER, the sign being no decoration."""
    r = ({1, 2, 3} - {i, j}).pop()
    zx = g_mul(c, g_mul(xi(i), xi(j)))
    delta = g_mul(g_mul(xi(i), xi(j)), xi(r))
    zxi = {r: g_mul(coef(-2, Fraction(ks[r - 1], 2)), delta)}
    return zx, zxi


say("=" * 92)
say("C. Dimension 1|3: B--O's Theorem 1 against the cases a naive reading adds")
say("=" * 92)
say("  the class of the cocycle lies in H^1(M;T_2^{(ij)}), which by [BO, Prop.19]")
say("  is non-zero if and only if k_i + k_j >= 3, and carries an s-invariant")
say("  class if and only if k_i + k_j = 4")
say("")
say(f"{'(k1,k2,k3)':>12} {'pair':>7} {'k_i+k_j':>8} {'non-split':>10}"
    f" {'in Thm 1':>9} {'homogeneous':>12} {'sdim v(M,O)':>13} {'split':>13}")

cases = []
for k in (0, 1, 2, 3, 4, 5, 6):
    cases.append(((2, 2, k), 1, 2))
for k in (0, 1, 2, 3):
    cases.append(((3, 1, k), 1, 2))
    cases.append(((4, 0, k), 1, 2))
for k in (1, 2, 3):
    cases.append(((k, 1, 1), 2, 3))
cases.append(((3, 3, 2), 1, 2))
cases.append(((2, 2, 2), 1, 3))
cases.append(((2, 2, 2), 2, 3))
cases.append(((2, 3, 2), 1, 3))

for ks, i, j in cases:
    zx, zxi = cocycle_pair(i, j, coef(-1), list(ks))
    M = SM.from_cocycle(list(ks), zx, zxi)
    kij = ks[i - 1] + ks[j - 1]
    s3 = sorted(ks)
    inthm = any(s3[:t] + s3[t + 1:] == [2, 2] and s3[t] != 1 for t in range(3))
    say(f"{str(ks):>12} {str((i,j)):>7} {kij:>8} {str(kij >= 3):>10}"
        f" {str(inthm):>9} {str(M.transitive(DEG)):>12}"
        f" {str(M.sdim_vect(DEG)):>13} {str(SM.split(list(ks)).sdim_vect(DEG)):>13}")

say("")
say("  B--O, Theorem 1: the homogeneous non-split ones of dimension 1|3 are")
say("  exactly those with retract (2,2,k), k = 0 or k >= 2, one for each k.")

say("")
say("=" * 92)
say("D. The degree-2 obstructions of [LT] against the homogeneous ones")
say("=" * 92)
say(f"{'E = O(-a)+O(-b)':>16} {'dim H^1(O(2-a-b))':>18} {'[LT] moduli':>14}"
    f" {'homogeneous ones':>18}")
for a in range(0, 9):
    for b in range(0, a + 1):
        d = h1(2 - a - b)
        if d == 0:
            continue
        hom = 1 if (a, b) == (2, 2) else 0      # Part B of run.py
        say(f"{str((a,b)):>16} {d:>18} {'CP^' + str(d - 1):>14} {hom:>18}")
say("")
say("  [LT] Thm 2.2.1 (A2),(B2): the degree-2 obstructions form CP^{a+b-4}.")
say("  Exactly one of all these supermanifolds is homogeneous: the one over")
say("  (2,2), where CP^0 is a point -- B--O's superquadric.")
LOG.close()
