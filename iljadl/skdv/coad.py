"""The N=3 super KdV family as a coadjoint action of the centrally extended k^L(1|3),
in the convention cD_i = d_theta_i - theta_i d_x (cD_i^2 = -d_x); in sf.py this is .Q(i).
Pairing <U,G> = Res of the theta_1theta_2theta_3-coefficient of U*G.
A differential polynomial linear in the components of G is a total x-derivative
iff its Euler operators in those components vanish."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sf import SF
subsets = [()] + [(i,) for i in (1, 2, 3)] + [(1, 2), (1, 3), (2, 3)] + [(1, 2, 3)]
def gen(N, pre, p):
    """a generic N=3 superfield of parity p with components named pre + index"""
    return SF.field(N, {I: pre + ''.join(map(str, I)) if I else pre + '0' for I in subsets}, p)
from spdo import P, PAR
from fractions import Fraction
count = 0
def check(name, ok):
    global count; count += 1
    print('%-62s %s' % (name, ok))
cD = lambda X, i: X.Q(i)          # cD_i = d_theta_i - theta_i d_x
cDp = lambda X, i: X.D(i)         # the other convention, d_theta_i + theta_i d_x, square +d_x
def w(X, D=cD): return D(D(D(X, 3), 2), 1)
def V(F, U, D=cD, s=1):
    r = (U.dx() * F).scale(-2) - U * F.dx()
    for i in (1, 2, 3): r = r + (D(U, i) * D(F, i)).scale(s)
    return r
def br(F, G, D=cD, s=1):
    r = (F * G.dx() - F.dx() * G).scale(-2)
    for i in (1, 2, 3): r = r + (D(F, i) * D(G, i)).scale(s)
    return r
def K(F, W, D=cD):
    r = F * W.dx()
    for i in (1, 2, 3): r = r - (D(F, i) * D(W, i)).scale(Fraction(1, 2))
    return r
def brK(F, G, D=cD):
    r = F * G.dx() - F.dx() * G
    for i in (1, 2, 3): r = r - (D(F, i) * D(G, i)).scale(Fraction(1, 2))
    return r
top = lambda X: X.c.get((1, 2, 3), P())
def dxk(p, k):
    for _ in range(k): p = p.dx()
    return p
def euler(p, name):
    r = P()
    for (ev, od), c in p.t.items():
        for v, a in ev:
            if v[0] == name:
                ev2 = dict(ev); ev2[v] -= 1
                if not ev2[v]: del ev2[v]
                r = r + dxk(P({(tuple(sorted(ev2.items())), od): c * a}), v[1]) * ((-1) ** v[1])
        for idx, v in enumerate(od):
            if v[0] == name:
                r = r + dxk(P({(ev, od[:idx] + od[idx + 1:]): c * (-1) ** idx}), v[1]) * ((-1) ** v[1])
    return r
U = gen(3, 'U', 1); F = gen(3, 'F', 0); G = gen(3, 'G', 0); H = gen(3, 'H', 0)
W0 = gen(3, 'W', 0); W1 = gen(3, 'Y', 1)
Gn = [nm for nm in PAR if nm.startswith('G')]
exact = lambda p: all(euler(p, n).iszero() for n in Gn)
# conventions
check('cD_i^2 = -d_x (i=1,2,3)', all((cD(cD(U, i), i) + U.dx()).iszero() for i in (1, 2, 3)))
check('cD_i cD_j = -cD_j cD_i (i<j)', all((cD(cD(U, j), i) + cD(cD(U, i), j)).iszero() for i in (1, 2, 3) for j in (1, 2, 3) if i < j))
# item 1: contact bracket
for W, nm in ((W0, 'even'), (W1, 'odd')):
    c = K(F, K(G, W)) - K(G, K(F, W)) - K(brK(F, G), W)
    check('[K_F,K_G] = K_{{F,G}_K} on an %s superfield' % nm, c.iszero())
check('{F,G}_K = -1/2 {F,G}', (brK(F, G) + br(F, G).scale(Fraction(1, 2))).iszero())
check('Jacobi for {,} on even F,G,H', (br(F, br(G, H)) + br(G, br(H, F)) + br(H, br(F, G))).iszero())
# item 2
check('[V(F),V(G)] = V({F,G})', (V(F, V(G, U)) - V(G, V(F, U)) - V(br(F, G), U)).iszero())
check('<V(F)U,G> + <U,{F,G}> exact (invariance)', exact(top(V(F, U) * G) + top(U * br(F, G))))
check('  control: <V(F)U,G> - <U,{F,G}> exact', exact(top(V(F, U) * G) - top(U * br(F, G))))
# item 3
check('omega(F,G) + omega(G,F) exact', exact(top(F * w(G)) + top(G * w(F))))
check('<D123 F,G> + omega(F,G) exact', exact(top(w(F) * G) + top(F * w(G))))
check('  control: <D123 F,G> - omega(F,G) exact', exact(top(w(F) * G) - top(F * w(G))))
check('V(F)D123(G) - V(G)D123(F) = D123{F,G}', (V(F, w(G)) - V(G, w(F)) - w(br(F, G))).iszero())
# the conclusion: ad*_F(U,c)(G) = -<U,{F,G}> - c omega(F,G); RHS = V(F)U + D123 F
for c in (1, -1):
    lhs = top((V(F, U) + w(F)) * G)
    rhs_ = top(U * br(F, G)) * (-1) - top(F * w(G)) * c
    check('RHS(SKdV3) paired with G = coadjoint action at (U,%d)' % c, exact(lhs - rhs_))
# the other convention: square +d_x, sum with sign minus
check('square +d_x conv.: [V-(F),V-(G)] = V-({F,G}-) (sum sign minus)',
      (V(F, V(G, U, cDp, -1), cDp, -1) - V(G, V(F, U, cDp, -1), cDp, -1) - V(br(F, G, cDp, -1), U, cDp, -1)).iszero())
check('  control: square +d_x conv. with sum sign plus', (V(F, V(G, U, cDp), cDp) - V(G, V(F, U, cDp), cDp) - V(br(F, G, cDp), U, cDp)).iszero())
print('checks run:', count)
# item 1 proof: cD_j(K_F W) - K_F(cD_j W) = -1/2 sum_i (cD_j cD_i F) cD_i W
for W, nm in ((W0, 'even'), (W1, 'odd')):
    ok = True
    for j in (1, 2, 3):
        r = cD(K(F, W), j) - K(F, cD(W, j))
        for i in (1, 2, 3): r = r + (cD(cD(F, i), j) * cD(W, i)).scale(Fraction(1, 2))
        ok = ok and r.iszero()
    check('cD_j K_F - K_F cD_j = -1/2 sum (cD_j cD_i F) cD_i on %s W' % nm, ok)
print('checks run:', count)
