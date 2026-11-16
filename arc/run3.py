"""The n = 3 half of run.py at the truncation order 2 -- same checks, cheaper."""
import sys
import sympy as sp
import radul as R
from radul import (t, e, d, num, th, eadd, esub, emul, escal, eeq, d_t,
                   Diffeo, Composite, multiplier, einv, epow, D_op, clean)
from test_radul import gen, first_order
from run import (check, amatrix, vfield, apply_to_row, sprod, S3_frame,
                 cocycle)
import run as RUN

R.setup(3)
R.set_order(2)
n = 3
G = Diffeo(gen(3), e, 3)
m = multiplier(G, n)
mi = einv(m)
a = amatrix(G, n)
v = [vfield(a[i], n) for i in range(n)]
print("frame built", flush=True)

good = True
for i in range(n):
    for j in range(n):
        good &= eeq(sprod(a[i], a[j], n), mi if i == j else {})
check("(v_i, v_j) = m^{-1} delta_ij", good)

vv = [[apply_to_row(v[i], a[j], n) for j in range(n)] for i in range(n)]
check("v_i(v_j) = - v_j(v_i) for i != j",
      all(all(eeq(vv[i][j][k], escal(-1, vv[j][i][k])) for k in range(n))
          for i in range(n) for j in range(n) if i != j))
check("v_i(v_i) = v_j(v_j)",
      all(all(eeq(vv[i][i][k], vv[0][0][k]) for k in range(n))
          for i in range(n)))

c1 = escal(sp.Rational(1, 2), emul(mi, v[1](m)))
c2 = escal(sp.Rational(-1, 2), emul(mi, v[0](m)))
c3 = emul(m, sprod(vv[0][1], a[2], n))
rhs = [eadd(emul(c1, a[0][k]), emul(c2, a[1][k]), emul(c3, a[2][k]))
       for k in range(n)]
check("v_1(v_2) = (m^{-1}/2)(v_2(m) v_1 - v_1(m) v_2) + m (v_1(v_2), v_3) v_3",
      all(eeq(vv[0][1][k], rhs[k]) for k in range(n)))

rhs = [eadd(*[emul(escal(sp.Rational(-1, 2), emul(mi, v[i](m))), a[i][k])
              for i in range(n)]) for k in range(n)]
check("v_1(v_1) = -(m^{-1}/2) sum_i v_i(m) v_i",
      all(eeq(vv[0][0][k], rhs[k]) for k in range(n)))

# the constant in S_3
g = gen(3)
want = D_op(1)(D_op(2)(D_op(3)(g)))
got = {S: sp.expand(sp.diff(c, e).subs({e: 0, d: 0}))
       for S, c in S3_frame(G, 3, 1).items()}
got = clean({S: c for S, c in got.items() if c != 0})
ratios = set()
for S in set(list(got) + list(want)):
    x, y = got.get(S), want.get(S)
    if y in (None, 0):
        ratios.add(sp.nan if x else sp.Integer(0))
    else:
        ratios.add(sp.simplify(sp.cancel((x or 0) / y)))
ratios.discard(sp.Integer(0))
print("     m^{3/2}(v_1(v_2),v_3) linearized / D_1D_2D_3(g) = %s"
      % sorted(ratios, key=str), flush=True)
check("the constant continuing the normalization of S_1, S_2 is -1 "
      "(Radul prints -3)",
      len(ratios) == 1 and sp.simplify(list(ratios)[0] + 1) == 0)

# S_3 as a cocycle of weight 1/2
S3 = lambda X: S3_frame(X, 3, -3)
check("S_3 is not identically zero", bool(S3(G)))
H = Diffeo(gen(3, seed=3), d, 3)
check("S_3(GH) = S_3(H) + m_H^{1/2} H^*(S_3(G))",
      cocycle(S3, sp.Rational(1, 2), 3, G, H))
for w in (sp.Rational(3, 2), sp.Integer(1)):
    check("   and of no other weight: %s fails" % w, not cocycle(S3, w, 3, G, H))

print("\n%d ok, %d FAILED" % (RUN.OK, RUN.BAD))
