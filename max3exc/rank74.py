#!/usr/bin/env python3
"""ag(2), the graded answers sl(2)+sl(2|1) of sdim (7|4): u_ev = sl(2)+gl(2) has
a 3-dim toral subalgebra (a maximal torus of ag(2)): u cap h of rank 2 plus the
center z of u_ev, with ad z semisimple on g.  So each copy is conjugate to h+S,
S closed of sdim (7|4), and check74.py shows every such lies in osp(3|2)+sl(2)."""
import sys, os, pickle
from fractions import Fraction as F
HERE = os.path.dirname(os.path.abspath(__file__))
import exc3, sympy
from sweep3 import Ech, cartan
from lin import kernel
g = exc3.build_G3()
d = pickle.load(open(os.path.join(HERE, "runs", "G3-100.pkl"), "rb"))
for tags, basis, s, pc, v in d["answers"]:
    if tuple(s) != (7, 4): continue
    E = Ech(g.N, [[F(x) for x in w] for w in basis])
    ue = Ech(g.N, [[x if g.PAR[i] == 0 else F(0) for i, x in enumerate(w)] for w in E.basis()]).basis()
    # center of u_ev
    conds = []
    for y in ue:
        for t in range(g.N):
            row = [g.br(x, y)[t] for x in ue]
            if any(row): conds.append(row)
    Z = kernel(conds, len(ue))
    z = [sum(c * ue[i][t] for i, c in enumerate(Z[0])) for t in range(g.N)]
    M = sympy.Matrix(g.N, g.N, lambda i, j: 0)
    for j in range(g.N):
        ej = [F(0)] * g.N; ej[j] = F(1)
        col = g.br(z, ej)
        for i in range(g.N): M[i, j] = sympy.Rational(col[i].numerator, col[i].denominator)
    t = sympy.Symbol("t")
    cp = sympy.factor_list(M.charpoly(t).as_expr())[1]
    rad = sympy.prod([f for f, m in cp])
    radM = sympy.zeros(g.N, g.N); P = sympy.Poly(rad, t)
    for k, c in enumerate(reversed(P.all_coeffs())): radM += c * M ** k
    H0 = cartan(g); hb = E.basis()
    Mk = [[H0[i][t] for i in range(len(H0))] + [-hb[j][t] for j in range(len(hb))] for t in range(g.N)]
    K = kernel(Mk, len(H0) + len(hb))
    hcap = [[sum(k[i] * H0[i][t] for i in range(len(H0))) for t in range(g.N)] for k in K]
    comm = all(not any(g.br(z, x)) for x in hcap)
    print(tags[0], "dim Z(u_ev) =", len(Z), " ad z semisimple:", radM.is_zero_matrix,
          " [z, u cap h]=0:", comm, " dim span(u cap h, z) =", Ech(g.N, hcap + [z]).dim())
