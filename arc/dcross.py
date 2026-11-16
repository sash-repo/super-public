#!/usr/bin/env python3
"""Cross-check: the 4-variable model of dform.py against the 1-variable engine
of check_grc_assoc.py, in which the printed formulas (133), (137) were
verified.

Two questions:
  1) is the printed even bracket (133) for k = 1 invariant in the 4-variable
     model (sigma = +1)?
  2) is the solution the 4-variable model returns for N = 2 invariant in the
     1-variable convention  Q(B(f,g)) = B(Qf,g) + (-1)^{p(f)} B(f,Qg)?

Run:  ../venv/bin/python dcross.py
"""
import sympy as sp

import dform as M
from check_grc_assoc import add, mul, dth, thmul, dt, dtn, Qminus, Qplus

mu1, mu2 = M.mu1, M.mu2
ZERO, ONE = sp.Integer(0), sp.Integer(1)


def D1v(u):                                     # D_theta in the 1-var engine
    out = dth(u)
    return add(out, thmul(dt(u)), 1)


def smul1(c, u):
    return {k: sp.expand(c * v) for k, v in u.items() if sp.expand(c * v) != 0}


def op_from_lamrho(N, lam, rho):
    """Realize sum lam_ij D_1^i D_2^j + th sum rho_ij D_1^i D_2^j in the
    1-variable engine, with the Koszul sign (-1)^{j p(f)} written out."""
    def B(f, g, pf):
        out = {}
        for (i, j), c in list(lam.items()) + [(k, None) for k in []]:
            pass
        for (i, j), c in lam.items():
            if c == 0:
                continue
            s = (-1) ** (j * pf)
            u, v = f, g
            for _ in range(i):
                u = D1v(u)
            for _ in range(j):
                v = D1v(v)
            add(out, smul1(s * c, mul(u, v)))
        th = {}
        for (i, j), c in rho.items():
            if c == 0:
                continue
            s = (-1) ** (j * pf)
            u, v = f, g
            for _ in range(i):
                u = D1v(u)
            for _ in range(j):
                v = D1v(v)
            add(th, smul1(s * c, mul(u, v)))
        add(out, thmul(th))
        return {k: sp.expand(v) for k, v in out.items() if sp.expand(v) != 0}
    return B


def check_1var(B, N, A=6):
    """Invariance in the 1-variable convention."""
    nu = mu1 + mu2 + N
    bad = []
    for Q in (Qminus, Qplus):
        for a in range(A + 1):
            for e in (0, 1):
                for b in range(A + 1):
                    for e2 in (0, 1):
                        f, g, pf = {(a, e): ONE}, {(b, e2): ONE}, e
                        sg = (-1) ** pf
                        res = dict(Q(B(f, g, pf), nu))
                        add(res, B(Q(f, mu1), g, (pf + 1) % 2), -1)
                        add(res, B(f, Q(g, mu2), pf), -sg)
                        res = {k: sp.simplify(v) for k, v in res.items()}
                        res = {k: v for k, v in res.items() if v != 0}
                        if res:
                            bad.append((Q.__name__, (a, e), (b, e2), res))
    return bad


def check_4var(N, lam, rho, sigma=1, A=None):
    A = A if A is not None else N + 3
    nu = mu1 + mu2 + N
    bad = []
    for u in M.monomials(A):
        for sign, Q1 in ((-1, lambda w: M.Qm1(w)), (1, lambda w: M.Qp1(w, nu))):
            lhs = M.apply_B(M.cQ(u, sign, (mu1, mu2)), N, lam, rho)
            rhs = M.smul(sigma, Q1(M.apply_B(u, N, lam, rho)))
            res = dict(lhs)
            M.add1(res, rhs, -1)
            res = {k: sp.simplify(v) for k, v in res.items()}
            res = {k: v for k, v in res.items() if v != 0}
            if res:
                bad.append((sign, u, res))
    return bad


if __name__ == "__main__":
    # (1) the printed even bracket for k = 1, in the D-language:
    #     [f,g]_1 = -(-1)^p D f D g + mu1 f g' - mu2 f' g
    #             = -D_1 D_2 + mu1 D_2^2 - mu2 D_1^2
    lam_txt = {(2, 0): -mu2, (1, 1): -ONE, (0, 2): mu1}
    print("printed (133) k=1, in the 4-variable model:",
          "INVARIANT" if not check_4var(2, lam_txt, {}) else "NOT invariant")
    print("printed (133) k=1, in the 1-variable engine:",
          "INVARIANT" if not check_1var(op_from_lamrho(2, lam_txt, {}), 2)
          else "NOT invariant")

    # (2) what the 4-variable solver returns for N = 2 (scaled by mu2)
    lam_sol = {(2, 0): mu2, (1, 1): -ONE, (0, 2): -mu1}
    print("solver N=2,           in the 4-variable model:",
          "INVARIANT" if not check_4var(2, lam_sol, {}) else "NOT invariant")
    print("solver N=2,           in the 1-variable engine:",
          "INVARIANT" if not check_1var(op_from_lamrho(2, lam_sol, {}), 2)
          else "NOT invariant")
