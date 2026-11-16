#!/usr/bin/env python3
"""The odd analog of Skryabin's forms of the 2nd type, omega = d(exp(v) alpha),
v = q_j^{(p^{u_j})}, for BOTH primitives of omega_0,

    alpha_0  = sum_i q_i dxi_i,        alpha_0' = sum_i xi_i dq_i,

with the auxiliary height of q_j raised by r = 1, 2, 3.  For each case: the
superdimension of the algebra of the fields of vect(n|n;u) preserving the
twisted form, against that of the algebra preserving omega_0.

Usage:  python3 twist2nd.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from peri import Omega, omega0, alpha0, vfield_basis, is_closed      # noqa: E402
from perirun import h_omega, sdim_graded                              # noqa: E402


def alpha0p(alg):
    """the second primitive of omega_0:  sum xi_i dq_i."""
    out = {}
    for i in range(alg.n):
        out = alg.add(out, alg.mul(alg.xi(i), alg.dq(i)))
    return out


def expv(alg, j, uj):
    """exp(q_j^{(p^{uj})}) = sum_{0<=a<p} q_j^{(a p^{uj})}, in an algebra whose
    j-th height exceeds uj."""
    N = alg.p ** uj
    out = alg.zero()
    for a in range(alg.p):
        out = alg.add(out, alg.q(j, a * N))
    return out


def run(n, p, u, j, r, prim):
    ubig = list(u); ubig[j] += r
    big = Omega(n, p, tuple(ubig))
    alpha = alpha0(big) if prim == "alpha_0" else alpha0p(big)
    om2 = big.d()(big.mul(expv(big, j, u[j]), alpha))
    om0 = omega0(big)
    small = set(m for m in big.fun_basis()
                if all(m[0][i] < p ** u[i] for i in range(n)))
    fields = [D for D in vfield_basis(big)
              if all(set(x) <= small for x in D[0] + D[1])]
    ns2, _ = h_omega(big, om2, fields)
    ns0, _ = h_omega(big, om0, fields)
    e2, o2 = sdim_graded(big, ns2, fields)
    e0, o0 = sdim_graded(big, ns0, fields)
    print(f"n={n} p={p} u={u} j={j} raised by {r}, primitive {prim}: "
          f"closed={is_closed(big, om2)}  sdim h_exp = ({e2}|{o2})  "
          f"sdim h(omega_0) = ({e0}|{o0})", flush=True)


if __name__ == "__main__":
    for (n, p, u) in [(1, 3, (1,)), (1, 5, (1,)), (2, 3, (1, 1))]:
        for prim in ("alpha_0", "alpha_0'"):
            for r in (1, 2, 3):
                run(n, p, u, 0, r, prim)
