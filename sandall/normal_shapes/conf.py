#!/usr/bin/env python3
r"""Conformal fields of a periplectic form:  c(omega) = {D | L_D omega in K omega}.

Why this is the mechanism behind the drop of dim Der by one (see `leiso.py`):
by Cartan's formula L_D = [iota_D, d] and closedness of omega,

    L_D omega = +- d(iota_D omega)   is EXACT,

so L_D omega = lambda omega with lambda =/= 0 forces omega to be exact.  For
omega_0 = sum dq_i dxi_i the Euler field E_q = sum q_i d/dq_i is conformal with
lambda = 1; for the exceptional shape omega_Xi = omega_0 + xi_1 (dxi_1)^{(2)},
whose class in H^2_odd is non-zero (Theorem 5.1 of the text), no conformal field
with lambda =/= 0 can exist.  So c(omega_0)/h(omega_0) = K while
c(omega_Xi) = h(omega_Xi), which is where the even outer derivation of
le(n;\un) that le_{omega_Xi}(n;\un) lacks comes from.

This script verifies both statements by exact linear algebra over F_p.

Usage:  python3 conf.py
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from peri import (Omega, omega0, omegaXi, vfield_basis, vec, rank_p,      # noqa: E402
                  nullspace_p, is_exact, is_closed)

LOG = None


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    if LOG:
        LOG.write(s + "\n")
        LOG.flush()


def conformal(alg, om):
    """(dim h(omega), dim c(omega)): the fields with L_D om = 0, resp. in K om."""
    B2 = alg.basis(2)
    i2 = {m: i for i, m in enumerate(B2)}
    basis = vfield_basis(alg)
    rows = [vec(alg, alg.lie(D)(om), i2) for D in basis]
    cols = [[rows[j][i] for j in range(len(rows))] for i in range(len(B2))]
    h = nullspace_p(cols, alg.p)
    # now the extended system with the unknown lambda: L_D om - lambda om = 0
    rows2 = rows + [[(-x) % alg.p for x in vec(alg, om, i2)]]
    cols2 = [[rows2[j][i] for j in range(len(rows2))] for i in range(len(B2))]
    c = nullspace_p(cols2, alg.p)
    return len(h), len(c), basis


def euler_q(alg):
    """E_q = sum q_i d/dq_i."""
    fs = tuple(alg.q(i) for i in range(alg.n))
    gs = tuple({} for _ in range(alg.n))
    return (fs, gs, 0)


def euler_xi(alg):
    """E_xi = sum xi_i d/dxi_i."""
    fs = tuple({} for _ in range(alg.n))
    gs = tuple(alg.xi(i) for i in range(alg.n))
    return (fs, gs, 0)


def main():
    global LOG
    here = os.path.dirname(os.path.abspath(__file__))
    LOG = open(os.path.join(here, "conf.log"), "w")
    from exactlin import backend
    say(f"backend: {backend()}")
    for (n, p, u) in [(1, 3, (1,)), (1, 3, (2,)), (2, 3, (1, 1)), (3, 3, (1, 1, 1))]:
        alg = Omega(n, p, u)
        o0, oX = omega0(alg), omegaXi(alg)
        say("")
        say(f"--- n={n} p={p} u={u}")
        for name, om in (("omega_0", o0), ("omega_Xi", oX)):
            say(f"    {name}: closed = {is_closed(alg, om)}, "
                f"exact = {is_exact(alg, om)}")
        for name, om in (("omega_0", o0), ("omega_Xi", oX)):
            dh, dc, _ = conformal(alg, om)
            say(f"    {name}: dim h(omega) = {dh}, dim c(omega) = {dc}, "
                f"conformal weights beyond 0: {dc - dh}")
        # the Euler fields, explicitly
        for nm, D in (("E_q", euler_q(alg)), ("E_xi", euler_xi(alg))):
            for name, om in (("omega_0", o0), ("omega_Xi", oX)):
                L = alg.lie(D)(om)
                rel = "0" if not L else ("omega" if L == om else "other")
                if L and rel == "other":
                    # is it a multiple of om, or of omega_0?
                    if L == o0:
                        rel = "omega_0"
                say(f"    L_{nm} {name} = {rel}")
    LOG.close()


if __name__ == "__main__":
    main()
