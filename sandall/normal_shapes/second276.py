#!/usr/bin/env python3
"""The 2nd-type (exponential) symplectic forms at p = 2.

The form is  omega_{2,j} = d( exp(v) alpha_0 ) = exp(v)(omega_0 + ubar_j du_j
alpha_0),  where v = u_j^(p^{u_j}) and alpha_0 = sum_i u_i du_{~i}.  Its
coefficients lie in the completion, not in O(m;u), so a truncated computation
would be unreliable.  It is not needed: since D(exp v) = D(v) exp(v) for the
divided powers, and exp(v) is invertible,

    L_D omega_{2,j} = exp(v) ( D(v) eta + L_D eta ),   eta := omega_0 +
                                                       ubar_j du_j alpha_0,

so that the whole computation -- the preserved algebra, its dimension and its
restrictedness -- takes place inside O(m;u), EXACTLY.  Here D(v) = D_j ubar_j.

Run:  python3 second276.py
"""
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nalt2
from nalt2 import (Alg, d2, lie, act, bracket, gram, det, reduce_basis,
                   g2_basis, flatten2, addform, smul, mul1, dvec, unit1)
from nonalt276 import square, in_span, ad_inner

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "lib"))
from exactlin import nullspace                                  # noqa: E402


def eta(alg, k, j, N):
    """omega_0 + ubar_j du_j alpha_0,  alpha_0 = sum_i u_i du_{~i}."""
    m = 2 * k
    om = {}
    for i in range(k):
        om[('a', i, i + k)] = alg.one
    ubar = alg.mono(tuple(N[t] - 1 if t == j else 0 for t in range(m)))
    # ubar_j du_j ^ sum_i u_i du_{~i}
    p = [0] * m
    p[j] = ubar
    q = [0] * m
    for i in range(k):
        q[i + k] = alg.gen(i)
    addform(om, mul1(alg, p, q))
    return {key: c for key, c in om.items() if c}


def twisted_lie(alg, D, om, j, ubar):
    """D(v) om + L_D om  with D(v) = D_j ubar_j."""
    out = dict(lie(alg, D, om))
    dv = alg.mul(D[j], ubar)
    if dv:
        addform(out, smul(alg, dv, om))
    return {k: c for k, c in out.items() if c}


def algebra(alg, k, j):
    m = 2 * k
    N = alg.N
    et = eta(alg, k, j, N)
    ubar = alg.mono(tuple(N[t] - 1 if t == j else 0 for t in range(m)))
    b2 = g2_basis(m)
    gens, cols = [], []
    for i in range(m):
        for t in range(alg.dim):
            D = [0] * m
            D[i] = 1 << t
            gens.append(D)
            cols.append(flatten2(alg, twisted_lie(alg, D, et, j, ubar), b2))
    M = [[cols[c][r] for c in range(len(cols))] for r in range(len(cols[0]))]
    ns = nullspace(M, p=2)
    out = []
    for v in ns:
        D = [0] * m
        for c, coef in enumerate(v):
            if int(coef) % 2:
                for i in range(m):
                    D[i] ^= gens[c][i]
        out.append(D)
    return et, reduce_basis(alg, out)


def report(k, u, j):
    m = 2 * k
    alg = Alg(m, u)
    et, h = algebra(alg, k, j)
    nd = alg.const_term(det(alg, gram(alg, et))) == 1
    # the undeformed Hamiltonian algebra, for comparison
    om0 = {('a', i, i + k): alg.one for i in range(k)}
    b2 = g2_basis(m)
    gens, cols = [], []
    for i in range(m):
        for t in range(alg.dim):
            D = [0] * m
            D[i] = 1 << t
            gens.append(D)
            cols.append(flatten2(alg, lie(alg, D, om0), b2))
    M = [[cols[c][r] for c in range(len(cols))] for r in range(len(cols[0]))]
    h0 = []
    for v in nullspace(M, p=2):
        D = [0] * m
        for c, coef in enumerate(v):
            if int(coef) % 2:
                for i in range(m):
                    D[i] ^= gens[c][i]
        h0.append(D)
    h0 = reduce_basis(alg, h0)
    bad = [D for D in h if not ad_inner(alg, h, square(alg, D))]
    bad0 = [D for D in h0 if not ad_inner(alg, h0, square(alg, D))]
    print("  2k=%d u=%-12s j=%d: eta non-degenerate=%-5s  dim h_exp=%-4d "
          "(dim h(omega_0)=%-4d)  fields with (ad_D)^2 not inner: %d "
          "(undeformed: %d)"
          % (m, str(tuple(u)), j + 1, nd, len(h), len(h0), len(bad), len(bad0)),
          flush=True)
    return len(bad), len(bad0)


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 78)
    print("The 2nd-type (exponential) forms at p = 2")
    print("=" * 78)
    for k, u in [(1, (1, 1)), (1, (2, 2)), (1, (2, 1)), (1, (1, 2)),
                 (2, (1, 1, 1, 1)), (2, (2, 2, 2, 2)), (2, (2, 1, 1, 2))]:
        for j in range(2 * k):
            report(k, u, j)
    print("\n[%.1f s]" % (time.time() - t0))
