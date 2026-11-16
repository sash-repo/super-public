#!/usr/bin/env python3
"""A) The divided power de Rham cohomology of O(n|n;u), p > 2.

Machine dimensions of H^k and of its even/odd parts against the closed formula
predicted by Kuenneth:

    H^*(O(n|n;u)) = Lambda[theta_1..theta_n] (x) (x)_j H^*(Lambda(xi_j)),

    theta_i = [qbar_i dq_i],  qbar_i = q_i^{(p^{u_i}-1)},  deg 1, parity ODD,
    H^k(Lambda(xi)) = <(dxi)^{(k)}>  if p | k      (parity even)
                    + <xi (dxi)^{(k)}>  if p | k+1 (parity odd).

For p > 2 and k <= 3 the odd factor contributes only

    zeta_j = [xi_j (dxi_j)^{(2)}],  deg 2, parity ODD,  iff p = 3,
    eta_j  = [(dxi_j)^{(3)}],       deg 3, parity even, iff p = 3.

Hence the two statements the text rests on:

    dim H^2_odd = 0 for p > 3,      dim H^2_odd = n for p = 3.

Usage: python3 cohom.py
"""
import sys, os, itertools
from math import comb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from peri import Omega, cohomology, is_closed, is_exact   # noqa


def predict(n, p, u, deg, par=None):
    """Kuenneth prediction: enumerate products of the generators."""
    # generators: theta_i (deg 1, par 1), and for each j the classes of the odd
    # factor: (dxi_j)^{(k)} for p|k (par 0), xi_j(dxi_j)^{(k)} for p|k+1 (par 1)
    gens = []
    for i in range(n):
        gens.append(("theta", i, 1, 1))
    # for each odd factor j, the allowed exponents k with a class, k <= deg
    odd_classes = []
    for j in range(n):
        cl = [(0, 0, 0)]                       # (form degree, parity, kind) = 1
        for k in range(1, deg + 1):
            if k % p == 0:
                cl.append((k, 0, 1))           # (dxi_j)^{(k)}
            if (k + 1) % p == 0:
                cl.append((k, 1, 2))           # xi_j (dxi_j)^{(k)}
        odd_classes.append(cl)
    total = 0
    # choose a subset of the thetas and one class per odd factor
    for S in itertools.chain.from_iterable(
            itertools.combinations(range(n), k) for k in range(n + 1)):
        for choice in itertools.product(*odd_classes):
            dg = len(S) + sum(c[0] for c in choice)
            pr = (len(S) + sum(c[1] for c in choice)) % 2
            if dg == deg and (par is None or pr == par):
                total += 1
    return total


def main():
    log = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "cohom.log"), "w")

    def out(s=""):
        print(s)
        log.write(s + "\n")
        log.flush()

    out("A) de Rham cohomology of the divided power complex over O(n|n;u), p>2")
    out("   columns: dim H^k (machine) / prediction, then the odd part")
    bad = 0
    cases = [(1, 3, (1,)), (1, 3, (2,)), (2, 3, (1, 1)), (2, 3, (1, 2)),
             (1, 5, (1,)), (2, 5, (1, 1)), (1, 7, (1,)), (3, 3, (1, 1, 1))]
    for (n, p, u) in cases:
        alg = Omega(n, p, u)
        out(f"\n  n={n}  p={p}  u={u}   dim O = {len(alg.fun_basis())}")
        for k in range(0, 4):
            if n == 3 and k == 3:
                continue                      # too big, and nothing new
            h, dim, r1, r0 = cohomology(alg, k)
            ho, *_ = cohomology(alg, k, par=1)
            he, *_ = cohomology(alg, k, par=0)
            ph, po, pe = (predict(n, p, u, k), predict(n, p, u, k, 1),
                          predict(n, p, u, k, 0))
            flag = ""
            if (h, ho, he) != (ph, po, pe):
                flag = "   <<< MISMATCH"
                bad += 1
            out(f"    k={k}:  dim Omega^k={dim:5d}   H^k = {h}/{ph}   "
                f"H^k_even = {he}/{pe}   H^k_odd = {ho}/{po}{flag}")
            if h != ho + he:
                out("      <<< parity split inconsistent")
                bad += 1
    out(f"\n  mismatches: {bad}")

    # B) the exceptional class at p = 3
    out("\nB) the odd classes in H^2 at p = 3")
    for (n, u) in [(1, (1,)), (1, (2,)), (2, (1, 1)), (2, (1, 2)), (3, (1, 1, 1))]:
        alg = Omega(n, 3, u)
        for j in range(n):
            z = alg.mul(alg.xi(j), alg.dxi(j, 2))
            cl, ex = is_closed(alg, z), is_exact(alg, z)
            out(f"    n={n} u={u}: xi_{j}(dxi_{j})^(2)  closed={cl}  exact={ex}")
            if not cl or ex:
                bad += 1
        # the off-diagonal ones ARE exact
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                z = alg.mul(alg.xi(i), alg.dxi(j, 2))
                cl, ex = is_closed(alg, z), is_exact(alg, z)
                out(f"    n={n} u={u}: xi_{i}(dxi_{j})^(2)  closed={cl}  exact={ex}")
    out(f"\n  total mismatches: {bad}")
    log.close()


if __name__ == "__main__":
    main()
