#!/usr/bin/env python3
"""Independent verification: take an explicit solution (coefficient vector in
the ansatz basis) and check the invariance identity

  L^{lam}_f(B(phi,psi)) = (-1)^{p(f)p(B)}[B(L_f phi, psi)
                                          + (-1)^{p(f)p(phi)} B(phi, L_f psi)]

for EVERY generating function f of the subalgebra, on a large set of monomials.
Written independently of knclass.build (no caching, no shortcuts)."""
from fractions import Fraction as Fr
from superN import add, smul, mul, mono, parity, par_mon
from knclass import terms_of_order, label


def check(alg, ts, coeffs, lam1, lam2, N, fields, tmax=4, verbose=True):
    pB = N % 2
    lam = lam1 + lam2 + N
    Ps = [alg.Dword(A, kA) for (kA, A, kB, B) in ts]
    Qs = [alg.Dword(B, kB) for (kA, A, kB, B) in ts]
    pQ = [len(B) % 2 for (kA, A, kB, B) in ts]

    def Bop(u, v, pu):
        out = {}
        for i, c in enumerate(coeffs):
            if c == 0:
                continue
            s = -1 if (pQ[i] and pu) else 1
            out = add(out, smul(s * c, mul(Ps[i](u), Qs[i](v))))
        return out

    MS = alg.monos(tmax)
    bad = []
    for nm, f in fields.items():
        pf = parity(f)
        sB = -1 if (pf and pB) else 1
        L1, L2, Lt = alg.L(f, lam1), alg.L(f, lam2), alg.L(f, lam)
        for m1 in MS:
            u = mono(*m1)
            pu = par_mon(m1)
            s2 = -1 if (pf and pu) else 1
            for m2 in MS:
                v = mono(*m2)
                res = add(Lt(Bop(u, v, pu)),
                          smul(-sB, add(Bop(L1(u), v, (pu + pf) % 2),
                                        smul(s2, Bop(u, L2(v), pu)))))
                if res:
                    bad.append((nm, m1, m2))
                    break
            if bad and bad[-1][0] == nm:
                break
    if verbose:
        if bad:
            for nm, m1, m2 in bad:
                print(f"   NOT invariant under K_{nm}: first failure at "
                      f"phi = {m1}, psi = {m2}")
        else:
            print(f"   invariant under all of {list(fields)} "
                  f"(monomials up to t^{tmax})")
    return not bad


def osp_fields(alg):
    """ALL the generating functions of i(osp(n|2)), by name."""
    n = alg.n
    out = {'1': mono(0, 0), 't': mono(1, 0), 't2': mono(2, 0)}
    for i in range(n):
        out[alg.names[i]] = mono(0, 1 << i)
        out['t' + alg.names[i]] = mono(1, 1 << i)
    for i in range(n):
        for j in range(i + 1, n):
            out[alg.names[i] + alg.names[j]] = mono(0, (1 << i) | (1 << j))
    return out


if __name__ == '__main__':
    import sys
    from run_kn import ALGS
    from knclass import Classifier, osp_gens
    key = int(sys.argv[1])
    N = int(sys.argv[2])
    alg = ALGS[key]
    l1, l2 = Fr(3, 7), Fr(-5, 11)
    C = Classifier(alg, osp_gens(alg), l1, l2)
    ts, ns = C.solve(N)
    print(f"k(1|{key}), order {N}: dim = {len(ns)} (positive generators only)")
    for v in ns:
        print("  solution:", {label(t, alg.names): str(c)
                              for t, c in zip(ts, v) if c != 0})
        check(alg, ts, v, l1, l2, N, osp_fields(alg))
