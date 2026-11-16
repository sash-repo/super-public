#!/usr/bin/env python3
"""n = 4: how the binary (GRC) operators and the unary (Bol) operators of the
family h(a) = osp_a(4|2) are related.

  (1) is the order-4 binary space the same for different a?
  (2) at lam1 = -3 (the Bol weight) does the binary space contain
      B_0(a) (x) 1,  B_0(a) = T_4 + (a/2) dt^2 ?
  (3) the order-2 and order-4 weight scans (exceptional weights).
"""
import sys
from fractions import Fraction as Fr
from sympy import Matrix, S, symbols
from knclass import Classifier, terms_of_order, label
from k14family import ALG
from bol_n4 import wgens, hfields
from verify_kn import check

NM = ALG.names
GEN = lambda a: list(hfields(a).values())


def bol_vec(a, N, k):
    """B_k(a) (x) 1  in the order-N binary ansatz, N = 4 + 2k."""
    ts = terms_of_order(4, N)
    v = [0] * len(ts)
    want = {(k + 2, (), 0, ()): S(a + k + 2) / (k + 2),
            (k + 1, (0, 1), 0, ()): -1,
            (k + 1, (2, 3), 0, ()): -1,
            (k, (0, 1, 2, 3), 0, ()): 1}
    for i, t in enumerate(ts):
        if t in want:
            v[i] = want[t]
    return ts, v


def in_span(vecs, w):
    M = Matrix([[S(x) for x in u] for u in vecs])
    M2 = M.col_join(Matrix([[S(x) for x in w]]))
    return M.rank() == M2.rank()


if __name__ == '__main__':
    job = sys.argv[1]

    if job == 'span':                     # (1)
        l1, l2 = Fr(3, 7), Fr(-5, 11)
        N = int(sys.argv[2]) if len(sys.argv) > 2 else 4
        avals = [Fr(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 \
            else [Fr(0), Fr(2)]
        sols = {}
        for a in avals:
            ts, ns = Classifier(ALG, GEN(a), l1, l2).solve(N, tmax=2)
            sols[a] = ns
            print(f"  a={a}: dim={len(ns)}")
            sys.stdout.flush()
        allv = [v for a in avals for v in sols[a]]
        print("  rank of the joint span:", Matrix([[S(x) for x in v]
                                                   for v in allv]).rank())

    elif job == 'bolweight':              # (2)
        N = int(sys.argv[2]) if len(sys.argv) > 2 else 4
        k = (N - 4) // 2
        l2 = Fr(-5, 11)
        for a in [Fr(x) for x in sys.argv[3].split(',')]:
            l1 = Fr(-3 - k)
            ts, ns = Classifier(ALG, GEN(a), l1, l2).solve(N, tmax=2)
            tsb, w = bol_vec(a, N, k)
            assert tsb == ts
            print(f"  a={a}, lam1={l1} (Bol weight), lam2={l2}: dim={len(ns)},"
                  f"  B_{k}(a)(x)1 in the span: {in_span(ns, w)}")
            # is the Bol vector itself invariant as a binary operator?
            ok = check(ALG, ts, w, l1, l2, N, hfields(a), tmax=3, verbose=False)
            print(f"      B_{k}(a)(x)1 invariant (direct check): {ok}")
            sys.stdout.flush()

    elif job == 'scan':                   # (3)
        N = int(sys.argv[2])
        a = Fr(sys.argv[3]) if len(sys.argv) > 3 else Fr(0)
        lo, hi = (-8, 8) if len(sys.argv) <= 4 else \
            (int(sys.argv[4]), int(sys.argv[5]))
        GRID = [Fr(x, 2) for x in range(lo, hi + 1)] + [Fr(3, 7)]
        dims = {}
        for i, l1 in enumerate(GRID):
            for l2 in GRID[i:]:
                d = len(Classifier(ALG, GEN(a), l1, l2).solve(N, tmax=2)[1])
                dims[(l1, l2)] = d
            sys.stdout.flush()
        base = max(set(dims.values()), key=list(dims.values()).count)
        print(f"order {N}, a={a}: generic dim = {base}")
        for kk, v in sorted(dims.items(), key=str):
            if v != base:
                print(f"   lam1={kk[0]} lam2={kk[1]}: dim={v}")

    elif job == 'dump':                   # save the solution vectors for analysis
        import json
        N = int(sys.argv[2])
        l1 = Fr(sys.argv[3]) if len(sys.argv) > 3 else Fr(3, 7)
        l2 = Fr(sys.argv[4]) if len(sys.argv) > 4 else Fr(-5, 11)
        avals = [Fr(x) for x in sys.argv[5].split(',')]
        out = {}
        for a in avals:
            ts, ns = Classifier(ALG, GEN(a), l1, l2).solve(N, tmax=2)
            out[str(a)] = [[str(x) for x in v] for v in ns]
            print(f"  a={a}: dim={len(ns)}")
            sys.stdout.flush()
        json.dump({'terms': [str(t) for t in terms_of_order(4, N)],
                   'labels': [label(t, NM) for t in terms_of_order(4, N)],
                   'l1': str(l1), 'l2': str(l2), 'sols': out},
                  open(f'sols-n4-N{N}-{l1}-{l2}.json'.replace('/', '_'), 'w'))
