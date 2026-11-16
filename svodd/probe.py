#!/usr/bin/env python3
"""probe.py -- graded anatomy of a type: the pieces of the layer, the pieces of
rho(s_f), transitivity of G_0 on each piece, sweepability, and an orbit census."""
import sys, random, itertools
from fractions import Fraction as F
import stab, orb, sv

TYPES = {
    "II":     lambda n, m: (1, 2, [(1, (n,), ()), (1, (m,), (0, 1))]),
    "III_1":  lambda n, m: (1, 4, [(1, (n,), ()), (1, (1,), (0, 1)), (1, (m,), (2, 3))]),
    "III_2":  lambda n, m: (1, 4, [(1, (n,), ()), (1, (1,), (0, 1))]),
    "III_3":  lambda n, m: (1, 4, [(1, (3,), ()), (1, (0,), (0, 1, 2, 3))]),
    "III_4":  lambda n, m: (1, 4, [(1, (3,), ()), (1, (1,), (0, 1, 2, 3))]),
    "III_5":  lambda n, m: (1, 4, [(1, (3,), ())]),
    "IV_1":   lambda n, m: (1, 5, [(1, (n,), ()), (1, (1,), (0, 1)), (1, (m,), (2, 3)), (1, (0,), (0, 2, 3, 4))]),
    "IV_2":   lambda n, m: (1, 5, [(1, (n,), ()), (1, (1,), (0, 1)), (1, (m,), (2, 3))]),
    "IV_3":   lambda n, m: (1, 5, [(1, (n,), ()), (1, (1,), (0, 1))]),
    "IV_4":   lambda n, m: (1, 5, [(1, (n,), ()), (1, (2,), (0, 1)), (1, (m,), (2, 3)), (1, (0,), (0, 2, 3, 4))]),
    "IV_5":   lambda n, m: (1, 5, [(1, (3,), ()), (1, (1,), (0, 1, 2, 3))]),
    "IV_6":   lambda n, m: (1, 5, [(1, (3,), ())]),
    "V":      lambda n, m: (1, 6, [(1, (n,), ()), (1, (1,), (0, 1)), (1, (1,), (2, 3)), (1, (1,), (4, 5))]),
}


def probe(tname, n=3, m=1, D=None, samples=40, verbose=True):
    p, q, terms = TYPES[tname](n, m)
    if D is None:
        D = 2 * n + 4
    res = stab.analyse(f"{tname}({n},{m})", p, q, terms, D, verbose=False)
    mats, basis = res["mats"], res["basis"]
    N = len(basis)
    degs = [stab.mdeg(b) for b in basis]
    pieces = orb.graded_pieces(mats, degs)
    name = f"{tname}({n},{m})"
    print(f"\n=== {name}  p|q={p}|{q}  sdim O_f=({res['sdim'][0]}|{res['sdim'][1]})"
          f"  dim Nt={N}  dim rho={res['rhodim']}  gorb={res['gdim']}"
          f"  mod={res['modality']}  {res['verdict']}")
    bydeg = {}
    for i, d in enumerate(degs):
        bydeg.setdefault(d, []).append(i)
    for d in sorted(bydeg):
        print(f"    Nt^({d}): dim {len(bydeg[d]):2d}   "
              + ", ".join(sv.show_mon(basis[i], "x", "xi") for i in bydeg[d]))
    print("    rho(s_f) graded pieces: " +
          ", ".join(f"s_{d}:{len(pieces[d])}" for d in sorted(pieces)))

    # transitivity of g_0 on each graded piece (tangent-space test)
    p0 = pieces.get(0, [])
    rnd = random.Random(5)
    for d in sorted(bydeg):
        idx = bydeg[d]
        k = len(idx)
        # restricted matrices of g_0 on this piece
        sub = []
        for M in p0:
            S = [[M[i][j] for j in idx] for i in idx]
            if any(any(r) for r in S):
                sub.append(S)
        dimsub = stab.rank_rows([sum(S, []) for S in sub]) if sub else 0
        # minimal orbit dimension over many probe vectors, incl. sparse ones
        worst = None
        probes = []
        for r in range(1, min(k, 3) + 1):
            for c in itertools.combinations(range(k), r):
                v = [F(0)] * k
                for j in c:
                    v[j] = F(1)
                probes.append(v)
                if r > 1:
                    v2 = [F(0)] * k
                    for t, j in enumerate(c):
                        v2[j] = F(1 + t)
                    probes.append(v2)
        for _ in range(60):
            probes.append([F(rnd.randint(-3, 3)) for _ in range(k)])
        for v in probes:
            if all(c == 0 for c in v):
                continue
            od = stab.rank_rows([[sum(S[i][j] * v[j] for j in range(k))
                                  for i in range(k)] for S in sub]) if sub else 0
            if worst is None or od < worst[0]:
                worst = (od, v)
        print(f"    g_0 on Nt^({d}): dim image {dimsub:3d}, min orbit dim over "
              f"{len(probes)} probes = {worst[0] if worst else 0}/{k}"
              + ("   TRANSITIVE on nonzero" if worst and worst[0] == k else
                 f"   worst at {worst[1] if worst else None}"))

    # sweepability: for each graded piece and each probe leading vector,
    # is s_s . v = Nt^(e+s) for all s >= 1?
    for d in sorted(bydeg):
        idx = bydeg[d]
        k = len(idx)
        allsw = True
        detail = []
        for s in sorted(x for x in pieces if x >= 1):
            tgt = bydeg.get(d + s)
            if not tgt:
                continue
            worst = None
            for _ in range(30):
                v = [F(0)] * N
                for i in idx:
                    v[i] = F(rnd.randint(-3, 3))
                if all(v[i] == 0 for i in idx):
                    continue
                rows = []
                for M in pieces[s]:
                    u = orb.apply_mat(M, v)
                    rows.append([u[i] for i in tgt])
                rk = stab.rank_rows(rows) if rows else 0
                if worst is None or rk < worst:
                    worst = rk
            if worst is not None and worst < len(tgt):
                allsw = False
                detail.append(f"s={s}: {worst}/{len(tgt)}")
        print(f"    sweep from Nt^({d}): "
              + ("full" if allsw else "PARTIAL  " + " ".join(detail)))

    cache = {}
    cen = orb.orbit_census(mats, N, samples=samples, degs=degs, cache=cache)
    print(f"    orbit census ({samples} random points): {len(cen)} classes")
    for c, dd, h in cen:
        nz = [(sv.show_mon(basis[i], "x", "xi"), str(c[i]))
              for i in range(N) if c[i]]
        print(f"        dim {dd:2d} hits {h:3d}  "
              + (" + ".join(f"{a}.{b}" for b, a in nz) if nz else "0"))
    return res, pieces, bydeg


if __name__ == "__main__":
    t = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    m = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    probe(t, n, m)
