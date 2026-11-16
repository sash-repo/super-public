#!/usr/bin/env python3
"""For random split 2-dim tori of each algebra: the codim<=3 subalgebras graded
by the torus (deform.weisfeiler with the weight classes of that torus)."""
import sys, time
import numpy as np
from collections import Counter
import deform as D
from k3 import LieAlg
from iso6 import T, LABEL, rank3, ad
from iso6_tori import nullsp, inv3, toral_in, weight_decomp, eig_dims
P = 3


def rebase(C, B):
    """structure constants in the basis given by the rows of B."""
    Bi = inv3(B.T)                      # coords of v: Bi @ v
    n = B.shape[0]
    tab = np.einsum("ai,bj,ijk->abk", B, B, C) % P
    tab = np.einsum("ka,xya->xyk", Bi, tab) % P
    return tab


def alg_from(tab, wts):
    n = tab.shape[0]
    def brk(i, j):
        return {k: int(tab[i, j, k]) for k in range(n) if tab[i, j, k]}
    return LieAlg(list(range(n)), brk, wts, [f"b{i}" for i in range(n)])


def random_tori(C, k, seed):
    rng = np.random.default_rng(seed)
    out = []
    tries = 0
    while len(out) < k and tries < 5000:
        tries += 1
        x = rng.integers(0, P, 26)
        Z = nullsp(ad(C, x))
        if len(Z) > 4:
            continue
        ts = toral_in(C, Z)
        if not ts:
            continue
        t1 = ts[rng.integers(len(ts))]
        Z1 = nullsp(ad(C, t1))
        # random toral t2 in C(t1): random combos until toral & independent
        for _ in range(3000):
            h = sum(int(c) * b for c, b in zip(rng.integers(0, P, len(Z1)), Z1)) % P
            A = ad(C, h)
            if rank3(np.array([t1, h])) == 2 and np.array_equal(np.linalg.matrix_power(A, 3) % P, A):
                out.append((t1, h))
                break
    return out


def search(C, t1, t2):
    sp = weight_decomp(C, t1, t2)
    rows, wts, cls = [], [], {}
    for w, vs in sorted(sp.items()):
        for v in vs:
            cls.setdefault(w, []).append(len(rows))
            rows.append(v); wts.append(w)
    B = np.array(rows, dtype=np.int64)
    tab = rebase(C, B)
    alg = alg_from(tab, wts)
    D.cls = cls
    ws = D.weisfeiler(alg, maxcodim=3)
    res = Counter()
    for c, L0 in ws:
        res[(c, tuple(D.filtration(alg, L0)))] += 1
    return res, (alg, ws, B)


if __name__ == "__main__":
    k = int(sys.argv[1]); names = sys.argv[2:]
    for nm in names:
        C = T[nm]
        t0 = time.time()
        tori = random_tori(C, k, seed=5)
        summary = Counter()
        print("=" * 72); print(LABEL[nm], f": {len(tori)} random split tori")
        for t1, t2 in tori:
            res, _ = search(C, t1, t2)
            key = tuple(sorted(res.items()))
            summary[key] += 1
            print("   ", dict(res) if res else "NONE", flush=True)
        print("  summary:")
        for key, v in summary.most_common():
            print(f"   {v:3d} x {dict(key) if key else 'no codim<=3 graded subalgebra'}")
        print(f"  ({time.time()-t0:.0f}s)", flush=True)
