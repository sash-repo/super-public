#!/usr/bin/env python3
"""Tori of k^(1)(3;1), er^(1)(3;1) and the (-6;0) deform, found WITHOUT a
preferred torus: centralizers C(x) of random regular x, their toral elements
(ad h)^3 = ad h, and the weight-space dims.  For each torus type found, the
codim<=3 subalgebras graded by that torus are searched (deform.weisfeiler)."""
import sys, time
import numpy as np
from collections import Counter, defaultdict
from iso6 import T, LABEL, rank3, ad, adbasis
P = 3


def nullsp(M):
    """basis (rows) of {v : M v = 0} mod 3."""
    M = np.array(M, dtype=np.int64) % P
    rows, cols = M.shape
    piv = []
    r = 0
    for c in range(cols):
        nz = np.nonzero(M[r:, c])[0] if r < rows else []
        if len(nz) == 0:
            continue
        p = r + nz[0]
        M[[r, p]] = M[[p, r]]
        if M[r, c] == 2:
            M[r] = 2 * M[r] % P
        f = M[:, c].copy(); f[r] = 0
        M = (M - np.outer(f, M[r])) % P
        piv.append(c); r += 1
        if r == rows:
            break
    out = []
    for fc in [c for c in range(cols) if c not in piv]:
        v = np.zeros(cols, dtype=np.int64); v[fc] = 1
        for i, pc in enumerate(piv):
            v[pc] = (-M[i, fc]) % P
        out.append(v)
    return out


def inv3(B):
    n = B.shape[0]
    M = np.concatenate([B % P, np.eye(n, dtype=np.int64)], axis=1)
    for c in range(n):
        p = c + np.nonzero(M[c:, c])[0][0]
        M[[c, p]] = M[[p, c]]
        if M[c, c] == 2:
            M[c] = 2 * M[c] % P
        f = M[:, c].copy(); f[c] = 0
        M = (M - np.outer(f, M[c])) % P
    return M[:, n:]


def bracket(C, x, y):
    return np.einsum("i,j,ijk->k", x, y, C) % P


def torus_of(C, x):
    """if C(x) is a 2-dim split torus: return (t1,t2) toral basis, else None."""
    n = C.shape[0]
    Z = nullsp(ad(C, x))
    if len(Z) != 2:
        return None
    tor = []
    for a in range(P):
        for b in range(P):
            h = (a * Z[0] + b * Z[1]) % P
            A = ad(C, h)
            if np.array_equal(np.linalg.matrix_power(A, 3) % P, A):
                tor.append(h)
    if len(tor) != 9:
        return ("nonsplit", len(tor))
    # choose basis among toral elements
    t1 = next(h for h in tor if h.any())
    t2 = next(h for h in tor if rank3(np.array([t1, h])) == 2)
    return (t1, t2)


def weight_decomp(C, t1, t2):
    n = C.shape[0]
    A1, A2 = ad(C, t1), ad(C, t2)
    I = np.eye(n, dtype=np.int64)
    spaces = {}
    for a in range(P):
        for b in range(P):
            ns = nullsp(np.concatenate([(A1 - a * I) % P, (A2 - b * I) % P]))
            if ns:
                spaces[(a, b)] = ns
    return spaces


def signature(spaces):
    return tuple(sorted(len(v) for v in spaces.values()))


def gl3_orbit_sig(spaces):
    """weight dims up to GL(2,F_3) acting on the weights: the multiset of dims
    together with the dim of the zero weight space."""
    return (len(spaces.get((0, 0), [])), signature(spaces))


def survey(nm, ntry=400, seed=7):
    C = T[nm]
    rng = np.random.default_rng(seed)
    types = Counter()
    reps = {}
    for _ in range(ntry):
        x = rng.integers(0, P, 26)
        r = torus_of(C, x)
        if r is None:
            types["C(x) not 2-dim"] += 1
            continue
        if r[0] == "nonsplit":
            types[f"nonsplit ({r[1]} toral)"] += 1
            continue
        sp = weight_decomp(C, *r)
        s = gl3_orbit_sig(sp)
        types[s] += 1
        reps.setdefault(s, r)
    return types, reps


if __name__ == "__main__":
    for nm in sys.argv[1:] or ["k1", "E6_1", "erdef_p"]:
        t0 = time.time()
        types, reps = survey(nm)
        print("=" * 72)
        print(LABEL[nm])
        for s, v in types.most_common():
            print(f"   {v:5d}  {s}")
        print(f"   ({time.time()-t0:.1f}s)")
        sys.stdout.flush()


def toral_in(C, basis):
    """all nonzero toral elements in the span of `basis` (enumeration)."""
    from itertools import product
    out = []
    for co in product(range(P), repeat=len(basis)):
        if not any(co):
            continue
        h = sum(c * b for c, b in zip(co, basis)) % P
        A = ad(C, h)
        if np.array_equal(np.linalg.matrix_power(A, 3) % P, A):
            out.append(h)
    return out


def eig_dims(C, t):
    A = ad(C, t)
    I = np.eye(26, dtype=np.int64)
    return tuple(26 - rank3((A - a * I) % P) for a in range(P))


def split_tori(nm, ntry=300, seed=11):
    C = T[nm]
    rng = np.random.default_rng(seed)
    t1s = {}
    for _ in range(ntry):
        x = rng.integers(0, P, 26)
        Z = nullsp(ad(C, x))
        if len(Z) > 4:
            continue
        for h in toral_in(C, Z):
            t1s.setdefault(eig_dims(C, h), h)
    res = {}
    for e1, t1 in sorted(t1s.items()):
        Z = nullsp(ad(C, t1))
        tor = toral_in(C, Z) if len(Z) <= 9 else []
        found = Counter()
        for t2 in tor:
            if rank3(np.array([t1, t2])) < 2:
                continue
            sp = weight_decomp(C, t1, t2)
            if sum(len(v) for v in sp.values()) != 26:
                continue
            s = gl3_orbit_sig(sp)
            found[s] += 1
            res.setdefault(s, (t1, t2))
        print(f"   toral t1 with eigen-dims {e1}: dim C(t1) = {len(Z)}, "
              f"split 2-tori through it: {dict(found)}")
    return res
