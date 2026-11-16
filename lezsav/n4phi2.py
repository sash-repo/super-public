#!/usr/bin/env python3
"""n4phi2.py -- Phi(lam) := Y . R1(x0 + K lam), the projection of
the step-1 residual onto the cokernel of the (constant) level-4 matrix M4.
Step 1 is solvable at lam iff Phi(lam) = 0.

  coker   : left null space of M4 over Q(i), component by component
  sample  : Phi at random kernel points (exact)
  jac     : rank of dPhi/dlam at a random point (5-point formula, exact for
            degree <= 4 in lam)
Usage: ../venv/bin/python n4phi2.py <stage> [loop] [a0] [args]
"""
import os, sys, pickle, random, time
import multiprocessing as mp
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from fractions import Fraction as Q
import n4noeth as N
from n4rec import qadd, qsmul, resid_full
from n4noeth import mzero, C
from n4loop import step_of
import exactlin

HERE = os.path.dirname(os.path.abspath(__file__))


def load(tag):
    with open(os.path.join(HERE, "build-%s.pkl" % tag), "rb") as fh:
        return pickle.load(fh)


def cnull_left(keys, cols):
    """complex left null space of the |keys| x |cols| matrix: list of dicts
    key -> C.  Via the real form of A^T: [[Ar^T,-Ai^T],[Ai^T,Ar^T]]."""
    K = len(keys); kidx = {k: i for i, k in enumerate(keys)}
    rows = []
    for c in cols:                       # row of A^T  (real & imag eqns)
        r1 = [0] * (2 * K); r2 = [0] * (2 * K)
        for k, v in c.items():
            i = kidx[k]
            r1[i] = v.re; r1[K + i] = -v.im
            r2[i] = v.im; r2[K + i] = v.re
        rows.append(r1); rows.append(r2)
    if not rows:
        rows = [[0] * (2 * K)]
    ns = exactlin.nullspace(rows)
    # choose a C-basis: the real span is J-stable, J(yr,yi)=(-yi,yr)
    basis, span = [], []
    def rk(M):
        return exactlin.rank(M) if M else 0
    for v in ns:
        if rk(span + [list(v)]) > rk(span):
            basis.append(v)
            span += [list(v), [-x for x in v[K:]] + list(v[:K])]
    out = []
    for v in basis:
        out.append({keys[i]: C(v[i], v[K + i]) for i in range(K)
                    if v[i] or v[K + i]})
    return out


def coker(tag):
    t0 = time.time()
    B = load(tag)
    M4 = B["M4"]
    comp = N._components(M4)
    groups = {}
    for root, mem in comp.items():
        ks = [m[1] for m in mem if m[0] == "k"]
        js = [m[1] for m in mem if m[0] == "c"]
        if ks:
            groups[root] = (sorted(ks), js)
    Y = []
    for root, (ks, js) in groups.items():
        ys = cnull_left(ks, [M4[j] for j in js])
        Y += ys
    covered = {k for ks, js in groups.values() for k in ks}
    print("M4: %d columns, %d row keys in %d components; coker dim inside"
          " them %d [%.0fs]" % (len(M4), len(covered), len(groups), len(Y),
                                time.time() - t0), flush=True)
    with open(os.path.join(HERE, "coker-%s.pkl" % tag), "wb") as fh:
        pickle.dump(dict(Y=Y, covered=covered), fh)


_G = {}


def _init(tag):
    B = load(tag)
    with open(os.path.join(HERE, "coker-%s.pkl" % tag), "rb") as fh:
        Cc = pickle.load(fh)
    _G.update(B); _G.update(Cc)


def phi_of_coeffs(co):
    """Phi as a dict: ('y',i) -> C for coker vectors, ('free',key) -> C for
    step-1 keys not touched by M4 (each such key is an equation by itself)."""
    q = _G["q0"]
    u = mzero()
    for j, c in enumerate(co):
        if c:
            u = qadd(u, qsmul(c, _G["d2"][j]))
    R = resid_full(qadd(q, u))
    out = {}
    bad0 = sum(1 for k in R if step_of(k) == 0)
    R1 = {k: v for k, v in R.items() if step_of(k) == 1}
    for i, y in enumerate(_G["Y"]):
        s = C()
        for k, c in y.items():
            if k in R1:
                s = s + c * R1[k]
        if s:
            out[("y", i)] = s
    for k, v in R1.items():
        if k not in _G["covered"]:
            out[("free", k)] = v
    return out, bad0


def coeffs_of_lam(lam):
    co = list(_G["x0"])
    for l, v in zip(lam, _G["ker0"]):
        if l:
            co = [a + l * b for a, b in zip(co, v)]
    return co


def _sample(seed):
    rng = random.Random(seed)
    lam = [C(rng.randint(-5, 5), rng.randint(-5, 5)) for _ in _G["ker0"]]
    if seed == 0:
        lam = [C() for _ in _G["ker0"]]
    ph, b0 = phi_of_coeffs(coeffs_of_lam(lam))
    nfree = sum(1 for k in ph if k[0] == "free")
    byz = {}
    for k in ph:
        if k[0] == "free":
            byz[k[1][2][0]] = byz.get(k[1][2][0], 0) + 1
    return seed, len(ph), b0, (nfree, sorted(byz.items()))


def _dir(args):
    j, lam = args
    e = [C() for _ in lam]
    vals = {}
    for t in (-2, -1, 1, 2):
        l2 = list(lam); l2[j] = l2[j] + t
        vals[t], _ = phi_of_coeffs(coeffs_of_lam(l2))
    keys = set().union(*vals.values())
    d = {}
    for k in keys:
        f = {t: vals[t].get(k, C()) for t in vals}
        v = ((f[1] - f[-1]) * 8 - (f[2] - f[-2])) * Q(1, 12)
        if v:
            d[k] = v
    return j, d


if __name__ == "__main__":
    st, tag = sys.argv[1], sys.argv[2]
    if st == "coker":
        coker(tag)
    elif st == "sample":
        n = int(sys.argv[3])
        _init(tag)
        with mp.Pool(8, _init, (tag,)) as p:
            for s, nph, b0, ex in p.imap_unordered(_sample, range(n)):
                print("sample %d: step-0 rows nonzero %d; Phi has %d nonzero"
                      " components; e.g. %s" % (s, b0, nph, ex), flush=True)
    elif st == "jac":
        _init(tag)
        rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
        lam = [C(rng.randint(-5, 5), rng.randint(-5, 5)) for _ in _G["ker0"]]
        t0 = time.time()
        cols = {}
        with mp.Pool(8, _init, (tag,)) as p:
            for j, d in p.imap_unordered(_dir, [(j, lam) for j in
                                                range(len(lam))], chunksize=2):
                cols[j] = d
        keys = sorted(set().union(*cols.values()), key=str)
        kidx = {k: i for i, k in enumerate(keys)}
        # complex rank via real form
        rows = []
        for j in range(len(lam)):
            r1 = [0] * (2 * len(keys)); r2 = [0] * (2 * len(keys))
            for k, v in cols[j].items():
                i = kidx[k]; L = len(keys)
                r1[i] = v.re; r1[L + i] = v.im
                r2[i] = -v.im; r2[L + i] = v.re
            rows += [r1, r2]
        rk = exactlin.rank(rows) if keys else 0
        print("Jacobian: %d components of Phi, %d directions, complex rank %d"
              " [%.0fs]" % (len(keys), len(lam), rk // 2, time.time() - t0),
              flush=True)
        with open(os.path.join(HERE, "jac-%s.pkl" % tag), "wb") as fh:
            pickle.dump(dict(lam=lam, cols=cols, keys=keys), fh)
