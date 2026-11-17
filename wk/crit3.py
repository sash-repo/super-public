#!/usr/bin/env python3
"""crit3.py -- the irreducibility criterion of Statement 3.2.4 for n = 3
(and the attainment of Statement 3.2.2 for n = 3).

M(lambda) over o^{(1)}_I(7) is 2^{12} = 4096-dimensional.  The module is
graded by the root lattice (the relations of u_{2,4}(g; chi) with chi = 0 on
n_+- are homogeneous), so the n_+-singular vectors are computed block by
block: the weights of D are taken to be (1, 100, 10000), so that the integer
degree determines the Z^3-degree.

Tested claim:  M(lambda) irreducible  <=>  lambda_i and lambda_i + lambda_j
(i != j) all lie outside F_2.

Output: crit3.log
"""
import itertools, random, sys, time
from collections import defaultdict
from wk import GF, Induced, oI, order_of, triangular, nullspace

LOG = open("crit3.log", "a")


def out(s=""):
    print(s, flush=True)
    LOG.write(s + "\n"); LOG.flush()


def setup(F, n):
    alg, deg = oI(F, n, weights=[100 ** i for i in range(n)])
    alg.build_struct()
    neg, tor, pos = triangular(alg, deg)
    orders = [order_of(alg, i) for i in neg]
    return alg, deg, neg, tor, pos, orders


def nsing(F, n, data, lam):
    alg, deg, neg, tor, pos, orders = data
    M = Induced(alg, neg, orders, tor, pos, list(lam))
    mdeg = [sum(a[r] * deg[neg[r]] for r in range(len(neg))) for a in M.mons]
    blocks = defaultdict(list)
    for i, d in enumerate(mdeg):
        blocks[d].append(i)
    total = 0
    for d, idx in blocks.items():
        loc = {i: c for c, i in enumerate(idx)}
        rows = defaultdict(dict)   # (pos gen, target index) -> {col: coeff}
        for c, i in enumerate(idx):
            for p in pos:
                for t, x in M.act(M._e(p), M.mons[i]).items():
                    rows[(p, t)][c] = x
        R = []
        for key, dct in rows.items():
            r = [0] * len(idx)
            for c, x in dct.items():
                r[c] = x
            R.append(r)
        total += len(nullspace(F, R, len(idx))) if R else len(idx)
        if total > 1:
            return total
    return total


def pred(F, lam):
    vals = list(lam) + [F.add(lam[i], lam[j]) for i in range(len(lam))
                        for j in range(i + 1, len(lam))]
    return all(v not in (0, 1) for v in vals)


def main():
    n = int(sys.argv[1]); k = int(sys.argv[2])
    mode = sys.argv[3] if len(sys.argv) > 3 else "all"
    F = GF(k)
    data = setup(F, n)
    out("n = %d over %s, mode %s, dim M = %d" % (n, F.name, mode,
        2 ** data[5].count(2) * 4 ** data[5].count(4)))
    lams = list(itertools.product(F.els, repeat=n))
    if mode.startswith("sample"):
        random.seed(1)
        m = int(mode[6:])
        good = [l for l in lams if pred(F, l)]
        bad = [l for l in lams if not pred(F, l)]
        lams = random.sample(good, min(m, len(good))) + \
            random.sample(bad, min(m, len(bad)))
    nirr = npred = 0; mism = []
    t0 = time.time()
    for lam in lams:
        s = nsing(F, n, data, lam)
        irr = (s == 1); p = pred(F, lam)
        nirr += irr; npred += p
        if irr != p:
            mism.append(lam)
    out("  %d weights tested: irreducible %d, predicted %d, mismatches %s"
        " (%.0f s)" % (len(lams), nirr, npred, mism or "none", time.time() - t0))


if __name__ == "__main__":
    main()
