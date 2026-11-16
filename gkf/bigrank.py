"""Rank of d on one large block, filled straight into a FLINT nmod_mat.

The dense Python list-of-lists that exactlin takes would need ~4 GB for the block
(n=5, k=0, q=7); the differential is sparse, so we fill a FLINT matrix entry by
entry instead.  Only the rank is wanted.

Usage:  bigrank.py <n> <k> <q> <p>
"""

import sys
import time
import os
from flint import nmod_mat
from po import Po
from ce import CE


def main():
    n, k, q, p = (int(x) for x in sys.argv[1:5])
    g = Po(n, "h1")
    ce = CE(g)
    r = len(ce.cwt[0])
    w = tuple([0] * r)
    os.makedirs("runs", exist_ok=True)
    out = open(f"runs/bigrank-n{n}-k{k}-q{q}.log", "a")

    def say(s):
        print(s, flush=True)
        out.write(s + "\n")
        out.flush()

    t = time.time()
    src = ce.basis_block(q, k, w)
    tgt = ce.basis_block(q + 1, k, w)
    say(f"n={n} k={k} q={q} p={p}: C^{q} = {len(src)}, C^{q+1} = {len(tgt)}"
        f"   [{time.time()-t:.1f} s]")
    ti = {m: i for i, m in enumerate(tgt)}
    t = time.time()
    M = nmod_mat(len(src), len(tgt), p)
    nz = 0
    for i, mon in enumerate(src):
        for m2, v in ce.d_monomial(mon).items():
            M[i, ti[m2]] = int(v) % p
            nz += 1
        if i % 2000 == 0:
            say(f"   filled {i}/{len(src)} rows, {nz} nonzeros [{time.time()-t:.1f} s]")
    say(f"   filled: {nz} nonzeros, density {nz/(len(src)*len(tgt)):.5f}"
        f"   [{time.time()-t:.1f} s]")
    t = time.time()
    rk = M.rank()
    say(f"   rank = {rk}   [{time.time()-t:.1f} s]")
    out.close()


if __name__ == "__main__":
    main()
