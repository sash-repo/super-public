"""One bigraded column of H^q_k(h^{(1)}(0|n); K), filled straight into FLINT nmod_mat.

Same mathematics as column.py; the difference is bookkeeping and memory.  column.py
builds the dense Python list-of-lists that exactlin takes, which is ~1 GB already at
(n=5, k=-9, q=7) and out of reach one degree higher; here the sparse differential is
poured entry by entry into an nmod_mat, and every q prints its dimensions, its rank
and its timing at once, flushed, so that a long run leaves a readable log behind.

The torus weight is 0 throughout: the maximal torus of g_0 = o(n) is inner, so
L_x = d i_x + i_x d is homotopic to zero and no cohomology sits outside weight 0
(verified for n = 4, where the whole complex is within reach in all weights).

Over F_p the complex S^*(Pi g^*) is the honest one only for q < p -- above that the
divided powers part company -- so p is passed in and must exceed qmax.

    ../venv/bin/python colbig.py <n> <k> <qmax> <p> [p2 ...]
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flint import nmod_mat                                        # noqa: E402
from po import Po                                                 # noqa: E402
from ce import CE                                                 # noqa: E402


def rank_block(ce, src, tgt, p, say):
    """rank of d : span(src) -> span(tgt) over F_p, filled sparsely."""
    if not src or not tgt:
        return 0
    ti = {m: i for i, m in enumerate(tgt)}
    t = time.time()
    M = nmod_mat(len(src), len(tgt), p)
    nz = 0
    for i, mon in enumerate(src):
        for m2, v in ce.d_monomial(mon).items():
            j = ti.get(m2)
            if j is None:                      # d cannot leave the block
                raise AssertionError("d left the block")
            M[i, j] = int(v) % p
            nz += 1
    say(f"      filled {len(src)}x{len(tgt)}, {nz} nonzeros, "
        f"density {nz/(len(src)*len(tgt)):.5f}   [{time.time()-t:.1f} s]")
    t = time.time()
    rk = M.rank()
    say(f"      rank = {rk}   [{time.time()-t:.1f} s]")
    return rk


def main():
    n, k, qmax = (int(x) for x in sys.argv[1:4])
    primes = [int(x) for x in sys.argv[4:]]
    g = Po(n, "h1")
    ce = CE(g)
    assert not ce.check_d2()
    w = tuple([0] * len(ce.cwt[0]))
    os.makedirs("runs", exist_ok=True)
    out = open(f"runs/colbig-n{n}-k{k}.log", "a")

    def say(s):
        print(s, flush=True)
        out.write(s + "\n")
        out.flush()

    say(f"\n==== n={n} k={k} qmax={qmax} primes={primes} ====")
    bas = {}
    for q in range(qmax + 2):
        t = time.time()
        bas[q] = ce.basis_block(q, k, w)
        say(f"  dim C^{q}_{{{k},0}} = {len(bas[q])}   [{time.time()-t:.1f} s]")

    for p in primes:
        if p <= qmax:
            say(f"  SKIPPED p={p}: the divided powers part company at q=p")
            continue
        say(f"  --- over F_{p}")
        ranks = {}
        for q in range(qmax + 1):
            say(f"    d^{q} : {len(bas[q])} -> {len(bas[q+1])}")
            ranks[q] = rank_block(ce, bas[q], bas[q + 1], p, say)
        say(f"  h^(1)(0|{n}), internal degree k = {k}, weight 0, mod {p}:")
        for q in range(qmax + 1):
            h = len(bas[q]) - ranks[q] - (ranks[q - 1] if q else 0)
            say(f"    dim H^{q}_{{{k}}} = {h}"
                f"      (dim C = {len(bas[q])}, rank out = {ranks[q]},"
                f" rank in = {ranks[q-1] if q else 0})")
    out.close()


if __name__ == "__main__":
    main()
