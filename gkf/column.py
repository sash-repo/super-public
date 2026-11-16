"""One bigraded column of H^*(h^{(1)}(0|n); K): fixed internal degree k, torus weight 0.

This is the direct analogue of the single surviving bidegree (0, 0) of h(2), where
dim H^q = 1 for q = 0 and q = 7 and 0 otherwise.

Usage:  column.py <n> <k> <qmax> [p]
"""

import sys
import time
import os
from po import Po
from ce import CE
from exactlin_shim import rank


def main():
    n, k, qmax = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    p = int(sys.argv[4]) if len(sys.argv) > 4 else None
    g = Po(n, "h1")
    ce = CE(g)
    assert not ce.check_d2()
    r = len(ce.cwt[0])
    w = tuple([0] * r)
    os.makedirs("runs", exist_ok=True)
    out = open(f"runs/col-n{n}-k{k}.log", "a")
    out.write(f"\n==== n={n} k={k} qmax={qmax} p={p} ====\n")
    bas = {}
    for q in range(qmax + 2):
        t = time.time()
        bas[q] = ce.basis_block(q, k, w)
        msg = f"  dim C^{q}_{{{k},0}} = {len(bas[q])}   [{time.time()-t:.1f} s]"
        print(msg, flush=True)
        out.write(msg + "\n")
        out.flush()
    ranks = {}
    for q in range(qmax + 1):
        if not bas[q] or not bas[q + 1]:
            ranks[q] = 0
            continue
        t = time.time()
        ti = {m: i for i, m in enumerate(bas[q + 1])}
        M = ce.matrix(bas[q], ti)
        ranks[q] = rank(M, p=p)
        msg = (f"  rank d^{q} : {len(bas[q])} -> {len(bas[q+1])}  =  {ranks[q]}"
               f"   [{time.time()-t:.1f} s]")
        print(msg, flush=True)
        out.write(msg + "\n")
        out.flush()
    msg = f"\n  h^(1)(0|{n}), internal degree k = {k}, torus weight 0" \
          f"{'  (mod ' + str(p) + ')' if p else '  (over Q)'}:"
    print(msg)
    out.write(msg + "\n")
    for q in range(qmax + 1):
        h = len(bas[q]) - ranks[q] - (ranks[q - 1] if q else 0)
        line = f"    dim H^{q}_{{{k}}} = {h}"
        print(line)
        out.write(line + "\n")
    out.close()


if __name__ == "__main__":
    main()
