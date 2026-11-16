"""dim H^{q,N}(h(2), sl(2); R) for one internal degree N, printed degree by degree.

Same mathematics as h2.relative_cohomology; the only difference is bookkeeping, so that
a run of several hours leaves a readable log behind instead of printing once at the end:

  1) the image of the relative basis under d is accumulated from a SPARSE differential
     (the dense |tgt| x |src| matrix of h2.differential is ~10^8 entries at N=7, q=7 and
     the triple Python loop over it is ~3*10^10 operations -- days);
  2) every q prints its dimensions, its rank and its timing at once, flushed.

The two assertions of h2.relative_cohomology are kept: d must land inside the relative
complex, and the rank is taken of the image expressed in the condition-1 basis.

    python3 cohN.py <N> [qmax]        (use ../venv/bin/python for the FLINT backend)
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h2                                                        # noqa: E402
from h2 import basis, bracket, keyof, perm_sign, relative_space   # noqa: E402
from exactlin import rank                                         # noqa: E402


def differential_sparse(q, N):
    """d: C^{q,N} -> C^{q+1,N} in the condition-1 bases, as (row, col, coeff) triples."""
    src = basis(q, N, True)
    tgt = basis(q + 1, N, True)
    sidx = {S: i for i, S in enumerate(src)}
    acc = {}
    for r, T in enumerate(tgt):
        for u in range(len(T)):
            for v in range(u + 1, len(T)):
                c, w = bracket(T[u], T[v])
                if c == 0:
                    continue
                rest = [T[i] for i in range(len(T)) if i != u and i != v]
                cand = [w] + rest
                sign = perm_sign(cand)
                if sign is None:
                    continue
                S = tuple(sorted(cand, key=keyof))
                j = sidx.get(S)
                if j is None:
                    continue
                acc[(r, j)] = acc.get((r, j), 0) + ((-1) ** (u + v)) * sign * c
    return [(r, j, c) for (r, j), c in acc.items() if c], src, tgt


def image_rows(vecs, trips, ntgt):
    """The images d(v), v in vecs, as rows in the condition-1 basis of degree q+1."""
    img = [[0] * ntgt for _ in vecs]
    for r, j, c in trips:
        for k, v in enumerate(vecs):
            a = v[j]
            if a:
                img[k][r] += c * a
    return img


def main():
    N = int(sys.argv[1])
    qmax = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    print("N = %d, qmax = %d, backend = %s" % (N, qmax, __import__("exactlin").backend()),
          flush=True)

    spaces = {0: ([], [])}
    for q in range(1, qmax + 2):
        t0 = time.time()
        spaces[q] = relative_space(q, N)
        print("  space q=%-3d cond1 %-6d dim C~ %-5d %.1fs"
              % (q, len(spaces[q][1]), len(spaces[q][0]), time.time() - t0), flush=True)

    ranks = {}
    for q in range(1, qmax + 1):
        vsrc, bsrc = spaces[q]
        vtgt, btgt = spaces[q + 1]
        if not vsrc:
            ranks[q] = 0
            continue
        t0 = time.time()
        trips, src, tgt = differential_sparse(q, N)
        assert src == bsrc and tgt == btgt
        img = image_rows(vsrc, trips, len(tgt))
        ranks[q] = rank(img) if img else 0
        if img and vtgt:
            assert rank(vtgt) == rank(list(vtgt) + img), \
                "d does not preserve the relative complex (q=%d, N=%d)" % (q, N)
        elif img:
            assert rank(img) == 0, "d does not land in the relative complex (q=%d)" % q
        print("  rank  q=%-3d nnz %-8d rank d %-5d %.1fs"
              % (q, len(trips), ranks[q], time.time() - t0), flush=True)

    dims = {q: len(spaces[q][0]) for q in range(1, qmax + 1)}
    coh = {q: dims[q] - ranks.get(q, 0) - ranks.get(q - 1, 0) for q in range(1, qmax + 1)}
    print()
    print("N = %d" % N)
    print("q        " + "".join("%5d" % q for q in range(1, qmax + 1)))
    print("dim C~   " + "".join("%5d" % dims[q] for q in range(1, qmax + 1)))
    print("rank d   " + "".join("%5d" % ranks[q] for q in range(1, qmax + 1)))
    print("dim H    " + "".join("%5d" % coh[q] for q in range(1, qmax + 1)), flush=True)


if __name__ == "__main__":
    main()
