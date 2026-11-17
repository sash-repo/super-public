"""New identities of Volichenko algebras, degree by degree.

    I_n = all multilinear identities of degree n
    C_n = the ones that follow from the identities of degree < n
    I_n / C_n = the NEW identities of degree n

Baranov's theorem says a new identity appears in degree 5.  This locates it.
"""

import sys
import time
from fractions import Fraction as Fr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exactlin import rank as ex_rank, rref as ex_rref  # noqa: E402

import volid
import videal
from videal import MSpace, identities, consequences


def echelon_add(basis, chunk, ncols, p=None):
    """Fold `chunk` (list of rows) into the echelon basis."""
    rows = basis + [r for r in chunk if any(r)]
    if not rows:
        return []
    R, _ = ex_rref(rows, p=p) if p else ex_rref(rows)
    return [r for r in R if any(r)]


def run(nmax=5, chunk_size=400, verbose=True, p=None, keep=None):
    lower = {}
    out = {}
    for n in range(3, nmax + 1):
        t0 = time.time()
        ms = MSpace(n)
        I = identities(n, ms)
        dI = len(I)
        ncols = len(ms.trees)
        basis, chunk, count, seen = [], [], 0, set()
        gens = [] if keep == n else None
        for f in consequences(lower, n):
            v = ms.vec(f)
            if not any(v):
                continue
            key = tuple(v)
            if key in seen:
                continue
            seen.add(key)
            if gens is not None:
                gens.append(v)
            chunk.append([int(c) % p for c in v] if p else v)
            count += 1
            if len(chunk) >= chunk_size:
                basis = echelon_add(basis, chunk, ncols, p)
                chunk = []
                if verbose:
                    print("    ... %6d consequences folded, rank %d/%d"
                          % (count, len(basis), dI), flush=True)
        basis = echelon_add(basis, chunk, ncols, p)
        dC = len(basis)
        if verbose:
            print("degree %d:  dim I_n = %-5d  consequences of lower degrees:"
                  " %-5d  NEW: %d      [%d generators, %.1fs]"
                  % (n, dI, dC, dI - dC, count, time.time() - t0), flush=True)
        out[n] = dict(I=I, dI=dI, dC=dC, basis=basis, ms=ms, gens=gens)
        lower[n] = I
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    p = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(n, chunk_size=2000, p=p)
