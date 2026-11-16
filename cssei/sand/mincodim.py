#!/usr/bin/env python3
r"""Smallest codimension of a proper subalgebra over F_2, exhaustively, by the isotropy
bound: a subalgebra M of codim c lies in Ker f and is totally
isotropic for B_f(x,y) = f([x,y]) there, so rank(B_f|Ker f) <= 2(c-1) for every
0 != f in M^perp.  rank(B_f|Ker f) >= rank(B_f) - 2, so the numba walk `lowrank` with
rmax = 2c lists a superset of the candidates; the restricted rank is then exact here."""
import sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from walk import lowrank
from sand import Alg, Span, kernel


def restricted_rank(A, f):
    n = A.n
    a = (f & -f).bit_length() - 1
    kb = [(1 << i) ^ (((f >> i) & 1) << a) for i in range(n) if i != a]
    rows = []
    for x in kb:
        v = 0
        for k, y in enumerate(kb):
            if bin(f & A.br(x, y)).count('1') & 1:
                v |= 1 << k
        rows.append(v)
    return len(Span(rows))


def annihilated(fs, n):
    # the subspace {x | f(x) = 0 for f in fs}
    imgs = []
    for j in range(n):
        v = 0
        for k, f in enumerate(fs):
            if (f >> j) & 1:
                v |= 1 << k
        imgs.append(v)
    return kernel(imgs, n)


def span_elems(basis):
    out = [0]
    for b in basis:
        out += [e ^ b for e in out]
    return out


def smallest(tab, cmax=3, verbose=True):
    A = Alg(tab)
    n = A.n
    for c in range(1, cmax + 1):
        t0 = time.time()
        sup = lowrank(tab, 2 * c)
        cand = [f for f in sup if restricted_rank(A, f) <= 2 * (c - 1)]
        cs = set(cand)
        found = []

        def rec(basis, start):
            if len(basis) == c:
                M = annihilated(basis, n)
                S = Span(M)
                if all(A.br(x, y) in S for x in M for y in M):
                    found.append(tuple(sorted(span_elems(basis)[1:])))
                return
            for i in range(start, len(cand)):
                f = cand[i]
                if f in Span(basis):
                    continue
                if all((e ^ f) in cs for e in span_elems(basis) if e):
                    rec(basis + [f], i + 1)
        rec([], 0)
        found = sorted(set(found))
        if verbose:
            print('  c=%d: %d functionals with rank(B_f)<=%d, %d candidates, %d subalgebras (%.0fs)'
                  % (c, len(sup), 2 * c, len(cand), len(found), time.time() - t0), flush=True)
        if found:
            return c, found
    return None, []


if __name__ == '__main__':
    cmax = int(os.environ.get('CMAX', '3'))
    for p in sys.argv[1:]:
        d = json.load(open(p))
        print(d.get('name', p), len(d['tab']), flush=True)
        c, found = smallest(d['tab'], cmax)
        print('  => smallest codim', c, 'number', len(found), flush=True)
