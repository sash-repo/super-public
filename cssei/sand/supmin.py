#!/usr/bin/env python3
r"""Smallest codimension of a proper subsuperalgebra of S = s(g,D), exhaustively over F_2:
a subsuperalgebra is a Lie subalgebra of (S, commutator), so the isotropy bound of
mincodim.py applies; the subalgebras found are then filtered: graded, and closed under
the squaring of odd elements.  Also: is n_S(s(S)) among the subsuperalgebras of least
codimension?  usage: supmin.py TABLE WORDS [sample]"""
import sys, os, json, random
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from supsand import superization, SAlg, analyse, intersect, bits
from walk import lowrank, nil2_rows
from mincodim import restricted_rank, annihilated, span_elems
from sand import Span, Alg


def graded_sq(A, M):
    S = Span(M)
    hb = A.homog_basis(S)
    if len(hb) != len(S):
        return False
    return all(A.sq(v) in S for v in hb if v & A.par)


def smallest_super(T, par, sq, cmax=6):
    A = SAlg(T, par, sq)
    L = Alg(T)
    n = A.n
    for c in range(1, cmax + 1):
        sup = lowrank(T, 2 * c, ch=min(8, n - 1))
        cand = [f for f in sup if restricted_rank(L, f) <= 2 * (c - 1)]
        cs = set(cand)
        found = []

        def rec(basis, start):
            if len(basis) == c:
                M = annihilated(basis, n)
                S = Span(M)
                if all(L.br(x, y) in S for x in M for y in M) and graded_sq(A, M):
                    found.append(tuple(sorted(Span(M).basis())))
                return
            for i in range(start, len(cand)):
                f = cand[i]
                if f in Span(basis):
                    continue
                if all((e ^ f) in cs for e in span_elems(basis) if e):
                    rec(basis + [f], i + 1)
        rec([], 0)
        found = sorted(set(found))
        if found:
            out = []
            for M in found:
                hb = A.homog_basis(Span(M))
                e = sum(1 for v in hb if not v & par)
                ev_tot = n - bin(par).count('1')
                out.append((ev_tot - e, bin(par).count('1') - (len(hb) - e)))
            return c, found, out
    return None, [], []


if __name__ == '__main__':
    tab = json.load(open(sys.argv[1]))['tab']
    W = json.load(open(sys.argv[2]))
    DER, words = W['der'], W['words']
    n = len(tab)
    samp = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    random.seed(1)
    if samp:
        words = random.sample(words, min(samp, len(words)))
    from collections import Counter
    cnt = Counter()
    for w in words:
        M = [0] * n
        for k in bits(w):
            M = [M[j] ^ DER[k][j] for j in range(n)]
        D = tuple(M)
        T, par, sq, a, b = superization(tab, D)
        r = analyse(tab, D)
        if 'codim_n' not in r:
            cnt[json.dumps(r, sort_keys=True)] += 1
            continue
        c, found, sd = smallest_super(T, par, sq, cmax=sum(r['codim_n']))
        r['min_codim'] = c
        r['min_sdims'] = sorted(set(sd))
        r['n_is_min'] = None
        if 'codim_n' in r and c is not None:
            r['n_is_min'] = sum(r['codim_n']) == c
        assert c is not None     # n itself is found at c = codim n at the latest
        cnt[json.dumps(r, sort_keys=True)] += 1
    print(os.path.basename(sys.argv[1]), 'gradings', len(words))
    for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
        print('  %5d  %s' % (v, k), flush=True)
