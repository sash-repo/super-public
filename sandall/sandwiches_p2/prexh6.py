#!/usr/bin/env python3
"""
prexh6.py --- Problem "PrExh" of the Sandwiches text: the exhaustive list of
the maximal subalgebras in codimension >= 6.

The search is the one of Subsec. "ssHow" 1): a subalgebra of codimension c is
the annihilator of a c-dimensional subspace U of L^* every non-zero element f
of which satisfies rank(B_f|ker f) <= 2(c-1); such U are enumerated by a
backtracking search in which a new basis vector is admitted only if it is the
smallest element of its own coset.  A subalgebra found in codimension c is
maximal if and only if it lies in no subalgebra of codimension < c, and the
list of those is exhaustive, so the test is rigorous.

One algebra per process; the answers are appended to
runs/prexh-<name>.log as they come, so that a killed run leaves usable output.
"""
import os as _os_runs; _os_runs.makedirs('runs', exist_ok=True)  # logs go to ./runs
import os
import sys
import time

from wall import build, subalgebras_of_codim, wgradings_of, key_of
from maxsub import restrict_rank
from css import in_span


def main():
    path = sys.argv[1]
    cmax = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 4_000_000_000
    name = os.path.basename(path)[:-4]
    out = open('runs/prexh-%s.log' % name.replace('/', '_'), 'w')

    def say(*a):
        s = ' '.join(str(x) for x in a)
        out.write(s + '\n')
        out.flush()

    L = build(path)
    t0 = time.time()
    ranks = {f: restrict_rank(L, f) for f in range(1, 1 << L.n)}
    say('%s   dim %d   (the %d ranks in %.0f s)'
        % (L.name, L.n, (1 << L.n) - 1, time.time() - t0))
    smaller = []
    for c in range(1, cmax + 1):
        cand = sum(1 for f in range(1, 1 << L.n) if ranks[f] <= 2 * (c - 1))
        t = time.time()
        try:
            subs = subalgebras_of_codim(L, ranks, c, budget=budget)
        except TimeoutError as e:
            say('  codim %2d: %6d candidate functionals -- SEARCH ABANDONED '
                '(%s) after %.0f s' % (c, cand, e, time.time() - t))
            break
        mx = [M for M in subs
              if not any(all(in_span(v, P) for v in M) for P in smaller)]
        smaller += subs
        say('  codim %2d: %6d candidate functionals, %5d subalgebras, '
            '%4d maximal   [%.0f s]'
            % (c, cand, len(subs), len(mx), time.time() - t))
        for M in mx:
            for w in wgradings_of(L, M):
                dg, dd, dn, sol = w['g0']
                say('      MAXIMAL codim %d  g_- = %s  g_>=0 = %s  growth %s'
                    '  g_0: dim %d, dim g_0\' %d, 2-nilp %d, solvable %s'
                    % (w['codim'], w['neg'], w['pos'], w['growth'], dg, dd,
                       dn, sol))
    say('DONE %.0f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
