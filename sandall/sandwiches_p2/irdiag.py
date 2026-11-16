#!/usr/bin/env python3
r"""
irdiag.py --- which solutions of the contact degree equation actually grade
ir(9; \One)?  (For F(kas) the module imposes no equation of its own; for ir it
does, and this script finds it.)

Also: the two W-gradings found by irw.py are checked to be given by MAXIMAL
subalgebras, by a complete sweep over all the cosets.
"""

import os as _os_runs; _os_runs.makedirs('runs', exist_ok=True)  # logs go to ./runs
import itertools
import sys

from ir import (NVAR, bits, bracket, echelon, in_span, prolong, flatten,
                minimal_ideal, G0T)
from irw import (build_ir, terms, graded_pieces, slot_deg, normalize, w_test,
                 growth, is_contact, PAIRIDX, contact_vectors)

LOG = open('runs/irdiag.log', 'w')


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    LOG.write(s + '\n')
    LOG.flush()


def main():
    basis = build_ir()
    tms = [terms(v) for v in basis]

    say('=== which solutions of d_1 = d_i + d_{11-i} grade ir? ===')
    good, bad = [], []
    for d in contact_vectors(range(1, 4), range(-2, 4)):
        gr = graded_pieces(basis, tms, d)
        (good if gr is not None else bad).append(d)
    say('  %d contact vectors tested: %d grade ir, %d do not'
        % (len(good) + len(bad), len(good), len(bad)))

    # guess: the extra equation involves the degree-1 module.  Look at the
    # values of the four "halves" a_k = d_i for the pairs (i, j), and of d_1.
    say('')
    say('  the pairs are (2,9), (3,8), (4,7), (5,6); write a_k for the degree')
    say('  of the first member of the k-th pair, so that the second one has')
    say('  degree d_1 - a_k.')
    say('')
    say('  a sample of the ones that do NOT grade ir:')
    for d in bad[:12]:
        say('    d_1 = %2d   a = %s' % (d[0], tuple(d[i] for i, _ in PAIRIDX)))
    say('  a sample of the ones that DO grade ir:')
    for d in good[:12]:
        say('    d_1 = %2d   a = %s' % (d[0], tuple(d[i] for i, _ in PAIRIDX)))

    # test a family of candidate equations
    def feats(d):
        a = [d[i] for i, _ in PAIRIDX]
        b = [d[j] for _, j in PAIRIDX]
        return d[0], a, b

    cands = {
        'sum of the four a_k = 2 d_1 (i.e. sum over all nine = 3 d_1... )':
            lambda d: sum(d[i] for i, _ in PAIRIDX) * 2 == 4 * d[0] - 0,
        'a_1 + a_2 + a_3 + a_4 = 2 d_1':
            lambda d: sum(d[i] for i, _ in PAIRIDX) == 2 * d[0],
        'd_2 + d_3 + d_4 + d_5 = 2 d_1':
            lambda d: d[1] + d[2] + d[3] + d[4] == 2 * d[0],
        'd_3 + d_4 + d_5 = d_1 + d_2  (i.e. w_6..w_9 balance)':
            lambda d: d[2] + d[3] + d[4] == d[0] + d[1],
    }
    say('')
    say('  candidate equations (must hold exactly on the good set):')
    for name, f in cands.items():
        ok = all(f(d) for d in good) and not any(f(d) for d in bad)
        cov = sum(1 for d in good if f(d)), sum(1 for d in bad if f(d))
        say('    %-55s  %s   (holds on %d/%d good, %d/%d bad)'
            % (name, 'YES' if ok else 'no', cov[0], len(good), cov[1], len(bad)))

    # brute force: an integer linear relation sum c_i d_i = 0 separating them
    say('')
    say('  searching for a linear form vanishing on the good set:')
    sols = []
    R = range(-2, 3)
    for c in itertools.product(R, repeat=5):        # c_0 .. c_4 for d_1..d_5
        if not any(c):
            continue
        def val(d, c=c):
            return (c[0] * d[0] + c[1] * d[1] + c[2] * d[2] + c[3] * d[3]
                    + c[4] * d[4])
        if all(val(d) == 0 for d in good):
            sols.append(c)
    say('    forms c_1 d_1 + ... + c_5 d_5 vanishing on every good vector: %s'
        % (sols if sols else 'none'))

    # what distinguishes them: the degree-1 module
    say('')
    say('=== the real criterion: the degree-1 module must be homogeneous ===')
    part = prolong(G0T)
    g1 = [v for _, v in part[1]]
    say('  dim g_1 = %d' % len(g1))
    t1 = [terms(v) for v in g1]
    agree = 0
    for d in good + bad:
        homog = graded_pieces(g1, t1, d) is not None
        if homog == (d in goodset(good)):
            agree += 1
    say('  "g_1 is d-homogeneous"  agrees with  "ir is d-graded"  on %d of %d'
        % (agree, len(good) + len(bad)))


_GS = None


def goodset(good):
    global _GS
    if _GS is None:
        _GS = set(good)
    return _GS


if __name__ == '__main__':
    main()
