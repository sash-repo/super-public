#!/usr/bin/env python3
r"""
irsand.py --- the complete sandwich scan of ir(9; \One), all 2^26 - 1 elements.

ir(9; \One) is restricted and centreless (checked in ir.py), so for X in ir

        (ad_X)^2 = ad_{X o X},   hence   (san) <=> X o X = 0,

where X o X is the composition square of the vector field X.  In coordinates
with respect to a basis (e_1, ..., e_26) of ir, writing S for the support of
X = sum_{i in S} e_i, the map is quadratic with polarization the bracket:

        sq(X) = sum_{i in S} sq(e_i) + sum_{i < k in S} [e_i, e_k],

so a Gray-code walk over the 2^26 subsets updates sq(X) with |S| exclusive-ors
of 26-bit integers per step.  Every X with sq(X) = 0 is then tested for the
second condition (san2), [[ir, X], [ir, X]] = 0.

Output goes to runs/irsand.log (never to /tmp), with a progress line every
2^22 steps so that a killed run still leaves a usable partial result.

Usage:  python3 irsand.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ir import (GM1, G0T, bracket, compose_square, echelon, flatten,     # noqa: E402
                in_span, minimal_ideal, prolong, reduce_vec, FDEG, bits)

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'irsand.log')


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    log = open(LOG, 'w', buffering=1)

    def say(msg):
        print(msg)
        log.write(msg + '\n')

    part = prolong(G0T)
    pelems = flatten(part)
    idl, uniq, _ = minimal_ideal(pelems)
    ir = [v for _, v in idl]
    n = len(ir)
    say('ir(9; \\One): dim %d, unique minimal ideal: %s' % (n, uniq))

    # coordinates
    rec = []
    for j, u in enumerate(ir):
        v, cm = u, 1 << j
        for p, b, c in rec:
            if v >> p & 1:
                v ^= b
                cm ^= c
        if v:
            rec.append((v.bit_length() - 1, v, cm))
            rec.sort(key=lambda t: -t[0])

    def coords(v):
        cm = 0
        for p, b, c in rec:
            if v >> p & 1:
                v ^= b
                cm ^= c
        return None if v else cm

    sq = [coords(compose_square(e)) for e in ir]
    br = [[coords(bracket(ir[i], ir[j])) for j in range(n)] for i in range(n)]
    assert all(x is not None for x in sq), 'ir is not restricted?!'
    assert all(x is not None for row in br for x in row), 'not closed?!'

    t0 = time.time()
    total = 1 << n
    cur = 0
    subset = []
    prev = 0
    nilp = []
    say('scanning %d nonzero elements ...' % (total - 1))
    for g in range(1, total):
        gray = g ^ (g >> 1)
        j = (gray ^ prev).bit_length() - 1
        upd = sq[j]
        brj = br[j]
        for i in subset:
            upd ^= brj[i]
        cur ^= upd
        if j in subset:
            subset.remove(j)
        else:
            subset.append(j)
        prev = gray
        if cur == 0:
            nilp.append(gray)
        if g % (1 << 22) == 0:
            say('  %3d%%  %.0f s  %d elements with X o X = 0 so far'
                % (100 * g // total, time.time() - t0, len(nilp)))

    say('elements with (ad_X)^2 = 0: %d  (%.0f s)' % (len(nilp), time.time() - t0))

    # now (san2): [[L, X], [L, X]] = 0, i.e. the image of ad_X is abelian.
    # All of it in coordinates, with early exit on the first nonzero bracket.
    def mul(u, v):
        out = 0
        for i in bits(u):
            bri = br[i]
            for j in bits(v):
                out ^= bri[j]
        return out

    t1 = time.time()
    sand = []
    for mask in nilp:
        cols = []
        for j in range(n):
            c = 0
            for i in bits(mask):
                c ^= br[i][j]
            if c:
                cols.append(c)
        # row echelon of the image
        img = []
        for c in cols:
            for p, b in img:
                if c >> p & 1:
                    c ^= b
            if c:
                img.append((c.bit_length() - 1, c))
                img.sort(key=lambda pb: -pb[0])
        ok = True
        for a in range(len(img)):
            for b2 in range(a + 1, len(img)):
                if mul(img[a][1], img[b2][1]):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            s = 0
            for k in bits(mask):
                s ^= ir[k]
            sand.append(s)
    say('(san2) tested on all %d candidates in %.0f s' % (len(nilp), time.time() - t1))
    say('sandwich elements (san) and (san2): %d, spanning dim %d'
        % (len(sand), len(echelon(sand))))
    if sand:
        salg = echelon(sand)
        changed = True
        while changed:
            changed = False
            for _, a in list(salg):
                for _, b in list(salg):
                    x = bracket(a, b)
                    if x and not in_span(x, salg):
                        salg = echelon([u for _, u in salg] + [x])
                        changed = True
        say('the sandwich subalgebra s(ir) has dim %d' % len(salg))
    else:
        say('s(ir) = 0: ir(9; \\One) has NO sandwich element over F_2 at all.')
    log.close()


if __name__ == '__main__':
    main()
