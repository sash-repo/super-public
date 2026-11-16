#!/usr/bin/env python3
r"""Rows obtained by REGRADING a standard algebra built at unconstrained heights:
the non-positive part of the regrading, as an abstract algebra, then realized on
O(m;1).  If g_{<=0} admits no realization with N~ = 1 (no compatible 2-map on
g_-), the largest compatible g_0' is found by best_core and that is realized.

usage: python3 regraded_rows.py vle~9 | vle9 | ...
"""
import sys
from f2vf import Space, parse_field, find_weights
from prolong import prolong, log
from run_row import analyse, save
from regrade import regrade, nonpos_abstract
from blattner import Realization, best_core, sub_abstract
import rows as R


def standard_vle(M):
    """vle(4;M|3) in vect(7): heights M on x1,x2,x3, 1 on y, xi (y forced to 1)"""
    spec = R.ROWS['vle7']
    nm = {c: i for i, c in enumerate(spec['coords'])}
    H = [M if len(c) == 2 else 1 for c in spec['coords']]      # y1,y2,y3 = x's
    sp0 = Space(H, spec['degs'], spec['par'])
    neg = [parse_field(sp0, s, nm) for s in spec['neg']]
    g0 = [parse_field(sp0, s, nm) for s in spec['g0']]
    W = find_weights(sp0, neg + g0)
    sp = Space(H, spec['degs'], spec['par'], W)
    neg = [parse_field(sp, s, nm) for s in spec['neg']]
    g0 = [parse_field(sp, s, nm) for s in spec['g0']]
    G = prolong(sp, neg, g0, verbose=False)
    return sp, G, spec


# vle7 coordinates: y1,y2,y3 = x1,x2,x3; y12 = xi3, y13 = xi2, y23 = xi1; y123 = y
def vle_R(xdeg, xideg, ydeg):
    """coordinate degrees in the order of rows.gen_vle7: y1,y2,y3,y12,y13,y23,y123"""
    return [xdeg[0], xdeg[1], xdeg[2], xideg[2], xideg[1], xideg[0], ydeg]


REGR = {
    # (8.3): deg xi1 = 0, x1 = 2, xi2 = xi3 = x2 = x3 = 1, y = 0
    'vle~9': ('vle', vle_R((2, 1, 1), (0, 1, 1), 0)),
    # R(K) of (25.4): x = 2,2,2, y = 0, xi = 1,1,1
    'vle9': ('vle', vle_R((2, 2, 2), (1, 1, 1), 0)),
}


def run(name, M=2):
    fam, r = REGR[name]
    sp, G, spec = standard_vle(M)
    log('standard %s at heights %s: dim %d, comps %s' % (fam, sp.h, G.dim(),
        {d: G.dim(d) for d in G.degrees()}))
    parts, dims = regrade(sp, G, r)
    log('regraded components (deg: [ev, od]):', dims)
    A = nonpos_abstract(sp, parts)
    res = {'row': name, 'source': 'standard %s at heights %s regraded by %s' % (fam, sp.h, r),
           'regraded dims': dims}
    try:
        Rz = Realization(A)
        D = Rz.realize()
        core = None
    except (ValueError, AssertionError) as e:
        log('no realization of the full g_<=0 on O(m;1): %s' % e)
        best, found, total = best_core(A)
        log('best compatible g_0\': dim %d of %d (%d of %d p-maps attain it)'
            % (best, sum(1 for i in range(A.n) if A.deg[i] == 0), len(found), total))
        res['full g0 realizable'] = False
        res["g0' dim"] = best
        pm, g0p = found[0]
        A = sub_abstract(A, None, g0p)
        Rz = Realization(A)
        D = Rz.realize()
    fl = [D[i] for i in range(A.n)]
    W = find_weights(Rz.sp, fl)
    sp2 = Space([1] * Rz.m, Rz.sp.degs, Rz.sp.par, W)
    neg = [D[i] for i in range(A.n) if A.deg[i] < 0]
    g0 = [D[i] for i in range(A.n) if A.deg[i] == 0]
    G2 = prolong(sp2, neg, g0)
    out = analyse(sp2, G2, name + ' (regraded standard, N~=1)')
    res['A'] = out
    save(name + '_regr', res)


if __name__ == '__main__':
    for n in sys.argv[1:]:
        run(n)
