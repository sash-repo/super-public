#!/usr/bin/env python3
"""Structure tables (over F_2) of the simple cores J of the Table 25.2 families of
dim <= 31, for cmp/count.py.  usage: export_series.py"""
import sys, os, json, io, contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import series as S
from f2vf import Space, parse_field, find_weights
from prolong import prolong, generators, ad_closure
from absutil import Coords


def core(spec):
    X = spec['coords']; nm = S.names(X); m = len(X)
    complete = spec['g0'] == 'complete'
    sp0 = Space([1] * m, spec['degs'])
    neg = [parse_field(sp0, s, nm) for s in spec['neg']]
    g0 = [] if complete else [parse_field(sp0, s, nm) for s in spec['g0']]
    W = find_weights(sp0, neg + g0)
    sp = Space([1] * m, spec['degs'], None, W)
    neg = [parse_field(sp, s, nm) for s in spec['neg']]
    g0 = None if complete else [parse_field(sp, s, nm) for s in spec['g0']]
    G = prolong(sp, neg, g0, verbose=False)
    J = ad_closure(sp, generators(G), G.basis(-1))
    return sp, J


def export(label, spec, out):
    sp, J = core(spec)
    B = J.basis(); n = len(B); co = Coords(B)
    tab = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            c = co(sp.bracket(B[i], B[j]))
            tab[i][j] = c if isinstance(c, int) else sum(1 << k for k in c)
    json.dump({'name': label, 'dim': n, 'tab': tab}, open(out, 'w'))
    print(label, n)


with contextlib.redirect_stdout(io.StringIO()):
    pass
export('vect(2;1)', S.spec_vect(2), 'cmp/vect2.json')
export('vect(3;1)', S.spec_vect(3), 'cmp/vect3.json')
export('svect(3;1)', S.spec_vect(3, special=True), 'cmp/svect3.json')
export('hP(4;1)', S.spec_ham(2, 0), 'cmp/hP4.json')
export('hI(4;1)', S.spec_ham(1, 2), 'cmp/hI4.json')
export('hI(5;1)', S.spec_ham(2, 1), 'cmp/hI5.json')
export('shP(6;1)', S.spec_ham(3, 0, sl_half=True), 'cmp/shP6.json')
export('k(5;1)', S.spec_k(2), 'cmp/k5.json')
