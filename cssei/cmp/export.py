#!/usr/bin/env python3
"""Export structure tables (tab[i][j] = bitmask of [e_i,e_j]) over F_2 of the
BGLLS algebras of dim <= 31 (sb~(7) core, Shen's gs(2)) to cmp/*.json."""
import sys, json, os, io, contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from der import core
from absutil import Coords

def export_row(name, out):
    sp, G, J = core(name)
    B = J.basis(); n = len(B); co = Coords(B)
    tab = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            c = co(sp.bracket(B[i], B[j]))
            tab[i][j] = c if isinstance(c, int) else sum(1 << k for k in c)
    json.dump({'name': name, 'dim': n, 'tab': tab}, open(out, 'w'))
    print(name, n)

def export_gs2(out):
    g = {'__name__': 'shen_exec'}
    sys.argv = ['shen.py']
    with contextlib.redirect_stdout(io.StringIO()):
        exec(open('shen.py').read(), g)
    k = g['k']; qbr = g['qbr']
    tab = [[qbr(i, j) for j in range(k)] for i in range(k)]
    json.dump({'name': 'gs(2)', 'dim': k, 'tab': tab}, open(out, 'w'))
    print('gs(2)', k)

if __name__ == '__main__':
    export_row('sb~7', 'cmp/sbt7.json')
    export_gs2('cmp/gs2.json')
