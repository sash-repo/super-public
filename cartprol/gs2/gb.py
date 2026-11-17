#!/usr/bin/env python3
"""Groebner basis mod 2 of a system written by gen_eqs.py (sympy); prints whether it is {1},
i.e., whether the system has no solution over any extension of F_2.
usage: gb.py eqs_1111.m [eqs_2111.m ...]
(gb.wl does the same in Mathematica: wolframscript -file gb.wl eqs_3111.m)
In the text: Statement 4.4."""
import sys, re, time, sympy as sp
for f in sys.argv[1:]:
    txt = open(f).read()
    V = sp.symbols([v.strip() for v in re.search(r'vars = \{(.*?)\};', txt, re.S).group(1).split(',')])
    E = [sp.sympify(e) for e in re.search(r'eqs = \{(.*)\};', txt, re.S).group(1).replace('^', '**').split(',\n')]
    t = time.time()
    G = sp.groebner(E, *V, modulus=2, order='grevlex')
    print('%s  unknowns %d  equations %d  GB = {1}? %s  (%.1f s)' % (f, len(V), len(E), list(G.exprs) == [1], time.time() - t))
