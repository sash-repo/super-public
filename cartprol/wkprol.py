#!/usr/bin/env python3
"""p = 2: Cartan prolongs of the least modules of wk(n;al), bgl(n;al) over F_2(al).
In the text: Statement 3.1, the rows wk(3;alpha), bgl(3;alpha), wk(4;alpha), bgl(4;alpha) of Table (1)."""
import sys
from ratf import F2alpha, AL, ONE, ZERO
from prol import Rep, Image, prolong
K = F2alpha()
a = AL
W3 = [[0, a, 0], [a, 0, 1], [0, 1, 0]]
W4 = [[0, a, 0, 0], [a, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
def run(name, A, par, lam, kmax=3):
    for ident in (False, True):
        r = Rep(A, par, 2, lam, field=K)
        img = Image(r, extra_identity=ident, extra_grading=ident)
        print("%s lam=%s M=%s g0=%s%s" % (name, lam, r.sdim(), img.dims(), " +Id,+torus" if ident else ""), flush=True)
        prolong(r, img, kmax=kmax)
which = sys.argv[1]
if which == "3":
    for lam in ([0,0,1], [0,0,1+a], [a,0,0], [1+a,0,0], [0,a,1+a], [1+a,1,0]):
        run("wk(3)", W3, [0,0,0], lam); run("bgl(3)", W3, [1,0,0], lam)
if which == "4":
    for lam in ([0,0,0,1], [0,0,0,a]):
        run("wk(4)", W4, [0,0,0,0], lam); run("bgl(4)", W4, [1,0,0,0], lam)
