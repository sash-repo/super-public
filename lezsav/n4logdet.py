#!/usr/bin/env python3
"""n4logdet.py -- the on-shell identity behind the logarithmic completion.

Checks, at the vacuum and across the family, that

        qbar = -(1/(4 i m)) ( cD_+^a c_b ),      c_b = the E2-current,

whence  T = -u = (1/2) log det q  is half the log-determinant of the matrix
of the plus-half odd derivatives of that current -- which is why the
completion of T is logarithmic and why logarithms of the SECOND-order
currents are what closes it.
"""
import pickle
import sys
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from fractions import Fraction as Q_
from n4noeth import (C, mono, mul, smul, sub, eq, minv, qbar, cvec, cDp,
                     mdet, mtr, is_solution, msmul, msub, meq)
from n4rec import qadd, qsmul

OUT = []
def log(*a):
    s = " ".join(str(x) for x in a); OUT.append(s); print(s, flush=True)
    open("n4logdet-out.txt", "w").write("\n".join(OUT) + "\n")

q = pickle.load(open("n4rec11-vacuum.pkl", "rb"))
tans = pickle.load(open("n4cons-tan.pkl", "rb"))
fam = [("vacuum", q)]
for d in tans:
    for lam in (1, -1, Q_(1, 2), 2, C(0, 1)):
        qq = qadd(q, qsmul(lam, d))
        if is_solution(qq):
            fam.append(("q+(%s)d" % lam, qq))
log("family: %d exact solutions" % len(fam))
ok = 0
for nm, qq in fam:
    qi = minv(qq)
    c = cvec(qq, qi)
    M = [[cDp(a, c[b]) for b in (0, 1)] for a in (1, 2)]
    pref = mul(mono(nm=-1), {(0, 0, 0, ()): C(0, Q_(1, 4))})   # -1/(4im) = i/(4m)
    claim = [[mul(pref, M[i][j]) for j in (0, 1)] for i in (0, 1)]
    same = meq(claim, qbar(qq))
    ok += same
    if not same and nm == "vacuum":
        D = msub(claim, qbar(qq))
        log("   at the vacuum the two differ; sample terms:")
        for i in (0, 1):
            for j in (0, 1):
                if D[i][j]:
                    log("      [%d][%d]: %d terms" % (i, j, len(D[i][j])))
log("qbar = (i/4m)(cD_+^a c_b) holds on %d of %d solutions" % (ok, len(fam)))
# and det of that matrix against det q
qi = minv(q)
c = cvec(q, qi)
M = [[cDp(a, c[b]) for b in (0, 1)] for a in (1, 2)]
log("det(cD_+^a c_b): %d terms; det q: %d terms; det qbar: %d terms"
    % (len(mdet(M)), len(mdet(q)), len(mdet(qbar(q)))))
log("det q == det qbar: %s" % eq(mdet(q), mdet(qbar(q))))

log("")
log("variants:")
import itertools
for tr in (False, True):
    for cf, cname in (({(0,0,0,()): C(0, Q_(1,4))}, "i/4"),
                      ({(0,0,0,()): C(0, Q_(-1,4))}, "-i/4"),
                      ({(0,0,0,()): C(Q_(1,4), 0)}, "1/4"),
                      ({(0,0,0,()): C(Q_(-1,4), 0)}, "-1/4")):
        pref = mul(mono(nm=-1), cf)
        good = 0
        for nm, qq in fam:
            qi = minv(qq); c = cvec(qq, qi)
            M = [[cDp(a, c[b]) for b in (0,1)] for a in (1,2)]
            if tr:
                M = [[M[j][i] for j in (0,1)] for i in (0,1)]
            claim = [[mul(pref, M[i][j]) for j in (0,1)] for i in (0,1)]
            good += meq(claim, qbar(qq))
        log("   transpose=%s coeff=%s/m : %d of %d" % (tr, cname, good, len(fam)))
