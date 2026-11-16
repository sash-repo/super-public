#!/usr/bin/env python3
r"""osprep.py --- Q2 of "what ARE the answers = sdims of irreps, both typical ... and atypical?"
for the two algebras of Subsection 4.3, osp(3|2) and osp_a(4|2)=D(2,1;a).

Everything is exact (Fraction arithmetic).  The irreducible modules are built
degree by degree from the Cartan matrix by hwmod.py (Verma
modulo the radical of the contravariant form); the root-system data (the form,
rho, typicality) are built here from scratch and cross-checked against them.
"""
import sys
from fractions import Fraction as Fr
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from hwmod import irrep

MAXDIM = 6000

def sdim_of(A, par, lam, maxdim=MAXDIM):
    try:
        M = irrep(A, par, 0, list(lam), maxdim=maxdim)
        return M.sdim()
    except Exception:
        return None

# ====================================================================
# 1.  osp(3|2) = B(1,1)
# ====================================================================
# weight space  a*eps + b*delta,  (eps,eps)=1, (delta,delta)=-1, (eps,delta)=0
def form3(u, v):
    return u[0]*v[0] - u[1]*v[1]

EPS, DEL = (Fr(1), Fr(0)), (Fr(0), Fr(1))
def add(*ws):
    return (sum(w[0] for w in ws), sum(w[1] for w in ws))
def smul(c, w):
    return (c*w[0], c*w[1])

# all-odd simple system:  al1 = eps - delta (isotropic), al2 = delta (non-isotropic)
AL1 = (Fr(1), Fr(-1))
AL2 = (Fr(0), Fr(1))
POS_EV = [(Fr(1), Fr(0)), (Fr(0), Fr(2))]                       # eps, 2 delta
POS_OD = [AL1, AL2, (Fr(1), Fr(1))]                             # eps-delta, delta, eps+delta
POS_OD_ISO = [r for r in POS_OD if form3(r, r) == 0]
RHO3 = smul(Fr(1, 2), add(*POS_EV, *[smul(-1, r) for r in POS_OD]))

A3 = [[0, 1], [-1, 1]]
PAR3 = [1, 1]
D3 = [Fr(1), Fr(-1)]        # symmetrizers: (al_i,al_j) = d_i * A[i][j]

def check_cartan3():
    als = [AL1, AL2]
    ok = True
    for i in range(2):
        for j in range(2):
            if form3(als[i], als[j]) != D3[i]*A3[i][j]:
                ok = False
    return ok

def lam3(a, b):
    """highest weight a*eps+b*delta  ->  (lambda(h_1), lambda(h_2))"""
    w = (a, b)
    return (form3(w, AL1)/D3[0], form3(w, AL2)/D3[1])

def typical3(a, b):
    lr = add((a, b), RHO3)
    return all(form3(lr, r) != 0 for r in POS_OD_ISO)

def kacdim3(a, b):
    lr = add((a, b), RHO3)
    p = Fr(2)**len(POS_OD_ISO)
    for al in POS_EV:
        p *= form3(lr, al)/form3(RHO3, al)
    return p

print("="*72)
print("1.  osp(3|2) = B(1,1),  all-odd Cartan matrix A =", A3, " par =", PAR3)
print("    Delta_0^+ =", POS_EV, "  Delta_1^+ =", POS_OD)
print("    isotropic odd positive roots:", POS_OD_ISO)
print("    rho =", RHO3, "   Cartan matrix consistent with the form:", check_cartan3())
print()
print("    m := lambda(h_eps) = 2a  (o(3)-label),   n := lambda(h_2delta) = b  (sp(2)-label)")
print("    %-9s %-13s %-10s %-9s %-11s %s" % ("(m,n)", "(lam1,lam2)", "sdim", "dim", "typical?", "Kac dim"))
rows3 = []
for m in range(0, 11):
    for n in range(0, 6):
        a, b = Fr(m, 2), Fr(n)
        l = lam3(a, b)
        sd = sdim_of(A3, PAR3, l)
        typ = typical3(a, b)
        kd = kacdim3(a, b)
        rows3.append((m, n, l, sd, typ, kd))
        if sd is None:
            continue
        print("    %-9s %-13s %-10s %-9s %-11s %s" %
              ("(%d,%d)" % (m, n), "(%s,%s)" % l, "(%d|%d)" % sd, sum(sd), typ, kd))

# ====================================================================
# 2.  osp_a(4|2) = D(2,1;a)
# ====================================================================
# weight space c1 eps1 + c2 eps2 + c3 eps3, form diag(s1,s2,s3),
# s1 = -(1+a), s2 = 1, s3 = a   (so s1+s2+s3 = 0: the odd roots are isotropic)
def mk4(a):
    s = [-(1+a), Fr(1), a]
    def frm(u, v):
        return sum(u[i]*v[i]*s[i] for i in range(3))
    POSEV = [(2,0,0), (0,2,0), (0,0,2)]
    POSOD = [(1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1)]
    RHO = (-1, 1, 1)                      # distinguished Borel
    # distinguished simple system: g1 = eps1-eps2-eps3 (odd iso), g2 = 2eps2, g3 = 2eps3
    A = [[0, 1, a], [-1, 2, 0], [-1, 0, 2]]
    par = [1, 0, 0]
    return s, frm, POSEV, POSOD, RHO, A, par

def lam_dist(a, m):
    """(m1,m2,m3) = the three sl(2)-labels  ->  (lambda(h_1),lambda(h_2),lambda(h_3))"""
    m1, m2, m3 = m
    return [Fr(m1*(1+a) + m2 + m3*a, 1)/2, Fr(m2), Fr(m3)]

print()
print("="*72)
print("2.  osp_a(4|2) = D(2,1;a),  distinguished A = [[0,1,a],[-1,2,0],[-1,0,2]], par=[1,0,0]")
print("    rho = -eps1+eps2+eps3;  p:=m1-1, q:=m2+1, r:=m3+1")
print()
for a in [Fr(1), Fr(2), Fr(3), Fr(1,2), Fr(5,3), Fr(-3)]:
    s, frm, POSEV, POSOD, RHO, A, par = mk4(a)
    print("  --- a = %s ---" % a)
    print("    %-11s %-16s %-11s %-8s %-10s %-9s" %
          ("(m1,m2,m3)", "lambda(h_i)", "sdim", "dim", "typical?", "16pqr"))
    for m1 in range(0, 5):
        for m2 in range(0, 3):
            for m3 in range(0, 3):
                m = (m1, m2, m3)
                lam = lam_dist(a, m)
                sd = sdim_of(A, par, lam, maxdim=2000)
                if sd is None:
                    continue
                lr = tuple(m[i] + RHO[i] for i in range(3))
                typ = all(frm(lr, al) != 0 for al in POSOD)
                p, q, r = m1-1, m2+1, m3+1
                print("    %-11s %-16s %-11s %-8s %-10s %-9s" %
                      (str(m), str(tuple(str(x) for x in lam)), "(%d|%d)" % sd,
                       sum(sd), typ, 16*p*q*r))

# ====================================================================
# 3.  Verification of the closed formulas
# ====================================================================
print()
print("="*72)
print("3.  Verification")
fails = []

# --- osp(3|2): typical dim = 4(m-1)(2n+1), sdim = 0; atypical iff m = 2n+2 ---
print("  osp(3|2): typical dim = 4(m-1)(2n+1) and sdim = 0;  atypical <=> m = 2n+2 (or m=n=0)")
nchk = 0
for m in range(0, 13):
    for n in range(0, 7):
        a, b = Fr(m, 2), Fr(n)
        sd = sdim_of(A3, PAR3, lam3(a, b), maxdim=8000)
        fin = sd is not None
        if not fin:
            if m >= 2:
                fails.append("osp(3|2) (m,n)=(%d,%d) expected finite-dim" % (m, n))
            continue
        if m < 2 and (m, n) != (0, 0):
            fails.append("osp(3|2) (m,n)=(%d,%d) unexpectedly finite-dim" % (m, n))
        typ = typical3(a, b)
        if typ != (m != 2*n+2 and (m, n) != (0, 0)):
            fails.append("osp(3|2) (m,n)=(%d,%d) typicality mismatch" % (m, n))
        if typ:
            d = 4*(m-1)*(2*n+1)
            if sum(sd) != d or sd[0] != sd[1]:
                fails.append("osp(3|2) (m,n)=(%d,%d): got (%d|%d), want (%d|%d)"
                             % (m, n, sd[0], sd[1], d//2, d//2))
        nchk += 1
print("     %d finite-dimensional modules checked" % nchk)

print("  osp(3|2) atypical (m = 2n+2):")
for n in range(0, 6):
    m = 2*n+2
    sd = sdim_of(A3, PAR3, lam3(Fr(m, 2), Fr(n)), maxdim=8000)
    pred = (4*(n+1)**2, 4*(n+1)**2 - 2)
    print("     n=%d  m=%-3d sdim = (%d|%d)   dim = %-4d sdim(L)=%-3d  (4(n+1)^2|...-2) = (%d|%d) %s"
          % (n, m, sd[0], sd[1], sum(sd), sd[0]-sd[1], pred[0], pred[1],
             "OK" if sd == pred else ("<-- exception" if n == 0 else "MISMATCH")))
    if sd != pred and n >= 1:
        fails.append("osp(3|2) atypical n=%d" % n)

# --- D(2,1;a): typical dim = 16(m1-1)(m2+1)(m3+1), sdim = 0 ---
print("  osp_a(4|2): typical dim = 16(m1-1)(m2+1)(m3+1) and sdim = 0")
nchk = 0
for a in [Fr(1), Fr(2), Fr(3), Fr(1,2), Fr(5,3), Fr(-3), Fr(7,2)]:
    s, frm, POSEV, POSOD, RHO, A, par = mk4(a)
    for m1 in range(0, 5):
        for m2 in range(0, 4):
            for m3 in range(0, 4):
                m = (m1, m2, m3)
                sd = sdim_of(A, par, lam_dist(a, m), maxdim=2500)
                if sd is None:
                    continue
                lr = tuple(m[i] + RHO[i] for i in range(3))
                typ = all(frm(lr, al) != 0 for al in POSOD)
                if typ:
                    d = 16*(m1-1)*(m2+1)*(m3+1)
                    if sum(sd) != d or sd[0] != sd[1]:
                        fails.append("D(2,1;%s) m=%s: got (%d|%d), want dim %d"
                                     % (a, m, sd[0], sd[1], d))
                    nchk += 1
                else:
                    if sd[0] == sd[1]:
                        fails.append("D(2,1;%s) m=%s atypical but sdim = 0" % (a, m))
print("     %d typical modules checked over 7 values of a" % nchk)

# --- sdim L != 0  <=>  L atypical  (Kac-Wakimoto; defect 1 for both algebras) ---
print("  |sdim L| <= 2 throughout, and sdim L = 0 exactly for the typical L: see above")

print()
print("  FAILURES:", len(fails))
for f in fails:
    print("   ", f)
