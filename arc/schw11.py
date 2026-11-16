"""
The Schwarzian of vect^L(1|1) written directly in the coordinates (x|theta)


Table (61) gives the cocycle on vect^L(1|1) only through the embedding
i: vect^L(1|1) -> k^L(1|2), as  c(X_1,X_2) = Res( i(X_1) D_xi D_eta d_t i(X_2) ).
Here we look for an intrinsic expression.  The answer, with 0 mismatches over
all 1296 pairs of basis elements with |deg x| <= 4, is

    c(X_1, X_2) = Res( d_theta(f_1) . S(X_2) ),   S(X) := d_x( d_x(f) - d_theta(g) ),

where X_i = f_i d_x + g_i d_theta and Res is the residue in x of the
theta-free part.  Note that d_x(f) - d_theta(g) is the divergence of X, so
S = d_x o Div, and that c is an ODD bilinear form: c(X_1,X_2) = 0 unless
p(X_1) + p(X_2) = 1.
"""
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from fractions import Fraction as F
from check_bases import make_i, vect_basis, name_vect, D_xi, D_eta
from check_cocycles import L_res
from stringy import mono, add, mul, smul, d_t, d_odd
i_dx,i_dth = make_i(1)
def emb(tok):
    kind,m,j,T = tok
    return i_dx(m,T) if kind=='dx' else i_dth(m,j,T)
def s(f): return D_xi(1)(D_eta(1)(d_t(f)))
def fg(tok):
    kind,m,j,T = tok
    mon = mono(m, tuple(1 for _ in T))
    return (mon,{}) if kind=='dx' else ({},mon)
def RES0(h): return h.get((-1,()),0)
toks = vect_basis(1,4)
for sg in (1,-1):
  for c0 in (1,-1):
    bad=0; ex=None
    for a in toks:
        for b in toks:
            lhs = L_res(1, mul(emb(a), s(emb(b))))
            f1,g1 = fg(a); f2,g2 = fg(b)
            div = add(d_t(f2), smul(sg, d_odd(1,g2)))
            rhs = c0*RES0(mul(d_odd(1,f1), d_t(div)))
            if lhs!=rhs: bad+=1; ex = ex or (name_vect(a),name_vect(b),lhs,rhs)
    print("Div=d_x f %s d_th g, overall %+d : %d mismatches / %d %s"
          % ('+' if sg>0 else '-', c0, bad, len(toks)**2, ex or ""))
