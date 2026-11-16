"""Verification of the m(1) binary dimensions with ALL EIGHT generators of pe(2)
and random rational weights (the classification used only M_q2, M_qtau, M_tauxi)."""
from sympy import symbols, S, Matrix, expand, Rational, factor, simplify
from m1 import PE
from classify_m1 import words, eqs_bin
ALL = tuple(PE)
W = [(S(0),S(0),S(0),S(0)), (Rational(1,2),Rational(-1,3),Rational(5,2),Rational(2,3)),
     (S(3),S(1),S(-2),S(4)), (S(1),S(0),S(1),S(0))]
for s,w in ((-1,1),(-2,0),(-3,-1),(-3,1)):
    terms=[]
    for sp in range(0,-s+1):
        for wp in range(-4,5):
            for n1,o1,p1 in words(-sp,wp):
                for n2,o2,p2 in words(s+sp,w-wp):
                    terms.append((f"{n1 or 'id'}|{n2 or 'id'}",o1,o2,p2,(p1+p2)%2))
    for par in (0,1):
        tt=[(n,o1,o2,pq) for (n,o1,o2,pq,pt) in terms if pt==par]
        if not tt: continue
        dims=[]
        for (A1,B1,A2,B2) in W:
            At,Bt = A1+A2+S(s+w)/2, B1+B2-S(w)/2
            cs,eqs = eqs_bin(tt,A1,B1,A2,B2,At,Bt,par,(-s)+2,ALL)
            M = Matrix([[expand(e).coeff(c) for c in cs] for e in eqs]) if eqs else Matrix(0,len(cs),[])
            dims.append(len(cs) - (M.rank() if M.rows else 0))
        print(f"  s={s} w={w:+d} par={par}: {len(tt)} terms, dims {dims}")
