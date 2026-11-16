#!/usr/bin/env python3
"""Check the closed formulas of Statement 2.10 of the paper for the
osp(2|2)-invariant binary operators on k(1|2), orders 2q+1 and 2q, against
the engine invar.py, with all eight generators of i(osp(2|2)) and three
random weight quadruples.

Usage:  python check_k12_explicit.py [QMAX] [odd|even|all]     # default 3 all
"""
import sys, os
from sympy import S, Rational, binomial, prod, expand, symbols
from invar import eqs_binary, matrix_of
from k12 import D_xi, D_eta, d_t

def Tn(j):
    def f(u):
        for _ in range(j): u = d_t(u)
        return u
    return f
def P(u): return d_t(u) and None
def Pop(u):
    v = D_xi(D_eta(u)); w = d_t(u)
    from super12 import add, smul
    return add(v, smul(-1, w))
def comp(*ops):
    def f(u):
        for o in reversed(ops): u = o(u)
        return u
    return f
E = lambda j: Tn(j)
F = lambda j: comp(Pop, Tn(j-1))
X = lambda j: comp(D_xi, Tn(j))
Y = lambda j: comp(D_eta, Tn(j))

WEIGHTS = [(Rational(1,2), Rational(-1,3), Rational(5,2), Rational(2,3)),
           (Rational(-7,3), Rational(5,7), Rational(11,5), Rational(-1,2)),
           (Rational(3,4), Rational(2,5), Rational(-5,7), Rational(1,9))]

def run(name, s, w, terms, coeffs_fn, nmax):
    ok = 0
    for (l1,n1,l2,n2) in WEIGHTS:
        coeffs = coeffs_fn(l1,n1,l2,n2)
        cs, eqs = eqs_binary(terms, l1,n1,l2,n2, l1+l2-s, n1+n2-w, w%2, nmax=nmax)
        sub = dict(zip(cs, coeffs))
        bad = [e for e in eqs if expand(e.subs(sub)) != 0]
        allzero = all(c == 0 for c in coeffs)
        if not bad and not allzero: ok += 1
        else: print("   FAIL at", (l1,n1,l2,n2), "nbad", len(bad), "allzero", allzero)
    print(f"{'OK ' if ok==len(WEIGHTS) else 'BAD'} {name}: {ok}/{len(WEIGHTS)} weights"); sys.stdout.flush()

def odd(q, chir):
    Z = X if chir > 0 else Y
    def co(l1,n1,l2,n2):
        L1 = l1 + chir*n1; L2 = l2 + chir*n2
        p2 = lambda j: prod([l2+q-r for r in range(j)])
        p1 = lambda j: prod([l1+r for r in range(j+1, q+1)])
        out = []
        for j in range(q+1): out.append((-1)**j*binomial(q,j)*(L2+q-j)*p2(j)*p1(j))
        for j in range(q): out.append(chir*-(-1)**j*binomial(q,j)*(q-j)*p2(j)*p1(j))
        for j in range(q+1): out.append(-(-1)**j*binomial(q,j)*(L1+j)*p2(j)*p1(j))
        for j in range(1,q+1): out.append(chir*(-1)**j*binomial(q,j)*j*p2(j)*p1(j))
        return out
    terms = ([('a',Z(j),E(q-j),None,0) for j in range(q+1)] +
             [('b',Z(j),F(q-j),None,0) for j in range(q)] +
             [('c',E(j),Z(q-j),None,1) for j in range(q+1)] +
             [('d',F(j),Z(q-j),None,1) for j in range(1,q+1)])
    return terms, co

def even(q, U0, V0):
    def co(l1,n1,l2,n2):
        Lp1, Lm1, Lp2, Lm2 = l1+n1, l1-n1, l2+n2, l2-n2
        rat = lambda j: (-1)**j*binomial(q-1,j)*prod([l2+q-r for r in range(1,j+1)])/prod([l1+r for r in range(1,j+1)])
        U = [rat(j)*U0 for j in range(q)]; V = [rat(j)*V0 for j in range(q)]
        A = {}; B = {}; C = {}; D = {}
        B[0] = (Lm1*V[0]-Lp1*U[0])/(2*(l2+q))
        A[0] = -(Lp1*(Lm2+q)*U[0]+Lm1*(Lp2+q)*V[0])/(2*q*(l2+q))
        for j in range(1,q):
            B[j] = ((Lm1+j)*V[j]-(Lp1+j)*U[j])/(2*(l2+q-j))
            A[j] = -((Lp1+j)*(Lm2+q-j)*U[j]+(Lm1+j)*(Lp2+q-j)*V[j])/(2*(q-j)*(l2+q-j))
            D[j] = j*(U[j]+V[j])/(2*(l2+q-j))
            C[j] = j*((Lm2+q-j)*U[j]-(Lp2+q-j)*V[j])/(2*(q-j)*(l2+q-j))
        C[q] = (Lp2*V[q-1]-Lm2*U[q-1])/(2*(l1+q))
        A[q] = ((Lp1+q)*Lm2*U[q-1]+(Lm1+q)*Lp2*V[q-1])/(2*q*(l1+q))
        return ([A[j] for j in range(q+1)] + [B[j] for j in range(q)] +
                [C[j] for j in range(1,q+1)] + [D[j] for j in range(1,q)] + U + V)
    terms = ([('A',E(j),E(q-j),None,0) for j in range(q+1)] +
             [('B',E(j),F(q-j),None,0) for j in range(q)] +
             [('C',F(j),E(q-j),None,0) for j in range(1,q+1)] +
             [('D',F(j),F(q-j),None,0) for j in range(1,q)] +
             [('U',X(j),Y(q-j-1),None,1) for j in range(q)] +
             [('V',Y(j),X(q-j-1),None,1) for j in range(q)])
    return terms, co

QMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 3
ONLY = sys.argv[2] if len(sys.argv) > 2 else 'all'
for q in range(0, QMAX+1):
    for chir in ((1,-1) if ONLY in ('all','odd') else ()):
        t, co = odd(q, chir)
        run(f"odd q={q} order {2*q+1} w={chir:+d}", -(2*q+1), chir, t, co, nmax=q+3)
for q in (range(1, QMAX+1) if ONLY in ('all','even') else ()):
    for (u,v) in ((1,0),(0,1)):
        t, co = even(q, S(u), S(v))
        run(f"even q={q} order {2*q} (U0,V0)=({u},{v})", -2*q, 0, t, co, nmax=q+3)
