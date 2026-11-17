#!/usr/bin/env python3
"""Theorem 3.4.1 (Vinberg's three types read off the inverse matrix), checked on random
indecomposable Z-matrices of sizes 2-5 with a symmetric zero pattern (a_ij = 0 iff a_ji = 0),
plus four affine Cartan matrices.

The type (finite, affine, indefinite) is computed from the smallest real eigenvalue
lambda_0(A) of formula (29), via the Perron root of s1 - A (numpy, floating point); the
determinant, the adjugate and the inverse are computed exactly (fractions.Fraction).
Checked: items 1) and 2) (finite type iff A^{-1} >= 0, then > 0; affine type iff det A = 0
and adj A >= 0 non-zero, then > 0); item 3) (an invertible matrix of indefinite type has a
negative element in every row and every column of its inverse; a singular one has adj A = 0
or adj A with elements of both signs); item 4) (A^{-1} <= 0 iff A is of indefinite type and
every A_{hat i} is a direct sum of matrices of finite or affine type; A^{-1} < 0 iff,
moreover, every A_{hat i} is of finite type).

Usage:  python checkc.py
"""
import random, numpy as np
from fractions import Fraction as F
def det(M):
    M=[[F(x) for x in r] for r in M]; n=len(M); d=F(1)
    for c in range(n):
        p=next((r for r in range(c,n) if M[r][c]!=0),None)
        if p is None: return F(0)
        if p!=c: M[c],M[p]=M[p],M[c]; d=-d
        d*=M[c][c]
        for r in range(c+1,n):
            f=M[r][c]/M[c][c]
            for k in range(c,n): M[r][k]-=f*M[c][k]
    return d
def minor(M,i,j): return [[M[a][b] for b in range(len(M)) if b!=j] for a in range(len(M)) if a!=i]
def adj(M):
    n=len(M); return [[(-1)**(i+j)*det(minor(M,j,i)) for j in range(n)] for i in range(n)]
def typ(A):
    s=max(A[i][i] for i in range(len(A)))+1
    B=s*np.eye(len(A))-np.array(A,float); l0=s-max(abs(np.linalg.eigvals(B)))
    return 'fin' if l0>1e-9 else ('aff' if abs(l0)<1e-9 else 'ind')
def comps(A,S):
    rem=set(S); out=[]
    while rem:
        s=rem.pop(); c={s}; st=[s]
        while st:
            x=st.pop()
            for y in list(rem):
                if A[x][y]!=0: rem.discard(y); c.add(y); st.append(y)
        out.append(sorted(c))
    return out
random.seed(7); stats={}; fails=0
aff=[[2,-2],[-2,2]],[[2,-1,-1],[-1,2,-1],[-1,-1,2]],[[2,-1,0],[-4,2,-1],[0,-1,2]],[[2,-1,0,0],[-1,2,-1,0],[0,-2,2,-2],[0,0,-1,2]]
pool=list(aff)
for t in range(40000):
    n=random.choice([2,3,4,5]); A=[[0]*n for _ in range(n)]
    for i in range(n):
        A[i][i]=random.choice([2,2,2,1,0])
        for j in range(i+1,n):
            if random.random()<0.55: A[i][j]=-random.choice([1,1,2,3]); A[j][i]=-random.choice([1,1,2,3,4])
    pool.append(A)
for A in pool:
    n=len(A)
    if len(comps(A,range(n)))!=1: continue
    T=typ(A); d=det(A); J=adj(A)
    fl=[x for r in J for x in r]
    if d!=0:
        inv=[x/d for x in fl]
        ok1=(T=='fin')==all(x>=0 for x in inv) and (T!='fin' or all(x>0 for x in inv))
        rowcol=all(any(J[i][j]/d<0 for j in range(n)) for i in range(n)) and all(any(J[i][j]/d<0 for i in range(n)) for j in range(n))
        ok3= T!='ind' or rowcol
        almost=all(typ([[A[a][b] for b in c] for a in c])!='ind' for k in range(n) for c in comps(A,[x for x in range(n) if x!=k]))
        allfin=all(typ([[A[a][b] for b in c] for a in c])=='fin' for k in range(n) for c in comps(A,[x for x in range(n) if x!=k]))
        ok4=(all(x<=0 for x in inv)==(T=='ind' and almost)) and (all(x<0 for x in inv)==(T=='ind' and allfin))
        ok=ok1 and ok3 and ok4
    else:
        nonneg=any(x!=0 for x in fl) and all(x>=0 for x in fl)
        ok=(T=='aff')==nonneg and (T!='aff' or all(x>0 for x in fl))
        if T=='ind': ok=ok and (all(x==0 for x in fl) or (any(x>0 for x in fl) and any(x<0 for x in fl)))
    stats[(T,d==0)]=stats.get((T,d==0),0)+1
    if not ok: fails+=1; print('FAIL',A,T)
print('matrices checked, by (type, det A = 0):')
for k in sorted(stats): print('   ', k, stats[k])
print('failures of Theorem 3.4.1:', fails)
