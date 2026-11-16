"""Subsec. 8.3.13 of the text, k^L(1|3): the reflection s_beta: b^(n) -> b^(n-1) and the rule
(formula (38)) for the root vectors. Run: python3 kl13_beta.py"""
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
import kL; kL.SIGN=-1
from kL import br, m, add, sm
from fractions import Fraction as F
XI,ETA,TH=0,1,2; N=3
cnt=[0,0]
def ok(c,msg):
    cnt[0]+=1
    if not c: cnt[1]+=1; print("FAIL",msg)
def isprop(a,b):
    if not a or not b: return False
    ks=set(a)|set(b); k0=next(iter(b)); c=F(a.get(k0,0))/b[k0]
    return c!=0 and all(a.get(k,0)==c*b.get(k,0) for k in ks)
def vecs(n):
    k=n//2
    if n%2==0: return k,dict(a=m(k,ETA,TH),b=m(1-k,XI),ma=m(-k,XI,TH),mb=m(k,ETA))
    return k,dict(a=m(k+1,ETA),b=m(-k,XI,TH),ma=m(-k,XI),mb=m(k,ETA,TH))
Y1,Y2,Z1,Z2=m(1,TH),m(0,XI,ETA,TH),m(0,TH),m(-1,XI,ETA,TH)
B=lambda f,g: br(f,g,N)
gam=lambda k: (add(Y1,sm(-(k-1),Y2)),add(Z1,sm(-(k-1),Z2)))
bad=[]
for n in range(-9,10):
    k,X=vecs(n); G,MG=gam(k)
    k1,X1=vecs(n-1); G1,MG1=gam(k1)
    # roots of b^(n-1): alpha'=-beta, beta'=beta+eps
    ok(isprop(X['mb'],X1['a']),f"X_beta -> X_-beta is X_alpha' n={n}")
    ok(isprop(X['b'],X1['ma']),f"X_-beta -> X_beta is X_-alpha' n={n}")
    ok(isprop(B(G,X['b']),X1['b']),f"X_alpha -> [X_g,X_b] is X_beta' n={n}")
    ok(isprop(B(MG,X['mb']),X1['mb']),f"X_-alpha -> [X_-g,X_-b] is X_-beta' n={n}")
    if n%2:   # n=2k+1: beta even, X_g fixed
        ok(G==G1 and MG==MG1,f"X_g fixed n={n}")
    else:     # n=2k: beta odd, X_g -> [X_a,X_b]
        im,mim=B(X['a'],X['b']),B(X['ma'],X['mb'])
        ok(im!={} and mim!={},f"[X_a,X_b]!=0 n={n}")
        if not isprop(im,G1): bad.append((n,k,k1,im,G1))
        # still a valid gamma' for b^(n-1): not prop. to [X_a',X_b'], brackets nonzero
        ok(not isprop(im,B(X1['a'],X1['b'])),f"new X_g not prop [Xa',Xb'] n={n}")
        ok(B(im,X1['a'])!={} and B(im,X1['b'])!={},f"[new X_g, X_a'],[.,X_b'] != 0 n={n}")
        ok(B(mim,X1['ma'])!={} and B(mim,X1['mb'])!={},f"negative analog n={n}")
        # with the other choice X_gamma = theta t-(k+1) xi eta theta the rule closes
        Gs,MGs=gam(k1+2)
        ok(isprop(im,gam(k)[0]) is False,"")
print(f"{cnt[0]} checks, {cnt[1]} failures")
print("n even: [X_a,X_b] vs the X_gamma of b^(n-1) by (k13gamma):")
for n,k,k1,im,G1 in bad[:4]: print(" n=",n,"[Xa,Xb]=",im," X_g'=",G1)
print(len(bad),"of the even n mismatch")
# the choice adapted to s_beta: X_gamma = theta t-(k+1) xi eta theta, X_-gamma likewise, k=[n/2]
gamb=lambda k: (add(Y1,sm(-(k+1),Y2)),add(Z1,sm(-(k+1),Z2)))
c2=[0,0]
for n in range(-9,10):
    k,X=vecs(n); G,MG=gamb(k); k1,X1=vecs(n-1); G1,MG1=gamb(k1)
    for c,msg in [(not isprop(G,B(X['a'],X['b'])),"not prop"),(not isprop(MG,B(X['ma'],X['mb'])),"neg not prop"),
                  (B(G,X['a'])!={} and B(G,X['b'])!={},"brackets"),(B(MG,X['ma'])!={} and B(MG,X['mb'])!={},"neg brackets"),
                  (isprop(B(G,X['b']),X1['b']) and isprop(B(MG,X['mb']),X1['mb']),"X_+-a image"),
                  ((G==G1 and MG==MG1) if n%2 else (isprop(B(X['a'],X['b']),G1) and isprop(B(X['ma'],X['mb']),MG1)),"X_g image")]:
        c2[0]+=1
        if not c: c2[1]+=1; print("FAIL adapted",msg,n)
print(f"adapted choice for s_beta: {c2[0]} checks, {c2[1]} failures")
