"""Subsec. 8.3.13 of the text, k^L(1|3): the brackets of the root vectors, the choice (36) of
X_gamma, X_{-gamma}, and the rule (37) for the root vectors under s_alpha. Run: python3 kl13_refl.py"""
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
import kL; kL.SIGN=-1   # the sign of formula (kbr) of the text
from kL import br, m, add, sm
from fractions import Fraction as F
XI,ETA,TH=0,1,2; N=3
cnt=[0,0]
def ok(c,msg):
    cnt[0]+=1
    if not c: cnt[1]+=1; print("FAIL",msg)
def eq(a,b,msg): ok(add(a,sm(-1,b))=={},f"{msg}: {a} vs {b}")
def prop(a,b,msg):  # a = c*b, c != 0
    ks=set(a)|set(b)
    if not a or not b: ok(False,msg+" zero"); return
    k0=next(iter(b)); c=F(a.get(k0,0),1)/b[k0]
    ok(c!=0 and all(a.get(k,0)==c*b.get(k,0) for k in ks),f"{msg}: {a} vs {b}")
def vecs(n):
    k=n//2
    if n%2==0: return k,dict(a=m(k,ETA,TH),b=m(1-k,XI),ma=m(-k,XI,TH),mb=m(k,ETA))
    return k,dict(a=m(k+1,ETA),b=m(-k,XI,TH),ma=m(-k,XI),mb=m(k,ETA,TH))
Y1,Y2,Z1,Z2=m(1,TH),m(0,XI,ETA,TH),m(0,TH),m(-1,XI,ETA,TH)
for n in range(-9,10):
    k,X=vecs(n); B=lambda f,g: br(f,g,N)
    if n%2==0:
        eq(B(Y1,X['a']),m(k+1,ETA),f"[Y1,Xa] n={n}"); eq(B(Y2,X['a']),{},f"[Y2,Xa] n={n}")
        eq(B(Y1,X['b']),sm(k,m(1-k,XI,TH)),f"[Y1,Xb] n={n}"); eq(B(Y2,X['b']),m(1-k,XI,TH),f"[Y2,Xb] n={n}")
        eq(B(Z1,X['a']),m(k,ETA),f"[Z1,Xa] n={n}"); eq(B(Z2,X['a']),{},f"[Z2,Xa] n={n}")
        eq(B(Z1,X['b']),sm(k-1,m(-k,XI,TH)),f"[Z1,Xb] n={n}"); eq(B(Z2,X['b']),m(-k,XI,TH),f"[Z2,Xb] n={n}")
    else:
        eq(B(Y1,X['a']),sm(-k,m(k+1,ETA,TH)),f"[Y1,Xa] n={n}"); eq(B(Y2,X['a']),sm(-1,m(k+1,ETA,TH)),f"[Y2,Xa] n={n}")
        eq(B(Y1,X['b']),m(1-k,XI),f"[Y1,Xb] n={n}"); eq(B(Y2,X['b']),{},f"[Y2,Xb] n={n}")
        eq(B(Z1,X['a']),sm(-(k+1),m(k,ETA,TH)),f"[Z1,Xa] n={n}"); eq(B(Z2,X['a']),sm(-1,m(k,ETA,TH)),f"[Z2,Xa] n={n}")
        eq(B(Z1,X['b']),m(-k,XI),f"[Z1,Xb] n={n}"); eq(B(Z2,X['b']),{},f"[Z2,Xb] n={n}")
    s=(-1)**(n%2)
    eq(B(X['a'],X['b']),sm(s,add(m(1,TH),sm(-k,m(0,XI,ETA,TH)))),f"[Xa,Xb] n={n}")
    eq(B(X['ma'],X['mb']),sm(s,add(m(0,TH),sm(-k,m(-1,XI,ETA,TH)))),f"[X-a,X-b] n={n}")
    G=add(Y1,sm(-(k-1),Y2)); MG=add(Z1,sm(-(k-1),Z2))
    # Corollary 1) of and its analog 2)
    ok(B(G,X['a'])!={} and B(G,X['b'])!={},f"[Xg,Xa],[Xg,Xb]!=0 n={n}")
    ok(B(MG,X['ma'])!={} and B(MG,X['mb'])!={},f"[X-g,X-a],[X-g,X-b]!=0 n={n}")
    # s_alpha on root vectors (rule (37)) against b^{(n+1)} with the choice (36) there
    k1,X1=vecs(n+1); G1=add(Y1,sm(-(k1-1),Y2)); MG1=add(Z1,sm(-(k1-1),Z2))
    prop(B(G,X['a']),X1['a'],f"X_b -> [Xg,Xa] n={n}")
    prop(X['ma'],X1['b'],f"X_a -> X_-a n={n}")
    prop(B(MG,X['ma']),X1['ma'],f"X_-b -> [X-g,X-a] n={n}")
    prop(X['a'],X1['mb'],f"X_-a -> X_a n={n}")
    if n%2==0: eq(G,G1,f"X_g fixed n={n}"); eq(MG,MG1,f"X-g fixed n={n}")
    else: prop(B(X['b'],X['a']),G1,f"X_g -> [Xb,Xa] n={n}"); prop(B(X['mb'],X['ma']),MG1,f"X-g -> [X-b,X-a] n={n}")
# linear independence, as in Lemma 5): fails iff a+b+1=(-1)^a
print(f"{cnt[0]} checks, {cnt[1]} failures")
