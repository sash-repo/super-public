# Casimir of gl(n|n) on its adjoint module, and on S^2(psl(n|n)) (super-symmetric square).
# Form: str(XY).  Run: python pslnn.py n  (n = 2 or 3)
import sys, numpy as np, itertools
n=int(sys.argv[1]); N=2*n
par=lambda i: 0 if i<n else 1
def E(i,j):
    M=np.zeros((N,N)); M[i,j]=1; return M
basis=[(i,j) for i in range(N) for j in range(N)]
bpar=[(par(i)+par(j))%2 for i,j in basis]
mats=[E(i,j) for i,j in basis]
def str_(X): return sum(X[i,i]*(1 if par(i)==0 else -1) for i in range(N))
def br(a,X,b,Y): return X@Y-(-1)**(a*b)*Y@X
# coordinates in gl
def coord(X): return X.reshape(-1)
D=len(basis)
ad=[np.array([coord(br(bpar[k],mats[k],bpar[l],mats[l])) for l in range(D)]).T for k in range(D)]
G=np.array([[str_(mats[k]@mats[l]) for l in range(D)] for k in range(D)])
Gi=np.linalg.inv(G)
# dual: x^k = sum_l Gi[l,k] x_l  so that str(x_k x^m)=delta
# SG selects the convention for the dual basis: 1 (sign (-1)^{p(x_k)}) and 2 (dual taken on the
# other side) are the two equivalent invariant ones, [C,ad] = 0 below; 0 and 3 are controls,
# not invariant.  Default 2.
SG=int(sys.argv[2]) if len(sys.argv)>2 else 2
def cas(rep):
    out=0
    for k in range(D):
        dual=sum(Gi[l,k]*rep[l] for l in range(D)); dual2=sum(Gi[k,l]*rep[l] for l in range(D))
        sg=(-1)**bpar[k] if SG in (1,3) else 1
        out=out+sg*(rep[k]@(dual if SG<2 else dual2))
    return out
C=cas(ad)
ok=max(np.abs(C@a-a@C).max() for a in ad)
print("gl(%d|%d): [C,ad] max"%(n,n),ok, " C on adjoint: rank",np.linalg.matrix_rank(C,1e-8)," C^2 rank",np.linalg.matrix_rank(C@C,1e-8))
# psl: sl = ker str (as element), quotient by identity
I=coord(np.eye(N))
strrow=np.array([str_(m) for m in mats])
# basis of sl: off-diagonal + diag with str 0
sl=[k for k,(i,j) in enumerate(basis) if i!=j]
diag=[]
for a in range(N-1):
    v=np.zeros(D); i,j=a,a+1
    v[basis.index((i,i))]=1; v[basis.index((j,j))]= -1 if par(i)==par(j) else 1
    diag.append(v)
B=[np.eye(D)[k] for k in sl]+diag   # sl basis vectors in gl coords (dim N^2-1)
B=np.array(B).T
# quotient by I (I lies in sl since str(1)=0): choose complement: drop the last diag vector
Q=B[:,:-1]                           # psl representatives
proj=np.linalg.lstsq(np.column_stack([Q,I]),np.eye(D),rcond=None)[0][:-1]  # gl->coords in Q (mod I), valid on sl
P=[ (proj@ad[k]@Q) for k in range(D)]   # action of gl on psl
d=Q.shape[1]
qpar=[ (bpar[k]) for k in sl ]+[0]*(len(diag)-1)
Cp=cas(P)
print("psl: dim",d," C on adjoint max |.|",np.abs(Cp).max())
# tensor square with Koszul signs
S=np.diag([(-1)**p for p in qpar])
def tens(A,a):
    # (A x 1)(v x w)=A v x w ; (1 x A)(v x w)=(-1)^{a|v|} v x A w
    return np.kron(A,np.eye(d))+np.kron(np.linalg.matrix_power(S,a),A)
T=[tens(P[k],bpar[k]) for k in range(D)]
CT=cas(T); print("[CT,T] max",max(np.abs(CT@t-t@CT).max() for t in T[:6]))
# super flip
tau=np.zeros((d*d,d*d))
for i in range(d):
    for j in range(d):
        tau[j*d+i,i*d+j]=(-1)**(qpar[i]*qpar[j])
Sym=(np.eye(d*d)+tau)/2
U,s,_=np.linalg.svd(Sym); r=(s>1e-8).sum(); U=U[:,:r]
CS=U.T@CT@U
ev=np.linalg.eigvals(CS)
vals,cnt=np.unique(np.round(ev.real,6),return_counts=True)
print("S^2 dim",r," eigenvalues of C:",dict(zip(vals,cnt)))
for v in vals:
    A=CS-v*np.eye(r)
    print("  ev",v,"ranks of (C-ev)^k k=1,2,3:",[np.linalg.matrix_rank(np.linalg.matrix_power(A,k),1e-6) for k in (1,2,3)])
# superdimensions of the eigenspaces: projector by Lagrange interpolation (C semisimple)
SS=np.kron(S,S)
for v in vals:
    Pv=np.eye(r)
    for w in vals:
        if w!=v: Pv=Pv@(CS-w*np.eye(r))/(v-w)
    print("  ev",v,"dim",round(np.trace(Pv).real),"ddim = dim_ev - dim_od",round(np.trace(U.T@SS@U@Pv).real))
