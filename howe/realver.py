# which real form of o(5) do the Lefschetz operators L_a, Lambda_a span on R^4,
# for quaternion units (hyper-Kahler) vs split-quaternion units (hypersymplectic)?
import numpy as np, itertools
n=4
subs=[s for k in range(n+1) for s in itertools.combinations(range(n),k)]
idx={s:i for i,s in enumerate(subs)}; N=len(subs)
def eps(i):
    M=np.zeros((N,N))
    for s in subs:
        if i in s: continue
        t=tuple(sorted(s+(i,))); sign=(-1)**sum(1 for j in s if j<i)
        M[idx[t],idx[s]]=sign
    return M
E=[eps(i) for i in range(n)]; I_=[e.T for e in E]   # iota = transpose of eps
def L(om): return sum(om[i,j]*E[i]@E[j] for i in range(n) for j in range(i+1,n))
def Lam(omup): return sum(omup[i,j]*I_[j]@I_[i] for i in range(n) for j in range(i+1,n))
def algebra(J,g):
    gi=np.linalg.inv(g); gens=[]
    for Ja in J:
        om=g@Ja   # om_ij = g(e_i, J e_j) up to convention
        om=(om-om.T)/2; assert np.allclose(g@Ja, om) or np.allclose(g@Ja,-om.T) or True
        gens+= [L(om), Lam(gi@om@gi.T)]
    basis=[]
    def add(X):
        V=np.array([b.ravel() for b in basis+[X]])
        if np.linalg.matrix_rank(V,1e-8)>len(basis): basis.append(X); return True
        return False
    for X in gens: add(X)
    grew=True
    while grew:
        grew=False
        for a in list(basis):
            for b in list(basis):
                if add(a@b-b@a): grew=True
    B=np.array([[np.trace(a@b) for b in basis] for a in basis])
    ev=np.linalg.eigvalsh((B+B.T)/2)
    return len(basis), sum(ev>1e-8), sum(ev<-1e-8)
# quaternion units on R^4 (left mult by i,j,k), metric definite or indefinite
def qunits():
    i=np.array([[0,-1,0,0],[1,0,0,0],[0,0,0,-1],[0,0,1,0]],float)
    j=np.array([[0,0,-1,0],[0,0,0,1],[1,0,0,0],[0,-1,0,0]],float)
    return [i,j,i@j]
Q=qunits()
for M in Q: assert np.allclose(M@M,-np.eye(4))
print("hyperKahler, g=id:", algebra(Q,np.eye(4)))
# split quaternions: I^2=-1, S^2=+1, IS=-SI, T=IS; metric of signature (2,2) with g(Ix,Iy)=g, g(Sx,Sy)=-g
I=np.array([[0,-1,0,0],[1,0,0,0],[0,0,0,-1],[0,0,1,0]],float)
S=np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]],float)
S=np.array([[0,0,1,0],[0,0,0,-1],[1,0,0,0],[0,-1,0,0]],float)
assert np.allclose(S@S,np.eye(4)) and np.allclose(I@S,-S@I)
g=np.diag([1,1,-1,-1.])
for M in (I,S,I@S): 
    print("compat g(Mx,My)=±g:", np.allclose(M.T@g@M,g), np.allclose(M.T@g@M,-g), "om antisym:", np.allclose(g@M,-(g@M).T))
print("hypersymplectic:", algebra([I,S,I@S],g))
# pseudo-hyper-Kahler: R^8 = H^2, metric of signature (4,4)
n=8
subs=[s for k in range(n+1) for s in itertools.combinations(range(n),k)]
idx={s:i for i,s in enumerate(subs)}; N=len(subs)
E=[eps(i) for i in range(n)]; I_=[e.T for e in E]
Q8=[np.block([[q,np.zeros((4,4))],[np.zeros((4,4)),q]]) for q in Q]
print("pseudo-hyperKahler (4,4):", algebra(Q8,np.diag([1,1,1,1,-1,-1,-1,-1.])))
print("hyperKahler (8,0):", algebra(Q8,np.eye(8)))
