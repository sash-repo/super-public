"""Invariant counts of a registered pair of surfaces, by orbit dimension.

Data at a matched point, pulled back to one tangent space: symmetric 2-tensors
T^(1)=g_1, T^(2)=II_1, T^(3)=phi^*g_2, T^(4)=phi^*II_2, with their 1-jets.
Group: 2-jets at 0 of diffeomorphisms fixing 0:  X^i = a^i_j x^j + 1/2 b^i_{jk} x^j x^k.
Constraint: Codazzi for (g_1,II_1) and for (phi^*g_2, phi^*II_2)  (2 equations each).
invariants = dim(constraint surface) - rank(action at a generic point on it).
"""
import numpy as np
rng = np.random.default_rng(7)
IDX = [(0,0),(0,1),(1,1)]          # symmetric index pairs
def sym(T):                          # 2x2 symmetric from 3 comps
    return np.array([[T[0],T[1]],[T[1],T[2]]])
def rand_tensor_jet():
    T0 = rng.normal(size=3); T0[0]+=3; T0[2]+=3     # value (make metrics positive)
    T1 = rng.normal(size=(2,3))                     # dT_k[ij]
    return T0, T1
def christoffel(g0, g1):
    """Gamma^m_{kl} at 0 from the 1-jet of g (g1[k] = d_k g_{ij} as 3-vector)."""
    G = sym(g0); Gi = np.linalg.inv(G)
    dg = np.zeros((2,2,2))
    for k in range(2):
        dg[k] = sym(g1[k])
    Gam = np.zeros((2,2,2))
    for m in range(2):
        for k in range(2):
            for l in range(2):
                Gam[m,k,l] = 0.5*sum(Gi[m,n]*(dg[k][n,l]+dg[l][n,k]-dg[n][k,l]) for n in range(2))
    return Gam
def impose_codazzi(g0,g1,B0,B1):
    """solve the two Codazzi equations for d_2 II_11 and d_1 II_22."""
    Gam = christoffel(g0,g1); B = sym(B0)
    # nabla_k B_ij = d_k B_ij - Gam^m_{ki} B_mj - Gam^m_{kj} B_im ; Codazzi: nabla_1 B_2j = nabla_2 B_1j
    def dB(k,i,j):
        return B1[k][IDX.index(tuple(sorted((i,j))))]
    # j=0: d_1 B_21 - d_2 B_11 = sum_m (Gam^m_{12} B_m0 + Gam^m_{10} B_1m) - (Gam^m_{21} B_m0 + Gam^m_{20} B_0m)
    rhs0 = sum(Gam[m,0,1]*B[m,0]+Gam[m,0,0]*B[1,m]-Gam[m,1,0]*B[m,0]-Gam[m,1,0]*B[0,m] for m in range(2))
    B1[1][0] = dB(0,1,0) - rhs0           # d_2 B_11
    rhs1 = sum(Gam[m,0,1]*B[m,1]+Gam[m,0,1]*B[1,m]-Gam[m,1,0]*B[m,1]-Gam[m,1,1]*B[0,m] for m in range(2))
    B1[0][2] = dB(1,0,1) + rhs1           # d_1 B_22
    return B1
def action(tensors, order):
    """rows: jet components of all tensors up to `order`; cols: Lie algebra (a: 4, b: 6)."""
    rows=[]
    npar = 4 + (6 if order>=1 else 0)
    cols=[]
    for p in range(npar):
        a = np.zeros((2,2)); b = np.zeros((2,2,2))
        if p<4: a[p//2,p%2]=1
        else:
            q=p-4; i=q//3; jk=IDX[q%3]; b[i,jk[0],jk[1]]=1; b[i,jk[1],jk[0]]=1
        col=[]
        for (T0,T1) in tensors:
            T=sym(T0); dT=[sym(T1[k]) for k in range(2)]
            for (i,j) in IDX:                       # order 0: (L_X T)_ij = T_kj a^k_i + T_ik a^k_j
                col.append(sum(T[k,j]*a[k,i]+T[i,k]*a[k,j] for k in range(2)))
            if order>=1:
                for l in range(2):
                    for (i,j) in IDX:
                        v = sum(a[k,l]*dT[k][i,j] + dT[l][k,j]*a[k,i] + T[k,j]*b[k,i,l]
                                + dT[l][i,k]*a[k,j] + T[i,k]*b[k,j,l] for k in range(2))
                        col.append(v)
        cols.append(col)
    return np.array(cols).T
def count(tensors, ncod, order):
    M = action(tensors, order)
    dim = M.shape[0] - ncod
    r = np.linalg.matrix_rank(M, tol=1e-9)
    return dim, r, dim-r
g1=rand_tensor_jet(); B1=rand_tensor_jet(); g2=rand_tensor_jet(); B2=rand_tensor_jet()
B1=(B1[0], impose_codazzi(g1[0],g1[1],B1[0],B1[1]))
B2=(B2[0], impose_codazzi(g2[0],g2[1],B2[0],B2[1]))
print("order 0 (pointwise):")
print("  one surface (g,II)             dim, orbit, invariants =", count([g1,B1],0,0))
print("  isometric pair (g,II1,II2)     dim, orbit, invariants =", count([g1,B1,B2],0,0))
print("  registered pair (g1,II1,g2,II2)dim, orbit, invariants =", count([g1,B1,g2,B2],0,0))
print("order 1 (1-jets, Codazzi imposed):")
print("  one surface (g,II)             dim, orbit, invariants =", count([g1,B1],2,1))
print("  isometric pair (g,II1,II2)     dim, orbit, invariants =", count([g1,B1,B2],4,1))
print("  registered pair (g1,II1,g2,II2)dim, orbit, invariants =", count([g1,B1,g2,B2],4,1))
# sanity: a metric's 1-jet alone has no invariants
print("  metric 1-jet alone             dim, orbit, invariants =", count([g1],0,1))
