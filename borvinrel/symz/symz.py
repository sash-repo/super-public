"""Statement 9.1.3 of the paper: integer symmetrizable matrices of type Pos, A_ii=2, cond. b), indecomposable.
Census up to renumbering and sign changes A -> DAD, D=diag(+-1).
Coherent case (A_ij A_ji >= 0): identify the root system W.{alpha_i}."""
import itertools, sys
from fractions import Fraction as F

def det(M):
    n=len(M); M=[list(map(F,r)) for r in M]; d=F(1)
    for c in range(n):
        p=next((r for r in range(c,n) if M[r][c]!=0),None)
        if p is None: return F(0)
        if p!=c: M[c],M[p]=M[p],M[c]; d=-d
        d*=M[c][c]
        for r in range(c+1,n):
            f=M[r][c]/M[c][c]
            if f:
                for k in range(c,n): M[r][k]-=f*M[c][k]
    return d

def symmetrizer(A):
    """d with d_i A_ij = d_j A_ji  ->  B_ij = A_ij/d_i ... return d (A = D B, B sym) or None"""
    n=len(A); d=[None]*n; d[0]=F(1); stack=[0]
    while stack:
        i=stack.pop()
        for j in range(n):
            if A[i][j]!=0:
                # A = D B: A_ij = d_i B_ij, A_ji = d_j B_ij  => d_j = d_i A_ji/A_ij
                v=d[i]*F(A[j][i],A[i][j])
                if d[j] is None: d[j]=v; stack.append(j)
                elif d[j]!=v: return None
    if any(x is None for x in d): return None
    return d

def connected(A):
    n=len(A); seen={0}; st=[0]
    while st:
        i=st.pop()
        for j in range(n):
            if A[i][j] and j not in seen: seen.add(j); st.append(j)
    return len(seen)==n

def pos(A):
    n=len(A)
    for k in range(1,n+1):
        for I in itertools.combinations(range(n),k):
            if det([[A[i][j] for j in I] for i in I])<=0: return False
    return True

def canon(A):
    n=len(A); best=None
    for p in itertools.permutations(range(n)):
        P=[[A[p[i]][p[j]] for j in range(n)] for i in range(n)]
        # sign flips: fix sign of s_0=+1 wlog (global -1 does nothing)
        for s in itertools.product((1,-1),repeat=n-1):
            s=(1,)+s
            t=tuple(P[i][j]*s[i]*s[j] for i in range(n) for j in range(n))
            if best is None or t<best: best=t
    return best

def pair_options(coherent_only=True, N=3):
    opts=[(0,0)]
    for a in range(-N,N+1):
        for b in range(-N,N+1):
            if a==0 or b==0: continue
            if a*b>=4: continue
            if coherent_only and a*b<0: continue
            opts.append((a,b))
    return opts

def extend(reps, n, opts):
    """reps: set of canonical tuples of size n-1 (all Pos, maybe decomposable). Return size n."""
    out=set()
    for t in reps:
        m=n-1; A=[list(t[i*m:(i+1)*m]) for i in range(m)]
        for choice in itertools.product(opts,repeat=m):
            B=[r+[choice[i][0]] for i,r in enumerate(A)]
            B.append([choice[i][1] for i in range(m)]+[2])
            if symmetrizer(B) is None and connected(B): continue
            if not pos_last(B): continue
            out.add(canon(B))
    return out

def pos_last(B):
    # minors not containing last index were checked already
    n=len(B)
    for k in range(0,n):
        for I in itertools.combinations(range(n-1),k):
            J=list(I)+[n-1]
            if det([[B[i][j] for j in J] for i in J])<=0: return False
    return True

def symm_ok(A):
    # symmetrizable componentwise: check every connected component
    n=len(A); comp=[-1]*n; c=0
    for s in range(n):
        if comp[s]>=0: continue
        st=[s]; comp[s]=c
        while st:
            i=st.pop()
            for j in range(n):
                if A[i][j] and comp[j]<0: comp[j]=c; st.append(j)
        c+=1
    for k in range(c):
        I=[i for i in range(n) if comp[i]==k]
        if symmetrizer([[A[i][j] for j in I] for i in I]) is None: return False
    return True

def census(nmax, coherent_only=True, N=3):
    opts=pair_options(coherent_only,N)
    reps={(2,)}; res={1:reps}
    for n in range(2,nmax+1):
        new=set()
        m=n-1
        for t in reps:
            A=[list(t[i*m:(i+1)*m]) for i in range(m)]
            for choice in itertools.product(opts,repeat=m):
                B=[r+[choice[i][0]] for i,r in enumerate(A)]
                B.append([choice[i][1] for i in range(m)]+[2])
                if not symm_ok(B): continue
                if not pos_last(B): continue
                new.add(canon(B))
        reps=new; res[n]=reps
        print(n, len(reps), sum(1 for t in reps if connected([list(t[i*n:(i+1)*n]) for i in range(n)])), file=sys.stderr)
    return res

if __name__=="__main__":
    census(int(sys.argv[1]))
