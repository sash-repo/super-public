# Which subsets of check-Delta={-2e_i} occur in Borel subalgebras of pe(n)
# with the standard even part (e_i-e_j, i<j)?  Formula (14) of the text: exactly the final
# segment -2e_i, ht b < i <= n.  Matrix model with the roots of formula (12).
import itertools, numpy as np, sys
def run(n):
    N=2*n; basis=[]  # (matrix, root tuple, parity)
    def E(i,j): m=np.zeros((N,N)); m[i,j]=1; return m
    for i in range(n):
        for j in range(n):
            if i!=j: basis.append((E(i,j)-E(n+j,n+i), tuple((k==i)-(k==j) for k in range(n)),0))
    for i in range(n):
        for j in range(i+1,n):   # B antisymmetric: root e_i+e_j
            basis.append((E(i,n+j)-E(j,n+i), tuple((k==i)+(k==j) for k in range(n)),1))
    for i in range(n):
        for j in range(i,n):     # C symmetric: root -e_i-e_j
            m=E(n+i,j)+E(n+j,i)
            basis.append((m, tuple(-(k==i)-(k==j) for k in range(n)),1))
    # sanity: closed under supercommutator in pe(n) shape
    def br(x,px,y,py): return x@y-(-1)**(px*py)*y@x
    def ok(S):
        allowed=set(S)|{tuple([0]*n)}
        elems=[b for b in basis if b[1] in S]
        for (x,a,p),(y,b,q) in itertools.product(elems,repeat=2):
            z=br(x,p,y,q)
            if np.abs(z).max()<1e-12: continue
            r=tuple(u+v for u,v in zip(a,b))
            if r not in allowed: return False
        # also brackets with h: fine
        return True
    ev=[b[1] for b in basis if b[2]==0 and next(c for c in b[1] if c)>0]
    pairs=[b[1] for b in basis if b[2]==1 and sum(b[1])>0]
    chk=[tuple(-2*(k==i) for k in range(n)) for i in range(n)]
    found=set(); cnt=0
    for signs in itertools.product([1,-1],repeat=len(pairs)):
        odd=[tuple(s*c for c in r) for s,r in zip(signs,pairs)]
        for sub in itertools.product([0,1],repeat=n):
            S=ev+odd+[chk[i] for i in range(n) if sub[i]]
            if ok(S): found.add(sub); cnt+=1
    print(f"n={n}: {cnt} closed root sets; check-Delta patterns (1 = -2e_i present):")
    for s in sorted(found,key=lambda t:(-sum(t),t)): print("  ",s)
for n in (2,3,4): run(n)
