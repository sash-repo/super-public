import sys
from gA import build
def Bk(k):
    # nodes: a0..a_{k-2}, gamma, beta ; chain a0..a_{k-2},gamma ; beta - a2
    n=k+1; names=[f"a{i}" for i in range(k-1)]+["g","b"]
    B=[[0]*n for _ in range(n)]
    for i in range(k-1): B[i][i]=2
    B[k][k]=2; B[k-1][k-1]=0
    for i in range(k-1):
        B[i][i+1]=B[i+1][i]=-1
    B[2][k]=B[k][2]=-1
    par=[0]*n; par[k-1]=1
    return B,par
if __name__=="__main__":
    k=int(sys.argv[1]); H=int(sys.argv[2])
    B,par=Bk(k)
    d,l=build(B,par,H)
    for h in sorted(l):
        ev=sum(d[b] for b in l[h] if sum(b[t]*par[t] for t in range(len(par)))%2==0)
        od=sum(d[b] for b in l[h])-ev
        print(h,(ev,od),flush=True)
