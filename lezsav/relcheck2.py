import pickle
import n4vac, n4rec
from n4noeth import C, mzero, is_solution
from n4rec import qadd, qsmul

if __name__ == "__main__":
    q = pickle.load(open("n4rec11-vacuum.pkl","rb"))
    ms0 = n4vac.graded_monos(0,0,-5,1,mfree=(-1,-1))
    B = n4vac.constrained_basis(ms0)
    qb,kern = n4vac.build_embedded_base(B)
    print("embedded base is a solution of (IK4):", is_solution(qb))
    same=diff=onlyq=onlyb=0
    for i in (0,1):
        for j in (0,1):
            ks=set(q[i][j])|set(qb[i][j])
            for k in ks:
                a=q[i][j].get(k,C()); b=qb[i][j].get(k,C())
                if a and b: same += (a==b); diff += (a!=b)
                elif a: onlyq+=1
                else: onlyb+=1
    print("monomials: agreeing %d, differing %d, only in vacuum %d, only in base %d"%(same,diff,onlyq,onlyb))
    for lvl in (0,2,4,6,8):
        nq=sum(1 for i in (0,1) for j in (0,1) for k in q[i][j] if len(k[3])==lvl)
        nb=sum(1 for i in (0,1) for j in (0,1) for k in qb[i][j] if len(k[3])==lvl)
        print("  level %d: vacuum %3d, embedded base %3d"%(lvl,nq,nb))
