from ab3dom import *
D = (sc(h,(-1,-1,-1,1)), (F(0),F(0),F(1),F(0)), (F(0),F(1),F(-1),F(0)), (F(1),F(-1),F(0),F(0)))
Bs = bases_from(D)
print(len(Bs), "bases")
target = [[2,-1,0,0],[-3,0,1,0],[0,-1,2,-2],[0,0,-1,2]]
from itertools import permutations
for B in Bs:
    A = cartan(B)
    par = [0 if b in EVEN else 1 for b in B]
    for p in permutations(range(4)):
        Bp = [B[i] for i in p]; Ap = cartan(Bp)
        # rescale isotropic row
        ok = True
        for i in range(4):
            row = Ap[i]; t = target[i]
            if ip(Bp[i],Bp[i]) == 0:
                ratio = None
                for x, y in zip(row, t):
                    if x == 0 and y != 0 or x != 0 and y == 0: ok = False
                    if x != 0 and y != 0:
                        r = F(y)/x
                        if ratio is None: ratio = r
                        elif ratio != r: ok = False
            elif [F(x) for x in row] != [F(x) for x in t]: ok = False
        if ok and [0 if b in EVEN else 1 for b in Bp] == [0,1,0,0]:
            print("MATCH", Bp)
    print(par, [[str(x) for x in r] for r in A])
