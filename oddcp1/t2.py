"""Statement 16.1 and table (25): the lists of basis cocycles printed in [Ba5]
(the retract CP^{1|5}) tested for spanning H^1(T_q), q = 2, 3, 4, 5.  In degree 3
the printed list (10 cocycles) does not span; with the 40 cocycles (29) added it does.
"""
import bashkin as B
from itertools import combinations
ks=(1,1,1,1,1); ALL=(1,2,3,4,5)

# --- Theorem cp15t2, q=2, as printed: 20 v-type + 20 w-type
vlist=[(1,2,4),(1,2,5),(1,3,2),(1,3,5),(1,4,3),(1,4,5),(1,5,2),(1,5,3),
       (2,3,1),(2,3,4),(2,4,1),(2,4,3),(2,5,3),(2,5,4),(3,4,2),(3,4,5),
       (3,5,1),(3,5,4),(4,5,1),(4,5,2)]
coc2=[B.w((i,j,t),t,-1) for (i,j,t) in vlist]
coc2+=[B.w(J,r,-1) for J in combinations(ALL,3) for r in ALL if r not in J]
print("q=2  printed list:", len(coc2), "cocycles ->", B.in_H1_span(ks,2,coc2))

# --- Proposition cp15ht3, q=3, as printed
coc3=[B.w(J,r,-rr) for J in combinations(ALL,4) for r in ALL if r not in J
      for rr in (1,2)]
print("q=3  printed list:", len(coc3), "cocycles ->", B.in_H1_span(ks,3,coc3))

# the cocycles the Proposition omits, by his own Theorem (case k_I = 3):
extra=[B.w(tuple(I)+(t,),t,-rr) for I in combinations(ALL,3)
       for t in ALL if t not in I for rr in (1,2)]
print("q=3  + the omitted v-type:", len(coc3+extra), "->", B.in_H1_span(ks,3,coc3+extra))

# --- Theorem cp15t4, q=4
coc4=[B.u(I,-1) for I in combinations(ALL,4)]
coc4+=[B.w(ALL,k,-rr) for k in ALL for rr in (1,2,3)]
print("q=4  printed list:", len(coc4), "cocycles ->", B.in_H1_span(ks,4,coc4))

# --- Proposition cp15ht5, q=5
coc5=[B.u(ALL,-1), B.u(ALL,-2)]
print("q=5  printed list:", len(coc5), "cocycles ->", B.in_H1_span(ks,5,coc5))
