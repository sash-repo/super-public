import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exactlin import rank
from dervect import Super, Echelon, svect, poisson, contact
from centext import Cocycles

def check(tag, S, L):
    C = Cocycles(S, L); bad = 0; tested = 0
    for w in sorted({C.deg[i]+C.deg[j] for i in range(C.N) for j in range(C.N)}):
        for eps in (0,1):
            P = C.pairs(w, eps)
            if not P: continue
            idx = {q:t for t,q in enumerate(P)}
            # coboundaries in this block
            cob = []
            for l in range(C.N):
                if C.deg[l]!=w or C.par[l]!=eps: continue
                row = [C.st.cc(i,j)[l] % C.p for (i,j) in P]
                if any(row): cob.append(row)
            if not cob: continue
            # the cocycle equations of this block
            eqs = []
            for i in range(C.N):
                for j in range(i, C.N):
                    cij = C.st.cc(i,j)
                    for k in range(j, C.N):
                        if C.deg[i]+C.deg[j]+C.deg[k] != w: continue
                        if (C.par[i]+C.par[j]+C.par[k])%2 != eps: continue
                        row = [0]*len(P)
                        def add(coef,l,kk):
                            t,sg = C.coord_of(idx,l,kk)
                            if t is not None and sg: row[t] = (row[t]+coef*sg) % C.p
                        for l in range(C.N):
                            if cij[l]: add(cij[l], l, k)
                        cjk = C.st.cc(j,k)
                        for l in range(C.N):
                            if cjk[l]: add(-cjk[l], i, l)
                        sg0 = (-1)**(C.par[i]*C.par[j]); cik = C.st.cc(i,k)
                        for l in range(C.N):
                            if cik[l]: add(sg0*cik[l], j, l)
                        if any(row): eqs.append(row)
            if not eqs: continue
            for b in cob:
                tested += 1
                if any(sum(r[t]*b[t] for t in range(len(P))) % C.p for r in eqs):
                    bad += 1
    print(f"{tag}: dB=0 tested on {tested} coboundaries, failures={bad}")

for p in (3,5):
    S = Super(p,1,1,[1]); check(f"p={p} vect(1|1)", S, S.vect())
    S = Super(p,2,1,[1,1]); check(f"p={p} svect(2|1)", S, svect(S))
    S = Super(p,2,1,[1,1]); L,_ = poisson(S,'h',1); check(f"p={p} h(2|1)", S, L)
    S = Super(p,2,2,[1,1]); L,_ = poisson(S,'le',2); check(f"p={p} le(2|2)", S, L)
    S = Super(p,3,0,[1,1,1],wt=[2,1,1]); L,_ = contact(S,1); check(f"p={p} k(3|0)", S, L)
