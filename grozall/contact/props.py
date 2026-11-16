"""Properties of the exceptional operator kGz: antisymmetry, order, non-degeneracy."""
import os, sys
from fractions import Fraction as Fr
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from affine import pencil, ops_at, dim_at
from gen import apply_term, rand_field, term_basis
from contact import madd, mscal, pmul
from named import apply_word

terms,V0,N0,N1,N2 = pencil(2,2,0,2,nfield=1)

def op_of(lam):
    o = ops_at(terms,V0,N0,N1,N2,lam,-lam)
    assert len(o)==1
    return o[0]

def ev(vec,F,G):
    acc={}
    for cf,t in zip(vec,terms):
        if cf: acc=madd(acc,mscal(cf,apply_term(t,F,G)))
    return {k[:3]:v for k,v in acc.items()}

def norm(v):
    nz=[x for x in v if x]
    return [x/nz[0] for x in v] if nz else v

Fs=[rand_field(2,3,seed=41+i) for i in range(3)]
Gs=[rand_field(2,3,seed=77+i) for i in range(3)]

print("1) antisymmetry at lam=mu=0:")
o0=op_of(Fr(0))
bad=0
for F in Fs:
    for G in Gs:
        if madd(ev(o0,F,G), ev(o0,G,F)):   # B(F,G) + B(G,F) must vanish
            bad+=1
print("   B(F,G) = -B(G,F) on %d pairs: %s"%(len(Fs)*len(Gs), "yes" if bad==0 else "NO (%d)"%bad))

print("2) the family: kGz_lam(F,G) vs kGz_{-lam}(G,F), normalized:")
for lam in (Fr(1),Fr(1,2),Fr(-2),Fr(5,3)):
    a=op_of(lam); b=op_of(-lam)
    vals=[]
    for F in Fs[:2]:
        for G in Gs[:2]:
            x=ev(a,F,G); y=ev(b,G,F)
            ks=sorted(set(x)|set(y))
            xv=[x.get(k,Fr(0)) for k in ks]; yv=[y.get(k,Fr(0)) for k in ks]
            nz=[i for i,t in enumerate(xv) if t]
            vals.append(None if not nz else yv[nz[0]]/xv[nz[0]] if all(
                xv[i]*yv[nz[0]]==yv[i]*xv[nz[0]] for i in range(len(ks))) else "not proportional")
    print("   lam=%-5s ratios: %s"%(lam,vals))

print("3) order: does kGz see second derivatives of a single argument?")
o=op_of(Fr(0))
mx=0
for cf,t in zip(o,terms):
    if cf:
        m,n,D1,D2,c=t
        mx=max(mx, D1[0]+D1[1]+2*D1[2], D2[0]+D2[1]+2*D2[2])
print("   highest weighted order on one argument:",mx,
      "; classical order =", max((D1[0]+D1[1]+D1[2]) for (m,n,D1,D2,c),cf in zip(terms,o) if cf))

print("4) is it a multiple of the order-0 pairing composed with something?")
t0,V00,_,_,_ = pencil(2,2,0,0,nfield=1)
from affine import ops_at as oa
pair = oa(t0,V00,[],[],[],Fr(0),Fr(0))
print("   the order-0 pairing space has dim", len(pair))
p=pair[0]
def ev0(vec,F,G):
    acc={}
    for cf,t in zip(vec,t0):
        if cf: acc=madd(acc,mscal(cf,apply_term(t,F,G)))
    return {k[:3]:v for k,v in acc.items()}
# find a pair with <F,G> = 0 but kGz(F,G) != 0
found=0
for i in range(60):
    F=rand_field(2,2,seed=1000+i); G=rand_field(2,2,seed=2000+i)
    if not ev0(p,F,G) and ev(o,F,G):
        found+=1
print("   pairs with <F,G>=0 but kGz(F,G)!=0 among 60 random ones:",found)
