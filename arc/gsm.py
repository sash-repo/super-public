"""The same question on the general superstring S^{1|1}: bilinear operators
between the spaces GSM_k = Vol^k, invariant under lf(S^{1|1}) = sl(1|2),
the projective subalgebra of vect(1|1), eq. (96) of the text:

  Span( d_x, x d_x + (1/2) th d_th, x^2 d_x + x th d_th, th d_th |
        d_th, x d_th, th d_x, x th d_x ).

Monomials x^m th^e are keys (m,e); Vol^k = F vvol^k with
  L_X (phi vvol^k) = (X phi + k Div X phi) vvol^k,
  Div(P d_x + Q d_th) = d_x P + (-1)^{p(Q)} d_th Q.
"""
import itertools
import sympy as sp

def add(*fs):
    out={}
    for f in fs:
        for k,v in f.items(): out[k]=out.get(k,0)+v
    return {k:sp.expand(v) for k,v in out.items() if sp.expand(v)!=0}

def smul(c,f):
    c=sp.sympify(c)
    return {k:sp.expand(c*v) for k,v in f.items() if sp.expand(c*v)!=0}

def mul(f,g):
    out={}
    for (m1,e1),c1 in f.items():
        for (m2,e2),c2 in g.items():
            if e1+e2>1: continue
            out[(m1+m2,e1+e2)]=out.get((m1+m2,e1+e2),0)+c1*c2
    return {k:sp.expand(v) for k,v in out.items() if sp.expand(v)!=0}

def d_x(f):
    out={}
    for (m,e),c in f.items():
        if m: out[(m-1,e)]=out.get((m-1,e),0)+m*c
    return {k:sp.expand(v) for k,v in out.items() if sp.expand(v)!=0}

def d_th(f):
    out={}
    for (m,e),c in f.items():
        if e: out[(m,0)]=out.get((m,0),0)+c
    return {k:sp.expand(v) for k,v in out.items() if sp.expand(v)!=0}

def mono(m=0,e=0,c=1): return {(m,e):sp.sympify(c)} if c else {}
def parity(f):
    ps={e%2 for (m,e) in f}
    return ps.pop() if ps else 0

# a vector field is (P, Q, p) meaning P d_x + Q d_th of parity p
FIELDS = {
 "d_x":      (mono(0,0), {}, 0),
 "h":        (mono(1,0), smul(sp.Rational(1,2), mono(0,1)), 0),
 "x^2d_x+..":(mono(2,0), mono(1,1), 0),
 "th d_th":  ({}, mono(0,1), 0),
 "d_th":     ({}, mono(0,0), 1),
 "x d_th":   ({}, mono(1,0), 1),
 "th d_x":   (mono(0,1), {}, 1),
 "x th d_x": (mono(1,1), {}, 1),
}
ODD = ["d_th","x d_th","th d_x","x th d_x"]

def div(X):
    P,Q,p=X
    # p(Q) = p(X)+1
    s = 1 if (p+1)%2==0 else -1
    return add(d_x(P), smul(s, d_th(Q)))

def act(k, X, phi):
    P,Q,p=X
    out=add(mul(P,d_x(phi)), mul(Q,d_th(phi)))
    return add(out, smul(k, mul(div(X), phi)))

def check():
    """[X,Y] closure and the module homomorphism, symbolic k."""
    k=sp.Symbol('k')
    def br(X,Y):
        P1,Q1,p1=X; P2,Q2,p2=Y
        s=-1 if (p1 and p2) else 1
        P=add(mul(P1,d_x(P2)), mul(Q1,d_th(P2)),
              smul(-s, add(mul(P2,d_x(P1)), mul(Q2,d_th(P1)))))
        Q=add(mul(P1,d_x(Q2)), mul(Q1,d_th(Q2)),
              smul(-s, add(mul(P2,d_x(Q1)), mul(Q2,d_th(Q1)))))
        return (P,Q,(p1+p2)%2)
    mons=[mono(m,e) for m in range(4) for e in (0,1)]
    bad=0
    for n1,X in FIELDS.items():
        for n2,Y in FIELDS.items():
            s=-1 if (X[2] and Y[2]) else 1
            Z=br(X,Y)
            for phi in mons:
                lhs=add(act(k,X,act(k,Y,phi)), smul(-s, act(k,Y,act(k,X,phi))))
                if add(lhs, smul(-1, act(k,Z,phi))): bad+=1
    print("gsm check_module:", "OK" if bad==0 else "FAILED %d"%bad)
    return bad==0

def ansatz(M,T):
    """theta^s d_x^m1 d_th^e1 (x) d_x^m2 d_th^e2 with m1+m2=M, e1+e2-s=T."""
    out=[]
    for s,e1,e2 in itertools.product((0,1),repeat=3):
        if e1+e2-s!=T: continue
        for m1 in range(M+1):
            out.append((s,m1,e1,M-m1,e2))
    return out

def apply_term(term,phi,psi):
    s,m1,e1,m2,e2=term
    a=phi
    for _ in range(m1): a=d_x(a)
    if e1: a=d_th(a)
    b=psi
    for _ in range(m2): b=d_x(b)
    if e2: b=d_th(b)
    out=mul(a,b)
    if s: out=mul(mono(0,1),out)
    return out

def hom_dim(M,ka,kb,dmax=6,want=False):
    """T = -M is forced by the two Cartan elements; target k = ka+kb+M."""
    T=-M
    terms=ansatz(M,T)
    if not terms: return (0,[],[]) if want else 0
    kc=ka+kb+M
    mons=[mono(m,e) for m in range(dmax+1) for e in (0,1)]
    rows={}
    for gname in ODD:
        X=FIELDS[gname]
        for phi in mons:
            pphi=parity(phi)
            for psi in mons:
                for j,term in enumerate(terms):
                    lhs=act(kc,X,apply_term(term,phi,psi))
                    r1=apply_term(term,act(ka,X,phi),psi)
                    sgn=-1 if pphi else 1
                    r2=apply_term(term,phi,act(kb,X,psi))
                    res=add(lhs,smul(-1,r1),smul(-sgn,r2))
                    for key,val in res.items():
                        rows.setdefault((gname,tuple(sorted(phi)),
                                         tuple(sorted(psi)),key),
                                        [0]*len(terms))[j]+=val
    mat=[r for r in rows.values() if any(x!=0 for x in r)]
    if not mat:
        ns=sp.eye(len(terms)).columnspace()
    else:
        ns=sp.Matrix(mat).nullspace()
    return (len(ns),terms,ns) if want else len(ns)

if __name__=="__main__":
    check()
    ka,kb=sp.symbols('ka kb')
    print("\n dim Hom_{sl(1|2)}( Vol^ka (x) Vol^kb , Vol^{ka+kb+M} ) :")
    for M in range(0,5):
        d,terms,ns=hom_dim(M,ka,kb,dmax=5,want=True)
        print("  M=%d  #terms=%d  dim=%d"%(M,len(terms),d))
        for v in ns:
            print("      ", [(t,sp.factor(sp.simplify(c))) for t,c in zip(terms,v) if c!=0])
