"""Generate the LaTeX of kGz in components -- generated, never transcribed by hand."""
import os, sys, math
from fractions import Fraction as Fr
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from affine import pencil, ops_at

terms,V0,N0,N1,N2 = pencil(2,2,0,2,nfield=1)
op = ops_at(terms,V0,N0,N1,N2,Fr(0),Fr(0))[0]
piv=[c for c in op if c]; op=[c/piv[0] for c in op]
den=1
for c in op: den = den*c.denominator//math.gcd(den,c.denominator)
op=[int(c*den) for c in op]
g=0
for c in op: g=math.gcd(g,abs(c))
op=[c//g for c in op]
# make the coefficient of the (R F_0)(G_2)-type term positive for a canonical sign
if sum(op) < 0 or (op[[i for i,c in enumerate(op) if c][0]] < 0):
    op=[-c for c in op]

def dl(D):
    i,j,s=D; out=""
    if i: out += "e_1^{%d}"%i if i>1 else "e_1"
    if j: out += "e_2^{%d}"%j if j>1 else "e_2"
    if s: out += "R^{%d}"%s if s>1 else "R"
    return out

def fac(name,m,D):
    body = "%s_{%d}"%(name,m)
    d = dl(D)
    return "%s%s"%(d,body) if d else body

lines=[]
for cf,t in zip(op,terms):
    if not cf: continue
    m,n,D1,D2,c = t
    s = "%+d"%cf if abs(cf)!=1 else ("+" if cf>0 else "-")
    lines.append("%s\\,(%s)(%s)"%(s,fac("F",m,D1),fac("G",n,D2)))
per=3
rows=[" ".join(lines[i:i+per]) for i in range(0,len(lines),per)]
out = "&\\ " + "\\\\\n&\\ ".join(rows) + "\n"
open("kgzform.tex","w").write(out)
print(out)
print("%% %d terms"%len(lines))
