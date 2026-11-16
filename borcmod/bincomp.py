"""Which k(1;N|1)-invariant binary maps are compositions u3 o J_l o (u1 x u2), u in {id, d, int}, l in {0,1} (or l in
{0,1,2} with the fourth argument J2)?  Prints every case of the census (output of binkfull.py) not spanned by
compositions, then the number of such cases.
Usage: python3 binkfull.py p N > binkfull-p-N.txt; python3 bincomp.py p N binkfull-p-N.txt J2
Operators m o Q(D1,D2), Q in normal order D1^i D2^j (D1 D2 = -D2 D1), D^{2P}=0."""
import sys, re, itertools
import os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import rank
p, N, log = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
USEJ2 = len(sys.argv) > 4
P = p**N
def mulQ(X, Y):
    out = {}
    for (i1, j1), v in X.items():
        for (i2, j2), w in Y.items():
            if i1+i2 >= 2*P or j1+j2 >= 2*P: continue
            s = (-1)**(j1*i2)
            out[(i1+i2, j1+j2)] = (out.get((i1+i2, j1+j2), 0) + s*v*w) % p
    return {k: v for k, v in out.items() if v}
def powQ(X, r):
    R = {(0, 0): 1}
    for _ in range(r): R = mulQ(R, X)
    return R
D1, D2 = {(1, 0): 1}, {(0, 1): 1}
DS = {(1, 0): 1, (0, 1): 1}
def unary(w):   # list of (Q-exponent r, new weight) for invariant unary maps from F_w
    out = [(0, w)]
    if w % p == 0: out.append((1, (w+1) % p))
    if w % p == 1: out.append((2*P-1, (w-1) % p))
    return out
def candidates(a, b):
    res = []
    for (r1, a1), (r2, b1) in itertools.product(unary(a), unary(b)):
        pre = mulQ(powQ(D1, r1), powQ(D2, r2))
        J2 = {(0, 2): (-a1) % p, (1, 1): 1, (2, 0): b1 % p}
        mids = [({(0, 0): 1}, (a1+b1) % p), ({(1, 0): b1 % p, (0, 1): (-a1) % p}, (a1+b1+1) % p)]
        if USEJ2: mids.append((J2, (a1+b1+2) % p))
        for mid, c1 in mids:
            if not any(mid.values()): continue
            for r3, c in unary(c1):
                Q = mulQ(mulQ(powQ(DS, r3), mid), pre)
                if Q: res.append((c, Q))
    return res
census = {}
for l in open(log):
    m = re.search(r'a=(\d+) b=(\d+) c=(\d+) (\w+): (\{.*\})', l)
    a, b, c, par = int(m[1]), int(m[2]), int(m[3]), m[4] == 'odd'
    census.setdefault((a, b, c, par), []).append({k: v % p for k, v in eval(m[5]).items()})
keys = [(i, j) for i in range(2*P) for j in range(2*P)]
vec = lambda Q: [Q.get(k, 0) % p for k in keys]
extra = 0
for (a, b, c, par), vs in sorted(census.items()):
    cands = [Q for cc, Q in candidates(a, b) if cc == c and all((i+j) % 2 == par for i, j in Q)]
    rc = rank([vec(Q) for Q in cands], p) if cands else 0
    rall = rank([vec(Q) for Q in cands] + [vec(v) for v in vs], p)
    if rall != len(vs) or rc != rall:
        extra += 1
        print(f"a={a} b={b} c={c} {'odd' if par else 'even'}: census {len(vs)}, compositions {rc}, joint {rall}:", vs)
print("cases not explained:", extra)
