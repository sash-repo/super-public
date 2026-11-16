"""Check of the hand formula for the t-constant part C of  L^b_{K_{g xi}} B - B L^a_{K_{g xi}},  g = t^(r), r = s,
B = x D_xi D_eta d_t^{s-1} + y d_t^s on F_{-s,mu} -> F_{s,mu}.  The constant part is read off as phi -> (Z phi)(t=0).
Used in the proof of Theorem 6.4.  Usage: python3 ccheck2.py p N"""
import sys, itertools
import os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from osp2 import Alg
p, N = int(sys.argv[1]), int(sys.argv[2])
A = Alg(p, N); P = A.P; n = A.n; add, mul, M = A.add, A.mul, A.M
DD = mul(A.DXI, A.DETA)
pw = [A.I]
for i in range(1, P): pw.append(mul(pw[-1], A.Dt))
def at0(Z):  # rows of t-degree 0
    return [Z[i] for i in range(4)]
Kxi = add(mul(A.Mxi, A.Dt), A.Deta, coef=[1, -1])
Y = lambda c: add(mul(mul(A.Mxi, A.Meta), A.Deta), A.Mxi, coef=[1, c % p])
W = add(A.I, mul(A.Meta, A.Deta), mul(A.Mxi, A.Dxi), coef=[1, 1, -1])
XE = mul(A.Mxi, A.Meta)
bad = cnt = 0
for s in range(2, P):
    g = A.fn(s); gp = A.fn(s - 1)
    for mu, x, y in itertools.product(range(p), (0, 1), (0, 1)):
        a, b = -s, s
        La = add(mul(M(g), Kxi), mul(M(gp), Y(a + mu)))
        Lb = add(mul(M(g), Kxi), mul(M(gp), Y(b + mu)))
        B = add(mul(DD, pw[s-1]), pw[s] if s < P else A.zero(), coef=[x, y])
        Z = add(mul(Lb, B), mul(B, La), coef=[1, -1])
        c = (mu - s) % p
        # hand formula
        Bk, Bk1, Bk2 = mul(A.Dxi, A.Deta), add(W, A.I, coef=[x, y]), add(A.zero(), XE, coef=[1, -x])
        Bk = add(A.zero(), Bk, coef=[1, x])
        t1 = mul(add(Bk1, mul(Bk2, A.Dt), coef=[1, s + 1]), Kxi)
        t2 = mul(add(Bk, mul(Bk1, A.Dt), mul(Bk2, mul(A.Dt, A.Dt)), coef=[1, s, (s+1)*s//2]), Y(c))
        C = add(t1, t2, coef=[-1, -1])
        red = add(A.Deta, mul(mul(A.Mxi, A.Dxi), A.Deta), mul(mul(XE, A.Deta), A.Dt), mul(A.Mxi, A.Dt),
                  coef=[(2 + c)*x + y, -(2 + c)*x, -(s + 2)*x - s*(1 + c)*x - s*y, -y*(1 + s*c)])
        cnt += 1
        bad += (at0(Z) != at0(C)) + (C != red)
print(f"p={p} N={N}: {cnt} cases, {bad} mismatches")
# s = 1, r = 2: constant part = -((2+c)x+y) xi eta d_eta - c y xi
bad = cnt = 0
s = 1; g = A.fn(2); gp = A.fn(1)
for mu, x, y in itertools.product(range(p), range(p), range(p)):
    a, b = -1, 1
    La = add(mul(M(g), Kxi), mul(M(gp), Y(a + mu))); Lb = add(mul(M(g), Kxi), mul(M(gp), Y(b + mu)))
    B = add(DD, A.Dt, coef=[x, y]); Z = add(mul(Lb, B), mul(B, La), coef=[1, -1])
    c = (mu - 1) % p
    red = add(mul(XE, A.Deta), A.Mxi, coef=[-((2 + c)*x + y), -c*y])
    cnt += 1; bad += at0(Z) != at0(red)
print(f"s=1,r=2: {cnt} cases, {bad} mismatches")
