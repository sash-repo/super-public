"""Which of the osp(2|2)-invariant maps F_{a,mu} -> F_{b,nu} of the closed form (Proposition 6.3) are invariant
under the whole k(1;N|2)?  Every candidate family is tested against all K_f, f a basis monomial of O(1;N|2).
Usage: python3 kfull2.py p N      (weights a, mu in F_p)"""
import sys, itertools
import os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import nullspace
from osp2 import Alg
p, N = int(sys.argv[1]), int(sys.argv[2])
A = Alg(p, N); P = A.P; n = A.n
DD = A.mul(A.DXI, A.DETA)
c2 = lambda f: [(x - y) % p for x, y in zip(A.apply(A.Dt, f), A.apply(DD, f))]
allk = [(A.fn(k//4, (k%4)//2, k%2), A.par[k]) for k in range(n)]
Ks = [A.K(f, pf) for f, pf in allk]
def L(lam, mu):
    return [(A.add(K, A.M(A.apply(A.Dt, f)), A.M(c2(f)), coef=[1, lam % p, mu % p]), pf) for K, (f, pf) in zip(Ks, allk)]
pw = [A.I]
for i in range(1, P): pw.append(A.mul(pw[-1], A.Dt))
inv = lambda z: pow(z % p, p-2, p)
cands = []   # (name, a, mu, b, nu, parity, list of operator matrices spanning)
for a, m in itertools.product(range(p), repeat=2):
    cands.append((f"d_t^(pj) on F_({a},{m})", a, m, a, m, 0, [pw[p*j] for j in range(P//p)]))
for s in range(1, P+1):
    a = (-s) % p
    for m in range(p):
        if s % p:
            ops = [A.add(A.mul(DD, pw[s-1]), pw[s] if s < P else A.zero(), coef=[1, (-1 - m*inv(s)) % p])]
        elif m == 0:
            ops = [A.add(A.mul(DD, pw[s-1]), pw[s] if s < P else A.zero(), coef=[1, -1])]
        else: continue
        cands.append((f"Delta_(s={s}) F_({a},{m})->F_({s%p},{m})", a, m, s % p, m, 0, ops))
for a in range(p):
    cands.append((f"D_xi d_t^(pj) F_({a},{-a%p})->F_({(a+1)%p},{(-a-1)%p})", a, -a % p, (a+1) % p, (-a-1) % p, 1, [A.mul(A.DXI, pw[p*j]) for j in range(P//p)]))
    cands.append((f"D_eta d_t^(pj) F_({a},{a})->F_({(a+1)%p},{(a+1)%p})", a, a, (a+1) % p, (a+1) % p, 1, [A.mul(A.DETA, pw[p*j]) for j in range(P//p)]))
cache = {}
def getL(w):
    if w not in cache: cache[w] = L(*w)
    return cache[w]
for name, a, m, b, nu, par, ops in cands:
    La, Lb = getL((a, m)), getL((b, nu))
    rows = []
    for (X, px), (Y, _) in zip(La, Lb):
        sg = -1 if (par and px) else 1
        # sum_i c_i (Y T_i - sg T_i X) = 0
        cols = [[x for r in A.add(A.mul(Y, T), A.mul(T, X), coef=[1, -sg]) for x in r] for T in ops]
        rows += [list(r) for r in zip(*cols)]
    ns = nullspace([r for r in rows if any(r)], p) if any(any(r) for r in rows) else [[1]*len(ops)]
    if ns and not (par == 0 and a == b and m == nu and all(len(v) == len(ops) for v in ns) and len(ns) == 1 and ops[0] == A.I and len(ops) == 1):
        print(f"{name}: k-invariant dim {len(ns)}", [v for v in ns])
