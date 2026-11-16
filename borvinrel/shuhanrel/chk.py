"""Statement 5.3.10 of the paper: g(A(t)), A(t)=2E-t(2E-A), A of type Fin, t rational, t!=1.
Compare dims of g(A(t))_alpha (engine eng.py, over F_p) with the Gabber-Kac prediction:
relations [e_i,e_j]=0 for A_ij=0 and (ad e_i)^{1-A_ij(t)} e_j only when -A_ij(t) in Z_{>0}.
Independent prediction: rank 2 -> K e_i |x Free(V), V=<(ad e_i)^k e_j>, k<N;
no integral entry -> free partially commutative Lie algebra (clique formula)."""
import sys, itertools
from fractions import Fraction as F
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE, os.path.join(_HERE, "..", "evenpart")]
from eng import G

def series_inv_log_dims(Q, n, H):
    """Q: dict multideg->coef with Q[0]=1. Return d_alpha with prod(1-x^a)^{-d_a}=1/Q, |a|<=H."""
    degs = [a for h in range(1, H+1) for a in itertools.product(range(h+1), repeat=n) if sum(a) == h]
    # log(1/Q) = -log Q ; compute L = -log(Q) via L' relation: use logarithmic derivative wrt total degree
    # Euler operator: sum |a| L_a x^a = -(E Q)/Q.  P=1/Q first.
    zero = (0,)*n
    P = {zero: F(1)}
    for a in degs:
        s = F(0)
        for b, c in Q.items():
            if b == zero: continue
            r = tuple(x-y for x, y in zip(a, b))
            if min(r) >= 0 and r in P: s -= c*P[r]
        P[a] = s
    EQ = {b: sum(b)*c for b, c in Q.items() if b != zero}
    L = {}
    for a in degs:
        s = F(0)
        for b, c in EQ.items():
            r = tuple(x-y for x, y in zip(a, b))
            if min(r) >= 0 and r in P: s -= c*P[r]
        L[a] = s/sum(a)   # coefficient of x^a in log(1/Q)
    # log(1/Q) = sum_a d_a sum_k x^{ka}/k  -> Moebius
    d = {}
    for a in degs:
        s = L[a]
        for k in range(2, sum(a)+1):
            if all(x % k == 0 for x in a):
                b = tuple(x//k for x in a)
                s -= F(d[b], k)
        assert s.denominator == 1, (a, s)
        d[a] = int(s)
    return d

def predict(A, H):
    n = len(A); zero = (0,)*n
    integral = [(i, j) for i in range(n) for j in range(n) if i != j and A[i][j] != 0
                and A[i][j].denominator == 1]
    def u(i): return tuple(1 if t == i else 0 for t in range(n))
    if not integral:
        Q = {zero: F(1)}
        for C in itertools.chain.from_iterable(itertools.combinations(range(n), r) for r in (1, 2, 3, 4)):
            if all(A[i][j] == 0 for i in C for j in C if i != j):
                Q[tuple(1 if t in C else 0 for t in range(n))] = F((-1)**len(C))
        return series_inv_log_dims(Q, n, H)
    assert n == 2 and len(integral) == 1
    i, j = integral[0]; N = int(1 - A[i][j])
    # n_+ = K e_i |x Free(V): 1/Q = (1-x_i)^{-1} * 1/(1-V)
    V = {}
    for k in range(N):
        a = [0, 0]; a[i] = k; a[j] = 1; V[tuple(a)] = F(1)
    # Q = (1-x_i)(1-V)
    Q = {}
    for b1, c1 in [(zero, F(1)), (u(i), F(-1))]:
        for b2, c2 in [(zero, F(1))] + [(b, -c) for b, c in V.items()]:
            b = tuple(x+y for x, y in zip(b1, b2)); Q[b] = Q.get(b, 0) + c1*c2
    return series_inv_log_dims(Q, n, H)

FIN = {
 'A2': [[2,-1],[-1,2]], 'A3': [[2,-1,0],[-1,2,-1],[0,-1,2]],
 'D4': [[2,-1,0,0],[-1,2,-1,-1],[0,-1,2,0],[0,-1,0,2]],
 'B2': [[2,-2],[-1,2]], 'G2': [[2,-3],[-1,2]],
}
CASES = [('A2', F(1,2), 13), ('A2', F(2,3), 13), ('A2', F(3,2), 13),
         ('A3', F(1,2), 9), ('A3', F(4,3), 9), ('D4', F(1,2), 8), ('D4', F(2,3), 8),
         ('B2', F(1,2), 13), ('B2', F(1,3), 13), ('B2', F(4,3), 13),
         ('G2', F(1,3), 13), ('G2', F(2,3), 13), ('G2', F(1,2), 13)]
PRIMES = [1000003, 3000017]

def At(A, t):
    n = len(A)
    return [[F(2) if i == j else t*A[i][j] for j in range(n)] for i in range(n)]

total = 0; bad = 0
for name, t, H in CASES:
    A = At(FIN[name], t); n = len(A)
    pred = predict(A, H)
    for p in PRIMES:
        Am = [[x.numerator*pow(x.denominator, p-2, p) % p for x in r] for r in A]
        g = G(Am, [0]*n, p); g.build(H)
        mism = [(a, g.dim.get(a, 0), d) for a, d in pred.items() if g.dim.get(a, 0) != d]
        total += len(pred); bad += len(mism)
        tot = [sum(d for a, d in pred.items() if sum(a) == h) for h in range(1, H+1)]
        print(name, 't=%s' % t, 'p=%d' % p, 'H=%d' % H, 'multidegrees', len(pred),
              'mismatches', len(mism), mism[:3], 'dims by height', tot, flush=True)
print('TOTAL multidegree comparisons', total, 'mismatches', bad)
