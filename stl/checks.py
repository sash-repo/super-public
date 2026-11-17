"""Section 10: the sl(lambda)-hierarchy dL/dt_m = [(L^{m/lambda})_+, L], L = D^lambda + sum u_k D^{lambda-k}.
1. the sl-condition u_1 = 0 is preserved; 2. Theorem 10.4.3: at lambda = n the subspace u_k = 0 (k > n)
is invariant; 3. at lambda = 2 the flow t_3 is the KdV equation (275), at lambda = 3 the Boussinesq flow agrees
with the integer Gelfand-Dickey computation; 4. q = L^{1/lambda} evolves by KP; 5. the flows t_2,
t_3 commute; 6. Res L^{m/lambda} are conserved densities; 7. Statement 10.6.2: the Dirac-reduced
bracket (278) is Virasoro with central term lambda(lambda^2-1)/12.  Prints the case count of each.
Run:  python3 checks.py"""
import sympy as sp
from psdo import Sym, comm, jet, Dx, Dxn, lam
from hier import build, invert, qsubs, flows
import adler

N = 8
cc, qu, fl = flows(N, 4)
ok = 0

def zero(e):
    return sp.simplify(sp.expand(e)) == 0

def restrict(e, n):
    """lam = n and u_k = 0 for k > n"""
    sub = {s: 0 for s in sp.sympify(e).free_symbols if s.name.startswith('u') and int(s.name[1:].split('_')[0]) > n}
    return sp.expand(sp.S(sp.S(e).xreplace(sub)).subs(lam, n))

# 1. u_1 stays 0, i.e. sl is preserved, all flows
for m in fl:
    assert zero(fl[m][1]); ok += 1
print('1. du1/dt_m = 0 for m=1..4:', ok)

# 2. the Gelfand-Dickey locus is invariant at lam = n
c2 = 0
for n in (2, 3, 4, 5):
    for m in fl:
        for k in fl[m]:
            if k > n:
                assert restrict(fl[m][k], n) == 0, (n, m, k); c2 += 1
print('2. u_k=0 (k>n) invariant at lam=n, n=2..5:', c2)

# 3. the reductions: KdV and Boussinesq
u, ux, uxxx = jet('u',2,0), jet('u',2,1), jet('u',2,3)
assert restrict(fl[2][2], 2) == 0
assert restrict(fl[3][2], 2) == sp.Rational(1,4)*uxxx + sp.Rational(3,2)*u*ux
print('3a. lam=2: t2 trivial, t3 = KdV  u_t = u_xxx/4 + 3/2 u u_x')
# Boussinesq: L = D^3 + u D + v; compare with direct Gelfand-Dickey at n=3
def gd(n, m):
    Ln = Sym(n, 0, {0: 1, **{k: jet('u', k, 0) for k in range(2, n + 1)}}, N)
    Qn, ccn = build(N)  # Q symbolic in q; instead compute L^{m/n} by solving directly:
    return Ln
b2 = {k: restrict(fl[2][k], 3) for k in (2, 3)}
print('3b. lam=3, t2:', b2)
# independent integer computation of [(L^{2/3})_+, L] at n=3
Q, ccq = build(N); qu3 = {j: sp.expand(e.subs(lam, 3)) for j, e in qu.items()}
f3 = lambda e: restrict(qsubs(qu)(e), 3)
Qu = Sym(1, 0, {k: f3(v) for k, v in Q.c.items()}, N)
L3 = Sym(3, 0, {0: 1, 2: jet('u',2,0), 3: jet('u',3,0)}, N)
C = comm((Qu * Qu).plus(), L3)
for k in (2, 3):
    assert sp.expand(C.c.get(2 + k, 0) - b2[k]) == 0
assert sp.expand((Qu*Qu*Qu).plus().c.get(0) - 1) == 0
# and (Q^3)_+ = L3 exactly
Q3p = (Qu*Qu*Qu).plus()
assert all(sp.expand(Q3p.c.get(k,0) - L3.c.get(k,0)) == 0 for k in range(0, 4))
print('3c. lam=3: Q^3 = L, and the t2-flow agrees with the integer Gelfand-Dickey computation')

# 4. KP equivalence: the q's computed from u's evolve by KP: dQ/dt_m = [(Q^m)_+, Q]
Qs = Sym(1, 0, {0: 1, **{j + 1: jet('q', j, 0) for j in range(1, N)}}, N)
c4 = 0
for m in (2, 3):
    P = Qs
    for _ in range(m - 1): P = P * Qs
    K = comm(P.plus(), Qs)          # base m+1; coeff of D^{-j} is K.c[m+1+j]
    for j in range(1, N - m - 1):
        kp = qsubs(qu)(K.c.get(m + 1 + j, 0))
        # chain rule: dq_j/dt = sum d q_j / d u_{k,i} * Dx^i(du_k/dt)
        e = qu[j]; dt = 0
        for s in e.free_symbols - {lam}:
            k, i = map(int, s.name[1:].split('_'))
            dt += sp.diff(e, s) * Dxn(fl[m][k], i)
        assert zero(dt - kp), (m, j); c4 += 1
print('4. q = L^{1/lam} evolves by KP (m=2,3):', c4, 'coefficients')

# 5. commuting flows t2, t3 on u2, u3
def dflow(e, m):
    r = 0
    for s in sp.S(e).free_symbols - {lam}:
        k, i = map(int, s.name[1:].split('_'))
        r += sp.diff(e, s) * Dxn(fl[m][k], i)
    return sp.expand(r)
c5 = 0
for k in (2, 3):
    assert zero(dflow(fl[2][k], 3) - dflow(fl[3][k], 2)); c5 += 1
print('5. [d/dt2, d/dt3] = 0 on u2, u3:', c5)

# 6. conserved densities Res L^{m/lam} (coefficient of D^{-1} in Q^m): time derivative exact
def euler(e):
    """variational derivatives w.r.t. every u_k; all zero iff e is a total derivative"""
    e = sp.expand(e); out = []
    ks = {int(s.name[1:].split('_')[0]) for s in e.free_symbols - {lam}}
    for k in ks:
        r = 0
        for i in range(0, 12):
            s = jet('u', k, i)
            d = sp.diff(e, s)
            if d != 0:
                r += (-1) ** i * Dxn(d, i)
        out.append(sp.expand(r))
    return out
c6 = 0
P = Sym(0, 0, {0: 1}, N)
Qu = Sym(1, 0, {k: qsubs(qu)(v) for k, v in Q.c.items()}, N)
for m in (1, 2, 3):
    P = P * Qu
    h = sp.expand(P.c.get(m + 1, 0))   # coeff of D^{-1}
    for t in (2, 3):
        if all(int(s.name[1:].split('_')[0]) <= N - t for s in sp.S(h).free_symbols - {lam}):
            assert all(zero(x) for x in euler(dflow(h, t))), (m, t); c6 += 1
print('6. d/dt Res L^{m/lam} exact (m=1..3, t=2,3):', c6)

# 7. the Virasoro bracket of the reduced Adler map
P11 = adler.asop(adler.P(1, 1)); P12 = adler.asop(adler.P(1, 2))
P21 = adler.asop(adler.P(2, 1)); P22 = adler.asop(adler.P(2, 2))
z = {jet('u',1,i): 0 for i in range(6)}
a11 = P11[1]; a12 = P12[2].subs(z); a21 = P21[2].subs(z)
assert set(P11) == {1} and a11 == -lam
c3 = sp.factor(P22[3] - a21 * a12 / a11)
assert sp.expand(c3 - lam*(lam**2-1)/12) == 0
assert sp.expand(P22[1].subs(z) - 2*jet('u',2,0)) == 0 and sp.expand(P22[0].subs(z) - jet('u',2,1)) == 0
print('7. Dirac-reduced {u2,u2} = lam(lam^2-1)/12 d^3 + 2u d + u_x;  c =', c3)
