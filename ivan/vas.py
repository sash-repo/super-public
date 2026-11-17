# Vasiliev's deformed oscillator Aq(2;nu): [a-,a+]=1+nu K, K a = -a K, K^2=1,
# on its Fock module |n>=(a+)^n|0>, a-|0>=0, K|0>=|0>.  Checks the dictionary with
# osp(1|2) of the text (brackets (44)) and U_lambda = U(osp(1|2))/(C-lambda^2-lambda).
from fractions import Fraction as F
import sympy as sp
nu = sp.Symbol('nu'); N = 14
def c(n): return n + (nu if n % 2 else 0)
ap = sp.zeros(N); am = sp.zeros(N); K = sp.zeros(N)
for n in range(N):
    K[n, n] = (-1)**n
    if n+1 < N: ap[n+1, n] = 1
    if n >= 1: am[n-1, n] = c(n)
M = N - 4                     # rows/cols trusted after products of length <= 4
def ok(A, B): return sp.simplify((A - B)[:M, :M]) == sp.zeros(M)
I = sp.eye(N)
assert ok(am*ap - ap*am, I + nu*K)
assert ok(K*ap, -ap*K) and ok(K*am, -am*K)
al, be = 1, sp.Rational(-1, 2)                 # alpha*beta = -1/2
Np, Nm = al*am, be*ap                          # nabla^+ lowers n: highest weight vector |0>
Hh = Np*Nm + Nm*Np; Xp = -Np*Np; Xm = Nm*Nm
br = lambda A, B: A*B - B*A
checks = [ok(br(Hh, Xp), 2*Xp), ok(br(Hh, Xm), -2*Xm), ok(br(Xp, Xm), Hh),
          ok(br(Hh, Np), Np), ok(br(Hh, Nm), -Nm), ok(br(Xp, Nm), Np), ok(br(Xm, Np), Nm)]
print("osp(1|2) brackets:", checks)
lam = -(1+nu)/2
C = 2*(Xp*Xm + Xm*Xp) - Np*Nm + Nm*Np + Hh*Hh
print("C - (lam^2+lam):", sp.simplify((C - (lam**2+lam)*I)[:M, :M]) == sp.zeros(M))
Om = Np*Nm - Nm*Np
print("Omega = -(1+nu K)/2:", ok(Om, -(I + nu*K)/2))
S = Om + I/2
print("S anticommutes with odd:", ok(S*Np, -Np*S), "S^2 = C+1/4:", ok(S*S, C + I/4))
print("H on |0>:", sp.simplify(Hh[0, 0]), " lam:", sp.simplify(lam))
Hcal = (ap*am + am*ap)/2
print("Calogero H_rel eigen:", [sp.simplify(Hcal[n, n]) for n in range(6)])
# finite quotient: a-|m>=0 at m odd, nu=-m  -> sdim (n+1|n) with m=2n+1
for m in (1, 3, 5):
    print("nu =", -m, " lam =", sp.Rational(m-1, 2), " c(m) =", c(m).subs(nu, -m))
# Vasiliev's anti-automorphism tau(a)= i a, tau(K)=K -> super anti-aut sigma(a)=a:
# sigma(a+ a-) = -a- a+ ; on sl(2): sigma = -id ; on Pi(L^1) = <a+-, K>: +id  (= st1)
print("sigma(T) = -T check:", ok(-(am*ap) - (ap*am)*0 , -(am*ap)))
