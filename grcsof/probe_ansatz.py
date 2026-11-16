"""Corollary 2.11 of the paper: does the ansatz of Theorem 2.6(i) contain an
invariant operator when nu_1, nu_2 are not both 0?

The printed ansatz for the even operators consists of the terms
   sum phi^(a) psi^(b),   sum (D_th1 f)(D_th1 g) + (D_th2 f)(D_th2 g),
   sum (D_th1 D_th2 f)(D_th1 D_th2 g);
in the xi-eta realisation the middle one is D_xi (x) D_eta + D_eta (x) D_xi
(exactly, see the dictionary in Section 2 of the paper), and the ansatz contains NO terms
D_xi D_eta (x) 1 or 1 (x) D_xi D_eta.  So for order 2 the printed ansatz is
   A = < phi' psi, phi psi', D_xi(f)D_eta(g) + D_eta(f)D_xi(g) >   (3-dimensional)
while the full space of order-2 terms of charge 0 is 6-dimensional.
"""
from sympy import symbols, S, Matrix, expand, simplify, factor, solve, together
from invar import gen_binary, eqs_binary, matrix_of
l1,n1,l2,n2 = symbols('lambda_1 nu_1 lambda_2 nu_2')
FIELDS=('txi','teta')
terms = gen_binary(-2,0)
names=[t[0] for t in terms]
cs,eqs = eqs_binary(terms,l1,n1,l2,n2,l1+l2+2,n1+n2,0,nmax=3,fields=FIELDS)
M = matrix_of(eqs,cs); ns=M.nullspace()
print("order 2, charge 0: names =", names)
print("dim =", len(ns))
al,be = symbols('alpha beta')
v = [al*ns[0][i]+be*ns[1][i] for i in range(len(names))]
d = dict(zip(names,v))
# the paper's ansatz: no (Dxe (x) 1), no (1 (x) Dxe), and the two D-terms equal
conds = [together(d['(d^0Dxe)x(d^0D1)']), together(d['(d^0D1)x(d^0Dxe)']),
         together(d['(d^0Dx)x(d^0De)'] - d['(d^0De)x(d^0Dx)'])]
conds = [c.as_numer_denom()[0] for c in conds]
print("conditions for alpha*v1+beta*v2 to lie in the printed ansatz:")
for c in conds: print("   ", factor(c), "= 0")
sol = solve(conds+[al*be-al*be], [al,be], dict=True)
print("solutions in (alpha,beta):", sol)
print()
print("=> substituting nu_1=nu_2=0:")
conds0=[factor(c.subs({n1:0,n2:0})) for c in conds]
for c in conds0: print("   ", c, "= 0")
print("   solutions:", solve(conds0,[al,be],dict=True))
