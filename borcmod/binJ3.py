"""J_3 on a=0, b=0, a+b=-2 vs J_2 o (d x 1), J_2 o (1 x d), d o J_2 (as coefficient vectors in B_ij).
Usage: python3 binJ3.py   (each line: J_3 restricted | the composition; they are proportional)"""
import sympy as sp
from binC import J, a, b
from binop import mul, M, add
def asQ(cs): return {(0, 0, i, j): v for (i, j), v in cs.items()}
def sub(cs, s): return {k: sp.factor(sp.sympify(v).subs(s)) for k, v in cs.items() if sp.sympify(v).subs(s) != 0}
J3 = J(3)
# J_2 o (d x 1): D_1 applied first, J_2 weights (1,b)
c1 = mul(asQ(sub(J(2), {a: 1})), M(0, 0, 1, 0)); print("a=0:", sub(J3, {a: 0}), "|", c1)
c2 = mul(asQ(sub(J(2), {b: 1})), M(0, 0, 0, 1)); print("b=0:", sub(J3, {b: 0}), "|", c2)
c3 = mul(add(M(0, 0, 1, 0), M(0, 0, 0, 1)), asQ(sub(J(2), {b: -2-a}))); print("a+b=-2:", sub(J3, {b: -2-a}), "|", c3)
