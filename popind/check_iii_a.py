# Item (iii), part A: the linear picture in signature (1,1) over R.
# Null basis e1,e2:  <e1|e1>=<e2|e2>=0, <e1|e2>=1.  Gram = [[0,1],[1,0]].
from sympy import *

G = Matrix([[0, 1], [1, 0]])
mu, nu, lam, x, y = symbols('mu nu lam x y', nonzero=True)

def refl(e):
    """R_e v = v - 2<v|e>/<e|e> e, as a matrix acting on column vectors."""
    ee = (e.T * G * e)[0, 0]
    return eye(2) - (2 / ee) * e * (e.T * G)

# 1. Reflection in e=(a,b) is the antidiagonal S_mu with mu = -a/b.
a, b = symbols('a b', nonzero=True)
R = simplify(refl(Matrix([a, b])))
print("R_(a,b) =", R, "   expected antidiag(-a/b, -b/a)")

S = lambda m: Matrix([[0, m], [Rational(1, 1) / m, 0]])
print("check S_mu with mu=-a/b :", simplify(R - S(-a / b)) == zeros(2, 2))

# 2. Every antidiagonal S_mu is an isometry, an involution, a reflection.
Sm = S(mu)
print("S_mu isometry:", simplify(Sm.T * G * Sm - G) == zeros(2, 2),
      "  involution:", simplify(Sm * Sm - eye(2)) == zeros(2, 2))
print("fixed line of S_mu: ", nsimplify((Sm - eye(2)).nullspace()[0].T),
      " root line:", nsimplify((Sm + eye(2)).nullspace()[0].T))
e_root = Matrix([-mu, 1])
print("<root|root> =", simplify((e_root.T * G * e_root)[0, 0]), " => eps=+ iff mu<0")

# 3. Product of two reflections is the boost T_{mu/nu}.
print("S_mu S_nu =", simplify(S(mu) * S(nu)), " = diag(mu/nu, nu/mu)")
T = lambda l: diag(l, 1 / l)
print("check:", simplify(S(mu) * S(nu) - T(mu / nu)) == zeros(2, 2))
print("perpendicular <=> nu=-mu <=> product = -id:",
      simplify(S(mu) * S(-mu)) == -eye(2))
print("finite order of T_lam <=> lam=+-1 (eigenvalues lam, 1/lam):",
      T(lam).eigenvals())

# 4. Rank-1 translation lattices are impossible when Lin W is infinite:
#    T_lam w = +-w forces w to be an eigenvector, i.e. isotropic, with lam=+-1.
w = Matrix([x, y])
sol = solve(list(T(lam) * w - w) + list(T(lam) * w + w), [x, y], dict=True)
print("T_lam w = +-w  =>", sol, " (only the null lines, with lam=+-1)")
