"""The second avatar: curvature as a FUNCTION on the space of line elements.

On the space of unit tangent elements of a surface Sigma the tautological 1-form

        alpha = cos(th) dx + sin(th) dy      (normal coordinates at the point looked at)

is contact; its Reeb field is the geodesic spray R = cos(th) d_x + sin(th) d_y, and the
contact distribution D is framed by

        e_1 = d_th   (vertical),        e_2 = -sin(th) d_x + cos(th) d_y   (horizontal),

with  d alpha(e_1,e_2) = 1  and  [e_1,e_2] = -R  at the point (flat model).

A symmetric 2-tensor field A on Sigma has, beside its pullback (which is what legend.py
kills), a second avatar here: the FUNCTION

        kappa_A(x,th) = A(v,v)/g(v,v),      v = (cos th, sin th),

which for A = the second fundamental form is the normal curvature, i.e. Euler's formula
kappa = k_1 cos^2 + k_2 sin^2.  It is a genuine function, not a pullback: it depends on
the fiber.  The census of the contact-invariant pairings of two weighted densities gives
exactly one non-trivial family, of weighted order 2, whose value at lambda = mu = 0 is

        {F,G} = (e_1 F)(e_2 G) - (e_2 F)(e_1 G).

This file computes it on two curvature functions and reads off what it is.

Run:  python3 curv.py
"""
import sympy as sp

x, y, th = sp.symbols("x y theta", real=True)
c, s = sp.cos(th), sp.sin(th)

OK = [0, 0]


def check(name, cond):
    OK[0] += 1
    OK[1] += 0 if cond else 1
    print("   %-64s %s" % (name, "ok" if cond else "FAILED"))


def e1(F):
    return sp.diff(F, th)


def e2(F):
    return -s * sp.diff(F, x) + c * sp.diff(F, y)


def R(F):
    return c * sp.diff(F, x) + s * sp.diff(F, y)


def brk(F, G):
    return sp.simplify(sp.expand_trig(sp.simplify(e1(F) * e2(G) - e2(F) * e1(G))))


print("1) the frame and the form (flat model = normal coordinates at the point):")
# alpha(e_1) = alpha(e_2) = 0, alpha(R) = 1, d alpha(e_1,e_2) = 1 -- by hand:
check("alpha(e_2) = -cos sin + sin cos = 0", sp.simplify(-c * s + s * c) == 0)
check("alpha(R) = cos^2 + sin^2 = 1", sp.simplify(c ** 2 + s ** 2) == 1)
check("d alpha(e_1,e_2) = sin^2 + cos^2 = 1", sp.simplify(s ** 2 + c ** 2) == 1)

print()
print("2) a pullback has no fiber dependence, and two of them do not pair:")
f = sp.Function("f")(x, y)
g = sp.Function("g")(x, y)
check("e_1 (pi^* f) = 0", sp.simplify(e1(f)) == 0)
check("{pi^* f, pi^* g} = 0", sp.simplify(brk(f, g)) == 0)

print()
print("3) the curvature function of a symmetric 2-tensor A = (a,b,c):")
a1, b1, c1 = (sp.Function(n)(x, y) for n in ("a1", "b1", "c1"))
a2, b2, c2 = (sp.Function(n)(x, y) for n in ("a2", "b2", "c2"))
k1 = a1 * c ** 2 + 2 * b1 * c * s + c1 * s ** 2
k2 = a2 * c ** 2 + 2 * b2 * c * s + c2 * s ** 2
check("kappa_A is not a pullback: e_1 kappa_A =/= 0", sp.simplify(e1(k1)) != 0)

B = brk(k1, k2)
check("{kappa_A, kappa_B} =/= 0 for two generic tensors", sp.simplify(B) != 0)
check("antisymmetry: {A,B} = -{B,A}", sp.simplify(B + brk(k2, k1)) == 0)
check("{kappa_A, kappa_A} = 0", sp.simplify(brk(k1, k1)) == 0)

print()
print("4) what the bracket is: its Fourier content in the fiber variable.")
print("   kappa_A = H_A + Re(z_A e^{2i th}) has the harmonics 0, +-2 -- it descends to")
print("   the space of LINE elements.  e_2 shifts a harmonic by +-1, e_1 = d_th keeps it.")
Bex = sp.expand(sp.expand_trig(sp.expand(B)))
harm = {}
for n in range(0, 8):
    cn = sp.simplify(sp.integrate(Bex * sp.cos(n * th), (th, 0, 2 * sp.pi))
                     / (sp.pi if n else 2 * sp.pi))
    sn = sp.simplify(sp.integrate(Bex * sp.sin(n * th), (th, 0, 2 * sp.pi)) / sp.pi)
    if cn != 0 or sn != 0:
        harm[n] = (cn, sn)
print("   harmonics present:", sorted(harm))
check("the mean over the fiber vanishes", 0 not in harm)
check("only ODD harmonics occur (1,3,5)", set(harm) == {1, 3, 5})
print("   the first harmonic, cos(th)-coefficient:")
sp.pprint(sp.simplify(harm[1][0]))

print()
print("5) why odd: the output is a density of weight 1, not a function.")
print("   Under th -> th + pi the form alpha changes sign, so a section of L over the")
print("   space of line elements is an ODD function on the unit tangent bundle --")
print("   which is what the harmonics say.")
check("kappa_A is even (descends to the line elements)",
      sp.simplify(k1.subs(th, th + sp.pi) - k1) == 0)
check("the bracket is odd (a section of L there)",
      sp.simplify(sp.expand_trig(Bex.subs(th, th + sp.pi) + Bex)) == 0)

print()
print("checks: %d run, %d failed" % (OK[0], OK[1]))
