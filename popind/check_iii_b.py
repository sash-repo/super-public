# Item (iii), part B: the crystallographic rank-2 groups.
# A = [[2,a12],[a21,2]] integral; alpha_1,alpha_2 a Z-basis of L; R_i alpha_j = alpha_j - a_ij alpha_i.
from sympy import *


def data(a12, a21):
    R1 = Matrix([[-1, -a12], [0, 1]])          # columns = images of alpha_1, alpha_2
    R2 = Matrix([[1, 0], [-a21, -1]])
    d1, d2 = symbols('d1 d2')
    sol = solve(Eq(d1 * a12, d2 * a21), [d1, d2], dict=True)
    return R1, R2


def gram(a12, a21, d1, d2):
    return Matrix([[2 * d1, d1 * a12], [d2 * a21, 2 * d2]])


for (a12, a21, d1, d2, tag) in [(-3, -3, 3, 3, "GCM (2,-3;-3,2), ab=9>4"),
                                (-1, -6, 6, 1, "GCM (2,-1;-6,2), ab=6>4"),
                                (1, -1, 1, -1, "mixed sign (2,1;-1,2)"),
                                (2, -3, 3, -2, "mixed sign (2,2;-3,2)")]:
    R1, R2 = data(a12, a21)
    Gm = gram(a12, a21, d1, d2)
    ok = (R1.T * Gm * R1 == Gm) and (R2.T * Gm * R2 == Gm)
    P = R1 * R2
    print(f"{tag}: symmetrizable Gram={Gm.tolist()} det={Gm.det()} "
          f"isometries={ok} tr(R1R2)={P.trace()} det={P.det()} "
          f"|eig|={[nsimplify(v) for v in P.eigenvals()]}")
    print("   <a1|a1>=%s  <a2|a2>=%s  => eps=(%s,%s)"
          % (Gm[0, 0], Gm[1, 1], sign(Gm[0, 0]), sign(Gm[1, 1])))

print()
# The mixed-sign group of A=(2,1;-1,2): its eps=+ reflections generate an index-2 subgroup.
a12, a21, d1, d2 = 1, -1, 1, -1
R1, R2 = data(a12, a21)
Gm = gram(a12, a21, d1, d2)
al1, al2 = Matrix([1, 0]), Matrix([0, 1])
sq = lambda v: (v.T * Gm * v)[0, 0]
print("squares: alpha1:%s alpha2:%s  R2*alpha1=%s square %s"
      % (sq(al1), sq(al2), (R2 * al1).T.tolist(), sq(R2 * al1)))
b1, b2 = al1, R2 * al1                     # the two adjacent eps=+ roots
c12 = 2 * (b2.T * Gm * b1)[0, 0] / sq(b1)
c21 = 2 * (b1.T * Gm * b2)[0, 0] / sq(b2)
print("Cartan matrix of the eps=+ pair: [[2,%s],[%s,2]]  (product %s)" % (c12, c21, c12 * c21))
refl = lambda e: eye(2) - (2 / sq(e)) * e * (e.T * Gm)
Q1, Q2 = refl(b1), refl(b2)
print("tr(Q1Q2)=%s  vs  tr((R1R2)^2)=%s" % ((Q1 * Q2).trace(), ((R1 * R2) ** 2).trace()))
print("Q1 = R1 ?", Q1 == R1, "  Q2 = R2 R1 R2 ?", Q2 == R2 * R1 * R2)
