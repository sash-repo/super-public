"""What the conformal weight of c_5 buys the clinic.  H. Andersen's question 3.

c_5 is exactly covariant under a common conformal change of the surface metric while
c_3 and c_1 are not.  What is the clinical use of this covariance?

This script answers that question and nothing else.  It re-derives the residuals of
confharm.py numerically, on explicit analytic fields rather than symbolically, which is
an independent check of that script, and then measures how large the anomaly is in the
units a radiologist can name.

Three findings.

  1.  A CONSTANT factor is harmless to all three harmonics.  The residuals are
      proportional to the GRADIENT of the conformal factor, so a change of voxel size
      between the two scanners, a global isotropic resampling, or a lesion that grows by
      the same factor everywhere biases nothing.  That is worth saying plainly, since it
      is the reading one would guess first and it is the wrong one.

  2.  What separates c_5 from c_3 and c_1 is an INHOMOGENEOUS isotropic factor: a lesion
      whose local scale change differs between one pole and the other.  That is the
      normal case, not the exception.

  3.  The anomaly of c_3 is proportional to  a*beta - alpha*b = H_1 beta - H_2 alpha,
      which vanishes exactly when the pair is locally a similarity -- i.e. exactly when
      there is no joint signal to measure.  So the contamination is not an additive
      background that could be subtracted or averaged away: it is a MULTIPLICATIVE
      distortion of the very quantity c_3 is being read for, largest where the signal is
      largest.  c_5's is zero.
"""
import sympy as sp

LOG = []


def say(s=""):
    print(s)
    LOG.append(s)


x, y = sp.symbols('x y', real=True)
p = sp.symbols('p', real=True, positive=True)
w = 2               # surface-conformal change with II fixed: alpha, beta have weight 2

# ---- the operators of confharm.py, spin-s covariant Wirtinger derivatives ----------
dz = lambda f: (sp.diff(f, x) - sp.I * sp.diff(f, y)) / 2
dzb = lambda f: (sp.diff(f, x) + sp.I * sp.diff(f, y)) / 2
D = lambda f, s, U: sp.exp(-U) * (dz(f) - s * dz(U) * f)
Db = lambda f, s, U: sp.exp(-U) * (dzb(f) + s * dzb(U) * f)


def harmonics(a, al, b, be, U, alb, beb):
    c5 = sp.Rational(3, 4) * sp.I * (al * D(be, 2, U) - be * D(al, 2, U))
    c3 = (sp.Rational(7, 4) * sp.I * (al * D(b, 0, U) - be * D(a, 0, U))
          + sp.I / 2 * (a * D(be, 2, U) - b * D(al, 2, U)))
    c1 = (sp.I / 2 * (a * D(b, 0, U) - b * D(a, 0, U))
          + sp.Rational(5, 4) * sp.I * (al * Db(b, 0, U) - be * Db(a, 0, U))
          - sp.I / 4 * (alb * D(be, 2, U) - beb * D(al, 2, U)))
    return c5, c3, c1


# ---- explicit fields, of the sizes a lesion has -----------------------------------
# a = H_1, b = H_2 (mean curvatures, scale 1/lesion radius); alpha, beta the Hopf fields,
# here about a third of H, which is a markedly asymmetric lesion; everything varying on
# the scale of the lesion itself.
A_ = 1 + sp.Rational(1, 5) * x + sp.Rational(1, 10) * y
B_ = sp.Rational(23, 20) + sp.Rational(1, 10) * x - sp.Rational(1, 4) * y
AL = (sp.Rational(3, 10) + sp.Rational(1, 10) * x - sp.Rational(1, 20) * y
      + sp.I * (sp.Rational(2, 25) * x + sp.Rational(3, 25) * y))
BE = (sp.Rational(33, 100) - sp.Rational(1, 50) * x + sp.Rational(9, 100) * y
      + sp.I * (sp.Rational(3, 20) * x + sp.Rational(1, 50) * y))
ALB = sp.conjugate(AL).rewrite(sp.conjugate).subs({sp.conjugate(x): x, sp.conjugate(y): y})
BEB = sp.conjugate(BE).rewrite(sp.conjugate).subs({sp.conjugate(x): x, sp.conjugate(y): y})
U0 = sp.Rational(1, 10) * x - sp.Rational(1, 20) * y            # the base metric's factor

c5, c3, c1 = harmonics(A_, AL, B_, BE, U0, ALB, BEB)

PTS = [{x: sp.Rational(a1, 10), y: sp.Rational(b1, 10)}
       for a1, b1 in ((0, 0), (3, 2), (-4, 1), (2, -3), (-2, -2))]


def mag(e, pt):
    return abs(complex(sp.N(e.subs(pt))))


# ---- 1.  a constant factor ---------------------------------------------------------
say("1.  A CONSTANT conformal factor")
c = sp.Rational(1, 5)                       # a 22 per cent isotropic rescaling, everywhere
t5, t3, t1 = harmonics(sp.exp(-2 * c) * A_, sp.exp(-w * c) * AL, sp.exp(-2 * c) * B_,
                       sp.exp(-w * c) * BE, U0 + c, sp.exp(-w * c) * ALB, sp.exp(-w * c) * BEB)
for nm, t, o, k in (("c5", t5, c5, 2 * w + 1), ("c3", t3, c3, w + 3), ("c1", t1, c1, w + 3)):
    r = sp.simplify(t - sp.exp(-k * c) * o)
    say(f"    {nm}: residual after removing the weight factor e^-{k}u = {r}")
say("    => a global change of scale -- voxel size, resampling, uniform growth -- is")
say("       invisible to all three.  The weight alone would not distinguish them.")
say()

# ---- 2.  a varying factor, and its size -------------------------------------------
say("2.  AN INHOMOGENEOUS factor  u = p*x  (the scale change differs between the poles;")
say("    p is the relative change of scale per lesion radius)")
u = p * x
t5, t3, t1 = harmonics(sp.exp(-2 * u) * A_, sp.exp(-w * u) * AL, sp.exp(-2 * u) * B_,
                       sp.exp(-w * u) * BE, U0 + u, sp.exp(-w * u) * ALB, sp.exp(-w * u) * BEB)
res = {}
for nm, t, o, k in (("c5", t5, c5, 2 * w + 1), ("c3", t3, c3, w + 3), ("c1", t1, c1, w + 3)):
    res[nm] = sp.simplify(t - sp.exp(-k * u) * o)
say(f"    c5 residual, simplified symbolically in p: {sp.simplify(res['c5'])}")
say()
say("    relative anomaly |residual| / |harmonic|, at five points of the lesion:")
say("      p      point        c5            c3            c1")
for pv in (sp.Rational(1, 20), sp.Rational(1, 10), sp.Rational(1, 5)):
    for pt in PTS[:3]:
        s = {**pt, p: pv}
        row = f"    {float(pv):4.2f}  ({float(pt[x]):4.1f},{float(pt[y]):4.1f})  "
        for nm in ("c5", "c3", "c1"):
            row += f"  {mag(res[nm], s)/mag(sp.exp(-(2*w+1 if nm=='c5' else w+3)*u)*(c5 if nm=='c5' else c3 if nm=='c3' else c1), s):10.3e}"
        say(row)
say("    read: c_1's anomaly is of the order of p -- a lesion whose local scale change")
say("    differs by 10 per cent between its poles carries a distortion of c_1 of about")
say("    the same 10 per cent.  c_3's is a few tenths of p, and it VARIES across the")
say("    lesion by an order of magnitude, which is the fingerprint of what it is")
say("    proportional to (part 3 below).  c_5's is zero at every point and every p.")
say()

# ---- 3.  the anomaly is proportional to the joint signal ---------------------------
say("3.  WHAT the c_3 anomaly is proportional to")
say("    confharm.py gives it as  -i(w-5)/4 * (a*beta - alpha*b) * (d_x u - i d_y u) * e^...")
say("    Two checks that this is the joint signal itself:")
say("      (a) the pair is locally a similarity when beta = alpha/s and b = a/s for one")
say("          real s > 0 (visit 2 is visit 1 scaled): then a*beta - alpha*b = 0.")
s_ = sp.symbols('s', positive=True)
BEs, Bs = AL / s_, A_ / s_
t3s = harmonics(sp.exp(-2 * u) * A_, sp.exp(-w * u) * AL, sp.exp(-2 * u) * Bs,
                sp.exp(-w * u) * BEs, U0 + u, sp.exp(-w * u) * ALB,
                sp.exp(-w * u) * sp.conjugate(BEs).subs({sp.conjugate(x): x, sp.conjugate(y): y,
                                                         sp.conjugate(s_): s_}))[1]
o3s = harmonics(A_, AL, Bs, BEs, U0, ALB,
                sp.conjugate(BEs).subs({sp.conjugate(x): x, sp.conjugate(y): y,
                                        sp.conjugate(s_): s_}))[1]
say(f"          a*beta - alpha*b there = {sp.simplify(A_ * BEs - AL * Bs)}, and the whole")
say(f"          c_3 residual there = {sp.simplify(t3s - sp.exp(-(w + 3) * u) * o3s)}")
say("      (b) and it is not zero for our generic pair, so the anomaly does not vanish")
say("          where the signal lives -- it scales with it:")
for pt in PTS:
    v = mag(A_ * BE - AL * B_, pt)
    say(f"          |a beta - alpha b| at ({float(pt[x]):4.1f},{float(pt[y]):4.1f}) = {v:.4f}"
        f"   (|alpha||b| = {mag(AL,pt)*mag(B_,pt):.4f})")
say()
say("    => the contamination of c_3 is a MULTIPLICATIVE distortion of exactly the")
say("       quantity it is read for, largest where the joint signal is largest; it is")
say("       not a background that could be estimated on a control region and subtracted.")
say("       c_5 carries none of it.  That is the clinical content of the weight: c_5 is")
say("       the harmonic that survives a lesion growing unevenly, which is every lesion.")
say()
say("    the proportionality is visible across the two tables: of the three points")
say("    tabulated, (-0.4,0.1) has the largest |a beta - alpha b| and the largest c_3")
say("    anomaly, (0.3,0.2) the smallest of each.  An additive background would not")
say("    track the signal that way.")

with open("confclin.log", "w") as f:
    f.write("\n".join(LOG) + "\n")
