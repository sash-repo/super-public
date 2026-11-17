"""Rigid pose is a nuisance parameter: what the joint descriptors do with it.

Rigid motion and orientation are nuisance parameters: translation and rotation must be
estimated or eliminated so that a changed orientation is not read as a biological
change; a rotation is not in itself a sign of malignancy.

The claim to be checked is accordingly NOT that the descriptors ignore the pose --- a
descriptor that ignored the registration would ignore the data --- but the two halves
that decide whether a changed pose can be mistaken for biology:

  (P)  the pointwise data of the registered pair, hence Delta phi, J = conj(alpha) beta,
       rho and tau, are unchanged when the pose of EITHER visit is changed and the
       registration is changed with it;
  (Z)  Delta phi = 0, J > 0, rho = 1, tau = 0 when the two boundaries are related by
       a similarity of space and the registration IS that similarity.

and the caveat that keeps (P) and (Z) honest:

  (E)  if the registration misses the motion by a turn epsilon of the frame, Delta phi
       shifts by 2 epsilon -- exactly item 4) of Proposition 5.3 of the text.  So the
       pose enters the descriptors through the ACCURACY of the registration and never
       as biology.

Everything is computed from (g, II, d phi) alone, by the formulas of a parametric
surface, so the ambient coordinates never enter; the checks below verify that this is
so rather than assume it.  A control is run beside each claim, so that no check passes
because both sides are zero.

Conventions are those of the text:  in an orthonormal frame of the tangent plane,
      a = tr(A)/2,   alpha = (A11 - A22)/2 - i A12       (eq. (hopf)),
the second visit is read in the frame the ROTATION factor R of d phi = R U transports,
      beta = alpha_of(R^T A_2 R),      Delta phi = arg(beta) - arg(alpha).

Run:  python3 nuisance.py     (sympy)
"""

import cmath
import sys

import numpy as np
import sympy as sp

I = sp.I
LOG = None
FAILED = []


def say(*a):
    msg = " ".join(str(x) for x in a)
    print(msg)
    if LOG is not None:
        LOG.write(msg + "\n")
        LOG.flush()


def check(name, ok):
    say("   [%s] %s" % ("ok" if ok else "FAIL", name))
    if not ok:
        FAILED.append(name)


# ------------------------------------------------------------------ geometry
#
# The differentiation is symbolic and everything after it is done in floating point:
# a symbolic simplify of the frames produces 0/0 forms once the pose carries a real
# rotation angle, and the quantities compared below are read to 1e-14 anyway.

def fl(M):
    """A sympy matrix of numbers as a numpy array of floats."""
    return np.array([[float(sp.re(sp.N(M[i, j], 40))) for j in range(M.cols)]
                     for i in range(M.rows)])


def surface_data(r, U, pt):
    """(g, II) of the parametrized surface r(u,v) at the parameter point pt.

    r is a 3-vector of expressions, U = (u, v).  No graph is assumed: the same code
    serves the image of a surface under a motion of space, which is what the checks
    below need.  n is the unit normal r_u x r_v / |r_u x r_v|.
    """
    u, v = U
    ru, rv = sp.diff(r, u), sp.diff(r, v)
    n = ru.cross(rv)
    ruN, rvN, nN = (fl(sp.Matrix(x).subs(pt)).ravel() for x in (ru, rv, n))
    nN = nN / np.linalg.norm(nN)
    g = np.array([[a.dot(b) for b in (ruN, rvN)] for a in (ruN, rvN)])
    II = np.array([[fl(sp.Matrix(sp.diff(r, U[i], U[j])).subs(pt)).ravel().dot(nN)
                    for j in (0, 1)] for i in (0, 1)])
    return g, II


def orthonormalize(g):
    """Gram-Schmidt on the PARAMETER frame: P with P^T g P = 1.

    P is built from the parametrization alone; the ambient coordinates do not enter,
    which is the point of the whole construction.
    """
    e1 = np.array([1.0, 0.0]) / np.sqrt(g[0, 0])
    w = np.array([0.0, 1.0]) - (e1 @ g @ np.array([0.0, 1.0])) * e1
    e2 = w / np.sqrt(w @ g @ w)
    return np.column_stack([e1, e2])


def alpha_of(A):
    """(a, alpha) of a symmetric 2x2 matrix in an orthonormal frame -- eq. (hopf)."""
    return (A[0, 0] + A[1, 1]) / 2, (A[0, 0] - A[1, 1]) / 2 - 1j * A[0, 1]


def rot(t):
    t = float(t)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def rot3(axis, t):
    """Rotation of space about a unit axis, by Rodrigues -- exact, for the surfaces."""
    k = sp.Matrix(axis)
    k = k / sp.sqrt((k.T * k)[0, 0])
    K = sp.Matrix([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return sp.simplify(sp.eye(3) + sp.sin(t) * K + (1 - sp.cos(t)) * K * K)


def polar(M):
    """M = R U with R in SO(2), U symmetric positive definite."""
    W, S, Vt = np.linalg.svd(M)
    return W @ Vt, Vt.T @ np.diag(S) @ Vt


def datum(r1, r2, U, pt, dphi, twist=0):
    """The pointwise datum of the registered pair: (a, alpha, b, beta, J).

    dphi is the differential of the registration in the PARAMETER coordinates of the
    two surfaces, a 2x2 matrix; the identity means "the same parameter point".
    twist is an error of the registration IN THE TANGENT FRAME: the transport is
    composed with a turn of the frame by that angle, which is the shape a mis-estimated
    rigid rotation takes at the matched point.
    """
    g1, II1 = surface_data(r1, U, pt)
    g2, II2 = surface_data(r2, U, pt)
    P1, P2 = orthonormalize(g1), orthonormalize(g2)
    A1, A2 = P1.T @ II1 @ P1, P2.T @ II2 @ P2
    dphi = np.array(dphi, dtype=float)
    M = np.linalg.inv(P2) @ dphi @ P1 @ rot(twist)   # frame 1 comps -> frame 2
    R, Um = polar(M)
    a, al = alpha_of(A1)
    b, be = alpha_of(R.T @ A2 @ R)
    return dict(a=a, alpha=al, b=b, beta=be, J=np.conj(al) * be, R=R, U=Um, M=M)


def num(z):
    return complex(z)


def close(x, y, tol=1e-12):
    return abs(complex(x) - complex(y)) < tol


def ang(z):
    """The argument, computed on the number and not left as a symbolic arg()."""
    return cmath.phase(complex(z))


def close_ang(x, y, tol=1e-10):
    d = (x - y) % (2 * cmath.pi)
    return min(d, 2 * cmath.pi - d) < tol


# ---------------------------------------------------------------- the surfaces

u, v = sp.symbols("u v", real=True)
UV = [u, v]
ORIG = {u: 0, v: 0}


def graph(f):
    return sp.Matrix([u, v, f])


def moved(r, Q, t, lam=1):
    """The image of the surface under x |-> lam Q x + t, same parametrization."""
    return sp.simplify(lam * (Q * r) + sp.Matrix(t))


# baseline and follow-up: two genuine graphs, neither umbilic at the origin
F1 = sp.Rational(1, 4) * (3 * u ** 2 + v ** 2) + sp.Rational(1, 5) * u ** 3
F2 = (sp.Rational(1, 4) * (3 * u ** 2 + v ** 2) * sp.Rational(11, 10)
      + sp.Rational(1, 6) * u * v + sp.Rational(1, 7) * v ** 3)

# a pose: a rotation of space about a slanted axis, and a translation
QQ = rot3([1, 2, 3], sp.pi / 5)
TT = [sp.Rational(2, 3), -sp.Rational(1, 4), sp.Rational(5, 7)]


# ===================================================================== part 1

def part1():
    say("\n=== 1. the pointwise datum, and that it is not vacuous")
    d = datum(graph(F1), graph(F2), UV, ORIG, np.eye(2))
    say("      a = %.8f   |alpha| = %.8f" % (d["a"], abs(d["alpha"])))
    say("      b = %.8f   |beta|  = %.8f" % (d["b"], abs(d["beta"])))
    say("      Delta phi = %.8f rad" % ang(d["J"]))
    check("alpha does not vanish (the point is not umbilic on visit 1)",
          abs(d["alpha"]) > 1e-6)
    check("beta does not vanish (the point is not umbilic on visit 2)",
          abs(d["beta"]) > 1e-6)
    check("Delta phi is not 0 for this pair -- the control for part 3",
          abs(d["J"].imag) > 1e-6)
    return d


# ===================================================================== part 2

def part2(base):
    say("\n=== 2. (P) the pose of either visit, with the registration moved with it")

    # --- visit 2 re-posed, registration composed with the same motion
    r2m = moved(graph(F2), QQ, TT)
    d = datum(graph(F1), r2m, UV, ORIG, np.eye(2))
    for k in ("a", "b", "alpha", "beta", "J"):
        check("visit 2 re-posed: %s unchanged" % k, close(d[k], base[k]))

    # --- visit 1 re-posed likewise
    r1m = moved(graph(F1), QQ, TT)
    d = datum(r1m, graph(F2), UV, ORIG, np.eye(2))
    for k in ("a", "b", "alpha", "beta", "J"):
        check("visit 1 re-posed: %s unchanged" % k, close(d[k], base[k]))

    # --- both, by different motions
    Q2 = rot3([-2, 1, 1], -sp.pi / 7)
    d = datum(moved(graph(F1), QQ, TT), moved(graph(F2), Q2, [0, 1, -1]), UV, ORIG,
              np.eye(2))
    for k in ("a", "b", "alpha", "beta", "J"):
        check("both re-posed: %s unchanged" % k, close(d[k], base[k]))

    # --- a similarity of visit 2: the SCALE does show, the angle does not
    lam = sp.Rational(3, 2)
    d = datum(graph(F1), moved(graph(F2), QQ, TT, lam), UV, ORIG, float(lam) * np.eye(2))
    check("visit 2 rescaled by 3/2: b scales by 2/3", close(d["b"], base["b"] / float(lam)))
    check("visit 2 rescaled by 3/2: |beta| scales by 2/3",
          close(abs(d["beta"]), abs(base["beta"]) / float(lam)))
    check("visit 2 rescaled by 3/2: Delta phi unchanged",
          close_ang(ang(d["J"]), ang(base["J"])))

    # --- the parametrization, not the pose: turn the parameters of visit 2
    t0 = sp.pi / 6
    r2p = graph(F2).subs({u: sp.cos(t0) * u - sp.sin(t0) * v,
                          v: sp.sin(t0) * u + sp.cos(t0) * v}, simultaneous=True)
    d = datum(graph(F1), r2p, UV, ORIG, rot(-t0))
    for k in ("a", "b", "alpha", "beta", "J"):
        check("visit 2 reparametrized: %s unchanged" % k, close(d[k], base[k]))


# ===================================================================== part 3

def part3():
    say("\n=== 3. (Z) a similarity between the two boundaries reads as no change")
    for nm, lam, Q, t in (("rigid motion", 1, QQ, TT),
                          ("similarity, ratio 7/5", sp.Rational(7, 5), QQ, TT),
                          ("pure translation", 1, sp.eye(3), [1, -2, 3])):
        r2 = moved(graph(F1), Q, t, lam)
        d = datum(graph(F1), r2, UV, ORIG, float(lam) * np.eye(2))
        check("%s: Delta phi = 0" % nm, abs(ang(d['J'])) < 1e-10)
        check("%s: J is real and positive" % nm,
              abs(d["J"].imag) < 1e-12 and d["J"].real > 1e-9)
        check("%s: |beta| = |alpha|/lambda" % nm,
              close(abs(d["beta"]) * float(lam), abs(d["alpha"])))
    say("      (rho is |mean J| / mean |J| and tau is Im J, so a J of constant")
    say("       argument 0 over the boundary gives rho = 1 and tau = 0 identically)")

    # the control: the same follow-up SHAPE, turned about its own normal, is not
    # a similarity of the pair and must be seen
    t0 = sp.Rational(1, 3)
    r2 = graph(F1).subs({u: sp.cos(t0) * u - sp.sin(t0) * v,
                         v: sp.sin(t0) * u + sp.cos(t0) * v}, simultaneous=True)
    d = datum(graph(F1), r2, UV, ORIG, np.eye(2))
    check("control: the follow-up turned about its normal gives Delta phi = 2 t0",
          close_ang(ang(d["J"]), 2 * float(t0)))


# ===================================================================== part 4

def part4(base):
    say("\n=== 4. (E) a registration that misses the motion by epsilon")
    for eps in (sp.Rational(1, 20), sp.Rational(1, 7), -sp.Rational(2, 9)):
        d2 = datum(graph(F1), graph(F2), UV, ORIG, np.eye(2), twist=eps)
        shift = ang(d2["J"]) - ang(base["J"])
        say("      epsilon = %-6s  shift = %.12f   2 epsilon = %.12f"
            % (eps, shift, 2 * float(eps)))
        check("epsilon = %s: Delta phi shifts by 2 epsilon" % eps,
              close_ang(shift, 2 * float(eps)))
    say("      so the pose reaches the descriptors only through the accuracy of the")
    say("      registration -- which is item 4) of Proposition 5.3 of the text, read")
    say("      for the pose rather than for the noise.")

    # and the honest converse: a rigid pair whose motion the registration missed by
    # a turn is NOT seen as no change
    t0 = sp.Rational(1, 4)
    r2 = moved(graph(F1), QQ, TT)
    d = datum(graph(F1), r2, UV, ORIG, np.eye(2), twist=t0)
    check("a rigid pair whose motion is missed by t0 gives Delta phi = 2 t0, not 0",
          close_ang(ang(d["J"]), 2 * float(t0)))


def main():
    global LOG
    LOG = open("nuisance.log", "w")
    say("nuisance.py -- rigid pose and the joint descriptors")
    base = part1()
    part2(base)
    part3()
    part4(base)
    say("\n=== summary")
    if FAILED:
        say("   %d FAILED: %s" % (len(FAILED), "; ".join(FAILED)))
    else:
        say("   all checks passed")
    LOG.close()
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
