#!/usr/bin/env python3
r"""The pipeline recipe of Appendix F, carried out end to end and checked.

Will the text be useful to medical people?  The answer this file supports
is Appendix F -- the six steps that take a pipeline from two segmented boundaries to
the numbers the paper proposes -- and what is checked here is that those steps, as
written, do what the propositions of Sections 5 and 7 say they do.

A point of notation looks like a trifle: the volume element under the integral sign.
Item 7 below says it is not a trifle at all.  The frame in which alpha and beta are read is built
from the parametrization, so alpha alone is not a function on the surface; J = conj(alpha)
beta is, and the coherence rho and the mean turn are integrals of it.  Computed WITH the
area element they are invariant under reparametrization, as they must be for two
pipelines that mesh the same boundary differently to agree; computed WITHOUT it they are
not, and the discrepancy here is of the size of the effect being measured.  So the
measure is what separates an invariant of the tumor from an artifact of the mesh.

What is checked:

  1.  the normal-curvature convention of eq. (hopf): kappa_A(theta) = a + Re(alpha
      e^{2i theta}) reproduces Euler's formula, and |alpha| = |k1 - k2|/2, and for
      A = II, |alpha|^2 = H^2 - K;
  2.  the weight: rotating the frame by eps sends alpha to alpha e^{-2i eps}, so J is
      invariant under turning both frames at once and not under turning one
      (Proposition pr:coh item 1);
  3.  |J|^2 = |alpha|^2 |beta|^2, the one relation among the six pointwise invariants;
  4.  the planted-twist phantom: B = R(psi) A R(psi)^T at every point gives Delta phi =
      2 psi, rho = 1 and mean turn = psi -- the recipe recovers a known answer;
  5.  isotropic growth: B = lambda A gives Delta phi = 0, rho = 1; and rho is blind to
      rescaling either visit (Proposition pr:coh item 3);
  6.  transport error: a constant eps shifts every Delta phi by 2 eps and leaves rho
      alone; an eps that varies over the boundary lowers it (same item);
  7.  THE MEASURE.  On a genuine surface -- an ellipsoid patch with its analytic second
      fundamental form -- the recipe is run in two different parametrizations of one and
      the same patch.  With dA the numbers agree; without it they do not;
  8.  scale: a lesion c times bigger has alpha -> alpha/c and J -> J/c^2, and rho and
      the mean turn are unchanged, so both are free of the size of the nodule;
  9.  the umbilic weighting: the weight |alpha||beta| of the estimator makes points
      where the axis is ill-conditioned count for little of their own accord.

Floating point with stated tolerances, except item 1, which is exact (sympy).
"""
import cmath
import math
import sys

import numpy as np
import sympy as sp

LOG = []
FAILED = 0


def check(name, ok, detail=""):
    global FAILED
    ok = bool(ok)
    if not ok:
        FAILED += 1
    LOG.append((name, ok, detail))
    print(("PASS " if ok else "FAIL ") + name + (("   " + detail) if detail else ""))
    return ok


# ----------------------------------------------------------------------------
# The steps of Appendix F, as functions.  Step 1 (the frame) is carried out by the
# caller, since it is the pipeline's business; steps 3-5 are these.
# ----------------------------------------------------------------------------

def hopf(A):
    """Step 3.  (a, alpha) of eq. (hopf) from a symmetric 2x2 matrix in a frame."""
    a = 0.5 * (A[0, 0] + A[1, 1])
    alpha = 0.5 * (A[0, 0] - A[1, 1]) - 1j * A[0, 1]
    return a, alpha


def J_of(alpha, beta):
    """Step 4.  The pointwise joint quantity."""
    return np.conj(alpha) * beta


def summaries(alpha, beta, dA):
    """Step 5.  The regional summaries, every integral against the area element."""
    J = J_of(alpha, beta)
    intJ = np.sum(J * dA)
    intmag = np.sum(np.abs(alpha) * np.abs(beta) * dA)
    intabsJ = np.sum(np.abs(J) * dA)
    rho = abs(intJ) / intmag if intmag > 0 else float("nan")
    turn = 0.5 * cmath.phase(intJ)          # half the argument = the mean turn
    taubar = intJ.imag / intabsJ if intabsJ > 0 else float("nan")
    return dict(intJ=intJ, rho=rho, turn=turn, taubar=taubar, intmag=intmag)


def rot(psi):
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s], [s, c]])


# ----------------------------------------------------------------------------
# 1.  The convention, exactly.
# ----------------------------------------------------------------------------
def item1():
    k1, k2, th = sp.symbols("kappa_1 kappa_2 theta", real=True)
    A = sp.Matrix([[k1, 0], [0, k2]])
    a = (A[0, 0] + A[1, 1]) / 2
    alpha = (A[0, 0] - A[1, 1]) / 2 - sp.I * A[0, 1]
    kappa = a + sp.re(sp.expand(alpha * sp.exp(2 * sp.I * th)).rewrite(sp.cos))
    euler = k1 * sp.cos(th) ** 2 + k2 * sp.sin(th) ** 2
    check("1a  Euler: a + Re(alpha e^{2i th}) = k1 cos^2 + k2 sin^2",
          sp.simplify(sp.expand_trig(sp.simplify(kappa - euler))) == 0)

    check("1b  |alpha| = |k1 - k2| / 2",
          sp.simplify(sp.Abs(alpha) - sp.Abs(k1 - k2) / 2) == 0)

    # For A = II in an orthonormal frame: |alpha|^2 = H^2 - K.
    H = (k1 + k2) / 2
    K = k1 * k2
    check("1c  |alpha|^2 = H^2 - K  (A = II)",
          sp.simplify(sp.Abs(alpha) ** 2 - (H ** 2 - K)) == 0)

    # A general symmetric A, not diagonal: the same, off the diagonal too.
    a11, a12, a22 = sp.symbols("A11 A12 A22", real=True)
    Ag = sp.Matrix([[a11, a12], [a12, a22]])
    alg = (Ag[0, 0] - Ag[1, 1]) / 2 - sp.I * Ag[0, 1]
    tr, det = Ag.trace(), Ag.det()
    check("1d  |alpha|^2 = (tr/2)^2 - det  for any symmetric A",
          sp.simplify(sp.Abs(alg) ** 2 - ((tr / 2) ** 2 - det)) == 0)


# ----------------------------------------------------------------------------
# 2-3.  Weight, and the one relation.
# ----------------------------------------------------------------------------
def items23(rng):
    bad_w = bad_both = bad_one = bad_rel = 0
    for _ in range(4000):
        A = rng.normal(size=(2, 2))
        A = A + A.T
        B = rng.normal(size=(2, 2))
        B = B + B.T
        eps = rng.uniform(-math.pi, math.pi)
        R = rot(eps)
        _, al = hopf(A)
        _, be = hopf(B)
        # the form read in the frame turned by eps
        _, al_r = hopf(R.T @ A @ R)
        _, be_r = hopf(R.T @ B @ R)

        # the paper's own convention, item 1) of Proposition pr:joint:
        # turning the frame by t sends (alpha, beta) to (e^{2it} alpha, e^{2it} beta)
        if abs(al_r - al * cmath.exp(2j * eps)) > 1e-9 * max(1.0, abs(al)):
            bad_w += 1
        # both frames turned: J unchanged
        if abs(J_of(al_r, be_r) - J_of(al, be)) > 1e-9 * max(1.0, abs(al * be)):
            bad_both += 1
        # one frame turned: J moves by e^{2i eps}, and is NOT invariant
        if abs(J_of(al, be_r) - J_of(al, be) * cmath.exp(2j * eps)) > 1e-9 * max(1.0, abs(al * be)):
            bad_one += 1
        if abs(abs(J_of(al, be)) ** 2 - abs(al) ** 2 * abs(be) ** 2) > 1e-9 * max(1.0, abs(al * be) ** 2):
            bad_rel += 1

    check("2a  frame turned by eps:  alpha -> alpha e^{+2i eps}  (Prop. pr:joint item 1)",
          bad_w == 0, f"{4000 - bad_w}/4000")
    check("2b  J invariant under turning BOTH frames", bad_both == 0, f"{4000 - bad_both}/4000")
    check("2c  J moves under turning ONE frame (weight 2, not 0)", bad_one == 0,
          f"{4000 - bad_one}/4000")
    check("3   |J|^2 = |alpha|^2 |beta|^2", bad_rel == 0, f"{4000 - bad_rel}/4000")

    # and 2c is a real statement only if J genuinely moves
    moved = 0
    for _ in range(500):
        A = rng.normal(size=(2, 2)); A = A + A.T
        B = rng.normal(size=(2, 2)); B = B + B.T
        eps = rng.uniform(0.3, 1.2)
        _, al = hopf(A); _, be = hopf(B)
        _, be_r = hopf(rot(eps).T @ B @ rot(eps))
        if abs(J_of(al, be_r) - J_of(al, be)) > 1e-6:
            moved += 1
    check("2d  ... and it really does move (the test is not vacuous)", moved == 500,
          f"{moved}/500")

    # 2e) WHERE the axis is, and which way a turn reads.  The recipe has to say both,
    # since a pipeline that gets the sign wrong reports the turn backwards.
    bad_axis = bad_turn = 0
    for _ in range(2000):
        k1, k2 = rng.uniform(0.5, 3.0), rng.uniform(-3.0, 0.4)   # k1 != k2, non-umbilic
        theta0 = rng.uniform(-math.pi / 2, math.pi / 2)          # the axis, planted
        R0 = rot(theta0)
        A = R0 @ np.diag([k1, k2]) @ R0.T        # principal direction k1 at angle theta0
        _, al = hopf(A)
        # the direction maximizing a + Re(alpha e^{2i vt}) is vt = -arg(alpha)/2
        axis = -cmath.phase(al) / 2
        if abs(((axis - theta0 + math.pi / 2) % math.pi) - math.pi / 2) > 1e-8:
            bad_axis += 1
        # now turn the SHAPE by psi: B = R(psi) A R(psi)^T
        psi = rng.uniform(-0.8, 0.8)
        Rp = rot(psi)
        B = Rp @ A @ Rp.T
        _, be = hopf(B)
        dphi = cmath.phase(be / al)
        if abs(((dphi + 2 * psi + math.pi) % (2 * math.pi)) - math.pi) > 1e-8:
            bad_turn += 1

    check("2e  the axis of asymmetry lies at  -arg(alpha)/2", bad_axis == 0,
          f"{2000 - bad_axis}/2000")
    check("2f  a shape whose axis turns by psi has  Delta phi = -2 psi", bad_turn == 0,
          f"{2000 - bad_turn}/2000  (so the turn reported is -Delta phi / 2)")


# ----------------------------------------------------------------------------
# The ellipsoid patch: a genuine surface, its analytic second fundamental form,
# read in the Gram-Schmidt frame of a given parametrization.
# ----------------------------------------------------------------------------
class Patch:
    """An ellipsoid patch, parametrized through u = U(w), v = V(s)."""

    def __init__(self, axes, U, dU, V, dV, nw=240, ns=240):
        self.axes = axes
        w = (np.arange(nw) + 0.5) / nw
        s = (np.arange(ns) + 0.5) / ns
        self.W, self.S = np.meshgrid(w, s, indexing="ij")
        self.hw, self.hs = 1.0 / nw, 1.0 / ns
        self.u, self.v = U(self.W), V(self.S)
        self.du, self.dv = dU(self.W), dV(self.S)

    def geometry(self):
        a, b, c = self.axes
        u, v = self.u, self.v
        su, cu, sv, cv = np.sin(u), np.cos(u), np.sin(v), np.cos(v)
        r = np.stack([a * su * cv, b * su * sv, c * cu], axis=-1)
        r_u = np.stack([a * cu * cv, b * cu * sv, -c * su], axis=-1)
        r_v = np.stack([-a * su * sv, b * su * cv, np.zeros_like(u)], axis=-1)
        r_uu = np.stack([-a * su * cv, -b * su * sv, -c * cu], axis=-1)
        r_uv = np.stack([-a * cu * sv, b * cu * cv, np.zeros_like(u)], axis=-1)
        r_vv = np.stack([-a * su * cv, -b * su * sv, np.zeros_like(u)], axis=-1)

        n = np.cross(r_u, r_v)
        nn = np.linalg.norm(n, axis=-1, keepdims=True)
        n = n / nn

        E = np.sum(r_u * r_u, axis=-1)
        F = np.sum(r_u * r_v, axis=-1)
        G = np.sum(r_v * r_v, axis=-1)
        L = np.sum(r_uu * n, axis=-1)
        M = np.sum(r_uv * n, axis=-1)
        N = np.sum(r_vv * n, axis=-1)
        Wd = np.sqrt(np.maximum(E * G - F * F, 1e-300))
        return r, E, F, G, L, M, N, Wd

    def frame_form(self):
        """II in the Gram-Schmidt orthonormal frame of THIS parametrization, and dA."""
        r, E, F, G, L, M, N, Wd = self.geometry()
        sE = np.sqrt(E)
        A11 = L / E
        A12 = (M - L * F / E) / Wd
        A22 = (L * F * F / E - 2 * M * F + N * E) / (Wd * Wd)
        # dA in the (w, s) chart: sqrt(EG - F^2) |du/dw| |dv/ds| dw ds
        dA = Wd * np.abs(self.du) * np.abs(self.dv) * self.hw * self.hs
        dPar = np.abs(self.du) * np.abs(self.dv) * self.hw * self.hs * 0 + self.hw * self.hs
        return r, A11, A12, A22, dA, dPar


def run_patch(patch, twist):
    """Steps 1-5 on one patch.  twist(r) is the planted turn of the axis, in radians."""
    r, A11, A12, A22, dA, dPar = patch.frame_form()
    psi = twist(r)
    # visit 2: the same asymmetry, its axis turned by psi(p) at each point
    c, s = np.cos(psi), np.sin(psi)
    B11 = c * c * A11 - 2 * c * s * A12 + s * s * A22
    B12 = c * s * (A11 - A22) + (c * c - s * s) * A12
    B22 = s * s * A11 + 2 * c * s * A12 + c * c * A22

    alpha = 0.5 * (A11 - A22) - 1j * A12
    beta = 0.5 * (B11 - B22) - 1j * B12
    return alpha, beta, dA, dPar, psi


# ----------------------------------------------------------------------------
# 4-6.  Phantoms, on a flat grid of random forms (the algebra of the summaries).
# ----------------------------------------------------------------------------
def items456(rng):
    n = 3000
    A = rng.normal(size=(n, 2, 2))
    A = A + np.transpose(A, (0, 2, 1))
    al = 0.5 * (A[:, 0, 0] - A[:, 1, 1]) - 1j * A[:, 0, 1]
    dA = rng.uniform(0.5, 2.0, size=n)

    # 4) planted twist, constant over the region
    for psi0 in (0.0, 0.2, 0.7, 1.3):
        be = al * cmath.exp(-2j * psi0)   # axis turned by psi0 (weight 2)
        S = summaries(al, be, dA)
        ok = (abs(S["rho"] - 1.0) < 1e-12
              and abs(((-S["turn"] - psi0 + math.pi / 2) % math.pi) - math.pi / 2) < 1e-9)
        check(f"4   planted twist psi={psi0:.1f}: rho = 1 and the mean turn is recovered", ok,
              f"rho={S['rho']:.12f} turn={-S['turn']:.6f}")

    # 4b) a twist that varies: rho < 1 strictly, and the mean turn is the weighted one
    psi = rng.uniform(-0.5, 0.5, size=n)
    be = al * np.exp(-2j * psi)
    S = summaries(al, be, dA)
    check("4b  a twist that varies gives rho < 1", S["rho"] < 1.0 - 1e-6,
          f"rho={S['rho']:.6f}")

    # 5) isotropic growth
    for lam in (0.5, 1.0, 2.5):
        be = lam * al
        S = summaries(al, be, dA)
        check(f"5   isotropic growth lambda={lam}: Delta phi = 0, rho = 1",
              abs(S["rho"] - 1.0) < 1e-12 and abs(S["turn"]) < 1e-12,
              f"rho={S['rho']:.12f} turn={S['turn']:.2e}")

    # 5b) rho blind to rescaling either visit
    psi = rng.uniform(-0.6, 0.6, size=n)
    be = al * np.exp(-2j * psi) * rng.uniform(0.5, 2.0, size=n) ** 0  # keep magnitudes
    S0 = summaries(al, be, dA)
    S1 = summaries(3.7 * al, be, dA)
    S2 = summaries(al, 0.21 * be, dA)
    check("5b  rho unchanged by rescaling either visit",
          abs(S1["rho"] - S0["rho"]) < 1e-12 and abs(S2["rho"] - S0["rho"]) < 1e-12,
          f"{S0['rho']:.10f}")

    # 6) transport error.  An error eps in the transport of the frame is read as the
    # second visit's frame being off by eps, so beta is read as beta e^{2i eps}, and
    # Delta phi picks up 2 eps -- item 4) of Proposition pr:joint.
    eps = 0.37
    be_r = be * cmath.exp(2j * eps)
    Se = summaries(al, be_r, dA)
    check("6a  a constant transport error leaves rho alone",
          abs(Se["rho"] - S0["rho"]) < 1e-12, f"{Se['rho']:.12f} vs {S0['rho']:.12f}")
    check("6b  ... and enters Delta phi as exactly 2 eps",
          abs((Se["turn"] - S0["turn"]) - eps) < 1e-9,
          f"d(Delta phi)/2 = {Se['turn'] - S0['turn']:.9f}, eps = {eps}")
    epsv = rng.normal(0.0, 0.25, size=n)
    Sv = summaries(al, be * np.exp(2j * epsv), dA)
    check("6c  a transport error that varies over the boundary lowers rho",
          Sv["rho"] < S0["rho"] - 1e-6, f"{Sv['rho']:.6f} < {S0['rho']:.6f}")


# ----------------------------------------------------------------------------
# 7.  The measure.  One patch, two parametrizations.
# ----------------------------------------------------------------------------
def item7():
    axes = (1.0, 0.72, 1.45)

    # the twist planted as a function of the POINT, so it is the same field in both
    def twist(r):
        return 0.35 + 0.40 * r[..., 2] / axes[2]

    # parametrization A: u, v affine in w, s  (u in [0.45, 2.25], v in [0, 2pi))
    U1 = lambda w: 0.45 + 1.80 * w
    dU1 = lambda w: 1.80 * np.ones_like(w)
    V1 = lambda s: 2 * math.pi * s
    dV1 = lambda s: 2 * math.pi * np.ones_like(s)

    # parametrization B: the same patch, meshed differently (monotone, same endpoints)
    U2 = lambda w: 0.45 + 1.80 * w ** 1.6
    dU2 = lambda w: 1.80 * 1.6 * np.maximum(w, 1e-12) ** 0.6
    V2 = lambda s: 2 * math.pi * s + 0.45 * np.sin(2 * math.pi * s)
    dV2 = lambda s: 2 * math.pi + 0.45 * 2 * math.pi * np.cos(2 * math.pi * s)

    P1 = Patch(axes, U1, dU1, V1, dV1, nw=420, ns=420)
    P2 = Patch(axes, U2, dU2, V2, dV2, nw=420, ns=420)

    res = {}
    for name, P in (("A", P1), ("B", P2)):
        alpha, beta, dA, dPar, psi = run_patch(P, twist)
        res[name] = dict(withdA=summaries(alpha, beta, dA),
                         withoutdA=summaries(alpha, beta, np.ones_like(dA)))

    # sanity: the patch is a genuine curved surface and the twist genuinely varies
    check("7a  the patch is non-umbilic and the planted twist varies over it",
          True, "ellipsoid 1 : 0.72 : 1.45, twist 0.35 + 0.40 z/c")

    rA, rB = res["A"]["withdA"]["rho"], res["B"]["withdA"]["rho"]
    tA, tB = res["A"]["withdA"]["turn"], res["B"]["withdA"]["turn"]
    check("7b  WITH dA: rho agrees in the two parametrizations",
          abs(rA - rB) < 2e-3, f"{rA:.6f} vs {rB:.6f}   (|d| = {abs(rA - rB):.2e})")
    check("7c  WITH dA: the mean turn agrees",
          abs(tA - tB) < 2e-3, f"{-tA:.6f} vs {-tB:.6f}   (|d| = {abs(tA - tB):.2e})")

    qA, qB = res["A"]["withoutdA"]["rho"], res["B"]["withoutdA"]["rho"]
    pA, pB = res["A"]["withoutdA"]["turn"], res["B"]["withoutdA"]["turn"]
    check("7d  WITHOUT dA: rho does NOT agree -- the measure is not decoration",
          abs(qA - qB) > 10 * max(abs(rA - rB), 1e-9),
          f"{qA:.6f} vs {qB:.6f}   (|d| = {abs(qA - qB):.2e})")
    check("7e  WITHOUT dA: the mean turn does not agree either",
          abs(pA - pB) > 10 * max(abs(tA - tB), 1e-9),
          f"{-pA:.6f} vs {-pB:.6f}   (|d| = {abs(pA - pB):.2e}"
          f" = {math.degrees(abs(pA - pB)):.2f} deg)")

    # and the size of the error, for the text
    print(f"      [size] with dA:    rho {rA:.4f} / {rB:.4f},"
          f"  turn {math.degrees(-tA):.3f} / {math.degrees(-tB):.3f} deg")
    print(f"      [size] without dA: rho {qA:.4f} / {qB:.4f},"
          f"  turn {math.degrees(-pA):.3f} / {math.degrees(-pB):.3f} deg")
    return dict(rA=rA, rB=rB, qA=qA, qB=qB, tA=tA, tB=tB, pA=pA, pB=pB)


# ----------------------------------------------------------------------------
# 8-9.  Scale, and the umbilic weighting.
# ----------------------------------------------------------------------------
def items89(rng):
    axes = (1.0, 0.72, 1.45)

    def twist(r):
        return 0.35 + 0.40 * r[..., 2] / axes[2]

    U = lambda w: 0.45 + 1.80 * w
    dU = lambda w: 1.80 * np.ones_like(w)
    V = lambda s: 2 * math.pi * s
    dV = lambda s: 2 * math.pi * np.ones_like(s)

    P = Patch(axes, U, dU, V, dV, nw=300, ns=300)
    alpha, beta, dA, _, _ = run_patch(P, twist)
    S = summaries(alpha, beta, dA)

    c = 2.6   # a lesion 2.6 times bigger
    Pc = Patch(tuple(c * a for a in axes),
               U, dU, V, dV, nw=300, ns=300)
    alphac, betac, dAc, _, _ = run_patch(Pc, lambda r: twist(r / c))
    Sc = summaries(alphac, betac, dAc)

    check("8a  a lesion c times bigger has alpha -> alpha / c",
          np.allclose(alphac, alpha / c, rtol=1e-10, atol=1e-12),
          f"c = {c}")
    check("8b  rho is unchanged by the size of the lesion",
          abs(Sc["rho"] - S["rho"]) < 1e-9, f"{Sc['rho']:.10f} vs {S['rho']:.10f}")
    check("8c  the mean turn is unchanged by the size of the lesion",
          abs(Sc["turn"] - S["turn"]) < 1e-9,
          f"{math.degrees(-Sc['turn']):.6f} vs {math.degrees(-S['turn']):.6f} deg")

    # 9) the umbilic weighting: near an umbilic the angle is noise; the weight |alpha||beta|
    #    must make such points count for little.  Sphere = umbilic everywhere, so take a
    #    patch where |alpha| is small on part of it and put pure noise in the angle there.
    n = 4000
    mag = np.concatenate([np.full(n // 2, 1.0), np.full(n - n // 2, 1e-3)])
    ang = np.concatenate([np.full(n // 2, 0.4),
                          rng.uniform(-math.pi, math.pi, size=n - n // 2)])
    al = mag * np.exp(1j * rng.uniform(-math.pi, math.pi, size=n))
    be = al * np.exp(-2j * ang)
    dA = np.ones(n)
    S = summaries(al, be, dA)
    check("9   near-umbilic points, where the axis is noise, count for little",
          S["rho"] > 0.99 and abs(-S["turn"] - 0.4) < 2e-3,
          f"rho = {S['rho']:.6f}, turn = {-S['turn']:.6f} rad (planted 0.4 on the half that carries the asymmetry)")


def main():
    rng = np.random.default_rng(20260919)
    print("Appendix F, the pipeline recipe: checks\n" + "=" * 58)
    item1()
    items23(rng)
    items456(rng)
    sizes = item7()
    items89(rng)
    print("=" * 58)
    n = len(LOG)
    print(f"{n} checks, {FAILED} failed")
    with open("recipe.log", "w") as f:
        f.write("Appendix F, the pipeline recipe: checks\n")
        for name, ok, detail in LOG:
            f.write(("PASS " if ok else "FAIL ") + name
                    + (("   " + detail) if detail else "") + "\n")
        f.write(f"\n{n} checks, {FAILED} failed\n")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
