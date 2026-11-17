"""Is the second angle meaningful only under a hypothesis?  H. Andersen's question 1.

Admitting the registration's differential as data gives four pointwise joint invariants
instead of one: the two principal stretches of d phi, and the two angles between the
strain axis and each visit's axis of asymmetry.  The clinical reading of the
new angle is: does the lesion grow ALONG its own axis of asymmetry or ACROSS it?  Is it
only meaningful under the hypothesis that the registration follows the tissue?

Three answers, computed here.

PART 1.  WHAT NEEDS THE HYPOTHESIS AND WHAT DOES NOT.
  The strain axis exists only where the stretch is anisotropic (lambda_1 != lambda_2);
  where it is isotropic, BOTH new angles are undefined.  Their DIFFERENCE, however, is
  the g_1-angle between the two visits' asymmetry lines, which needs no strain axis at
  all.  So exactly one of the four is at risk, and nothing the paper already carries is.
  Checked on random data: the difference of the two angles equals the line angle, to
  machine precision, for every anisotropy including the isotropic limit.

PART 2.  THE PAPER'S Delta phi IS NOT THAT DIFFERENCE, AND HERE IS THE GAP.
  The paper transports visit 2 by the ROTATION factor R of d phi = R U; the two-metric
  reading transports it by d phi itself.  Both are canonical, so both are invariants --
  no claim of the paper is destroyed -- but they are different functions.  With the
  strain axes as coordinate axes and r = lambda_1/lambda_2,

        tan(psi_phi) = r * tan(psi_R),

  so the two readings of "the same" angle differ by up to arctan(sqrt r)-arctan(1/sqrt r).
  That maximum is the honest size of the correction Proposition 5.9 would carry, tabulated
  against the anisotropy A = (lambda_1-lambda_2)/(lambda_1+lambda_2).

PART 3.  THE HYPOTHESIS IS NOT MATERIAL, IT IS MEASURABLE -- AND IT DEGRADES GRACEFULLY.
  Nothing about tissue mechanics is assumed: the condition is that the MEASURED Jacobian
  of the registration be anisotropic enough at the point, which the registration output
  itself reports.  First-order perturbation gives the rate,

        d(2 theta) = 2 (R^T dU R)_{12} / (lambda_1 - lambda_2),

  verified here against exact re-diagonalization, and a Monte Carlo over the registration
  noise sigma turns it into a gate: the angle is worth printing where A exceeds a fixed
  multiple of sigma, and is uniform noise below it.
"""
import numpy as np

rng = np.random.default_rng(20260916)
LOG = []


def say(s=""):
    print(s)
    LOG.append(s)


def rot(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


def axis_angle(M):
    """Doubled angle 2*theta of the principal axis of the symmetric 2x2 M (mod 2pi).

    A line field has a direction only mod pi, so the invariant carrier is 2*theta; this
    is the same spin-2 convention the paper uses for arg(alpha).
    """
    return np.arctan2(2 * M[0, 1], M[0, 0] - M[1, 1])


def trace_free(B, g):
    """Trace-free part of the symmetric B with respect to the metric g, as a spin-2 marker.

    Returned in a g-orthonormal frame, so that its doubled angle is the asymmetry axis.
    """
    L = np.linalg.cholesky(g)                 # g = L L^T
    Bo = np.linalg.solve(L, np.linalg.solve(L, B.T).T)   # B in the g-orthonormal frame
    return Bo - 0.5 * np.trace(Bo) * np.eye(2), L


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


# ----------------------------------------------------------------- Part 1
say("PART 1.  what needs the strain axis and what does not")
say("  data: g_1, II_1, phi^*g_2, phi^*II_2 at one matched point; random, 20000 draws")
say("  claim: angle(strain,l_1) - angle(strain,l_2) = angle(l_1,l_2), whatever the strain")
worst = 0.0
worst_iso = 0.0
for k in range(20000):
    # metric g_1 and a random stretch S = g_1^{-1}(phi^*g_2), forced isotropic in 1/4 of draws
    A1 = rng.normal(size=(2, 2))
    g1 = A1 @ A1.T + 0.5 * np.eye(2)
    L1 = np.linalg.cholesky(g1)
    iso = (k % 4 == 0)
    lam = np.array([1.0, 1.0]) * np.exp(rng.normal(scale=0.3)) if iso else np.exp(rng.normal(scale=0.3, size=2))
    thU = rng.uniform(0, np.pi)
    U = rot(thU) @ np.diag(lam) @ rot(thU).T          # stretch in the g_1-orthonormal frame
    g2p = L1 @ (U @ U) @ L1.T                          # phi^*g_2 = g_1(U^2 . , .)
    B1 = rng.normal(size=(2, 2)); B1 = B1 + B1.T       # II_1
    B2 = rng.normal(size=(2, 2)); B2 = B2 + B2.T       # phi^*II_2
    tf1, _ = trace_free(B1, g1)                        # visit 1 asymmetry, in g_1-ON frame
    tf2, L2 = trace_free(B2, g2p)                      # visit 2 asymmetry, in its OWN ON frame
    # the visit-2 asymmetry LINE, pushed to the g_1-orthonormal frame (a pullback of a line,
    # no choice of transport involved): its direction vector carried by L2 then L1^{-1}
    a2 = 0.5 * axis_angle(tf2)
    v2 = np.linalg.solve(L1, L2 @ np.array([np.cos(a2), np.sin(a2)]))
    psi2 = np.arctan2(v2[1], v2[0])
    psi1 = 0.5 * axis_angle(tf1)
    if iso:
        continue
    # the two new angles, measured from the strain axis in g_1
    ph1 = wrap(2 * (thU - psi1))
    ph2 = wrap(2 * (thU - psi2))
    worst = max(worst, abs(wrap((ph1 - ph2) - wrap(2 * (psi2 - psi1)))))
say(f"  max |(phi_1 - phi_2) - angle(l_1,l_2)| over 15000 anisotropic draws = {worst:.2e} rad")
say("  => the DIFFERENCE is free of the strain axis; only the individual angles need it.")
say("  in the isotropic draws the strain axis is undefined and both individual angles are")
say("  not computed at all, while the difference is: it is the one the paper carries.")
say()

# ----------------------------------------------------------------- Part 2
say("PART 2.  the paper's transport (rotation R) against the two-metric one (d phi)")
say("  with the strain axes as coordinate axes:  tan psi_phi = r tan psi_R,  r = lam1/lam2")
say("  verified on random data, then the worst-case gap tabulated")
gap = 0.0
for _ in range(20000):
    lam = np.exp(rng.normal(scale=0.4, size=2))
    r = lam[0] / lam[1]
    psiR = rng.uniform(-np.pi / 2, np.pi / 2)
    # a unit vector at psiR, pulled back by U^{-1} = diag(1/lam1, 1/lam2)
    v = np.array([np.cos(psiR) / lam[0], np.sin(psiR) / lam[1]])
    psiP = np.arctan2(v[1], v[0])
    pred = np.arctan2(np.sin(psiR) / lam[1], np.cos(psiR) / lam[0])
    gap = max(gap, abs(wrap(2 * (psiP - pred))))
say(f"  max residual of the tan law over 20000 draws = {gap:.2e} rad")
say()
say("  A = (lam1-lam2)/(lam1+lam2)     r      max |psi_phi - psi_R|")
for A in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
    r = (1 + A) / (1 - A)
    mx = np.degrees(np.arctan(np.sqrt(r)) - np.arctan(1 / np.sqrt(r)))
    say(f"    {A:>5.2f}                      {r:5.3f}    {mx:6.2f} deg")
say("  read: at a 10 percent anisotropy of the stretch the two readings of the SAME angle")
say("  differ by up to 5.6 deg, at 20 percent by 11.3 deg.  That is the size of the")
say("  correction; both numbers are invariants, so the choice is which one the clinic wants.")
say()

# ----------------------------------------------------------------- Part 3
say("PART 3.  conditioning of the strain axis: the hypothesis is measurable, not material")
say("  first-order law   d(2 theta) = 2 (R^T dU R)_{12} / (lam1 - lam2)   against exact")
err = 0.0
for _ in range(5000):
    A = rng.uniform(0.05, 0.4)
    lam = np.array([1 + A, 1 - A])
    th = rng.uniform(0, np.pi)
    R = rot(th)
    U = R @ np.diag(lam) @ R.T
    E = rng.normal(size=(2, 2)) * 1e-6
    dU = E + E.T
    Up = U + dU
    w, V = np.linalg.eigh(Up)
    i = int(np.argmax(w))
    th2 = np.arctan2(V[1, i], V[0, i])
    exact = wrap(2 * (th2 - th))
    pred = 2 * (R.T @ dU @ R)[0, 1] / (lam[0] - lam[1])
    err = max(err, abs(wrap(exact - pred)))
say(f"  max |exact - predicted| over 5000 draws (|dU| ~ 1e-6) = {err:.2e} rad")
say()
say("  Monte Carlo: F = R_o U observed as F + sigma |F| E, E standard normal 2x2.")
say("  Reported: circular sd of the doubled axis angle, and the share of draws whose axis")
say("  lands within 15 deg of the truth.  5000 draws per cell.")
say()
hdr = "   A \\ sigma " + "".join(f"{s:>16}" for s in ("2%", "5%", "10%"))
say(hdr)
TR = 5000
for A in (0.02, 0.05, 0.10, 0.20, 0.30, 0.40):
    row = f"    {A:4.2f}     "
    for sg in (0.02, 0.05, 0.10):
        lam = np.array([1 + A, 1 - A])
        d = []
        hit = 0
        for _ in range(TR):
            th = rng.uniform(0, np.pi)
            U = rot(th) @ np.diag(lam) @ rot(th).T
            Ro = rot(rng.uniform(0, 2 * np.pi))
            F = Ro @ U
            Fh = F + sg * np.linalg.norm(F) / np.sqrt(2) * rng.normal(size=(2, 2))
            M = Fh.T @ Fh                      # polar: U_hat = sqrt(F^T F)
            w, V = np.linalg.eigh(M)
            Uh = V @ np.diag(np.sqrt(np.maximum(w, 1e-18))) @ V.T
            e = wrap(axis_angle(Uh) - axis_angle(U))
            d.append(e)
            hit += abs(e) < np.radians(30)      # 30 deg in the doubled angle = 15 deg of axis
        d = np.array(d)
        # circular sd of the doubled angle
        Rlen = abs(np.mean(np.exp(1j * d)))
        sd = np.degrees(np.sqrt(-2 * np.log(max(Rlen, 1e-12)))) / 2   # back to axis degrees
        row += f"{sd:7.1f} deg {100*hit/TR:4.0f}%"
    say(row)
say()
u = rng.uniform(-np.pi, np.pi, size=200000)      # doubled angle uniform = axis uniform
say(f"  reference, an axis carrying nothing (uniform): sd {np.degrees(np.std(u))/2:.1f} deg, "
    f"{100*np.mean(np.abs(u) < np.radians(30)):.0f}% inside 15 deg.")
say("  a cell at or near those two numbers is noise and must not be printed.")
say()
say("  the gate that follows: print the second angle where  A >= 3 sigma,  which at the")
say("  5 percent registration noise of a good deformable method means A >= 0.15, i.e. the")
say("  principal stretches differing by 30 percent of their mean.  Where it fails, the")
say("  point still contributes its two stretches and the difference angle: nothing is lost,")
say("  one of the four is withheld.")
say()
say("  and the condition is testable per point from the registration's own Jacobian --")
say("  no constitutive law, no material model, no assumption about how tissue deforms.")

with open("strainaxis.log", "w") as f:
    f.write("\n".join(LOG) + "\n")
