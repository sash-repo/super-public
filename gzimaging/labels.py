"""Imperfect labels, and what they can and cannot do to a centroid classifier.

Two questions about a classifier built on the descriptors of the text:

  (i)  clinician-defined labels are not a perfect ground truth and may introduce bias
       into the model; a model that merely learns the readers' interpretation of the
       images would inherit their blind spots;

  (ii) can the model be built from meaningful mathematical quantities from the start,
       as a concept-based explainable classifier?

Both have exact answers, and this script computes them.  The setting throughout is the
one of a minimal classifier: a feature vector X in R^p per longitudinal pair, two classes with
means mu_M, mu_B and a common covariance Sigma, and a rule fitted from the empirical
centroids -- which, with a pooled covariance, IS Fisher's linear discriminant,
w = Sigmahat^{-1} (muhat_M - muhat_B).

The label is not Y but Yhat, obtained from Y by flipping with probabilities that may
depend on the features:

        P(Yhat = B | Y = M, x) = h_M(x),    P(Yhat = M | Y = B, x) = h_B(x).

PART 1.  h constant -- a reader who errs at random.  Then, exactly and for every
         asymmetric pair of rates,

             Deltahat = kappa Delta,  kappa = w + w' - 1,
             Sigmahat = Sigma + c Delta Delta^T,
             what = kappa/(1 + c d^2) . Sigma^{-1} Delta.

         The estimated discriminant is PARALLEL TO THE TRUE ONE.  Random label noise
         attenuates and does not rotate: the reported separation shrinks by
         kappa^2/(1 + c d^2), the ranking of the lesions does not change at all, and the
         AUC against the TRUE classes is the noise-free one.  A study run on noisy labels
         is therefore conservative -- it can lose a finding, it cannot manufacture one.

PART 2.  h a function of the READER-VISIBLE coordinates only -- which is what a blind
         spot is: the reader errs where the image looks a certain way to him.  Write R
         for the projection onto those coordinates.  Then (Theorem, proved in the text):

             what in span{Sigma^{-1} Delta} + {vectors supported on R}.

         Whence, for every coordinate j the reader does NOT read,

             what_j = lambda . (Sigma^{-1} Delta)_j,  one and the same lambda.

         The blind spot cannot create, destroy or reverse the weight on a coordinate it
         cannot see; it can only act among the coordinates it can.  The proof is that
         the observed-class density is the true one tilted by a function of (Rx, u) with
         u = Delta^T Sigma^{-1} x, and such a tilt moves the mean and the covariance only
         inside the image of Sigma[R^T | Sigma^{-1}Delta] -- exactly the (r+1) directions
         Woodbury then returns to the same span.  Checked here to machine precision on
         probit and probit-mixture blind spots, where every Gaussian moment is closed
         form and nothing is sampled.

         The corollary is the design rule, and it is her own rule arriving from a second
         direction: the guarantee holds for the ADJUSTED coefficient, the one fitted with
         the reader-visible features in the model.  The UNADJUSTED association of a
         descriptor is biased, by an amount proportional to its covariance with the
         reader-visible block -- a quantity measurable on the cohort with no ground truth
         whatever.  So x_S, x_V, x_G are not only the control that makes a gain
         attributable to jointness; they are the shield against the reader.

PART 3.  What fails when the hypotheses do -- unequal class covariances, heavy tails,
         a blind spot that reads a coordinate we thought it could not.  Measured, not
         asserted.

PART 4.  The concept layer, and the audit no learned concept can have.  Each coordinate
         of the vector has a DECLARED transformation law under a group acting on the
         datum: c_n |-> e^{i n psi} c_n under a rotation of the frame, and under the swap
         of the two visits J |-> conj J, whence taubar |-> -taubar, rho invariant,
         x_G |-> -x_G, c_5 |-> -c_5.  A coordinate that has silently absorbed information
         beyond its definition -- concept leakage, the standard failure of a concept
         bottleneck -- fails one of these identities.  The test is run here, on the
         honest coordinates and on a leaked one, and the leak is caught.

Run:  python3 labels.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LOG = None
NCHECK = 0
NFAIL = 0


def say(*a):
    msg = " ".join(str(x) for x in a)
    print(msg)
    if LOG:
        LOG.write(msg + "\n")
        LOG.flush()


def check(name, ok):
    global NCHECK, NFAIL
    NCHECK += 1
    if not ok:
        NFAIL += 1
    say(("  OK   " if ok else "  FAIL ") + name)


def close(a, b, tol=1e-10):
    return np.max(np.abs(np.asarray(a) - np.asarray(b))) < tol


# ---------------------------------------------------------------------------
# Gaussian moments of a probit tilt, in closed form (nothing sampled).
#
#   x ~ N(mu, Sigma),  g(x) = Phi(a + b^T x)
#   E[g]      = Phi(u),                       u = m/sqrt(1+v), m = a + b^T mu, v = b^T Sigma b
#   E[x g]    = mu E[g] + Sigma b . phi(u)/sqrt(1+v)
#   E[(x-mu)(x-mu)^T g] = Sigma E[g] + Sigma b b^T Sigma . E[-s phi(s)],
#                         E[-s phi(s)] = -(m/(1+v)) phi(u)/sqrt(1+v)
# the last by the second-order Stein identity, grad^2 g = -(a+b^T x) phi(a+b^T x) b b^T.
# ---------------------------------------------------------------------------


def _Phi(z):
    from math import erf, sqrt

    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _phi(z):
    from math import exp, pi, sqrt

    return exp(-0.5 * z * z) / sqrt(2.0 * pi)


def probit_moments(a, b, mu, Sigma):
    """(E[g], E[x g], E[x x^T g]) for g = Phi(a + b^T x)."""
    from math import sqrt

    b = np.asarray(b, float)
    mu = np.asarray(mu, float)
    m = a + b @ mu
    v = float(b @ Sigma @ b)
    s = sqrt(1.0 + v)
    u = m / s
    G = _Phi(u)
    Sb = Sigma @ b
    E_xg = mu * G + Sb * (_phi(u) / s)
    E_spp = -(m / (1.0 + v)) * _phi(u) / s
    E_cc = Sigma * G + np.outer(Sb, Sb) * E_spp
    E_xxg = E_cc + np.outer(mu, E_xg) + np.outer(E_xg, mu) - np.outer(mu, mu) * G
    return G, E_xg, E_xxg


def mixture_moments(terms, mu, Sigma):
    """h(x) = sum_k wk Phi(a_k + b_k^T x); returns (E[h], E[x h], E[x x^T h])."""
    p = len(mu)
    G = 0.0
    Exg = np.zeros(p)
    Exxg = np.zeros((p, p))
    for wk, ak, bk in terms:
        g, xg, xxg = probit_moments(ak, bk, mu, Sigma)
        G += wk * g
        Exg += wk * xg
        Exxg += wk * xxg
    return G, Exg, Exxg


def observed(muM, muB, Sigma, piM, hM, hB):
    """Exact observed centroids and pooled covariance under the flip model.

    hM, hB are lists of probit terms (w, a, b); a constant rate eta is the single term
    (eta, +inf, 0), which we encode as (eta, 40.0, zeros) -- Phi(40) = 1 to 1e-300.
    """
    p = len(muM)
    piB = 1.0 - piM
    gM, xM, xxM = mixture_moments(hM, muM, Sigma)  # E_M[h_M], E_M[x h_M], ...
    gB, xB, xxB = mixture_moments(hB, muB, Sigma)  # E_B[h_B], ...
    # Yhat = M  <-  (Y=M, not flipped)  or  (Y=B, flipped)
    qM = piM * (1.0 - gM) + piB * gB
    qB = piM * gM + piB * (1.0 - gB)
    num_m = piM * (muM - xM) + piB * xB
    num_b = piM * xM + piB * (muB - xB)
    muhM = num_m / qM
    muhB = num_b / qB
    S_M = Sigma + np.outer(muM, muM)
    S_B = Sigma + np.outer(muB, muB)
    sec_m = piM * (S_M - xxM) + piB * xxB
    sec_b = piM * xxM + piB * (S_B - xxB)
    ShM = sec_m / qM - np.outer(muhM, muhM)
    ShB = sec_b / qB - np.outer(muhB, muhB)
    Sigmah = qM * ShM + qB * ShB
    return muhM, muhB, Sigmah, qM, qB


def const(eta):
    return [(eta, 40.0, None)] if eta > 0 else []


def _fill(terms, p):
    return [(w, a, np.zeros(p) if b is None else np.asarray(b, float)) for w, a, b in terms]


def rand_problem(rng, p, scale=1.0):
    A = rng.normal(size=(p, p))
    Sigma = A @ A.T / p + np.eye(p) * 0.5
    Delta = rng.normal(size=p) * scale
    muB = rng.normal(size=p) * 0.3
    muM = muB + Delta
    return muM, muB, Sigma


def fisher(muhM, muhB, Sigmah):
    return np.linalg.solve(Sigmah, muhM - muhB)


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------
# PART 1.  Random label noise: attenuation without rotation.
# ---------------------------------------------------------------------------


def part1():
    say("")
    say("=" * 78)
    say("PART 1.  A reader who errs at random: the rule does not rotate")
    say("=" * 78)
    rng = np.random.default_rng(20260915)

    for trial in range(6):
        p = 5 + trial
        muM, muB, Sigma = rand_problem(rng, p)
        Delta = muM - muB
        piM = float(rng.uniform(0.25, 0.75))
        etaM = float(rng.uniform(0.02, 0.30))
        etaB = float(rng.uniform(0.02, 0.30))
        muhM, muhB, Sigmah, qM, qB = observed(
            muM, muB, Sigma, piM, _fill(const(etaM), p), _fill(const(etaB), p)
        )
        piB = 1 - piM
        w = piM * (1 - etaM) / qM
        wp = piB * (1 - etaB) / qB
        kap = w + wp - 1.0
        c = qM * w * (1 - w) + qB * wp * (1 - wp)
        d2 = float(Delta @ np.linalg.solve(Sigma, Delta))

        if trial == 0:
            say("")
            say("  the closed forms, against the exact observed moments:")
        check(
            "p=%d  Deltahat = kappa.Delta       (kappa = %.4f)" % (p, kap),
            close(muhM - muhB, kap * Delta),
        )
        check(
            "p=%d  Sigmahat = Sigma + c.DD^T    (c     = %.4f)" % (p, c),
            close(Sigmah, Sigma + c * np.outer(Delta, Delta)),
        )
        what = fisher(muhM, muhB, Sigmah)
        wtrue = np.linalg.solve(Sigma, Delta)
        lam = kap / (1.0 + c * d2)
        check("p=%d  what = lambda.Sigma^{-1}Delta EXACTLY, lambda = %.4f" % (p, lam), close(what, lam * wtrue))
        check("p=%d  cos(what, w_true) = 1 to 1e-14" % p, abs(cosine(what, wtrue) - 1) < 1e-14)
        d2h = float((muhM - muhB) @ np.linalg.solve(Sigmah, muhM - muhB))
        check(
            "p=%d  dhat^2 = kappa^2 d^2/(1+c d^2): %.4f vs %.4f" % (p, d2h, kap**2 * d2 / (1 + c * d2)),
            abs(d2h - kap**2 * d2 / (1 + c * d2)) < 1e-10,
        )
        check("p=%d  dhat^2 <= d^2 (the study is conservative)" % p, d2h <= d2 + 1e-12)

    # the symmetric balanced case, where the constants are the familiar ones
    say("")
    say("  the balanced symmetric case eta_M = eta_B = eta, pi = 1/2:")
    p = 6
    muM, muB, Sigma = rand_problem(rng, p)
    Delta = muM - muB
    d2 = float(Delta @ np.linalg.solve(Sigma, Delta))
    say("     d = %.3f (the noise-free separation)" % np.sqrt(d2))
    say("     eta      kappa=1-2eta    c=eta(1-eta)   dhat/d")
    for eta in (0.0, 0.05, 0.10, 0.20, 0.30, 0.40):
        muhM, muhB, Sigmah, qM, qB = observed(
            muM, muB, Sigma, 0.5, _fill(const(eta), p), _fill(const(eta), p)
        )
        kap = 1 - 2 * eta
        c = eta * (1 - eta)
        ok = close(muhM - muhB, kap * Delta) and close(Sigmah, Sigma + c * np.outer(Delta, Delta))
        d2h = float((muhM - muhB) @ np.linalg.solve(Sigmah, muhM - muhB))
        say("     %.2f     %.4f          %.4f         %.4f" % (eta, kap, c, np.sqrt(d2h / d2)))
        check("eta=%.2f  the two closed forms" % eta, ok)

    # the operating point, which does move under asymmetric noise
    say("")
    say("  what DOES move: the threshold, under ASYMMETRIC rates")
    p = 4
    muM, muB, Sigma = rand_problem(rng, p)
    Delta = muM - muB
    mid0 = 0.5 * (muM + muB)
    for etaM, etaB in ((0.1, 0.1), (0.25, 0.05), (0.05, 0.25)):
        muhM, muhB, Sigmah, qM, qB = observed(
            muM, muB, Sigma, 0.5, _fill(const(etaM), p), _fill(const(etaB), p)
        )
        w = 0.5 * (1 - etaM) / qM
        wp = 0.5 * (1 - etaB) / qB
        t = 0.5 * (w - wp)
        mid = 0.5 * (muhM + muhB)
        check(
            "eta=(%.2f,%.2f)  midpoint shifts by (w-w')/2 = %+.4f along Delta" % (etaM, etaB, t),
            close(mid, mid0 + t * Delta),
        )

    # the ranking, hence the AUC against the TRUE labels, is untouched
    say("")
    say("  the consequence for a validation: the ranking is unchanged")
    p = 8
    muM, muB, Sigma = rand_problem(rng, p)
    Delta = muM - muB
    muhM, muhB, Sigmah, _, _ = observed(
        muM, muB, Sigma, 0.4, _fill(const(0.22), p), _fill(const(0.11), p)
    )
    what = fisher(muhM, muhB, Sigmah)
    wtrue = np.linalg.solve(Sigma, Delta)
    XM = rng.multivariate_normal(muM, Sigma, size=40000)
    XB = rng.multivariate_normal(muB, Sigma, size=40000)

    def auc(v):
        sM, sB = XM @ v, XB @ v
        allv = np.r_[sM, sB]
        ranks = np.empty(len(allv))
        ranks[np.argsort(allv)] = np.arange(1, len(allv) + 1)
        nM = len(sM)
        return float((ranks[:nM].sum() - nM * (nM + 1) / 2) / (nM * len(sB)))

    a_noisy, a_true = auc(what), auc(wtrue)
    d = float(np.sqrt(wtrue @ Delta))
    say("     AUC of the rule fitted on NOISY labels, scored on TRUE ones: %.6f" % a_noisy)
    say("     AUC of the noise-free rule                                 : %.6f" % a_true)
    say("     the value Phi(d/sqrt 2) for d = %.3f                        : %.6f"
        % (d, _Phi(d / np.sqrt(2))))
    check("the two rules rank identically (same AUC exactly)", abs(a_noisy - a_true) < 1e-12)
    check("and that AUC is the noise-free Phi(d/sqrt 2) to sampling error",
          abs(a_true - _Phi(d / np.sqrt(2))) < 5e-3)

    # finite samples: the bias is the population one, the cost is power
    say("")
    say("  finite samples, n = 200 per class, 1200 repetitions.  A single fit is noisy")
    say("  whatever the labels, so bias is read off the MEAN direction, not off one fit:")
    p = 6
    muM, muB, Sigma = rand_problem(rng, p)
    Delta = muM - muB
    wtrue = np.linalg.solve(Sigma, Delta)
    wtrue = wtrue / np.linalg.norm(wtrue)

    def series(eta, reps=1200, n=200):
        dirs = []
        for _ in range(reps):
            XM = rng.multivariate_normal(muM, Sigma, size=n)
            XB = rng.multivariate_normal(muB, Sigma, size=n)
            X = np.vstack([XM, XB])
            y = np.r_[np.ones(n), np.zeros(n)]
            yh = np.where(rng.random(2 * n) < eta, 1 - y, y)
            A, B = X[yh == 1], X[yh == 0]
            S = (np.cov(A.T) * (len(A) - 1) + np.cov(B.T) * (len(B) - 1)) / (2 * n - 2)
            v = np.linalg.solve(S, A.mean(0) - B.mean(0))
            dirs.append(v / np.linalg.norm(v))
        dirs = np.array(dirs)
        per = np.array([cosine(v, wtrue) for v in dirs])
        mean_dir = dirs.mean(0)
        return per.mean(), cosine(mean_dir, wtrue)

    say("     eta      mean cos of ONE fit    cos of the MEAN direction")
    ang = {}
    for eta in (0.0, 0.15, 0.30):
        per, mean_c = series(eta)
        ang[eta] = mean_c
        say("     %.2f     %.4f                 %.5f" % (eta, per, mean_c))
    check("the mean direction is the true one whatever eta (cos > 0.999)",
          min(ang.values()) > 0.999)
    say("     -- so the cost of label noise at finite n is VARIANCE, not a rotation.")


# ---------------------------------------------------------------------------
# PART 2.  A blind spot: h depends on the reader-visible coordinates only.
# ---------------------------------------------------------------------------


def part2():
    say("")
    say("=" * 78)
    say("PART 2.  A blind spot acts only among the coordinates the reader reads")
    say("=" * 78)
    rng = np.random.default_rng(892892)

    # first: the closed-form moments themselves, against Monte Carlo
    say("")
    say("  the closed forms for a probit tilt, against 4,000,000 samples:")
    p = 4
    muM, muB, Sigma = rand_problem(rng, p)
    a, b = 0.3, rng.normal(size=p)
    G, Exg, Exxg = probit_moments(a, b, muM, Sigma)
    X = rng.multivariate_normal(muM, Sigma, size=4_000_000)
    s = a + X @ b
    g = np.array([_Phi(z) for z in s[:400000]])
    Xs = X[:400000]
    check("E[g]      closed form %.6f vs MC %.6f" % (G, g.mean()), abs(G - g.mean()) < 4e-3)
    check("E[x g]    closed form vs MC (max dev %.4f)" % np.max(np.abs(Exg - (Xs * g[:, None]).mean(0))),
          np.max(np.abs(Exg - (Xs * g[:, None]).mean(0))) < 3e-2)
    mc2 = np.einsum("ni,nj,n->ij", Xs, Xs, g) / len(Xs)
    check("E[x x^T g] closed form vs MC (max dev %.4f)" % np.max(np.abs(Exxg - mc2)),
          np.max(np.abs(Exxg - mc2)) < 8e-2)

    # THE THEOREM
    say("")
    say("  THE THEOREM.  R = the reader-visible coordinates; h_M, h_B supported on them.")
    say("  For every j NOT in R, what_j / (Sigma^{-1}Delta)_j is one and the same lambda.")
    say("")
    for trial in range(8):
        p = 9
        r = 4  # the reader reads coordinates 0..3
        muM, muB, Sigma = rand_problem(rng, p)
        Delta = muM - muB
        # a genuinely non-monotone blind spot: a mixture of three probits on R only
        def mk():
            terms = []
            for _ in range(3):
                bb = np.zeros(p)
                bb[:r] = rng.normal(size=r) * 1.5
                terms.append((float(rng.uniform(0.05, 0.18)), float(rng.normal()), bb))
            return terms

        hM, hB = mk(), mk()
        muhM, muhB, Sigmah, qM, qB = observed(muM, muB, Sigma, 0.45, hM, hB)
        what = fisher(muhM, muhB, Sigmah)
        wtrue = np.linalg.solve(Sigma, Delta)
        ratios = what[r:] / wtrue[r:]
        spread = float(np.max(ratios) - np.min(ratios))
        check(
            "trial %d: lambda = %.5f, spread over the %d unseen coordinates = %.2e"
            % (trial, ratios.mean(), p - r, spread),
            spread < 1e-9,
        )
        if trial == 0:
            seen = what[:r] / wtrue[:r]
            say("           and among the SEEN ones the ratios are %s"
                % np.array2string(seen, precision=3, floatmode="fixed"))
            check("the seen coordinates ARE rotated (their ratios differ)",
                  float(np.max(seen) - np.min(seen)) > 1e-3)

    # the corollary: adjusted vs unadjusted, on the MVP's own vector
    say("")
    say("  THE COROLLARY, on the MVP's own vector.  p = 14:")
    say("     reader-visible R = x_V, x_G, x_S (1+1+8 = 10 coordinates)")
    say("     not read        = x_J, taubar, rho^2_adj, x_5 (4 coordinates)")
    p, r = 14, 10
    rng2 = np.random.default_rng(14141)
    for rho_dr in (0.0, 0.2, 0.5):
        # Sigma with a prescribed descriptor-reader correlation, built by construction:
        # each unseen coordinate is rho . f + sqrt(1-rho^2) . (its own part), where f is
        # one normalized combination of the reader-visible coordinates.
        q = p - r
        f = rng2.normal(size=r)
        f /= np.linalg.norm(f)
        Sig_RR = np.eye(r)
        C = rho_dr * np.outer(np.ones(q), Sig_RR @ f)
        Sig_DD = np.eye(q) * (1 - rho_dr**2) + rho_dr**2 * np.ones((q, q))
        Sigma = np.block([[Sig_RR, C.T], [C, Sig_DD]])
        ev = np.linalg.eigvalsh(Sigma)
        if ev.min() <= 1e-8:
            Sigma += np.eye(p) * (1e-8 - ev.min())
        Delta = np.zeros(p)
        Delta[:r] = 0.35            # the reader-visible signal
        Delta[r:] = np.array([0.10, 0.45, -0.30, 0.25])   # x_J, taubar, rho2, x_5
        muB = np.zeros(p)
        muM = muB + Delta
        # the blind spot: the reader errs on small, slowly growing lesions -- x_V, x_G
        bb = np.zeros(p)
        bb[0], bb[1] = -1.4, -1.1
        hM = [(0.40, 0.2, bb)]
        hB = [(0.25, -0.1, bb)]
        muhM, muhB, Sigmah, qM, qB = observed(muM, muB, Sigma, 0.5, hM, hB)
        what = fisher(muhM, muhB, Sigmah)
        wtrue = np.linalg.solve(Sigma, Delta)
        lam = float(np.mean(what[r:] / wtrue[r:]))
        spread = float(np.max(what[r:] / wtrue[r:]) - np.min(what[r:] / wtrue[r:]))
        # unadjusted: the marginal mean difference of each unseen coordinate
        unadj_true = Delta[r:]
        unadj_hat = (muhM - muhB)[r:]
        # scale both by the same attenuation, to compare directions not sizes
        rel = unadj_hat / unadj_true
        say("")
        say("     corr(descriptor, reader block) = %.1f   (cond Sigma = %.1f)"
            % (rho_dr, np.linalg.cond(Sigma)))
        say("       ADJUSTED   coefficients: lambda = %.4f, spread over the four = %.2e"
            % (lam, spread))
        check("corr=%.1f: lambda is not degenerate (%.4f)" % (rho_dr, lam), lam > 1e-3)
        say("       UNADJUSTED coefficients: ratios %s"
            % np.array2string(rel, precision=4, floatmode="fixed"))
        say("         -- their spread %.4f, i.e. %.1f%% of their mean"
            % (rel.max() - rel.min(), 100 * (rel.max() - rel.min()) / abs(rel.mean())))
        check("corr=%.1f: the adjusted coefficients are exactly proportional" % rho_dr, spread < 1e-9)
        if rho_dr == 0.0:
            check("corr=0: the unadjusted ones are proportional too (nothing to inherit)",
                  rel.max() - rel.min() < 1e-9)
        else:
            check("corr=%.1f: the unadjusted ones are NOT (the bias is real)" % rho_dr,
                  rel.max() - rel.min() > 1e-3)

    say("")
    say("  Whence the rule: report the coefficient of a descriptor ADJUSTED for what the")
    say("  reader reads, never the raw association; and the number that bounds the damage")
    say("  to the raw one -- the descriptor-reader covariance -- needs no ground truth.")


# ---------------------------------------------------------------------------
# PART 3.  Where the theorem stops.
# ---------------------------------------------------------------------------


def part3():
    say("")
    say("=" * 78)
    say("PART 3.  What fails when the hypotheses do")
    say("=" * 78)
    rng = np.random.default_rng(31313)
    p, r, n = 9, 4, 200_000

    def sim(kind):
        muM, muB, Sigma = rand_problem(rng, p)
        Delta = muM - muB
        L = np.linalg.cholesky(Sigma)
        if kind in ("gauss", "peek"):
            ZM, ZB = rng.normal(size=(n, p)), rng.normal(size=(n, p))
        elif kind == "t3":
            g = rng.chisquare(3, size=(n, 1)) / 3
            ZM = rng.normal(size=(n, p)) / np.sqrt(g)
            g = rng.chisquare(3, size=(n, 1)) / 3
            ZB = rng.normal(size=(n, p)) / np.sqrt(g)
            ZM /= np.sqrt(3.0)
            ZB /= np.sqrt(3.0)
        elif kind == "unequal":
            # ANISOTROPIC: an isotropic Sigma_M/Sigma_B leaves Sigma^{-1}Delta fixed up
            # to scale and so cannot break anything.  Stretch the two classes along
            # different coordinates instead.
            sM = np.ones(p)
            sB = np.ones(p)
            sM[: p // 2] = 1.8
            sB[p // 2:] = 1.8
            ZM = rng.normal(size=(n, p)) * sM
            ZB = rng.normal(size=(n, p)) * sB
        XM, XB = muM + ZM @ L.T, muB + ZB @ L.T
        X = np.vstack([XM, XB])
        y = np.r_[np.ones(n), np.zeros(n)]
        s = -1.4 * X[:, 0] - 1.1 * X[:, 1]
        if kind == "peek":
            # the reader turns out to read one coordinate we had put among the unseen
            s = s + 1.3 * X[:, r]
        h = np.array([0.40 * _Phi(0.2 + z) for z in s])
        yh = np.where(rng.random(2 * n) < h, 1 - y, y)
        A, B = X[yh == 1], X[yh == 0]
        S = (np.cov(A.T) * (len(A) - 1) + np.cov(B.T) * (len(B) - 1)) / (2 * n - 2)
        what = np.linalg.solve(S, A.mean(0) - B.mean(0))
        wtrue = np.linalg.solve(Sigma, Delta)
        # the theorem says the two agree in DIRECTION on the unseen block; a ratio is
        # unstable where a true coefficient is near zero, so measure the angle
        u, v = what[r:], wtrue[r:]
        cs = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
        return float(np.degrees(np.arccos(min(1.0, max(-1.0, cs)))))

    say("")
    say("  angle between the fitted and the true discriminant RESTRICTED to the")
    say("  coordinates the reader does not read -- 0 is the theorem, and with")
    say("  n = %d per class about a degree is sampling noise:" % n)
    res = {}
    for kind, label in (
        ("gauss", "Gaussian, common Sigma  (the hypotheses)"),
        ("t3", "t_3 tails, common Sigma  (heavy tails)"),
        ("unequal", "Gaussian, anisotropic Sigma_M vs Sigma_B"),
        ("peek", "the reader DOES read one 'unseen' coordinate"),
    ):
        vals = [sim(kind) for _ in range(10)]
        res[kind] = float(np.mean(vals))
        say("     %-42s %5.2f deg  (range %.2f-%.2f over 10 problems)"
            % (label, res[kind], min(vals), max(vals)))
    check("under the hypotheses the angle is sampling noise (< 1.5 deg)", res["gauss"] < 1.5)
    check("heavy tails degrade it (t_3 angle exceeds the Gaussian one)", res["t3"] > res["gauss"])
    check("a reader who reads an 'unseen' coordinate breaks it (angle > 2x the noise)",
          res["peek"] > 2 * res["gauss"])
    say("")
    say("  Read the third line as it stands: unequal spreads did NOT visibly degrade the")
    say("  direction in this configuration, and no claim is made that they cannot.  The")
    say("  fourth line is the failure that matters, and it is not a defect of the theorem")
    say("  but its hypothesis: R is whatever the reader ACTUALLY uses, and a coordinate")
    say("  wrongly placed outside it loses the guarantee entirely.")

    say("")
    say("  And the hypothesis that cannot be tested away: the model posits a true class Y")
    say("  that the reader mislabels.  Where the label is the only definition of the class,")
    say("  there is nothing for a descriptor to be unbiased FOR.  That is a matter of the")
    say("  endpoint -- histology, or growth over a long enough arm -- not of the algebra.")


# ---------------------------------------------------------------------------
# PART 4.  The concept layer and its audit.
# ---------------------------------------------------------------------------


def part4():
    say("")
    say("=" * 78)
    say("PART 4.  The equivariance audit: a test a learned concept cannot have")
    say("=" * 78)
    rng = np.random.default_rng(4444)
    N = 4000

    # a synthetic pair of Hopf differentials and mean curvatures over a patch
    al = rng.normal(size=N) + 1j * rng.normal(size=N)
    be = rng.normal(size=N) + 1j * rng.normal(size=N)
    d_al = rng.normal(size=N) + 1j * rng.normal(size=N)
    d_be = rng.normal(size=N) + 1j * rng.normal(size=N)
    m = np.abs(al) * np.abs(be)
    V1, V2 = 900.0, 1180.0

    def coords(al, be, d_al, d_be, V1, V2, leak=0.0):
        J = np.conj(al) * be
        res = np.sum(m * J / np.abs(J)) / np.sum(m)
        tau = res.imag + leak * float(np.mean(np.abs(al) ** 2))
        rho2 = abs(res) ** 2
        c5 = 0.75j * (al * d_be - be * d_al)
        x5 = np.sqrt(np.mean(np.abs(c5) ** 2))
        xG = np.log(V2 / V1)
        xV = 0.5 * (np.log(V1) + np.log(V2))
        return dict(tau=tau, rho2=rho2, x5=x5, xG=xG, xV=xV, c5mean=np.mean(c5))

    base = coords(al, be, d_al, d_be, V1, V2)

    # (a) a rotation of the frame by psi: a Hopf field has spin 2, c_5 has spin 5
    psi = 0.7
    alr, ber = al * np.exp(2j * psi), be * np.exp(2j * psi)
    d_alr, d_ber = d_al * np.exp(3j * psi), d_be * np.exp(3j * psi)
    rot = coords(alr, ber, d_alr, d_ber, V1, V2)
    check("frame rotation: taubar invariant", abs(rot["tau"] - base["tau"]) < 1e-10)
    check("frame rotation: rho^2 invariant", abs(rot["rho2"] - base["rho2"]) < 1e-10)
    check("frame rotation: |c_5| invariant, c_5 turns by e^{5 i psi}",
          abs(rot["x5"] - base["x5"]) < 1e-10
          and abs(rot["c5mean"] - base["c5mean"] * np.exp(5j * psi)) < 1e-10)

    # (b) the swap of the two visits
    sw = coords(be, al, d_be, d_al, V2, V1)
    check("visit swap: taubar |-> -taubar", abs(sw["tau"] + base["tau"]) < 1e-10)
    check("visit swap: rho^2 invariant", abs(sw["rho2"] - base["rho2"]) < 1e-10)
    check("visit swap: x_G |-> -x_G", abs(sw["xG"] + base["xG"]) < 1e-12)
    check("visit swap: x_V invariant", abs(sw["xV"] - base["xV"]) < 1e-12)
    check("visit swap: c_5 |-> -c_5, whence |c_5| invariant",
          abs(sw["c5mean"] + base["c5mean"]) < 1e-10 and abs(sw["x5"] - base["x5"]) < 1e-10)

    # (c) independent rescaling of the two visits: rho and taubar are normalized
    s1, s2 = 1.7, 0.6
    sc = coords(s1 * al, s2 * be, s1 * d_al, s2 * d_be, V1, V2)
    check("independent rescaling: taubar invariant", abs(sc["tau"] - base["tau"]) < 1e-10)
    check("independent rescaling: rho^2 invariant", abs(sc["rho2"] - base["rho2"]) < 1e-10)
    check("independent rescaling: |c_5| scales by s_1 s_2 (it is NOT normalized)",
          abs(sc["x5"] - s1 * s2 * base["x5"]) < 1e-8)

    # (d) the leak, and its detection
    say("")
    say("  a LEAKED coordinate -- taubar with 5%% of a separable magnitude added to it,")
    say("  which is what an end-to-end fit does to a concept layer that is not complete:")
    lk = coords(al, be, d_al, d_be, V1, V2, leak=0.05)
    lk_sw = coords(be, al, d_be, d_al, V2, V1, leak=0.05)
    resid = abs(lk_sw["tau"] + lk["tau"])
    say("     honest taubar:  |tau(swap) + tau| = %.2e" % abs(sw["tau"] + base["tau"]))
    say("     leaked taubar:  |tau(swap) + tau| = %.4f" % resid)
    check("the swap test catches the leak", resid > 1e-3)
    say("")
    say("  The point is not the size of the leak but that there IS a test.  Each")
    say("  coordinate of the vector carries a declared transformation law -- a theorem of")
    say("  the manuscript, not a property fitted from data -- and a coordinate that has")
    say("  absorbed anything beyond its definition violates one of them.  A concept")
    say("  learned from annotations has no such identity to violate, and its annotations")
    say("  come from the very reader whose blind spots are the question.")


def main():
    global LOG
    LOG = open("labels.log", "w")
    say("=" * 78)
    say("Imperfect labels and a centroid classifier")
    say("=" * 78)
    part1()
    part2()
    part3()
    part4()
    say("")
    say("=" * 78)
    say("%d checks, %d failed" % (NCHECK, NFAIL))
    LOG.close()
    return 1 if NFAIL else 0


if __name__ == "__main__":
    sys.exit(main())
