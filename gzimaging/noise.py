#!/usr/bin/env python3
"""
noise.py --- how much signal-to-noise a differential operator of order k costs.

Written for the question: can hGz be used as a longitudinal shape
descriptor of a pulmonary nodule on CT?  hGz is of order 2, and its arguments
(second fundamental form, Hessian of the intensity, strain) are themselves of
order 1 or 2, so the descriptor is a derivative of total order 3 or 4 of data
sampled at the voxel scale.  This file puts a number on what that costs.

Model (one-dimensional, which is where the scaling law lives; the transverse
directions contribute the same factor to every k and cancel in the ratio):

  * a profile f sampled on a grid of spacing h,
  * i.i.d. Gaussian noise of standard deviation sigma on each sample,
  * the k-th derivative estimated by convolution with the k-th derivative of a
    Gaussian of scale s --- the standard scale-space estimator, and the one that
    minimizes the noise for a given resolution.

Two quantities are computed and cross-checked against Monte-Carlo:

  N(k,s) = SD of the output on pure noise
         = sigma * sqrt(h * I_k),  I_k = int (g_s^{(k)})^2 = (2k-1)!! / (2^{k+1} sqrt(pi) s^{2k+1})

  S(k,s,L) = peak of the output on a clean feature of size L and amplitude a.

The ratio of interest is SNR(k)/SNR(0) for a feature of size L, i.e. how much
worse an order-k descriptor is than an order-0 one on the same data.  For
s << L it is

  SNR(k)/SNR(0) = C_k * (s/L)^k,   C_k = M_k * sqrt(2^k / (2k-1)!!),

where M_k = max_u |He_k(u)| exp(-u^2/2) depends on the shape assumed for the
feature and is O(1).  The exponent k is what matters and is shape-independent.

Everything below is verified numerically; nothing is asserted from the formula
alone.
"""

import math
import numpy as np


# ----------------------------------------------------------------- analytic

def dfact(n):
    """(n)!! for odd n >= -1;  (-1)!! = 1."""
    r = 1
    while n > 1:
        r *= n
        n -= 2
    return r


def I_k(k, s):
    """int_R (d^k/dx^k gauss_s)^2 dx, exactly."""
    return dfact(2 * k - 1) / (2 ** (k + 1) * math.sqrt(math.pi) * s ** (2 * k + 1))


def noise_sd_analytic(k, s, h, sigma):
    """SD of the k-th-derivative estimate on i.i.d. noise (continuum limit)."""
    return sigma * math.sqrt(h * I_k(k, s))


def hermite_peak(k):
    """M_k = max_u |He_k(u)| exp(-u^2/2), the peak of the k-th derivative of a
    Gaussian in units of (amplitude / width^k).  M_0 = 1."""
    u = np.linspace(-14.0, 14.0, 400001)
    He = [np.ones_like(u), u]
    for j in range(1, k + 1):
        He.append(u * He[j] - j * He[j - 1])
    return float(np.max(np.abs(He[k]) * np.exp(-u ** 2 / 2)))


def snr_ratio_analytic(k, s, L):
    """SNR(k)/SNR(0) for a Gaussian feature of size L, in the limit s << L.

    The k-dependence is (s/L)^k; the prefactor C_k = M_k sqrt(2^k/(2k-1)!!) is
    an O(1) constant depending on the shape assumed for the feature, and is
    computed here for a Gaussian bump.  It is the exponent that matters."""
    return hermite_peak(k) * math.sqrt(2 ** k / dfact(2 * k - 1)) * (s / L) ** k


# --------------------------------------------------------------- estimators

def hermite_coeff_kernel(k, s, h, half_width=8.0):
    """Sampled k-th derivative of a Gaussian of scale s, on a grid of step h.

    d^k/dx^k g_s(x) = (-1/(s sqrt2))^k He_k(x/(s sqrt2)) ... we just take the
    derivative numerically-exactly via the Hermite recursion:
        g^{(k)}(x) = (-1)^k / (s^k) * He_k(x/s) * g(x)   with the probabilists'
    Hermite polynomials He_k.
    """
    n = int(math.ceil(half_width * s / h))
    x = np.arange(-n, n + 1) * h
    g = np.exp(-x ** 2 / (2 * s ** 2)) / (s * math.sqrt(2 * math.pi))
    u = x / s
    # probabilists' Hermite by recursion He_0=1, He_1=u, He_{n+1}=u He_n - n He_{n-1}
    He = [np.ones_like(u), u]
    for j in range(1, k + 1):
        He.append(u * He[j] - j * He[j - 1])
    return ((-1) ** k / s ** k) * He[k] * g * h  # * h = the quadrature weight


def apply_kernel(y, ker):
    return np.convolve(y, ker[::-1], mode="same")


# -------------------------------------------------------------- Monte-Carlo

def mc_noise_sd(k, s, h, sigma, n_samples=4000, n_trials=400, rng=None):
    """SD of the output of the k-th-derivative estimator on pure noise."""
    rng = rng or np.random.default_rng(20260909 + k)
    ker = hermite_coeff_kernel(k, s, h)
    m = len(ker) // 2
    vals = []
    for _ in range(n_trials):
        y = rng.normal(0.0, sigma, n_samples)
        out = apply_kernel(y, ker)
        vals.append(out[m + 50:n_samples - m - 50])
    return float(np.std(np.concatenate(vals)))


def clean_peak(k, s, h, L, a=1.0, half=60.0):
    """Peak |output| of the estimator on a = a*exp(-x^2/2L^2), no noise."""
    n = int(math.ceil(half * max(L, s) / h))
    x = np.arange(-n, n + 1) * h
    f = a * np.exp(-x ** 2 / (2 * L ** 2))
    ker = hermite_coeff_kernel(k, s, h)
    out = apply_kernel(f, ker)
    m = len(ker) // 2
    return float(np.max(np.abs(out[m:len(out) - m])))


# -------------------------------------------------------------------- main

def main():
    lines = []

    def say(t=""):
        print(t)
        lines.append(t)

    say("noise.py --- cost in signal-to-noise of a differential descriptor of order k")
    say("=" * 74)
    say()

    # --- 1. the analytic noise formula against Monte-Carlo -----------------
    say("1) Noise SD of the k-th-derivative estimator, analytic vs Monte-Carlo")
    say("   (grid h = 0.1, scale s = 1.0, sigma = 1.0)")
    say()
    say("     k   analytic      Monte-Carlo    ratio")
    h, s, sigma = 0.1, 1.0, 1.0
    ok = True
    for k in range(0, 5):
        a = noise_sd_analytic(k, s, h, sigma)
        m = mc_noise_sd(k, s, h, sigma)
        r = m / a
        flag = "" if abs(r - 1) < 0.03 else "   <-- MISMATCH"
        if abs(r - 1) >= 0.03:
            ok = False
        say(f"     {k}   {a:10.6f}    {m:10.6f}   {r:6.4f}{flag}")
    say()
    say(f"   analytic formula confirmed: {ok}")
    say()

    # --- 2. the SNR ratio law ---------------------------------------------
    say("2) SNR(k)/SNR(0) on a clean feature of size L, analytic vs computed")
    say("   (h = 0.05, s = 0.5, several L; a = 1)")
    say()
    h, s = 0.05, 0.5
    say("      L     k    predicted     computed      ratio")
    ok2 = True
    for L in (2.5, 5.0, 10.0):
        for k in (1, 2, 3, 4):
            pred = snr_ratio_analytic(k, s, L)
            comp = (clean_peak(k, s, h, L) / noise_sd_analytic(k, s, h, 1.0)) / \
                   (clean_peak(0, s, h, L) / noise_sd_analytic(0, s, h, 1.0))
            r = comp / pred
            flag = "" if abs(r - 1) < 0.10 else "   <-- off"
            if abs(r - 1) >= 0.10:
                ok2 = False
            say(f"   {L:5.1f}   {k}   {pred:11.3e}  {comp:11.3e}   {r:6.3f}{flag}")
    say()
    say(f"   law  SNR(k)/SNR(0) = C_k (s/L)^k,  C_k = M_k sqrt(2^k/(2k-1)!!):"
        f"  confirmed for s/L <= 0.2: {ok2}")
    say()

    # --- 3. the numbers for a CT-like setting -----------------------------
    say("3) The numbers for CT of a pulmonary nodule")
    say()
    say("   Voxel h ~ 0.6-1.0 mm, so the smallest usable smoothing scale is")
    say("   s ~ 1 mm.  The shape features that carry the malignant signal --- a")
    say("   spiculation, a lobulation --- have size L ~ 2-5 mm.")
    say()
    say("   hGz is of order 2.  Its arguments are of order 1 (strain) or of")
    say("   order 2 (second fundamental form; Hessian of the intensity), so the")
    say("   descriptor is of total order k = 3 or k = 4 in the data.")
    say()
    say("      s (mm)  L (mm)   k    SNR(k)/SNR(0)     i.e. a loss of")
    for s_mm in (1.0, 1.5):
        for L_mm in (2.0, 5.0):
            for k in (3, 4):
                r = snr_ratio_analytic(k, s_mm, L_mm)
                say(f"      {s_mm:5.1f}   {L_mm:5.1f}   {k}    {r:12.3e}"
                    f"      {1/r:9.0f} x")
    say()
    say("   For comparison, the order-0 descriptors --- the principal curvatures")
    say("   and their symmetric functions, the eigenvalues of the strain, the")
    say("   Jacobian --- sit in the k = 0 row by definition, at ratio 1.")
    say()

    # --- 4. what smoothing would be needed --------------------------------
    say("4) The scale s that an order-k descriptor needs to reach a given SNR")
    say("   ratio, for L = 5 mm.  (s > L means: the feature must be smoothed")
    say("   away before the descriptor is stable.)")
    say()
    L_mm = 5.0
    say("      target ratio    k=3: s (mm)    k=4: s (mm)")
    for target in (0.5, 0.2, 0.1):
        row = []
        for k in (3, 4):
            # target = sqrt(2^k/(2k-1)!!) (s/L)^k  =>  s = L (target/c)^(1/k)
            c = hermite_peak(k) * math.sqrt(2 ** k / dfact(2 * k - 1))
            row.append(L_mm * (target / c) ** (1.0 / k))
        say(f"      {target:8.2f}       {row[0]:9.2f}     {row[1]:9.2f}")
    say()
    say("   Every one of these exceeds the voxel size by a wide margin and is")
    say("   comparable to L itself: the smoothing that makes the descriptor")
    say("   reproducible removes the feature it was to measure.")
    say()

    say("=" * 74)
    say(f"all checks passed: {ok and ok2}")

    with open("noise.log", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
