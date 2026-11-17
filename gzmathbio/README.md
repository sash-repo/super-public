# gzmathbio

**The turn of the curvature axis of a growing nodule: a joint descriptor of two registered tomograms**

H. Andersén, V. Grozman, D. Leites, O. Lozhechnyk. Status: in preparation.

The scripts check the computations of the text: the number of invariants of a registered pair of
surfaces at a matched point, on the values and on the 1-jets of the two second fundamental forms,
with the Codazzi equation imposed (Statement 2.5.2); the three phantoms, their joint angle and the
top harmonic c_5 (Example 3.1.1); the coherence rho of the phantoms over disks of growing radius, and
its noise floor by Monte Carlo (Remark 2.9.4, Example 3.1.2); and the simulated cohort of
Subsection 3.2 with its AUCs (Statement 3.2.1). The phantoms and the invariant counts are computed
exactly (sympy over Q and Q(i), floating point only for the final evaluation); `pairjets.py`
computes ranks of the infinitesimal action numerically (numpy, at a random point); the cohort and the
noise floor are Monte Carlo runs with fixed seeds (numpy, the standard `random` module).

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r gzmathbio/requirements.txt
cd gzmathbio
../venv/bin/python jointness.py
```

Every script is run from inside this folder, with no arguments. `jointness.py` and `coherence.py`
also write their output to `jointness.log` and `coherence.log` in this folder. Times are for a
laptop running one script at a time.

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script | Expected output |
|---|---|---|
| Statement 2.5.2, item 1) (in the frame that R transports, the separable invariants of the 1-jets number 6+6, those of the pair 13; arg ∂β − arg ∂α is a function of Δφ and of the separable data) | `jointness.py` (about 25 seconds), part 3; `pairjets.py` (under a second) | `jointness.py`: `[ok] one surface's 1-jet has exactly 7 - 1 = 6 independent invariants`, `[ok] the joint invariants number exactly 14 - 1 = 13`, `[ok] arg(d beta) - arg(d alpha) is NOT a further independent invariant: it is a function of dphi and the per-visit data`; `pairjets.py`: `one surface (g,II)             dim, orbit, invariants = (16, np.int64(10), np.int64(6))`, `isometric pair (g,II1,II2)     dim, orbit, invariants = (23, np.int64(10), np.int64(13))` |
| Statement 2.5.2, item 2) (with the stretch U admitted, the pair has 22 invariants on the 1-jets, so 10 are not separable) | `pairjets.py` | `registered pair (g1,II1,g2,II2)dim, orbit, invariants = (32, np.int64(10), np.int64(22))` (22 = 6 + 6 + 10) |
| Remark 2.9.4 (the deficit 1 − E ρ of a coherent pair is 0.0098 at σ = 0.10 and 0.061 at σ = 0.25 and does not shrink as N grows; the standard deviation falls like N^(-1/2)) | `coherence.py` (about 35 seconds), part 6 | `sigma = 0.10: floor 1-E[rho] = ['0.00904', '0.00968', '0.00981'], ratios over N x 4 = ['0.93', '0.99']`, `sigma = 0.25: floor 1-E[rho] = ['0.05824', '0.05990', '0.06070'], ratios over N x 4 = ['0.97', '0.99']` (N = 12, 48, 192), `sigma = 0.10: sd = ['0.00383', '0.00197', '0.00100'], ratios over N x 4 = ['1.95', '1.97']` |
| Example 3.1.1 (the three phantoms: dome against ridge; the congruent follow-ups separated by Δφ = 2t_0 alone; the incoherent pair with Δφ over [−0.66, 0.59], mean −0.015, matched by the coherent pair at t_0 = −0.0074, c_5 = (9/32)(1 − i) against c_5 = 0) | `jointness.py`, part 6 | `dome    H = 1   \|alpha\| = 0   K = 1`, `ridge   H = 1   \|alpha\| = 1   K = 0`; the twelve values of dphi around the circle of radius 1/4, from `-0.6629` to `+0.5896`; `the mean of dphi over the circle = -0.01475646`; `at t0 = -0.00737823, whose dphi is that constant and whose c_5 is 0.`; `c_5 at the centre, incoherent pair = 9/32 - 9*I/32`; `c_5 at the centre, coherent pair   = 0`; finally `=== 41 checks, 0 failed` |
| Example 3.1.2 (1 − ρ of the incoherent pair is 0.0255, 0.0976, 0.3010 at r = 1/8, 1/4, 1/2, of its coherent twin 0.00008, 0.0013, 0.0160; orders r^2 and r^4; the separations 5.2, 9.9, 19.5 standard deviations at σ = 0.10 and 1.8, 3.7, 7.2 at σ = 0.25) | `coherence.py`, parts 4 and 6 | `0.125     0.00008018      0.02553011      318.4`, `0.250     0.00126370      0.09757707       77.2`, `0.500     0.01598583      0.30102015       18.8`; `[ok] the incoherent pair's defect is O(r^2) -- the leading effect`; in the Monte Carlo table the last column (gap/sd) reads `5.2`, `9.9`, `19.5` for sigma 0.10 and `1.8`, `3.7`, `7.2` for sigma 0.25, at N = 12, 48, 192; finally `27 checks, 0 failed` |
| Statement 3.2.1 and table (4) (the AUCs of 1 − ρ, of the best separable classifier and of both, in designs A and B; correlated noise; the negative controls in [0.48, 0.52]; the size alone in [0.40, 0.58]) | `cohort.py` (about 2 minutes) | the selftest lines, then one line per row, e.g. `A    (0.0, 0.05)  (0.1, 0.2)  0.10 \|  0.941  0.506   0.505    0.982  \| 0.945 / 0.876` (columns: AUC of 1 − ρ, size, separable, separable + ρ, mean ρ benign / malignant); under `spatially correlated curvature noise`: `0.920`, `0.994`, `0.627`, `0.892` in the AUC rho column; `control A: both classes (0.0, 0.05): AUC rho 0.486, separable 0.501, sep+rho 0.520`, `control B: both classes (0.0, 0.15): AUC rho 0.500, separable 0.511, sep+rho 0.508` |

## Files

- `pairjets.py` — the infinitesimal action of the 2-jets of diffeomorphisms fixing a point on the
  values and 1-jets of the metrics and second fundamental forms, Codazzi imposed; invariants counted
  as dimension minus orbit dimension.
- `jointness.py` — the frame covariance of the Hopf differential, the invariant counts at orders 0
  and 1, the top harmonic c_5, and the three phantoms as explicit graphs.
- `coherence.py` — the field J = conj(alpha) beta, the coherence rho, the phantoms read in rho, and
  the Monte Carlo noise floor.
- `cohort.py` — the simulated cohort: model nodules, the two designs, registration and curvature
  noise, logistic regressions scored by cross-validated AUC.
