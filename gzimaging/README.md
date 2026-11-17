# gzimaging

**P. Grozman's operator and the shape of a growing nodule**

H. Andersén, V. Grozman, D. Leites, O. Lozhechnyk. Status: in preparation.

The scripts check the computations of the text: the contact geometry of the space of contact
elements PT^*Sigma of the boundary surface and the vanishing of every k(3)-invariant pairing on
pulled-back tensor fields (Section 3); the pairing of two curvature functions, the census of the
natural bilinear scalars of a pair of metrics on a surface under O(2) and under SO(2), the value of
the operator kGz on the tensor fields lifted by the metric, its three harmonics c_1, c_3, c_5 and
their conformal behavior (Section 3); the joint invariants of a registered pair of surfaces, the
two readings of the angle between the axes of asymmetry, the frame-free pairing under a constant
stretch (the Blaschke factor and the leak into higher harmonics), the change of pose, the three
phantoms, the coherence rho, its noise floor, the magnitude coherence, the focal change,
imperfect labels and the simulated cohort (Section 5); the signal-to-noise law of an order
(Appendix D), and the pipeline of Appendix F on an ellipsoid patch and on a triangulated mesh.
Exact arithmetic wherever a symbolic derivation exists (sympy over Q, `fractions.Fraction`, the
linear algebra of `exactlin.py`, fast when python-flint is installed and correct without it);
numpy/scipy floating point for the Monte Carlo simulations, the quadratures and the meshes, the
random seeds being fixed.

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r gzimaging/requirements.txt
cd gzimaging
../venv/bin/python legend.py
```

Every script is run from inside this folder, with no arguments. Most of them also write their
output to a `.log` file of the same name. `harm39.py` reuses the operator that `prop39.py`
solves for and caches in `prop39_ops.json`, so run `prop39.py` first. Times are for a laptop
running several scripts at a time.

The section, statement and formula numbers quoted in the docstrings and in the printed lines of
some scripts are those of earlier versions of the text; the table below gives the numbers of the
current one.

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Remark 2.3.1 (the finite-dimensional analog on T^*R: hGz vanishes on the pull-backs; its value on the Sasaki-type lift) and Remark 3.11 (the pairing of Statement 3.11 is -1/\|p\| times the Poisson bracket of T^*Sigma) | `tomega.py` (seconds) | `pull-backs only (lam=0): hGz(pi^*A, pi^*B) = [0, 0, 0]`; the horizontal block `lam*(3*a(q)*Derivative(b(q), (q, 2)) + 3*b(q)*Derivative(a(q), (q, 2)) + 4*Derivative(a(q), q)*Derivative(b(q), q))`, mixed block `0`, vertical block `lam**2*(a(q)*Derivative(b(q), (q, 2)) + b(q)*Derivative(a(q), (q, 2)))`; `Poisson bracket on T*R^2 / Statement-3.11 bracket = -1/r` |
| Proposition 3.6 (the adapted Darboux frame of PT^*Sigma, constant structure functions) and Statement 3.7 (on 16 pairs of pull-backs kGz and the order-zero contractions vanish, the symmetric product does not; controls; the naturality of the pull-back) | `legend.py` (10 s); `propchecks.py`, part D | `legend.py`: `kGz(pi^*A, pi^*B) = 0 on all 16 pairs  ok`, `two rank-one forms with different kernels: non-zero in 16 of 16  ok`, `kGz(pi^*A, pi^*B + (u^1)^2) non-zero in 9 of 9  ok`, `W=0, k_3=4: the symmetric product survives (9 of 9 values non-zero) ok`, `checks: 16 run, 0 failed`; `propchecks.py`: lines `D1`-`D12` all `ok` |
| Statement 3.11 (the pairing of two curvature functions: antisymmetric, zero for B = A and for two multiples of g, harmonics 1, 3, 5 only, mean 0, odd under theta -> theta + pi) | `curv.py` (3 min); `metriclift.py`, part 3 | `curv.py`: `only ODD harmonics occur (1,3,5)  ok`, `the bracket is odd (a section of L there)  ok`, `checks: 13 run, 0 failed`; `metriclift.py`: `PASS  Statement 3.11: the harmonics are 1, 3, 5 and no others` |
| Statement 3.12 (the O(2)-invariant scalars of weight 2: 15 = 10 symmetric + 5 antisymmetric, 4 + 5 + 4 + 2 curvature terms; the relation (dim2rel); 17 of weight at most 2; the SO(2) count: 1 at weight 0, 12 = 4 + 8 at weight 2, 30 coefficients of which 14 antisymmetric) | `metriclift.py` (10 s); `orient.py` (10 s) | `metriclift.py`: `TOTAL  15 = 10 symmetric + 5 antisymmetric`, `the 6 contractions span a space of dimension 5 -- so there is 1 relation`, `the relation, in the order listed:  -1  1  1  -1  -1  1`, `so the scalars of weight <= 2 make 17 = 12 symmetric + 5 antisymmetric`, `35 checks, 35 passed, 0 failed`; `orient.py`: `WEIGHT 2 TOTAL  12 = 4 symmetric + 8 antisymmetric`, `SO(2) total 30, of which 14 antisymmetric`, `43 checks, 0 failed` |
| Statement 3.13 (the lift (lambdag), the value (prop39), c = -2 lambda, no curvature term, harmonics 1, 3, 5) and the parity of (prop39) under a reflection | `prop39.py` (about 22 min; it writes `prop39_ops.json`, which `prop39curv.py` and `harm39.py` read); `prop39curv.py` (2 min); `lift.py` (parts 0-2 in about 5 min; part 3 did not finish, see below); `orient.py`, part 3 | `prop39.py`: `PASS  at W = 2 there is exactly one invariant, and it lands in F_3`, `PASS  the answer equals  2c s^{-3} [ (d_theta kA)(nabla kB) - (nabla kA)(d_theta kB) - (R_theta kA) kB + kA (R_theta kB) ]`, `kGz(Lam A, Lam B)            : [1, 3, 5]`, `40/40 checks pass`; `prop39curv.py`: `PASS  and they cancel: the curvature term of the answer is ZERO`, `18/18 checks pass`; `lift.py`: `PASS  the constant is -4, i.e. kGz(Lam A, Lam B) = -4 lambda ( ... ) / s^3`, `whence the paper's c and the vertical factor lambda are related by c = -2 lambda`, `PASS  hence the mixed component of a lift to the space of LINE elements is 0` (part 3, the other vertical candidates, ran over 40 min without finishing on this machine; formula (lambdag) is otherwise confirmed by `prop39.py` and `prop39curv.py`); `orient.py`: `[ok] F(sigma*A, sigma*B)(x,y,vartheta) = - F(A,B)(x,-y,-vartheta): it is ODD` |
| Statement 3.14 (the coefficients (cfive)-(cone), their reduction (creduced), (creducedone) under the Codazzi equation (codazzi), c_5 as the Wronskian of the two Hopf differentials) | `harm39.py` (15 s, after `prop39.py`) | `PASS  c_5 as given by H. Andersen is exactly the coefficient of z^5` (and for z^3, z^1), `PASS  c_3 collapses to (7i/4)(alpha d b - beta d a) + (i/2)(a d beta - b d alpha)`, `PASS  c_1 collapses to (i/2)(a d b - b d a) + (5i/4)(alpha dbar b - beta dbar a) - (i/4)(alpha-bar d beta - beta-bar d alpha)`, `PASS  surface A: the answer is left with 7 of its 9 real parameters, ...`, `PASS  c_5 = (3i/4) * (alpha d beta - beta d alpha), the Wronskian`, `26/26 checks pass`. Parseval's identity of Remark 3.14 is not printed by any script |
| Remark 3.14 (the conformal weight of c_5: exactly covariant, c_3 and c_1 not; what an uneven change of scale does to each) | `confharm.py`; `confclin.py` | `confharm.py` (1 min): `c5: residual after removing e^{-(2w+1)u} = 0`, `c3: residual ... = -I*(w - 5)*(a(x, y)*be(x, y) - al(x, y)*b(x, y))*(Derivative(u(x, y), x) - I*Derivative(u(x, y), y))*...`, and for different factors the c5 residual proportional to `al(x, y)*be(x, y)` times the gradient of `u - u2`; `confclin.py` (seconds): the table of `\|a beta - alpha b\|` at four points and the conclusion that the c_3 anomaly is multiplicative |
| Proposition 5.3 (the one joint quantity per point; item 5): parameter spaces of dimensions 6, 9, 12 at a point and 16, 23, 32 on the 1-jets, orbits of dimensions 4 and 10) and Proposition 5.9 item 1) | `pairjets.py` (seconds); `jointness.py` (20 s); `coherence.py`, parts 1-2 | `pairjets.py`: `one surface (g,II)  dim, orbit, invariants = (6, np.int64(4), np.int64(2))`, ..., `registered pair (g1,II1,g2,II2)dim, orbit, invariants = (32, np.int64(10), np.int64(22))`; `jointness.py`: `=== 41 checks, 0 failed` |
| Remark 5.3 on the two readings of the angle (tan psi_phi = (lambda_1/lambda_2) tan psi_R; the table of max \|psi_phi - psi_R\|; the first-order law of the axis; the Monte Carlo 23.8 deg at A = 0.05 and 5.3 deg at A = 0.20 for 5 per cent noise; 52 deg for a uniform axis) | `strainaxis.py` (10 s) | the column `2.87 deg`, `5.74 deg`, `8.63 deg`, `11.54 deg`, `17.46 deg`, `23.58 deg` for A = 0.05, ..., 0.40; in the Monte Carlo table, the 5% column: `0.05 ... 23.8 deg`, `0.20 ... 5.3 deg`; `reference, an axis carrying nothing (uniform): sd 52.0 deg` |
| Statement 5.3.1 (the frame-free pairing under a constant stretch: items 1)-5), the Blaschke factor (blaschke), the leak into the odd harmonics, the ratio of harmonic 7 to harmonic 5 varying with the 2-jets) | `blaschke.py` (seconds); `framefree.py` (10 s); `iso.py` (2 min) | `blaschke.py`: `PASS  the fiber map is z -> (z - A)/(1 - A z) with A = (r-1)/(r+1)`, `18 checks, 18 passed, 0 failed`; `framefree.py`: `r = 2    : a7/a5 over the samples = [0.02 0.1  0.49]`, `27 checks, 27 passed, 0 failed`; `iso.py`: `PASS  that factor is ell^{-2}, ...`, `19 checks, 19 passed, 0 failed` |
| Statement 5.3.2 (formula (leak), its three-line proof, the splitting of the pairing, formula (hopfstrain)) | `leak.py` (4 min); `iso.py`, last part | `leak.py`: `33 checks, 33 passed, 0 failed`; `iso.py`: `PASS  its Hopf number picks up b*(l_1^2-l_2^2)/2 -- the strain anisotropy entering through the TRACE of II_2` |
| Lemma 5.3.4 (a change of pose leaves the joint quantities unchanged) | `nuisance.py` (seconds) | lines `[ok] visit 2 re-posed: ... unchanged`, ending `all checks passed` |
| Lemma 5.6 (c_5 = (3i/4) alpha^2 d(beta/alpha); c_5 = 0 does not force a constant ratio) and the twist of Proposition 5.11 | `phase.py` (seconds); `orient.py`, part 1 | `phase.py`: `PASS c_5 = (3i/4) alpha^2 d(beta/alpha)`, `PASS yet beta/alpha = zbar is NOT constant`, `31 checks, 0 failed`; `orient.py`: `[ok] tau = 1/4 (k1-k2)_A (k1-k2)_B sin 2(phi_A - phi_B)` |
| Example 5.7 (the three phantoms: dphi over [-0.66, +0.59] on the circle of radius 1/4, mean -0.015, the coherent twin at t_0 = -0.0074, c_5 = (9/32)(1-i)) | `jointness.py`, part 6 | `-0.6267  -0.6629 ... +0.5896 ...`, `the mean of dphi over the circle = -0.01475646`, `at t0 = -0.00737823`, `c_5 at the centre, incoherent pair = 9/32 - 9*I/32` |
| Proposition 5.9 items 2)-3), Example 5.9 (1-rho = 0.0255, 0.0976, 0.3010 against 0.00008, 0.0013, 0.0160 at r = 1/8, 1/4, 1/2) and Remark 5.9 on the noise (the deficit 0.0098 and 0.061; the ratios 0.93, 0.99 and 0.97, 0.99; the spread 0.0038, 0.0020, 0.0010; the separations 5.2, 9.9, 19.5 and 1.8, 3.7, 7.2 standard deviations) | `coherence.py` (30 s) | `0.125  0.00008018  0.02553011`, `0.250  0.00126370  0.09757707`, `0.500  0.01598583  0.30102015`; `sigma = 0.10: floor 1-E[rho] = ['0.00904', '0.00968', '0.00981'], ratios over N x 4 = ['0.93', '0.99']`, `sigma = 0.25: floor 1-E[rho] = [..., '0.06070'], ratios ... ['0.97', '0.99']`, `sd = ['0.00383', '0.00197', '0.00100']`, the last column `5.2`, `9.9`, `19.5`, `1.8`, `3.7`, `7.2`; `27 checks, 0 failed` |
| Remark 5.9 on the magnitude coherence (\|alpha\|^2 = H^2 - K, rho_W = rho . rho_mag <= 1, rho = 1 forces beta = e^{i theta} h alpha, the umbilics) | `willmore.py` (seconds) | `[ok] rho_W = rho . rho_mag on every draw`, `31 checks, 0 failed` |
| Remark 5.9 on the magnitude coherence, last paragraph (the coherence deficit and the percentile of \|c_5\| cross at a patch of some 5 to 10 percent of the window under correlated noise) | `focal.py` (2 min) | the table `=== CORRELATED noise ...`: `focal` up to `eps = 0.050`, `global` from `eps = 0.100` on; under `=== WHITE noise ===` `global` throughout |
| Proposition 5.13 (E[rho^2] = 1/N_eff, rho^2_adj centered, N_eff <= N; the share of negative rho^2_adj, about 62 per cent) | `propchecks.py` (seconds), part E | `ok   E5  N = 10: E[rho^2_adj] = 0 and it goes negative   [mean 0.0002, negative in 62.2% of draws, least -0.111]`, `66 checks, 0 failed` |
| Propositions 5.15, 5.16 and Remarks 5.16 (imperfect labels: random flips attenuate and do not rotate; a blind spot acts only among the coordinates the reader reads; lambda = -0.37 for a reader flipping with probability 0.95) | `labels.py` (20 s); `propchecks.py`, parts A, B | `labels.py`: `84 checks, 0 failed`; `propchecks.py`: `ok   B6  and yet lambda < 0   [lambda = -0.370957, kappa_agg = -0.856544]` |
| Statement 5.18 (table (tab:cohort): the AUCs of the simulated cohort, the correlated-noise rows, the negative controls in [0.48, 0.52], size alone in [0.40, 0.58]) | `cohort.py` (2 min) | the rows, e.g. `A    (0.0, 0.05)  (0.1, 0.2)  0.10 \|  0.941  0.506   0.505    0.982`, `B    (0.0, 0.15)  (0.6, 1.2)  0.25 \|  0.913  0.546   0.781    0.991`; `control A: both classes (0.0, 0.05): AUC rho 0.486, separable 0.501, sep+rho 0.520` |
| Lemma A.4 (det omega = (b . e)^2; the stabilizer in the rotations is trivial for b, e in general position and 1-dimensional for b parallel to e) | `appendices.py` (30 s), part E | `[ok] det(omega) = (b.e)^2, so nondegeneracy is exactly b.e != 0`, `[ok] generic (b,e): the stabilizer in SO(3) has dimension 0`, `[ok] b parallel to e, still nondegenerate: dimension 1, a single axis` |
| Proposition B.3 (naturality under every diffeomorphism; tr S = 0, S^2 + K = 0, (IIIIK) <II,II> = 2K; the pointwise invariants generated by K, resp. by the three pairings; the cap and the ridge) | `appendices.py`, parts A-D | `[ok] <II,II>_omega = 2K  (arbitrary chart, symbolic)`, `[ok] one form: ambient 3, generic orbit 2 -> exactly 1 invariant (K)`, `[ok] a pair: ambient 6, generic orbit 3 -> exactly 3 invariants`, `[ok] K agrees (1 = 1) while H does not (1 against 5/4)` |
| Proposition C.3 (the volume may grow while the area falls) | `propchecks.py`, part F | `ok   F1  a needle and a ball, both diffeomorphic to S^2   [V: 0.8378 -> 0.9048 (grows), A: 9.8772 -> 4.5239 (falls)]` |
| Statement D.3 (the law (snrlaw), C_k = 1, 0.858, 1.155, 1.008, 1.171), table (tab:snr), Remark D.3 (the factor 25, 6.9 mm), Lemma D.4 (s^* = L sqrt(2k+1), the factors 1.2-1.5, the merge threshold 2 sqrt(L^2+s^2)), Remark D.4 on robustness (-5 and +13 per cent at s = h, 0.5 per cent at s = 1.25h; colored noise: 2.00 and 2.06) | `appendices.py` (seconds), part F; `noise.py` (seconds); `propchecks.py`, part C | `appendices.py`: `C_k = 0:1.00000, 1:0.85776, 2:1.15470, 3:1.00790, 4:1.17108`, the rows `1/11.1 1/2.6`, `1/21.3 1/4.9`, `1/131.5 1/5.8`, `1/577.2 1/25.6`, `exact 6.87 mm = 1.37 L`, `k = 3: SNR(k)/SNR(0) = 1/1.49, but at s* = 2.65 L`, `[ok] at s/h = 1 the k=3 ratio is 0.954 and the k=4 ratio 1.125`, `CT worse by 2.00x`, `CT worse by 2.06x`, `55 checks, 55 passed, 0 failed`; `noise.py`: `analytic formula confirmed: True`; `propchecks.py`: `ok   C4  L=2.0 s=3.0: merge threshold is 2 sqrt(L^2+s^2) = 7.211` |
| Table (tab:snr3) of Remark D.4 | `orient.py`, part 4 | `2.0 & 3.0 & 0.67 & 3 & 1/5.8 & 1/2.1`, `2.0 & 5.0 & 0.40 & 3 & 1/19.4 & 1/3.1`, `3.0 & 3.0 & 1.00 & 3 & 1/2.8 & 1/1.6`, `3.0 & 5.0 & 0.60 & 3 & 1/7.3 & 1/2.2` |
| Appendix F, steps 1)-6), and Remark F.1 (with dA the two parameterizations agree, rho = 0.9414, turn 22.818 deg; without dA, 24.63 deg against 28.94 deg) | `recipe.py` (seconds) | `[size] with dA:    rho 0.9414 / 0.9414,  turn 22.818 / 22.818 deg`, `[size] without dA: rho 0.9303 / 0.9302,  turn 24.633 / 28.936 deg`, `32 checks, 0 failed` |
| Statement F.5 (the exponents beta = 2, nu = 3 of a mesh estimator; the best radius; 24 deg at r = 0.3 against 8.4 deg at r = 0.45) and Remark F.5 (pointwise x21, regional turn 0.429 -> 0.413, shift 0.016 deg against 0.47; +2.2 and +14.4 per cent; the Rice mean +13.6 per cent, rho 0.829 against 0.841; at 1 per cent the numerator moves by 6.6 per cent) | `mesh.py` (2 min) | `beta = 1.984 (turn), 2.168 (J)`, `nu = 2.897 +- 0.161 over the three noise levels`, `24.7 deg at 0.30 against 8.4 at 0.45`, `pointwise x21, regional turn 4%`, `0.016 deg against the bound 0.470`, `denominator 2.93290 against 2.56308 (+14.4%, predicted +13.6%)`, `predicted 0.8289 against measured 0.8409`, `\|E sum\| is +6.6% off`, `26 checks, 0 failed`. The script also prints a predicted law `r* ~ sigma^0.099`, which is the exponent of an earlier draft; its own measurement (r* grows by 1.329 over a factor 4 in sigma) agrees with the text's sigma^(1/5) |
| Remark F.5 (the regional turn 0.461, 0.429, 0.364 deg at r = 0.2, 0.3, 0.4 against the analytic 0.537 deg) | not deposited | these four numbers come from H. Andersén's own script for a triangulated ellipsoid, which is hers and is not included here |

## Files

- `legend.py` -- the space of contact elements PT^*Sigma, the pull-back of tensor fields and
  kGz on them; uses `contact.py`, `gen.py`, `affine.py`, `fast.py` (the engine of the census of
  k(3)-invariant bilinear operators, shared with the deposit `grozall/contact/`).
- `curv.py` -- the curvature functions on the unit tangent bundle and their pairing.
- `metriclift.py`, `orient.py` -- the natural bilinear scalars of a pair of metrics on a surface,
  under O(2) and under SO(2); the twist; Appendix D at other scales. `orient.py` imports
  `metriclift.py`.
- `prop39.py`, `prop39curv.py`, `lift.py`, `harm39.py` -- kGz solved for from invariance and
  evaluated on the lifted tensor fields, flat and curved; the harmonics c_1, c_3, c_5.
  `prop39curv.py` imports `prop39.py`.
- `confharm.py`, `confclin.py` -- the conformal behavior of c_5, c_3, c_1.
- `pairjets.py`, `jointness.py`, `coherence.py`, `phase.py`, `willmore.py`, `strainaxis.py`,
  `nuisance.py` -- the joint invariants of a registered pair, the phantoms, the coherences, the
  pose, the strain axis.
- `blaschke.py`, `framefree.py`, `leak.py`, `iso.py` -- the frame-free pairing under a constant
  stretch.
- `focal.py`, `labels.py`, `cohort.py` -- the focal change, imperfect labels, the simulated cohort.
- `propchecks.py` -- checks of Propositions 3.6, 5.13, 5.15, 5.16, C.3 and Lemma D.4.
- `appendices.py`, `noise.py` -- the signal-to-noise law of Appendix D (and the pairing of
  Appendix B).
- `recipe.py`, `mesh.py` -- the pipeline of Appendix F on an ellipsoid patch and on a
  triangulated mesh.
- `tomega.py` -- the two identities of Remarks 2.3.1 and 3.11.
- `exactlin.py` -- exact rational linear algebra (python-flint if installed, pure Python
  otherwise).
