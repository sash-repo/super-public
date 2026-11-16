# shclumn

**The classification of simple complex Lie superalgebras of polynomial vector fields and their
deformations**

D. Leites, I. Shchepochkina. Status: survey for *Russian Math. Surveys* (*Uspekhi Mat. Nauk*);
an earlier version is arXiv:2406.16760. The text is printed from one source in two versions:
the journal version, and the arXiv version, which in addition carries Sections 12 and 13. The
numbers below are those of the text; Sections 1–11 are numbered alike in both versions.

Most of the survey rests on published results, many of them obtained with the *Mathematica*
package SuperLie (P. Grozman); that code is not included here. This folder holds only the
computations behind the claims proved by computation in the present text: the cohomology of
spe(n) and pe(n), the rigidity of sle'(n) in degree −2, the rigidity of four of the five
exceptional simple vectorial Lie superalgebras and the degree 0 for kle(9|6) (Section 9),
Golod's elliptic algebra (Section 12) and the deformations of the distinguished stringy Lie
superalgebras (Section 13). Everything is exact arithmetic over Q (via `../lib/exactlin.py`);
`family.py` and `lamscan.py` work over Q(lambda) with sympy.

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r shclumn/requirements.txt
cd shclumn
../venv/bin/python spechk.py
```

Every script is run from inside this folder; it finds `../lib/exactlin.py` by itself and prints
its results to the screen.

## Claims of the text and the scripts behind them

Times are wall-clock times measured on a laptop running about ten such jobs at once; on an idle
machine expect them to be several times shorter. Runs without a stated time take under five
minutes; the longest is about three hours (`weights.py`, `sd_census.py family`).

### Section 9 (both versions)

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Statement 9.2.1 (the corrected form of formula (155) for H(spe(n)), H(pe(n))): 1) H(spe(2)) = C[x_1,y_2]/(x_1^2 y_2); 2) H^1(pe(3)) = C, H^1(spe(3)) = 0, H^3(spe(3)) = C[det], H^3(pe(3)) = 0; 3) dim H^4(spe(4)) = 2, spanned by [Pf]^2, [det]; 4) H(pe(2)) = Lambda(xi_1) | `spechk.py` | `H^k(spe(2)) : {0: 1, 1: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2}`, `H^k(pe(2)) : {0: 1, 1: 1, 2: 0, ...}`, `H^k(spe(3)) : {0: 1, 1: 0, 2: 0, 3: 1, ...}`, `H^k(pe(3)) : {0: 1, 1: 1, 2: 0, 3: 0, ...}`, `H^k(spe(4)) : {0: 1, 1: 0, 2: 1, 3: 0, 4: 2}`; `verdict: the relation holds in every degree computed`; `x1^2*y2 : ZERO modulo coboundaries`; `d=0..6: ... OK`; `[Pf]^2 and [det] span 2 classes in H^4(spe(4))` |
| Lemma 9.5.1 (H^1(spe(n)) = 0, H^2 = C[Pf] for n = 4 and 0 otherwise, 3 <= n <= 8; H^3 = C[det] for n = 3, C[Pf] for n = 6, 0 for n = 4, 5); Remark 9.5.2 (H^p(pe(n)) = 0 for p = 2, 3, n = 3, 4; H^1(pe(n)) = C) | `spe.py 6 3` (degrees <= 3, n <= 6, about 5 min); `spe.py 8 2` (degrees <= 2, n <= 8, about 35 min) | `H^k(spe(3)) : {0: 1, 1: 0, 2: 0, 3: 1}`, `H^k(spe(4)) : {0: 1, 1: 0, 2: 1, 3: 0}`, `H^k(spe(5)) : {0: 1, 1: 0, 2: 0, 3: 0}`, `H^k(spe(6)) : {0: 1, 1: 0, 2: 0, 3: 1}`, `H^k(pe(3))`, `H^k(pe(4)) : {0: 1, 1: 1, 2: 0, 3: 0}`; then `H^k(spe(7)) : {0: 1, 1: 0, 2: 0}`, `H^k(spe(8)) : {0: 1, 1: 0, 2: 0}` |
| Lemma 9.4.2, heading 2) (H(Delta) = C Xi, [sb, sb] = Im Delta, codimension 1), a check for n <= 5 | `sle.py 5 8` (about 10 min) | for each n: `H(Delta) is nonzero exactly in degree(s) [n] (dim [1])`, `[sb,sb] = Im(Delta) in every computed degree: True`, `dim g_0 = ... , dim spe(n) = ...` equal |
| Theorem 9.5.4, heading 1): H^2(sle'(4); sle'(4))_{-2} = 0; Remark 9.4.3: the same for n = 3 and 5 | `sle4rig.py 4`, `sle4rig.py 3`, `sle4rig.py 5` (about 6, 1 and 12 min) | `RESULT for sle'(n), degree -2:  dim H^2(g;g) <= 0 (even 0, odd 0)` for n = 4, 3, 5 |
| Theorem 9.6.3: H^2(g; g) = 0 for vle(4\|3), vas(4\|4), kas(1\|6), mb(4\|5) | `run2.py` (vle, kas, vas; about 40 min); `run3.py` (mb(4\|5); about 45 min); an independent second implementation for the depth-1 cases: `run1.py` | `run2.py`: `dim Z\|det <= 46 dim B = 46 B in Z: True` (vle(4\|3) and kas(1\|6)), `<= 60 ... = 60` (vas(4\|4)); `run3.py`: `dim Z\|det <= 71 dim B = 71 B in Z: True`; `run1.py`: `dim Z <= 46, dim B = 46` (vle, kas), `dim Z <= 60, dim B = 60` (vas), `0` in parity 1 |
| Section 9.6, after Conjecture 9.6.4: the derivations of kle(9\|6;K) of degree 0 and weight 0 form a 5-dimensional space, the inner ones 4 | `kleder.py` (about 30 min) | `outer: 1` (5 against 4) |
| Lemma 9.6.6: a check of the model of kle(9\|6;K) (super Jacobi, transitivity, generation, deg = 2 sigma mod 5) | `klecheck.py` | `Jacobi ... 0 failures`; component dimensions `{-2: 5, -1: 10, 0: 24, 1: 40, 2: 70, 3: 105, 4: 160}`; the congruence holds in every degree |
| Statement 9.6.7: H^2(kle(9\|6); kle(9\|6))_0 = 0 | `klerun.py 0 4 2 0` | `k=0: cochains 939, ... -> cocycles <= 236, coboundaries 236, B not in Z: 0` |

### Section 12 (arXiv version): Golod's elliptic algebra

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Lemma 12.3.3 (the relations are brackets of functions on the curve; Jacobi; the loop algebra at A = 0); Lemma 12.4.1 (the deforming cocycle is a coboundary); Proposition 12.4.2 (the family is formally trivial); Lemma 12.5.2 and the paragraph after it (the centroid is C_A; in a finite range of degrees it has dimension 2D−1 for A ≠ 0 and 2D+1 for A = 0); Theorem 12.5.3 (genus of the curve) | `golod_run.py` | `35 checks, 35 passed, 0 failed` |

### Section 13 (arXiv version): deformations of the distinguished stringy Lie superalgebras

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Proposition 13.3.4 (the grading element is inner for every algebra of the list and for svect^L_0(1\|2), outer for svect^L_lambda(1\|2), lambda ≠ 0); the engine itself: super Jacobi for every algebra, the grading, and the calibration (the bracket is a cocycle and is delta(id)) | `test_defco.py` | `72/72 checks pass` |
| Statement 13.4.1 (dim H^2(g; C) = 1 for the eight Laurent algebras, 0 for the six polynomial counterparts) | `central.py` | a table with columns `N Z B h`: `h` = 1 for witt, k^L(1\|1), k^L(1\|2), k^L(1\|3), k^M(1\|1), k^M(1\|2), vect^L(1\|1), m^L(1), and `h` = 0 for vect(1), k(1\|1), k(1\|2), k(1\|3), vect(1\|1), m(1), each with `(N=3,4,5: 1,1,1)`, resp. `(N=3,4,5: 0,0,0)` |
| Statement 13.5.1 (formal rigidity of the eight algebras of (SD:rigidlist)), and Subsec. 13.10.9 (k^L(1\|2) and vect^L(1\|1) give the same counts) | `sd_census.py laurent` (about 2 hours); the polynomial counterparts: `sd_census.py poly` (about 7 min) | `h=0` for every algebra at the largest windows (N = 5, 6, resp. 6, 7) |
| Remark 2) after Statement 13.5.1, Proposition 13.11.1: H^2 vanishes in the weights w ≠ 0 for every algebra with an inner grading element | `weights.py` (about 3 hours) | `h(w) for w=[-2, -1, 0, 1, 2]:  [0, 0, 0, 0, 0]` for the seven algebras with inner E that it runs (k^L(1\|3) is not run; for it the claim rests on the proof) |
| Statement 13.6.3 (at lambda_0 = 1/3, d mu/d lambda is not a coboundary on the core \|d\| <= 1, so H^2(svect^L_lambda(1\|2); svect^L_lambda(1\|2)) ≠ 0) | `family.py` | `super Jacobi identically in lambda: OK`, `controls: ... YES; ... YES`, `d mu / d lambda is a coboundary on the core \|deg\| <= 1: NO` |
| Statement 13.8.5 (the same at lambda_0 = 1/2, 1/3, 2/3, 3/2, 5/7, 22/7, 2; the frame has a pole at 0 and −1) | `lamscan.py` | `NOT a coboundary (so H^2 != 0) at lambda_0 = 2, 1/2, 1/3, 2/3, 3/2, 5/7, 22/7`, `frame singular, not tested: lambda_0 = 0, -1` |
| Proposition 13.9.2, item 1) (in weight 0 the even classes of svect^L_lambda(1\|2), lambda_0 = 1/3, 2/5, form a 1-dimensional space); Subsec. 13.9.5 (at lambda_0 = 0 and 1 the counts jump to 2 and 0) | `sd_census.py family` (about 3 hours) | `h=1` for lambda = 1/3 and 2/5 at N = 3, 4, 5; `h=2` at lambda = 0 (N = 4, 5) and `h=0` at lambda = 1 |
| Proposition 13.9.2, item 2) (no class in the weights w ≠ 0, either parity); Subsec. 13.9.3 (in k^M(1\|n) and m^L(1) odd cochains of weight 0 exist and carry no cohomology); item 3) (no odd cochain of weight 0 for svect^L_lambda(1\|2)) | `others.py` (about 1.5 hours) | part A: `h=0` everywhere, with `unk > 0` exactly for k^M(1\|1), k^M(1\|2), m^L(1); part B: `unk=0`; part C: `h=0` for w = ±1, ±2 and both parities |
| Statement 13.10.3 (the charges C_g(a): −2(6a^2−6a+1) for witt, 3(4a−1) for k^L(1\|1) and k^M(1\|1), −6 for k^L(1\|2) and k^M(1\|2), 0 for k^L(1\|3)) | `ghost.py` (about 10 min) | `fit: C(a) = -12 a^2 + 12 a + -2` (witt), `0 a^2 + 12 a + -3` (k^L(1\|1), k^M(1\|1)), `0 a^2 + 0 a + -6` (k^L(1\|2), k^M(1\|2)), `0 a^2 + 0 a + 0` (k^L(1\|3)) |
| Subsec. 13.10.8 (the charge vanishes for n = 4 as well; also for k^M(1\|3), k^M(1\|4)) | `charges34.py` | `C=0` for every weight a, for k^M(1\|3), k^M(1\|4), k^L(1\|4) |
| Subsec. 13.10.7 (for k(1\|2) the charge does not depend on the second weight b either: C = −6) | `mumford.py` | for sigma = +1: `b=...  C for a=-1,0,1/2,1:  -6  -6  -6  -6` for every b |

## Files

- `spe.py` — Chevalley–Eilenberg cohomology of pe(n), spe(n) on the weight-0 subcomplex;
  `spechk.py` — Statement 9.2.1 on top of it.
- `sle.py` — b(n), the odd Laplacian Delta, sb(n), sle'(n); `sle4rig.py` — the degree −2 part of
  H^2(sle'(n); sle'(n)).
- `vfcore.py`, `excalg.py`, `models.py`, `mb.py` — polynomial vector fields on a superspace, the
  Cartan prolong, the exceptional algebras vle(4|3), kas(1|6), vas(4|4), mb(4|5);
  `cocyc.py` (generic depth) and `rigid1.py` (depth 1) — H^2(g; g)_0; drivers `run1.py`,
  `run2.py`, `run3.py`.
- `kle.py` — the model of kle(9|6;K) (Kac's realization); `klecoh.py` — H^2(g; g)_k;
  drivers `klerun.py` (`klerun.py census kmin kmax dmax` counts the weight-0 cochains),
  `kleder.py`, `klecheck.py`.
- `golod.py`, `golod_run.py` — Golod's elliptic su(2)-algebra, its trivializing series and its
  centroid.
- `stringy.py`, `yb.py` — functions on the superstring and the distinguished stringy Lie
  superalgebras; `defco.py` — H^2(g; g) by the window method, `test_defco.py`; `central.py`,
  `sd_census.py`, `weights.py`, `others.py` — the censuses; `family.py`, `lamscan.py` —
  svect^L_lambda(1|2) over Q(lambda); `ghost.py`, `xieta.py`, `mumford.py`, `charges34.py` —
  semi-infinite forms and the charges C_g(a). The last block of `ghost.py` tries the second
  weight b of k^L(1|2) in the orthonormal (theta) frame and reports that the operators do not
  form a representation there (`the b-family is not built`); the second divergence is written
  in the hyperbolic xi-eta frame, which `xieta.py` builds and `mumford.py` uses.
