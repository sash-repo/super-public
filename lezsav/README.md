# lezsav

**Superized Leznov-Saveliev equations are the zero-curvature condition on a reduced connection**

D. Leites. Status: in preparation.

The scripts check the computations of the text: the classical anomalies (central charges) of the
super Liouville equation (N = 1) and of the Ivanov-Krivonos equations (N = 2, 4), in the
Neveu-Schwarz and Ramond sectors; the constraints a contact algebra k^L(1|N) preserves; the
Ivanov-Krivonos algebra svect^L_lambda(1|2), its embedding into k^L(1|4), its Cartan matrices,
central extension and real forms; the N = 4 vacuum, the logarithmic completion of the stress
tensor and the level-1 obstruction for a non-degenerate leading loop; and the superdimensions and
characters of osp(3|2)- and osp_a(4|2)-modules. Exact arithmetic throughout (integers, fractions,
Gaussian rationals Q(i), sympy, and `../lib/exactlin.py`), except the minimizations of items 7)-9)
of Statement 7.2.7, which are floating-point (numpy, scipy).

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r lezsav/requirements.txt
cd lezsav
../venv/bin/python sliouv.py
```

Every script is run from inside this folder; it finds `../lib/exactlin.py` by itself. No arguments
are needed except where stated. Most scripts print their report and also write it to
`<name>-out.txt`; the scripts of Section 7 also write pickles, which later scripts read, so the
order "two steps", "three steps", ... below matters. Run times are those of a laptop with no other
load; where they exceed a few minutes they are stated.

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Theorem 4.2.1 (the classical anomaly of the super Liouville equation, c = -6; the super stress tensor and its law; the cocycle on k^L(1\|1)) | `sliouv.py`; `nscoc.py` | `sliouv.py`: `D^3 exp(-F) = T exp(-F) with T = (DF)(D^2F) - D^3F : True`; `nscoc.py`: the three relations of ns, `c = -6` in both sectors, `super-skew-symmetry: failures 0 of 64`, `cocycle identity   : failures 0 of 512` |
| Propositions 5.1.3 and 5.1.4 (which constraints k^L(1\|N) preserves: chirality for N = 2, the doublet for N = 4, no scalar constraint for N = 3 and N >= 5) | `ivkr.py` (about 2 min) | the table ending `N=3  V irreducible, so(3) simple    no SCALAR constraint`, `N=4  so(4)=su(2)+su(2), doublet-valued superfield`, `N>=5 V irreducible, so(N) simple    no SCALAR constraint` |
| Theorem 5.3.1 (the classical anomaly for N = 2, c = -3; the stress tensor is a conserved current) | `n2.py`; `n2cons.py`; `n2coc.py` (about 3 min) | `n2.py`: the operators annihilating Psi = exp(-v-v+); `n2cons.py`: `d_- T = 0 verified identically               18 checks   OK`; `n2coc.py`: `the three sectors agree:  c = -3`, `super-skew-symmetry of c                      196 checks   OK`, `the cocycle identity                          2744 checks   OK` |
| Remark 5.3.2: the central charge c(1-4 beta^2) of the embedding L^{alpha,beta} | `embcheck.py` | `pairs 81 bad 0 m^3 coeff*12 = -c*(2*beta - 1)*(2*beta + 1)` |
| Remark 5.3.2 b): spectral flow, inversion and mirror are automorphisms of k^L(1\|2) and act on (alpha, beta) as stated | `autcheck.py` | `Jacobi triples 21952 bad 0`, `pairs 784 bad 0` for U_1, P, M, and `U maps L^{a,b} to (alpha + 1, beta)`, `P maps ... to (-alpha, beta)`, `M maps ... to (-alpha, -beta)`, each `True` |
| Remark 5.3.2 d): for alpha = 1/2 the space H^1(witt; k^L(1\|2)_od) has sdim (0\|2) for beta = +-1/2, (0\|1) for beta = +-3/2, +-5/2, (0\|0) otherwise | `h1odd.py` | H^1 of witt with coefficients in the density module M_lam, lam = 1/2 + beta (the G-part) and lam = 1/2 - beta (the other part), window \|m\|, \|n\| <= 12: `H^1 2` for lam = 0, `H^1 1` for lam = -1, -2, `H^1 0` for the other lam in [-2, 4]; the sum over the two parts gives the stated superdimensions |
| Proposition 6.1.1 (the Ivanov-Krivonos algebra is svect^L_0(1\|2); relations (svectrel), (svectdoub)) | `sv2.py` (about 3 min) | `checks: 7931, failures: 0` |
| Proposition 6.1.2 (the Ivanov-Krivonos algebra inside k^L(1\|4), the generating functions (KAemb)) | `n4emb.py` | `checks: 1786,  failures: 0` |
| Proposition 6.1.3 (the central extension of svect^L_lambda(1\|2)) | `sv2coc.py` (about 6 min) | `dim H^2(svect^L_lambda(1|2); C) = 1 for every lambda.`, `checks: 2599, failures: 0` |
| Propositions 6.2.1 and 6.2.2 (a twist of the odd coordinates shifts lambda; the N = 4 equation for every lambda) | `twist.py` (about 4 min); `slnN.py` | `twist.py`: `checks: 60, failures: 0`; `slnN.py`: `[K_f,D] = -(d_t f) D  for all 12 monomials: failures 0` |
| Proposition 6.2.3 (the shift accompanying K_f for N = 4) | `n4shift.py` (several minutes) | `checks: 2433,  failures: 0` |
| Proposition 6.2.4, items 5) and 7) (the equation fixes the shift) | `n4dyn.py` (about 6 min) | `checks: 49,  failures: 0` |
| Lemma 7.1.2 and Theorem 7.1.3 (for N = 4 the recipe is the identity; \|c\| = 6) | `n4stress.py` (about 7 min) | `c = 6 four times over:   |c| = 6`, `32 checks, 0 failures` |
| Proposition 7.2.1, item 1) (the N = 4 vacuum: 82 monomials at the levels 0, 2, 4, 6, 8 as 2, 20, 38, 20, 2) | four steps: `n4rec9.py`, `n4rec10.py`, `n4rec11.py` (each writes the pickle the next reads; about 3 min together), then `n4ver.py` | `n4rec11.py`: `the solution has 82 nonzero monomial coefficients in its four entries`, `by theta-level: [(0, 2), (2, 20), (4, 38), (6, 20), (8, 2)]`; `n4ver.py`: `VERDICT: the N=4 vacuum is CONFIRMED by the independent checker` |
| Proposition 7.2.1, the displays (vac0), (vac2) | `n4disp.py` (after `n4rec11.py`) | `all displays reproduce the vacuum: True` |
| Proposition 7.2.1, item 2) (the leading level is solvable for every invertible A with kernel 56; the class is determined by the residual, and the recursion is linear at each step; the quadratic rows) | `n4rec.py`; `n4grade.py`; `n4rec6.py`; `n4rec8.py` (about 1-2 min each) | `n4rec.py`: `solvable; kernel dimension 56` for every A tried; `n4grade.py`: `TOTAL violations: 0  (the recursion is linear at each step)`; `n4rec6.py`, `n4rec8.py`: `verdict: solvable` |
| Proposition 7.2.1, item 3) (only the diagonal placement of the N = 2 vacuum admits a q satisfying the constraints) | `n4shape.py` | `no base` for every placement except the diagonal one |
| Proposition 7.2.1, item 4) (the tangent space is 1-dimensional over Q(i), in the levels 2, 4, 6; every tangent step integrates) | `n4fam.py` (after `n4rec11.py`; about 1 min) | `tangent space: 2 real dimensions = 1 over Q(i)`, `the tangent vectors involve theta-levels [2, 4, 6]`, `8 of 8 tangent steps land on an exact solution` |
| Proposition 7.2.1, item 5) (the embedded N = 2 vacuum is not a solution: 35 kept, 26 changed, 21 added) | `relcheck2.py` (after `n4rec11.py`) | `embedded base is a solution of (IK4): False`, `monomials: agreeing 35, differing 26, only in vacuum 21, only in base 0` |
| Remark 7.2.2 (level-wise solving reports a spurious obstruction) | `n4rec.py` | `INCONSISTENT: 6 blocked sectors, 38 rows`, `-> the recursion DIES at level 2` (the vacuum of `n4rec11.py` exists nevertheless) |
| Proposition 7.2.3, item 1) (the residue -1; no element of the ring completes T) | `n4cons.py` (after `n4fam.py`; it writes `n4cons-tan.pkl`, read by all the scripts below), then `n4cons3.py` | `n4cons.py`: `NO conserved completion inside this list of candidates.`; `n4cons3.py`: `still no solution` |
| Proposition 7.2.3, item 2) (none of the 1333 composites completes T; the level-0 condition is met, the next 24 are not) | `n4cons5.py`; `n4cons6.py` | `n4cons5.py`: `NO solution at the vacuum`; `n4cons6.py`: `levels <= 0 (   1 rows, 1333 unknowns): solvable`, `levels <= 2 (  25 rows, 1333 unknowns): INCONSISTENT` |
| Proposition 7.2.3, item 3) (logarithms complete T on the 17 members of the family; fitted on 14, the 3 held out obey; 14 relations) | `n4cons7.py`; `n4cons8.py` (about 3 min) | `n4cons7.py`: the combination with coefficients `-1/4`, ...; `n4cons8.py`: `holdout: fitted on 14, 3 of 3 held-out members obey` |
| Proposition 7.2.3, item 4) (qbar_b^a = (i/4m) cD_+^a c_b, the index placement) | `n4logdet.py` | `transpose=True coeff=i/4/m : 11 of 11` and `0 of 11` for the other placements and coefficients |
| Lemma 7.2.4 (the reflection sigma and the boost map solutions to solutions; the 51 solutions) | `n4mirror.py` (about 2 min); `n4boost.py` (about 4 min); both after `n4cons.py` | `n4mirror.py`: `holdout: fitted on 28, 6 of 6 held-out obey`; `n4boost.py`: on the `51 solutions, 19 log columns`, `3983 rows: SOLVED`, with the coefficients `-3/4` and `-1/4` of formula (completion4), `freedom: 28 real = 14 complex directions` |
| Statement 7.2.5 (the completion (completion4) holds on all 51 solutions, for both signs; the same 14 relations) and item 4) of Lemma 7.2.4 (d_+ = kappa d_- on every solution of the orbit) | `n4pin.py` (about 4 min); `n4both.py` (about 6 min); `n4prop.py`; all after `n4cons.py` | `n4pin.py`: `kernel: 28 real = 14 complex directions` for d_- and for d_+, `dim (K_- + K_+) = 14`; `n4both.py`: `both, i.e. T + X constant on the solution:      17 of 17` on each of the three families; `n4prop.py`: `d_+ T = kappa . d_- T` with kappa = 1, -1 and 1 + 2t (t = e1e2), each `on 17 of 17` |
| Lemma 7.2.6, items 4) and 5) (which loops have d_+ = kappa d_-) | `n4kap.py` | `F = [[1, zeta], [0, 1]]     (unipotent)    NO kappa`, ..., `checks: 10, failures: 0` |
| Lemma 7.2.6, item 6) (no boost-homogeneous solution) | `n4win0.py` | `There is NO boost-homogeneous solution, and the bigrading cannot cut the class down.` |
| Statement 7.2.7, item 1) (the level-0 system over the twelve loops: solvable, with kernel 56 for each value of n the window admits) | `n4loop.py -5 5` (about 2 min) | twelve lines, each `solvable, kernel 616` (= 11 x 56). With the narrower window `n4loop.py -3 3` the loops A zeta^(+-2) report `INCONSISTENT ... [2 rows beyond the window ...]`: there the window does not yet reach the rows the system carries |
| Statement 7.2.7, item 2) (the level-1 system is inconsistent for F = A zeta, diag(zeta, 1), the unipotent loop: 64, 59, 73 blocked rows) | `n4abs.py 1 -3 4 1 mono`, `n4abs.py 1 -3 4 1 diag`, `n4abs.py 1 -3 4 1 uni` (4-7 min each) | `INCONSISTENT: 1 blocked sectors, 64 rows`, resp. `59 rows`, `73 rows` |
| Statement 7.2.7, item 3) (the same 64 rows over the relative windows [-1,1], [-2,2]) | `n4two.py -1 1`; `n4two2.py 1 -1 1`; `n4two2.py 1 -2 2` (2-4 min each) | `n4two.py`: `a_0 = 1   : step 0 solvable for 10 of 10 level-0 data`; `n4two2.py`: `INCONSISTENT: 1 blocked sectors, 64 rows` for both windows |
| Statement 7.2.7, item 4) (the vacuum does not deform into the two-variable class) | `n4win.py 1` (after `n4rec11.py`; about 2 min) | `tangent space: 2 real dimensions = 1 over Q(i)`, `z-shifts touched by the tangent space as a whole: [0]`, `rank of the kernel projected on the s<0 block: 0 real = 0` |
| Statement 7.2.7, item 5) (the level-4 unknowns, 2240 of them, enter the level-1 rows linearly, and the level-1 system is inconsistent at a fixed level-2 part) | `n4rob.py 1 -3 4 1 mono` (about 3 min); the last argument is the number of sampled level-2 parts, and `n4rob.py 1 -3 4 6 mono` samples five random points of the kernel besides the particular solution (several minutes per point) | `level-4 unknowns: 2240`, `sample 0 (the particular solution): 64 rows of step 1, 2240 level-4 columns, 0 nonlinear : step 1 blocked`, `0 of 1 sampled level-2 parts leave step 1 solvable` |
| Statement 7.2.7, item 6) (Phi(lambda): the level-4 window [-10,7], the 8 blocks of 56, the rows -8 <= n_z <= 6, the 250 free rows: 160 affine and consistent, 90 quadratic) | seven steps, in order: `n4elim.py build mono 1 -10 7` (about 5 min), `n4phi2.py coker mono1-w-10_7`, `n4phi2.py sample mono1-w-10_7 16`, `rowrange.py mono1-w-10_7`, `freerows.py mono1-w-10_7`, `affine.py mono1-w-10_7`, `blocks.py mono1-w-10_7` (a few minutes each) | `step 0: 896 level-2 unknowns, kernel 448`, `M4 built: 5040 columns`, `M4 constancy: 0 of 24 sampled columns differ at a random point`; `coker dim inside them 6552`; `Phi has ... nonzero components` at each of the 16 points (22 at the particular solution, about 1300 elsewhere, never 0); `step 1 [(-8, ...), ..., (6, ...)]`; `free rows: 250 degrees along the line: [(1, 160), (2, 90)]`; `free rows 250: affine 160, quadratic 90`, `affine subsystem: complex rank of L 128, of [L|c] 128 -> consistent`; `top block n=4: 45 components of Phi depend on it alone` |
| Statement 7.2.7, item 7) (the minimal distance d is 14.90, 14.76, 14.67, 14.61, 14.56 over the windows [-3,4], ..., [-7,8]; 90% of d^2 in the rows -2 <= n_z <= 1) | `build.py -3 4 -10 7` (about 1 min), then `lm2.py w2_-3_4-w4_-10_7 0 1 40` (about 6 min), then `lmrows2.py w2_-3_4-w4_-10_7 lm2-w2_-3_4-w4_-10_7-0.pkl`. The other four windows: `build.py -4 5 -12 9`, `build.py -5 6 -14 11`, `build.py -6 7 -16 13`, `build.py -7 8 -18 15`, each followed by `lm2.py <tag> 0 1 40` with the tag the build prints (`w2_-4_5-w4_-12_9`, ...); these take about 9, 17, 23 and 32 min (not rerun for this deposit) | `lm2.py`: `final |Phi| = 14.9016`, `rank(J) 276/448`; `lmrows2.py`: `share of |Phi|^2 by n_z:` with `-2:0.125  -1:0.100  0:0.467  1:0.214` (together 0.906); the other windows: `final |Phi| = 14.7612`, `14.6709`, `14.6084`, `14.5635` |
| Statement 7.2.7, item 8) (Phi is quadratic; its grading by the powers of z; the edge quadrics, their radicals of dimension 25, their zeros outside the radicals) | `cert.py w2_-3_4-w4_-10_7 6` (about 10 min; it writes a pickle of 0.6 GB, read by `phistruct.py`), then `phistruct.py`, `grading.py`, `peel2.py`, `peel3.py` (about 30 min), each with the argument `w2_-3_4-w4_-10_7` | `cert.py`: `stacked rank 1010 of 6802 components -> common left null dim 5792`, `|P_W Phi(0)| = 0.000000`; `phistruct.py`: `|coef_k|` of order 1e-8 and below for k >= 3 along each line, `rank L = rank J(0) = 276`; `grading.py`: `const ... rows n_z: [-2, -1, 0, 1, 2]`, `L lam_n ... rows n_z: [n-2, ..., n+1]`, `Q a.b ... rows n_z: [a+b-2, a+b-1]`; `peel2.py`: `rank of v->B(v,.) = 31 (radical dim 25)` for k = -3, -2, -1, 3, 4; `peel3.py`: `min |Q(w)| on |w|=1, w in complement` of order 1e-17 to 1e-20 for each k |
| Statement 7.2.7, item 9) (with only lambda_0, lambda_1, lambda_2 free, min d = 16.11; the four middle rows alone come down to about 1.1) | `midmin.py w2_-3_4-w4_-10_7 red 0` (seeds 1, 2 likewise); `midmin.py w2_-3_4-w4_-10_7 mid 0` does not terminate by itself: it reaches about 1.1 after 10 min and is still decreasing (not rerun for this deposit) | `red seed 0 FINAL 16.1094` |
| Proposition 8.1.1 (the Cartan matrices of svect^L_lambda(1\|2), the reflections between them, the isomorphisms for lambda and lambda+n, n-lambda) | `berroots.py` (several minutes) | the three Cartan matrices read off the relations, the closure inside the window \|deg t\| <= 3 contains all basis fields, `checks: 162, failures: 0` |
| Remark 8.1.2 (the cocycle of Berezin's book, 2nd ed., in the basis of the text) | `bercoc.py` | `checks: 7, failures: 0` |
| Proposition 8.2.1 (which svect^L_lambda(1\|2) have a real form; the anti-linear automorphism C) | `antilin.py` (about 1 min) | `C preserves svect and C^2 = id: True` for each lambda tried, `60 checks, 0 failures` |
| Proposition 8.3.1 (the nine odd-reflected systems; the Cartan matrix of svect^L_lambda(1\|2) is not symmetrizable) | `olsh.py` (about 2 min) | `checks: 128, failures: 0` |
| Lemma 9.1.1, items 1) and 3), and Proposition 9.1.2, item 1) (the Ramond frame on the double cover, the twist by R, the identity omega_12(f) + t omega_21(f) = 0) | `ramondframe.py` (about 5 min) | the contact fields of k^M(1\|n) for n = 2, 3, 4: `[K^M_f, K^M_g] = K^M_{f,g}_R.b. on 1024 pairs: 0 failures` (4096, 16384 pairs); the double cover, `{hatD_a, hatD_b} = 2 delta_ab Dcal on all 16 pairs: 0 failures` (and 4, 9 pairs); the grading of omega by the monodromy, `c o(N)-shape: 0 failures; monodromy grading: 0 failures`; `omega_{12} + t*omega_{21} nonzero anywhere: False`; section (6), the reflection R = diag(1,1,1,-1). The lines `K_f normalises D_theta_i = theta_i d_t + d_theta_i: 8 failures; the other sign: 0 failures` compare the two signs: the derivations spanning the contact distribution are d_theta_i - theta_i d_t. |
| Theorem 9.2.1 (the classical anomaly in the Ramond sector: \|c\| = 6 for N = 1, \|c\| = 3 for N = 2) | `n2ram.py`; `ramond.py` | `n2ram.py`: `the Ramond value is c = -3`, `225 checks   OK`, `3375 checks   OK`; `ramond.py`: `c = -6` for N = 1, `c = -3` in every sector for N = 2, `TOTAL failures 0` |
| Statement 9.2.2 (the Ramond-type N = 4 system: the symmetry algebra of sdim (3\|3), its subalgebra g_R of sdim (2\|3)) | `ramond4.py`; `ram4all.py` (about 1 min) | `ramond4.py`: `dim S^M` equal to `(1|0)`, `(0|3)`, `(2|0)` in the weights 0, 1, 2 and `(0|0)` in the others, `closure failures: 0`, and the multipliers Lam(f); `ram4all.py`: `det(A^dag A) = w**2*(w - 2)**4*(w - 1)**2*...` (even) and `(w - 2)**2*(w - 1)**6*...` (odd), whose last factors are P(w) of the text, and the M-invariant kernels at w = 0, 1, 2 |
| Statements 12.1.1 and 12.1.3 (finite-dimensionality, atypicality and superdimension of the osp(3\|2)- and osp_a(4\|2)-modules) | `osprep.py` (several minutes) | `78 finite-dimensional modules checked` (osp(3\|2)), `290 typical modules checked over 7 values of a` (osp_a(4\|2)), `FAILURES: 0` |
| Statement 12.1.3, item 3) (the table (sdimOsp42at), k <= 7, a = 1, 2, 5/3) | `atyp4b.py` (about 2 min) | for each a: `k=1 ... (9|8)`, `k=2 ... (56|54)`, ..., `k=7 m=(8, 6, 6) (1666|1664) dim=3330` |
| Kac's character formula (kacChar) and its supercharacter twin (kacSuperChar) on the typical modules | `charcheck.py` | `osp_a(4|2): typical modules tested over 7 values of a: 156;  ch holds 156,  sch holds 156`, `failures: none` |
| Lemma 12.2.1 (the two-term relation for osp(3\|2), n = 1..5, and osp_a(4\|2), k = 2..6) | `eulerchk.py` (about 1 min) | `ch: True  sch: True` on every line, `passed 20 of 20` |

## Files

- `stringy.py` — the contact algebras with Laurent coefficients (shared calculus).
- `sliouv.py`, `nscoc.py` — Section 4 (N = 1).
- `ivkr.py`, `n2.py`, `n2cons.py`, `n2coc.py`, `embcheck.py`, `autcheck.py`, `h1odd.py` —
  Section 5 (N = 2).
- `sv2.py`, `sv2coc.py`, `n4emb.py`, `twist.py`, `slnN.py`, `n4shift.py`, `n4dyn.py` — Section 6.
- `n4stress.py` — Subsection 7.1.
- `n4noeth.py` — the ring of the N = 4 system (superfields over Q(i), the two halves, the
  constraints and equations of (IK4)); `n4vac.py`, `n4phi.py` — the graded class and its
  monomials (imported by the rest); `n4rec.py`, `n4rec2.py`, `n4rec6.py`, `n4rec8.py`,
  `n4rec9.py`, `n4rec10.py`, `n4rec11.py`, `n4ver.py`, `n4disp.py`, `n4grade.py`, `n4shape.py`,
  `n4shapeq.py`, `n4fam.py`, `relcheck2.py` — the vacuum (Proposition 7.2.1).
- `n4cons.py`, `n4cons2.py`, `n4cons3.py`, `n4cons5.py`, `n4cons6.py`, `n4cons7.py`,
  `n4cons8.py`, `n4logdet.py` — Proposition 7.2.3.
- `n4mirror.py`, `n4boost.py`, `n4pin.py`, `n4both.py`, `n4prop.py` — Lemma 7.2.4,
  Statement 7.2.5.
- `n4kap.py`, `n4win0.py`, `n4loop.py`, `n4abs.py`, `n4two.py`, `n4two2.py`, `n4win.py`,
  `n4rob.py` — Lemma 7.2.6, Statement 7.2.7, items 1)-5).
- `n4elim.py`, `n4phi2.py`, `rowrange.py`, `freerows.py`, `affine.py`, `blocks.py` — the map Phi,
  item 6) (exact).
- `fast.py`, `build.py`, `lanes.py`, `lm2.py`, `lmrows2.py`, `cert.py`, `phistruct.py`,
  `grading.py`, `peel.py`, `peel2.py`, `peel3.py`, `midmin.py` — items 7)-9) (floating point).
- `berroots.py`, `bercoc.py`, `antilin.py`, `olsh.py` — Section 8.
- `ramondframe.py`, `n2ram.py`, `ramond.py`, `ramond4.py`, `ram4all.py` — Section 9.
- `hwmod.py` (irreducible highest-weight modules over g(A) from the Cartan matrix, as the quotient
  of the Verma module by the radical of the contravariant form), `osprep.py`, `atyp4b.py`,
  `charcheck.py`, `eulerchk.py` — Section 12.
