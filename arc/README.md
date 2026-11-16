# arc

**Superized Schwarzians and linear fractional transformations of distinguished superstrings**

D. Leites, I. Shchepochkina. Status: in preparation.

The scripts check the computations of the text: the brackets and the cocycles of the distinguished
stringy Lie superalgebras k^L(1|n), k^M(1|n), vect^L(1|1), m^L(1) (Table (61) and its corrections),
the Berezinian of a linear fractional transformation, the annihilators of the Schwarzians and the
linear fractional supergroups, the weight pair on S_c^{1|2}, the Gordan-Rankin-Cohen operators on
weighted densities and the decomposition of their tensor products, the rank-2 case, the Schwarzians
of Radul and their Moebius analogs, and the Appendix on the cocycles of the stringy algebras.
Exact arithmetic throughout (integers, `fractions.Fraction`, sympy, and python-flint through
`../lib/exactlin.py`); mpmath is used only by `modweight.py`, for a numerical check of a modular
form, and is labeled as such in its output.

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r arc/requirements.txt
cd arc
../venv/bin/python check_equiv.py
```

Every script is run from inside this folder. No arguments are needed except where stated.
`stringy.py`, `radul.py`, `mobius.py`, `schwW.py`, `sl12.py`, `degen.py` are modules imported by the
others. Most runs take seconds to a few minutes on a laptop; the table gives the times of the longer
ones (measured with many scripts running at once, so they are upper bounds). `casimir.py`,
`multdeg.py`, `verma.py`, `sl12run.py`, `sweep1.py` also write what they print to a file in this
folder (`runs-casimir.txt`, `runs-multdeg.txt`, `verma.log`, `sl12run.log`, `sweep1.log`).

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Props. 2.1.1, 2.4.1, 2.6.3, 2.6.5, Remark 2.1.2 | `check1920.py` | `checks: 24585, failures: 0` |
| Statement 2.6.6 (the subalgebra kas^L is isotropic) | `kas1920.py` (about 2 min) | four lines `sign ... bracket c ... isotropic True non-closing brackets N of 16384`; N = 0 exactly for `sign -1 bracket c 1`, the conventions of the text (the other three are controls) |
| Subsec. 2.8.1 (the cocycles c_IS and c_KvdL are proportional, with the sign factors) | `check_cocycles.py` (about 25 min) | `cocycle identity ...: OK`; `c_IS/c_KvdL (with (-1)^p(f)): ratios {'1': 44, '-1': 44} ; one-sided-zero mismatches: 0` |
| Theorem 2.8.3, Table (61), eq. (48), Subsecs. 2.7.1, 2.7.2 | `check_table.py`, `check_components.py` (about 25 and 45 min) | the ratios computed/printed for each block of Table (61); `printed ... computed ...` rows for each component |
| eq. (58) (Delta_L as one operator, not the literal composition) | `check_alpha2.py` (about 2 min), `checkdelta.py` | `reading (b) [one operator], with (-1)^p(f) : COCYCLE`; `ALL CHECKS PASSED  (0 failures)` |
| eqs. (104) and (205) (the order of the factors of a Schwarzian) | `order3.py` | the two "no sign factor" readings fail (132 and 90 failures, by design); `1/2 (D_xi D_eta - D_eta D_xi) d_t, with (-1)^p(f)    COCYCLE`; last line `: OK` |
| eq. (206) | `schw11.py` | `Div=d_x f - d_th g, overall +1 : 0 mismatches / 1296` (the other conventions mismatch, by design) |
| The order of the factors in the k^L column of Table (61) and in Subsec. 9.3 | `ord958.py` (about 6 min) | `38 checks, 0 failures` |
| Theorem 3.5.3 (the Berezinian and the contact multiplier of a linear fractional transformation) | `check_lemsch.py`, `check_berezinian.py` | every line `True`; `check_berezinian.py` ends with `they agree ? False` (the naive candidate is not the Berezinian, by design) |
| The equivalence of the two systems of equations of Sect. 3 | `check_equiv.py` | `ALL CHECKS PASSED` |
| Props. 4.3 and 9.5 (projective structures, the c/2 in eq. (84)) | `proj_hill.py` | `checks passed: 10` |
| Theorem 4.5.1, eqs. (93)-(98) (kernels of s_n, Div K_f, the factor 2^{-m} in eq. (28)) | `check_misc.py`, `check_bases.py`, `check_ident.py` (about 13 min) | `with the factor 2^{-m}: ... : OK`, `withOUT the factor 2^{-m}: ... : 12 failures` (by design); explicit bases with `sdim = (4\|4)`; `sl(1\|2) = osp(2\|2): dim 8, sdim (4\|4), simple`, `psl(2\|2): dim 14, sdim (6\|8), simple` |
| Subsec. 4.5.2, eqs. (96), (97), (100) | `vpair979.py` | `11 ok, 0 FAILED` |
| eqs. (116)-(118), the m^L(1) row of Table (61) | `mL1.py` (about 2 min) | `(D_xi)^2  (-1)^p(f)  : antisym OK , 2-cocycle OK` (the other three readings are not cocycles); `radical ...: sdim (4\|4)`; `... SAME SPACE`; `eq. (116) closes under the bracket: closed` |
| Statement 4.6.2 (linear fractional transformations in coordinates), items 1)-5) | `lfgrp.py` (n = 1, 2, 3; about 25 min; `lfgrp.py 4` takes hours) | every line `OK`, except the control `m = f' - sum ph_i ph_i' : NO` |
| the same, in vector form | `vecform.py` (about 15 min) | `62 checks, 0 failures` |
| Lemma 4.6.4 and items 2)-4) of Statement 4.6.2 | `conf.py` (about 25 min) | `failures: 0` |
| The weight of the modular forms (+k/2) | `modweight.py` (numerical) | `w=+k/2 : ... INVARIANT`, `w=-k/2 : ... not invariant` |
| Statement 5.2.1, items 1)-4); eq. (125) | `pair2.py` (about 13 min), `pair994.py` | the multipliers mu, nu per case, `mu/nu = 1` on the components stated; `18 checks, 0 failed` |
| eqs. (132), (213); Subsec. 10.1 item 2 (no associative product for the odd brackets) | `check_grc_assoc.py` (about 5 min) | PART 1, 2a-2c reports |
| Prop. 6.1.4, eq. (137) | `oddclosed.py` | `Q_1 = (mu1 + 2)*(mu1 + 3)*(mu2 + 3)*(mu1 + mu2 + 3)`, `Q_2 = ...`, `Q_3 = ...` |
| Theorem 6.1.5 for orders k <= 6; Lemma 7.2.1 | `fullops.py 6` (about 50 min) | `# unexpected dimensions: 0` |
| Lemmas 7.2.2, 7.2.3 | `tensor.py` | `levels with an unexpected answer: 0` |
| Lemma 7.2.4, Theorem 7.2.5 | `structure.py` (about 25 min) | `FAILURES: 0` |
| Table (154) | `degen_par.py 3 8` (3-12 min on 8 cores), `checkdegen.py` | `# weight pairs differing from the generic answer: 9 of 169`; `checkdegen.py` reads `runs-degen-N3.txt`, `runs-degen-N4.txt` (`degen_par.py 4 8` regenerates the second, about 9 min) and prints `# 4 checks, 0 failures` |
| Lemma 7.2.9 | `casimir.py` | `SUMMARY: (a) True  (b) True  (c) True  (d) True` (in `runs-casimir.txt`) |
| Theorems 7.2.10, 7.2.11, Table (154) | `multdeg.py` (about 35 min), `verma.py` (about 30 min) | `SUMMARY: part I 9 failing pairs; part II True; ...; part VI True` (the 9 pairs are those of Table (154)); `TOTAL: 42314 checks, 0 mismatches -- ALL PASS` |
| Lemma 7.2.14 | `degen0.py` | `#  mismatches: 0` |
| Lemma 7.2.17 | `sign.py 3` (about 7 min) | tests T1-T6; `N=2 : the identity holds`, `N=1`, `N=3 : NOT an action ...` (by design) |
| Lemmas 7.2.18, 7.2.20, Theorem 7.2.22 | `dform.py`, `dproof.py` (about 7 min), `d1var.py` (about 4 min), `dcross.py` | the coefficient table; `N=3, 5, 7  (134) = -1, 1, -1 x uniform`; `with beta^N: invariant` for N = 3, 4, 5; the "NOT invariant" lines of `dcross.py` are controls |
| eq. (142) | `osp12gen.py` | `15 checks, 0 failed` |
| Lemma 8.1.3, eqs. (174)-(179) | `sl12.py` | `all structure checks: OK` |
| Theorem 8.3.1 (generic weights, N <= 4) | `sl12run.py gen` (about 8 min) | the table in `sl12run.log` |
| Theorem 8.3.1 (N <= 8 at a rational point) | `sweep1.py` (about 12 min) | `sweep1.log`: rows N = 1, ..., 8; for even N two operators at q = 0, for odd N one at q = +-1 |
| Theorem 8.4.1 | `ops1.py`, `opsD.py` (about 2 min each) | the coefficient lists |
| Statement 8.4.3 | `uni1.py` | `correction ansatz solution: []` |
| Theorem 8.5.1 | `gsm.py` | `dim=1` at M = 0 only, `dim=0` for M = 1, ..., 4 |
| Statement 9.4, item 1, Table (194) | `test_radul.py`, `run.py`, `recheck12.py` (these and the next five rows are sympy computations of 10-60 min each) | `19 ok, 0 FAILED`; `27 ok, 1 FAILED` (the one FAIL is the check that rejects the constant -3, by design); `8 ok, 0 FAILED` |
| Statement 9.4, item 2 | `hill959.py` | `10 ok, 0 FAILED` |
| Statement 9.7, eqs. (196), (199) | `run3.py`, `cyc959.py` | `10 ok, 0 FAILED`; every line `ok`, the last `S_3 with the constant -1 is a 1-cocycle of weight 1/2` |
| Statement 9.10, Table (201) | `test_mobius.py`, `runmob.py` | `30 ok, 0 FAILED`; `42 ok, 0 FAILED` |
| Statement 9.11, eq. (202) | `mob979.py`, `runmob3c.py` (about 7 min) | every line `ok`, the last `S^M_3: the UNTWISTED identity, weight 1/2  <-- (a)`; every line of `runmob3c.py` `ok` |
| Statement 9.13, eqs. (207), (211) | `test_schwW.py`, `runW.py` | `20 ok, 0 failed`; `16 ok, 0 failed` |
| Section 11 (the Appendix: Subsecs. 11.1.1, 11.1.2, eq. (214), the svect rows) | `appx.py` (about 40 min) | every line `OK` |
