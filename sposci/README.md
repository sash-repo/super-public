# sposci

**Spinor-oscillator representations of stringy Lie superalgebras**

D. Leites, I. Shchepochkina. Status: in preparation.

The scripts check the computations of the text: the cocycles of the distinguished simple stringy Lie
superalgebras in the xi-eta basis; the modules of tensor fields over k^L(1|n), k^M(1|n), vect^L(1|2)
and svect^L_lambda(1|2) that carry an invariant bilinear form, and the highest weights (c, h; H) of the
spinor and oscillator representations built on them; Xu's conformal oscillator representations of the
ortho-symplectic Lie superalgebras; the Mumford-type exponents; and the Fock spaces of an odd form
(the antibracket and its quantization). Exact arithmetic throughout (integers, `fractions.Fraction`,
sympy, and python-flint through `../lib/exactlin.py`); numpy only in `harm1806.py`.

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r sposci/requirements.txt
cd sposci
../venv/bin/python fock.py
```

Every script is run from inside this folder, with no arguments except where stated. Times are for a
laptop running one script at a time. The runs marked "hours" (`runns4.py`, `hw4.py`, `hw4b.py`,
`hw4d.py`, `hw4f.py`, `hw4g.py`, `harm1806.py`, and the main block of `kM4.py`) take several hours;
for them the expected output given below is that of the runs behind the text.

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Sect. 3 (the distinguished stringy superalgebras in the xi-eta basis: brackets, bases, cocycles) | `xieta.py` (about 4 minutes) | `TOTAL FAILURES: 0` |
| Statement 15.1 (the cocycle tables in the xi-eta basis: complete, one sign of the Neveu-Schwarz table wrong) | `kL4.py`, `solveL.py`, `analyzeL.py`, `fixL.py`, `solveM.py` (`solveM.py` imports `kM4.py`) | `kL4.py`: `Jacobi identity for the xi-eta contact bracket (kappa=1): OK`, then the failing triples of the printed table for each normalization kappa; `solveL.py`: `dim of the space of weight-0 cocycles, restricted to the B-window: 10`; `analyzeL.py`: `dim of coboundaries = 7`, `table's alpha part is a cocycle: False`, `beta ... True`, `gamma ... True`; `solveM.py`: `=> dim H^2 (weight 0) = 5` |
| Sect. 4 (the densities Vol^lambda and the modules cF_lambda) | `dens.py` | `0 failures` for each check, `control ... 60 failures (must be > 0)` |
| Sect. 5 (the central charge sees the superdimension of the ambient superspace) | `target.py` | `all values match  c_Spin = -2(6L^2-6L+1) = -c_Osc :  0 mismatches` |
| Sect. 6 (harmonic superpolynomials) | `harm1806.py` (hours) | for each osp(m\|2n): `k=...: dim H_k=...  IRR(M)` |
| Statement 7.1.3 (two signs in Xu's conformal oscillator representations) | `runxu_all.py` (about 2.5 minutes; `runxu.py` is one degree) | for every (n, m) and both families: `FORM  invariant forms: 1 ... True`, `HOM   failures over all ... pairs: 0` |
| Statement 7.1.5 (the conformal family over the deformation osp_a(4\|2)) | `run_ospa.py` | for each degree k the singular values of c, e.g. `k 2 dim 8 singular c (M): ([4/3, 4/5], ...)` |
| Subsec. 7.2 (the four self-dual vir-modules, the doubled module, the factor 1/2, the sum formula, the Witt subalgebras) | `fock.py`, `fock2.py`, `finspin.py`, `sumderiv.py`, `witt.py`, `ccemb.py`, `feigin.py` | `fock.py`: `(c, h) = (1, -1/16)  doubled: (2, -1/8)`; `fock2.py`: `c_computed / c_table = ['-1', ...]` (the table has the opposite sign); `sumderiv.py`: `0 failures` on each line; `witt.py`: the class of each Witt subalgebra; `feigin.py`: `10 checks passed` |
| Proposition 7.3, Statement 7.3.6 (one odd indeterminate: the invariant forms, and Spin of the doubled module over k^L(1\|1), k^M(1\|1)) | `runns1.py` (about 9 minutes), `census-ns1.py` | `runns1.py` ends with `NS  c = -3*1 + 6*lambda`, `NS  h = 1/2*mu`, `R   c = -3*1 + 6*lambda`, `R   h = 1/8*1 + -1/4*lambda`, each `(exact on all 24 points)` |
| Statements 7.3.10, 7.3.11, 7.3.13, Proposition 7.3.12 (two odd indeterminates) | `runns2.py` (about 20 minutes), `forms-ns2.py` (about 20 minutes), `census-ns2.py` | `runns2.py`: `L  c = -6*1`, `L  h = 0`, `L  H = 0`, `M  c = -6*1`, `M  h = 1/4*1`, each `(exact on all ... points)`; `forms-ns2.py`: `k^M(1\|2): 119 (lambda, mu) pairs tried, 4 carry a form`; `census-ns2.py`: `(c, h, H) = (-6, 0, 0) on 60 triples`, `(c, h) = (-6, 1/4) on 30 pairs` |
| Statements 7.3.19, 7.3.22, 7.3.24, Proposition 7.3.25 (four odd indeterminates) | `runns4.py` (long), `census-ns4.py`, `hw4.py 4`, `hw4b.py 4 L`, `hw4c.py`, `hw4d.py 5 2 L`, `hw4e.py`, `hw4f.py 5 1`, `hw4g.py 5` (long) | `runns4.py` (hours): every module carrying a form, and its double, gets the labels `(0, 0, (0, 0))` (c, h; H_1, H_2) for Spin and for Osc, `stable` as the window grows; `hw4f.py 5 1` (hours): `\|0> IS a highest weight vector`, `=> the trivial module is the unique irreducible quotient M / g.M`, `g-invariant vectors of M: 0`; `hw4g.py 5` (hours): `r = 1: pairs (x, y) with <One\| x y \|One> =/= 0: 0 []`; `hw4c.py` (about 2 minutes) and `hw4e.py` (about 3 minutes): the g_0-tower and the two polarizations |
| Statements 7.3.28, 7.3.30, 7.3.31, 7.3.32 (vect^L(1\|2), svect^L_lambda(1\|2)) | `runvect2.py` (about 6 minutes) | `0 failures` on every line, `0 mismatches` on every comparison |
| Statement 7.4.4 (the three central charges of Spin over k^L'(1\|4) vanish) | `runspin.py` (about 26 minutes) | `CONCLUSION.  alpha = beta = gamma = 0 for Spin(cF_{lambda;mu} (+) dual) over k^{L}'(1\|4), for every lambda and mu tested` |
| Statement 7.5.2 (every map of eq. (233) is a homomorphism) | `wittemb.py` | `0 failures on \|m\|,\|n\| <= 4` on every line |
| Statement 8.2 (the Mumford-type exponents) | `mumford.py`, `h2run.py` (about 7 minutes), `h2zero.py` (about 5 minutes) | `checks: 95220 of 95220 passed`; `checks: 380092 of 380092`; `checks: 90744 of 90744` |
| Sects. 9-14 (the Fock spaces of an odd form; the quantized Buttin superalgebra; the deformations of the antibracket; characteristic p) | `run_abfock.py`, `run_bn.py`, `fockq.py` | `passed 45, failed 0`; `passed 18, failed 0`; `passed 84, failed 0` |

## Files

- `stringy.py` — functions on the superstrings and the contact brackets; `kL4.py`, `kM4.py`,
  `xieta.py`, `dens.py`, `ccemb.py`, `solveL.py`, `solveM.py`, `analyzeL.py`, `fixL.py` use it.
- `ns1.py`, `ns2.py`, `ns4.py`, `vect2.py`, `spin.py` — the modules, their invariant forms and the
  spinor representations; `runns1.py`, `runns2.py`, `runns4.py`, `runvect2.py`, `runspin.py` drive them.
- `hw4*.py` — the module Spin(V) at four odd indeterminates, level by level.
- `xuconf.py` — Xu's representations; `runxu.py`, `runxu_all.py` check them.
- `ospa.py` — the modules T(K_c) over osp_a(4\|2); `run_ospa.py` checks them.
- `mumford.py` — the semi-infinite cocycles; `h2run.py`, `h2zero.py` use it.
- `abfock.py`, `bn.py` — the Fock spaces of the antibracket and the Buttin superalgebra;
  `run_abfock.py`, `run_bn.py`, `fockq.py` check them.
- `witt.py`, `wittemb.py`, `fock.py`, `fock2.py`, `finspin.py`, `sumderiv.py`, `target.py`,
  `feigin.py`, `census-ns*.py`, `forms-ns2.py`, `harm1806.py` — the checks listed above.
