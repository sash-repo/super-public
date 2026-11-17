# sugragl

**From Supergravity to Ballbearings**

P. Grozman, D. Leites. Status: in preparation.

The Table of Subsection 2.3 (the degrees and weights of the structure functions
H^2(g_-; (g_-,g_0)_*) of nine models of the N-extended Minkowski superspace) was computed by
P. Grozman with **the SuperLie package** for *Mathematica*; that computation is not deposited here.
The scripts below recompute it from scratch and independently: for a parity string Par they
reconstruct the marking of the simple roots (cut Par into consecutive blocks of size at most 2, a
change of parity forcing a cut), build the Z-grading of sl(N|4), g_- and g_0 (the "Riemannian" g_0 =
the semisimple stationary subalgebra, the "conformal" g_0 = the whole block diagonal), compute the
generalized Cartan prolong degree by degree as the space of derivations, then H^2 split by degree and
by the weight of the full diagonal torus of gl(N|4), and decompose it into irreducible
sl(2) x ... x sl(2)-modules R(w_1, ..., w_k), the format of the Table (subscript 1 = odd class,
exponent = multiplicity). The same engine computes the five models added in Subsection 2.5, the
census of Statement 2.6, the counts of Statements 2.8 and 2.9, and the restriction of the two scalar
curvatures of Par = (00001) to Lambda^2(M). Exact rational arithmetic throughout
(`fractions.Fraction`; python-flint's exact `fmpq_mat` via `../lib/exactlin.py` when installed,
otherwise the same results in pure Python, more slowly).

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r sugragl/requirements.txt
cd sugragl
../venv/bin/python run.py --calibrate
```

Every script is run from inside this folder; `models.py` imports `../lib/exactlin.py`. Times are for
a laptop with python-flint installed (measured with three of the runs below going at once).

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Theorem 2.2, Table 2.3 and Remark 2.4 (all degrees and weights of the structure functions of the nine tabulated models; the seven corrected entries; the boxes = conformally invariant classes) | `run.py --calibrate` (about 8 minutes, almost all of it the N = 8 column) | for each Par a block `Par = (...)   N = ...   blocks ...` with `sdim g_-`, the prolong dimensions and, per degree, the columns `Riemannian` and `full block diag` (the conformal case); e.g. for `(0000)`: `2  (04), (22), (40), (00)   (04), (40)`; for `(00001)`: `0  (30)_1`, `2  (32)_1, (22), (10)_1, (00)^2   (32)_1` |
| The same, cell by cell against the LaTeX of the Table | `emit.py 0000 00100 00001 001100 100001 000011 00111100 00001111 111100001111` (writes `out/<Par>.json`, about 8 minutes), then `checktable.py <LaTeX source of the text>` | `38 cells checked in 9 columns, 0 disagreements` |
| Theorem 2.7 (the structure functions of the five models (00110011), (0011111100), (1100110011), (001111111100), (110011110011)); "In all five, every cocycle of the conformal case is one of the Riemannian case as well" | `run.py --new` (about 19 minutes, the two N = 8 models 5 to 7 minutes each); `emit.py` on the five strings and `mktable.py` on them to regenerate the LaTeX of the five tables | the five blocks, in the format above, agreeing with the five tables of Subsection 2.7; `emit.py` stops with `conformal-only class ...` if a class of the conformal case is missing from the Riemannian one, and it does not stop |
| Statement 2.6 (the census: for N = 0, 2, 4, 6, 8 there are 1, 2, 3, 6, 9 models with all blocks of size 2, of them self-dual 1, 1, 1, 2, 3; six columns of the Table are of this type, (00100), (00001), (100001) are not) | `census.py` (instant) | `N = 0: 1 models, 1 self-dual, 1 in the Table`, ..., `N = 8: 9 models, 3 self-dual, 1 in the Table`; the list of the models not yet in the Table, and the three columns with a block of size 1 |
| Statement 2.8 (the number s of weight-0 tensors in degree 2: s = 3, 2, 1, 0, 0 for (001111111100), (0011111100), (110011110011), (00110011), (1100110011)) and Statement 2.9 (for the eleven models with all blocks of size 2, s = the number of copies of M in degree -1) | `compare.py cal.out new.out`, where `cal.out`, `new.out` are the outputs of `run.py --calibrate > cal.out` and `run.py --new > new.out`; the copies of M in degree -1 are the adjacent pairs of equal parity in the `blocks` line of `run.py` (for the Table's columns and (001111111100) also `sugra.py`, part (1), column `M(deg -1)`) | one line per model, the number s before `CP^{s-1}`, e.g. `(001111111100)  8 -3..2   3  CP^2`, `(0011111100)  6 -2..2   2  CP^1`, `(110011110011)  8 -3..2   1  CP^0`, `(00110011)  4 -1..2   0  none -- (B) fails`; for the eleven all-size-2 models s = 1, 0, 1, 1, 2, 3, 0, 2, 0, 3, 1 for (0000), (001100), (000011), (00111100), (00001111), (111100001111), (00110011), (0011111100), (1100110011), (001111111100), (110011110011) |
| Subsection 2.1 and the paragraph after Theorem 2.2 (the marking read off Par is consistent with the number of entries of the weight tuples in all nine columns; sdim g_-) | `sugra.py`, part (1) (instant) | `marking reconstruction consistent with every printed weight tuple: YES`; e.g. `100001  2 1o\|2e\|2e\|1o  2  2  3  (5\|8)` |
| The column (0000) (the classical Riemann tensor: 20 = (4,0)+(0,4)+(2,2)+(0,0), of which (4,0)+(0,4) conformal), computed by a second, independent method | `sugra.py`, part (2) | `H^(2,2) = (4,0) + (2,2) + (0,4) + (0,0)` and, for co(4), `H^(2,2) = (4,0) + (0,4)` |
| Subsection 2.1 (which tabulated column is an enlargement R(N) of M(N): only (00001111), the (4,2,2) harmonic superspace) | `rn.py` (instant) | `00001111 ... YES, a proper enlargement, N=4` and `p=2, q=2, p+q=4 <= N=4 : ok` |
| The paragraph after eq. (7) (Par = (00001), Riemannian g_0: R_1(00) has E_55-weight 0 and restricts to Lambda^2(M) with equal contractions s_L = s_R != 0; R_2(00) has E_55-weight -2 and vanishes on Lambda^2(M); no coboundary has a component on Lambda^2(M)) | `rscal.py` (instant) | `weight (1, 1, -1, -1, 0): ... dim H=2`, `coboundaries on Lambda^2 M: []`, `s_L = 6  s_R = 6` (the second basis cocycle of this weight space is the zero-weight vector of (2,2) plus R_1); `weight (0, 0, 1, 1, -2): ... dim H=1`, `Lambda^2M part nonzero: False` |
| (self-test of the engine) | `test_models.py` (seconds) | `30 passed, 0 failed` (Jacobi and super-antisymmetry in g_-, every prolong vector a derivation, d o d = 0) |

## Files

- `models.py` -- the model of a parity string: blocks, the Z-grading of sl(N|4), g_- and g_0, the
  generalized Cartan prolong degree by degree.
- `cohom.py` -- H^2(g_-; (g_-,g_0)_*), split by degree and torus weight, decomposed into
  irreducible sl(2)^k-modules.
- `run.py` -- prints the structure functions of the given models (or `--calibrate`, `--new`).
- `emit.py` -- writes them as `out/<Par>.json`; `mktable.py` -- the LaTeX tables generated from the
  JSON; `checktable.py` -- parses the Table out of the LaTeX of the text and compares every cell
  with the JSON.
- `compare.py` -- the number s of scalar curvatures per model, from the output of `run.py`.
- `census.py` -- the models with all blocks of size 2, up to the parity flip and reversal.
- `sugra.py` -- the marking and sdim g_- of each column; the column (0000) by an independent
  Spencer-cohomology computation (standard library only); `rscal.py` imports its `nullspace`.
- `rn.py` -- which column is an enlargement R(N).
- `rscal.py` -- the two scalar curvatures of Par = (00001) restricted to Lambda^2(M).
- `test_models.py` -- self-tests of `models.py`/`cohom.py`.
