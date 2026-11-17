# indecops

**Which invariant differential operators on the line are indecomposable**

D. Leites, O. Lozhechnyk. Status: in preparation.

The scripts check the computations of the text: the spaces of n-linear differential operators on
the line, invariant under all changes of the variable, between weighted densities F_{a_1} x ... x
F_{a_n} -> F_nu (the invariance system (6) of the text, solved exactly); the reduction of Malyshev's
operators W_n^{(k)} to operators on 1-forms (Lemma 2.3); the census of Table 1, i.e., the dimension
of the subspace of compositions (Definition 3.1) for n <= 5 and every order; the S_n-module
structure of the indecomposable operators; the upper bound on the order of an invariant operator
for positive weights; the symbols, the Vandermonde determinant and the harmonics; and that the
conditions for q = 2, 3 imply the others (Subsection 2.2). The invariance condition is verified
independently, from the definition, with sympy. Exact rational arithmetic throughout
(`fractions.Fraction`; ranks and kernels by `exactlin.py`, which uses python-flint's exact
`fmpq_mat` when installed and pure Python otherwise); sympy only in `verify.py`.

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r indecops/requirements.txt
cd indecops
../venv/bin/python run.py 5
```

Every script is run from inside this folder, with the arguments given below. Without python-flint
everything except `verify.py` runs on plain `python3`, only slower. Times are for a laptop running
one script at a time, with python-flint.

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| The invariance condition (6) and its symbol form (7), (8) used by all the scripts | `verify.py` (about 3 minutes); `test_line.py` (seconds) | `verify.py`: `Gz = (6)*u^(0)v^(3) + (9)*u^(1)v^(2) + (-9)*u^(2)v^(1) + (-6)*u^(3)v^(0)` and `77 checks passed, 0 failed`; `test_line.py`: `100 checks passed, 0 failed` |
| Subsection 2.2 (the conditions for q = 2 and q = 3 imply the others, for all weights) | `thm3.py` (about 1 minute) | `152 checks passed, 0 failed` |
| Lemma 2.3 (W_n^{(n+s)} = the invariant operators on 1-forms of order s) | `run.py 5` (6 to 12 minutes; `run.py 4`, the same for n <= 4, seconds), STEP 1 | for every n <= 5 and k: `n=... k=...: dim W_n^(k) = d, dim inv(cF_1,s=k-n) = d  ok` |
| Theorem 3.2 (Grozman's binary operators on densities; its indecomposability clause) | `test_line.py`; `binary.py` (seconds) | `test_line.py`: order 1 of dimension 2 exactly at a = b = 0, order 2 exactly on ab(a+b+1) = 0, Gz at (-2/3,-2/3) with the profile (2,3,-3,-2), indecomposable; `binary.py`: the order-2 operators are compositions (`indecomposable 0`), `(-2/3,-2/3): dim 1, decomposable 0, indecomposable 1` at order 3 |
| Statement 4.1 and Table 1 (dimension of the space and of the quotient by the compositions, n <= 5); Corollary 4.2 | `run.py 5` (STEP 2) | e.g. for n = 5: `s= 7  k=12   dim= 15  decomposable=  9  INDECOMPOSABLE=  6`, `s= 8  k=13   dim=  9  decomposable=  0  INDECOMPOSABLE=  9`; the totals `total indecomposable = 1`, `4`, `20` for n = 3, 4, 5 (n = 2: both orders indecomposable) |
| Statement 4.9 (the indecomposables as S_n-modules) | `atomstruct.py 5` (under 1 minute) | for each (n, s): the decomposition of the whole space and of the decomposable subspace into irreducibles; the indecomposables are the difference, e.g. n = 5: `s=7: dim 15 = [3,1,1] + [2,2,1] + [2,1,1,1]   \| decomposable 9 = [2,2,1] + [2,1,1,1], INDECOMPOSABLE 6`; then the atoms for n <= 4 written out explicitly |
| Statement 4.5 (the bound M(m) = m(m-1)/2 for the listed positive weights; an operator of order 4 for (-1,-1,-1)); formula (13) (the largest order of a composite) and Problem 6.2 (the dimensions for the weights (2,1,1,1,1)) | `weights.py` (about 4 minutes) | for each weight tuple the list of dimensions in orders 0, ..., m(m-1)/2 + 2, e.g. `('7', '1', '1', '1'): [1, 3, 5, 6, 5, 3, 1, 0, 0]`, `('-1', '-1', '-1'): [1, 2, 2, 1, 1, 0]   <-- differs`, `('2', '1', '1', '1', '1'): [1, 4, 9, 15, 20, 22, 20, 12, 7, 1, 1, 0, 0]   <-- differs`; then `n=5: max composite order 7 (blocks [4, 1]), top order 10,  forced indecomposable in orders 8..10` and the same for n = 2, ..., 8. A `FAIL` line here means only "the dimensions differ from the Mahonian numbers", not a failed claim (see the note below) |
| Remark 4.7, items 2) and 3) | `degchk.py` (seconds) | `(Fraction(1, 3), 2, Fraction(5, 2)) [..., (3, 1, 3), ...]`; `(Fraction(1, 3), Fraction(1, 2), 2, Fraction(1, 5)) [(0, 1, 0), (1, 3, 1), (2, 5, 2), (3, 6, 3), (4, 5, 4), (5, 2, 5), (6, 0, None), ...]` — triples (order, dimension, highest power of the last variable) |
| Section 5: Table 1 agrees with Theorem 5.2 for n <= 5; Lemma 5.4 (the symbol of the Wronskian is the Vandermonde determinant); Remark 5.5 (the symbols are not the harmonics) | `harmonic.py` (about 2 minutes) | a table `n s \| dim symbols dim harmonics a_(n-1,s) same subspace?` with `False` exactly at (3,2), (4,2..5), (5,2..9), and `134 checks passed, 0 failed` |

Note on `weights.py`: its checks test whether the dimensions in orders 0, ..., m(m-1)/2 + 2 are the
Mahonian numbers. The tuples (-2/3,-2/3), (-1,-1,-1), (-1,-1,-1,-1) (negative weights),
(2,1,1,1,1), (3,1,1,1,1) (Problem 6.2) and (1,2,3,4) are reported as `FAIL`, ending with
`24 checks passed, 6 failed`; what Statement 4.5 asserts is read off the printed lists.

## Files

- `line.py` — the engine: the invariance system for given weights and order, composition of
  operators, the span of the composites (Definition 3.1), the indecomposable quotient.
- `exactlin.py` — exact rank, row echelon form and kernel over Q (python-flint if present).
- `test_line.py` — calibration against Grozman's classification and the Mahonian numbers;
  `verify.py` — the invariance condition checked from the definition, infinitesimally and under
  finite changes of the variable (sympy).
- `run.py` — Lemma 2.3 and the census of Table 1; `atomstruct.py` — S_n-characters of the census
  and the atoms written out.
- `weights.py`, `degchk.py` — the bound of Statement 4.5 and Remark 4.7; `thm3.py` — Subsection
  2.2; `binary.py` — the binary atoms; `harmonic.py` — symbols, Vandermonde, harmonics.
