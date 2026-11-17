# donin

**Skew diagrams and decomposition into irreducible components of exterior and symmetric powers of the adjoint representation of gl(p|q)**

J. Donin, edited by D. Leites. Status: in preparation.

The scripts check the computations of Section 7 of the text ("Comments of the editor"). The
combinatorics of Sections 2--4 is implemented from the definitions printed there -- diagrams,
diagram mappings, the groups Aut, R and C, tabloids, the spaces T_B(A), S_B(A), S(A,B), S'(A,B)
and their triple analogues -- so that the Littlewood--Richardson rule is an output of the check,
not an input: Theorem 3.4 (dim S(A,B) = nu(A,B)), Remark 3.4.3, Theorem 3.4.2, Lemma 3.2,
Proposition 2.1 and Theorem 4.1 on the skew diagrams with at most five cells (Statement 7.1).
For gl(n), formulas (5.7) and (5.8) of Theorem 5.2 are compared with the dimension of the span of
the vectors (3), computed from honest polynomials in S(gl(n)), and with the decomposition of
S^k(gl(n)) and E^k(gl(n)) into irreducibles, computed from characters and reduced by Kostant's
theorem (Statement 7.2, Subsection 7.3, Proposition 7.4, Remarks 7.4.1 and 7.4.2). Exact
arithmetic throughout (integers and `fractions.Fraction`); the ranks are taken by
`../lib/exactlin.py`, standard library only (faster when python-flint is installed, correct
without it).

## How to run

```bash
cd donin
python3 run4.py
```

Every script is run from inside this folder, with no arguments; it finds `../lib/exactlin.py`
by itself. Each script takes a few seconds on a laptop.

`run3.py` imports `run2.py`, and `run4.py` imports `run3.py`, so their output begins with the
output of the scripts they import.

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Statement 7.1, items 1)--3) (Theorem 3.4, Remark 3.4.3 a)--e) and its B-nice reading, Theorem 3.4.2) over all pairs of the 31 skew diagrams with at most five cells | `allpairs.py` | `31 skew diagrams with at most five cells, 217 ordered pairs of equal order`, `1647 passed, 0 failed` |
| Statement 7.1, item 4) (Lemma 3.2: 163 semi-ordered tableaux with cycles) and the last paragraph (Theorem 4.1 in ten configurations of C) | `run2.py` | `ok   tableaux with cycles that also have regular ones: got 163, want 163`, `(over 31 diagrams, 163 tableaux with cycles)`; ten lines `ok   nu(C...)` under `Theorem 4.1`; `224 passed, 0 failed` |
| Statement 7.1, items 2) and 5) (Remark 3.4.3 d), e) against independently computed Littlewood--Richardson coefficients; Proposition 3.1 against the Specht module dimensions; the count of ordered tableaux against the hook length formula) | `test_skew.py` | sections `1.`--`9.`, every line `ok`, ending `108 passed, 0 failed` |
| Proposition 2.1, the vanishing in its proof (including the smallest instance, one row of two cells); Lemma 3.3 in its smallest instance; the Example after Remark 3.4.4 (four semi-ordered A-tableaux of type A, three of them with cycles) | `run1.py` | `semi-ordered A-tableaux of type A: 4`, `nu(A,A) = 3     dim S(A,A) = 3`; `nu(A,B) = 0`; for every lam the `Garnir situations tested` equals the number in which the `inner sum vanished` |
| Statement 7.2, first sentence (n = 3, 4, k <= 4: the span of the vectors (3) has dimension (5.7)); Remark 7.4.1 for gl(4) | `run4.py` | the table `n k f dim HS (5.7) harmonic verdict`; `cases where (5.7) differs from dim HS: 0`; `cases where dim HS differs from the harmonic count: 3` (n = 4: k = 3, f = (1, 1, -1, -1); k = 4, f = (2, 1, -1, -2) and (1, 1, -1, -1)) |
| Statement 7.2, second sentence ((5.8) for n = 4, k <= 3 and n = 5, k <= 4); Subsection 7.3 and Remark 7.4.1 ((5.7) against the harmonics: no discrepancy for gl(3), k <= 6; for gl(4) the two part company at k = 3 and k = 4) | `run3.py` | under `SYMMETRIC POWERS`, n = 3: every `k` line `ok`; n = 4: `FAIL` lines at k = 3, 4, 5 with `MISMATCH f=(1, 1, -1, -1): Kostant 0, formula 1` (k = 3) and `MISMATCH f=(2, 1, -1, -2): Kostant 1, formula 2`, `MISMATCH f=(1, 1, -1, -1): Kostant 1, formula 2` (k = 4) -- these are the discrepancy the text describes; under `EXTERIOR POWERS` every line `ok`; final line `15 passed, 3 failed` |
| Proposition 7.4 (gl(4), f = (1,1,-1,-1), k = 3: the vector (3) is a multiple of w * tr, non-zero, spanning HS_f^3, and dim PS_f^3 = 1); Statement 7.2 and Remark 7.4.2 for the Example of Section 5 (gl(4), k = 6, f = (3,1,-2,-2)) | `example6.py` | `w primitive: True`, `number of monomials of v: 8`, `v == w * tr: True`, `vector (3) for k = 2 equals 2w: True`, `v is the vector (3) for kappa = (2,1): True`, `v != 0: True`, `dim HS_f^3 = 1`, `dim PS_f^3 (multiplicity of f in S^3(gl(4))) = 1`; `dim HS_f^6 (span of the vectors (3))  = 2`, `formula (5.7)                         = 2`, `multiplicity of f in S^6(gl(4))       = 2`, `multiplicity of f in the harmonics    = 1` |

## Files

- `skew.py` -- diagrams, diagram mappings, the groups Aut, R, C, tableaux and tabloids, the spaces
  T_B(A), S_B(A), S(A,B) and the number nu(A,B), all from the definitions of the text.
- `test_skew.py` -- calibration of `skew.py` against Specht module dimensions, the hook length
  formula, Kostka numbers and Littlewood--Richardson coefficients.
- `run1.py` -- the Example after Remark 3.4.4, Lemma 3.3, the vanishing in the proof of
  Proposition 2.1.
- `run2.py` -- Lemma 3.2, Theorem 3.4.2 (the odd tabloids), Theorem 4.1 (triples of diagrams).
- `allpairs.py` -- Statement 7.1 over all pairs of diagrams with at most five cells; imports
  `run2.py`.
- `gln.py` -- the decomposition of S^k(gl(n)) and E^k(gl(n)) into irreducibles, by characters.
- `prim.py` -- S^k(gl(n)) by hand: monomials, the operators delta_m, the adjoint action, the
  vector (3).
- `run3.py` -- formulas (5.7)/(5.8) against the decomposition reduced by Kostant's theorem.
- `run4.py` -- the dimension of the span of the vectors (3) against (5.7) and against the
  harmonics.
- `example6.py` -- Proposition 7.4 and the Example of Section 5.
