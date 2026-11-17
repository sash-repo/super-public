# invcm

**Inverses of Cartan matrices of Lie algebras and Lie superalgebras**

D. Leites, O. Lozhechnyk. Status: published, *Linear Algebra Appl.* **583** (2019), 195--256;
arXiv:1905.12408. The scripts check the revised version of the text; all numbers below
(Sections 3.2--3.4 and 9.4 among them, which are new in it) are those of the revised version.

The scripts check the tables of Subsections 9.1 and 9.2 (the Cartan matrices A, their inverses
and determinants for the 132 almost affine Lie superalgebras and the 238 hyperbolic Lie algebras),
the symmetrizability and signature of these 370 matrices, the hypotheses of Theorems 3.2.1 and
3.3.1 on them, the path expansion of the adjugate (Lemma 1.2.2), Theorem 3.4.1 on random matrices,
the four Cartan matrices of each of NS3_82 -- NS3_86 in the real quadratic field Q(alpha)
(Subsections 9.3, 9.4), and formula (22) for the length of an alpha_k-string over F_p and F_{p^2}.
Exact arithmetic throughout (`fractions.Fraction`; sympy for Q(alpha); integers mod p), except
that `signe.py` and `checkc.py` also compute the smallest real eigenvalue lambda_0(A) of formula
(29) with numpy, in floating point (in `signe.py` only as a cross-check of an exact witness).

`tables.json` is the transcription of the 370 table entries of Subsections 9.1 and 9.2 (name, the
mark !!!, A, the printed inverse with the scalar in front of it, the printed determinant); it is
read by `tables.py`. The 20 Cartan matrices of Subsection 9.3 are typed into `cases.py` and
`galois.py`.

References: [BGL] = Bouarroudj, Grozman, Leites, SIGMA 5 (2009) 060, arXiv:0710.5149;
[Leb] = A. Lebedev, On reflections of roots in positive characteristic, arXiv:2311.02710;
[LT] = Lusztig, Tits, The inverse of a Cartan matrix, An. Univ. Timisoara 30 (1992) 17--23.

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r invcm/requirements.txt
cd invcm
../venv/bin/python invcm.py
```

Every script is run from inside this folder, with no arguments. Timings on one core: `invcm.py`,
`symm.py`, `leb.py`, `signe.py` under 1 s each; `mmat.py` 13 s; `checkc.py` 16 s;
`cases.py` and `galois.py` about 30 s each.

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script | Expected output |
|---|---|---|
| Tables of Subsections 9.1 and 9.2: every printed determinant and inverse; every determinant negative; the mark !!! exactly on the inverses with no zero element (39 of them), whose elements are then all negative; the zeros of the inverses only on the main diagonal | `invcm.py` | `read 370 table entries (132 from Subsection 9.1, 238 from Subsection 9.2)`, `marked !!!: 39`, three `OK`, `0 problems` |
| Statement 3.1.2 (239 symmetrizable: all 142 H, 97 of the 132 S; n_- = 1 for each, formula (28) holds; the other 131 not symmetrizable) | `symm.py` | `132 matrices, 97 symmetrizable, 35 not`, `238 matrices, 142 symmetrizable, 96 not`, `n_- = 1 :  239 matrices`, `fails for: nothing` |
| Corollary 3.2.2 (all 370 are of indefinite type, every A_{hat i} is a sum of matrices of finite or affine type, det A < 0; for the 132 superalgebras this is the computation the text refers to) | `signe.py` | `matrices checked: 370`, `violations ...: 0` |
| Lemma 1.2.2 (adjugate as a sum over paths), Proposition 1.2.3 (the tree hypothesis can be weakened to connectedness), Remark 1.2.5 (condition a) cannot be dropped) | `mmat.py` (PARTs 1, 2) | `1200 trials, 0 mismatches -> PASS`; `inverses with a non-positive entry: 0 -> PASS`; `the inverse has 3 zeros ... -> condition a) is needed` |
| Statement 3.3.2 (the hypotheses of Theorem 3.3.1 hold for all 370; the sign pattern of A^{-1} is the predicted one; the 39 marked !!! are exactly those with every A_{hat i} of finite type) | `mmat.py` (PART 3) | every count `370`, `all 370 agree`, `0 problems`, `PART 1 PASS   PART 2 PASS   PART 3 PASS` |
| Theorem 3.4.1 (Vinberg's types read off the inverse or adjugate), checked on about 26 000 random matrices | `checkc.py` | `failures of Theorem 3.4.1: 0` |
| Statement 2.7.5 (formula (22) agrees with the recursion of [Leb] on 6246 triples, p = 2, 3, 5, 7, over F_p and F_{p^2}; [BGL] (7.4) disagrees in 2762) | `leb.py` | `checked 6246 triples (i_k, A_kk, A_kj); mismatches with the recursion: 0`, `triples where [BGL] (7.4) disagrees with (22): 2762` |
| Subsection 9.3: -1 < alpha < 0; A_{k+1} = r_k(A_1), formula (54), with the kth node the only isotropic one; the printed determinants, inverses and marks; det A_1 < 0, det A_2 > 0, det A_3 > 0, det A_4 < 0 (the sign statement of Corollary 9.4.2) | `cases.py` | `OK` on every line, e.g. for NS3_84-3): `det = 3*(2*alpha + 3)`, `+5.60555 (positive)` |
| Statement 9.4.3 (A_1, ..., A_4 pairwise non-equivalent for NS3_82 -- NS3_85; A_2, A_3, A_4 pairwise equivalent for NS3_86); the four systems of simple roots give all the Cartan matrices obtained by reflections | `cases.py` | `NS3_86   equivalent pairs among A_1..A_4: [(2, 3), (2, 4), (3, 4)]   closure: 2 classes [[1], [2, 3, 4]]  OK`, `FAILURES: 0` |
| Lemma 9.4.1 (sigma permutes A_1, ..., A_4, fixing the classes of A_1, A_2 and interchanging those of A_3, A_4; its action on A_1) | `cases.py` | `sigma(A_i) ~ A_j for i = 1..4: j in [[1], [2], [4], [3]]` (NS3_82 -- NS3_85) |
| Formulas (55), (56) (alpha a unit of norm 1); formula (57) (det A_4 = sigma(det A_3)); Corollary 9.4.2, formula (58) (norms -3, -6, -9, -8, -16) | `galois.py` | `N(alpha) = 1`, `formula (55): ... = 0`, `det 4) = sigma(det 3)): True`, `N(det) = -3` ... `N(det) = -16` |

Not deposited here: the inverses of the Cartan matrices of the finite-dimensional Lie
(super)algebras (Sections 5, 7, 8; formulas (41) and the like of Section 6 are proved in the
text); Statement 2.3.1 (a general statement, not a computation). No claim of the text was
computed with *SuperLie*.

## Files

- `tables.json`, `tables.py` -- the 370 entries of Subsections 9.1, 9.2 and exact determinants.
- `invcm.py` -- the check of those tables.
- `symm.py` -- symmetrizability (a symmetrizer by propagation along the graph) and the inertia
  of the symmetrized matrix.
- `signe.py` -- the hypotheses and the conclusion of Theorem 3.2.1 / Corollary 3.2.2.
- `mmat.py` -- the path expansion of the adjugate, random M-matrices, and Theorem 3.3.1 on the
  tables.
- `checkc.py` -- Theorem 3.4.1 on random matrices.
- `leb.py` -- formula (22) against the recursion for the alpha_k-strings over F_p, F_{p^2}.
- `cases.py` -- Subsection 9.3, Lemma 9.4.1, Statement 9.4.3 (in Q(alpha), sympy).
- `galois.py` -- formulas (55)--(58): the Galois involution and the norms; also the orbit of
  A_1 under repeated reflections with a finer normalization (rows with a 0 on the diagonal
  not rescaled), and the signs of the determinants met in it.
