# dsmode

**Duflo--Serganova homology for exceptional modular Lie superalgebras with Cartan matrix**

A. Krutov, D. Leites, J. Shang. Status: in preparation.

The scripts check the computations of the text that concern the matrix Lie superalgebras and the
symmetric pairs: the DS-homology g_x = Ker ad_x / Im ad_x of gl, sl, psl(n|n) and of gl, psl(n|n+pk)
over Q and F_p (Sections 3 and 4, and Sherman's symmetries of Statement 1.7.4); the squaring on g_x
for p = 2 (Examples 1.4.3, 1.4.4, Proposition 1.4.8); the homological elements of gl(m|n), their
normal shapes, the Weyl-group step and the cone of psl(2|2) (Subsec. 1.7.7); the atypicality tables
of brj(2;3) and brj(2;5) (Subsecs. 6.3.1, 6.3.2); and the invariants, ranks and relative cohomology
of the Z/2-graded Lie algebras and of the queer series (Section 5). Exact arithmetic throughout
(integers, `fractions.Fraction`, arithmetic in GF(2^k) by bitmasks, and python-flint through
`../lib/exactlin.py`).

The tables of Section 2 (the exceptional g(A)) and every claim built on the structure constants of
those algebras were computed with the *SuperLie* package and are not deposited here; see the list
at the end.

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r dsmode/requirements.txt
cd dsmode
../venv/bin/python dscheck.py
```

Every script is run from inside this folder, with no arguments except `sherman.py npk`. `sym.py`
takes about 4 minutes, `princ.py` about 1 minute; every other script finishes in under 40 seconds.

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Examples 1.4.3 and 1.4.4 (the squaring is lost on g_x for p = 2, also in an algebra with an even NIS) | `alsq.py` (parts 1, 2) | `theta_x=NON-ZERO`, `witness y = y,  y^2 = b  not in Im ad_x` for both; `NIS: violations {'sym': 0, 'even': 0, 'inv': 0}, rank of the Gram matrix 8 of 8` |
| Proposition 1.4.8 (gl, sl inherit the squaring for p = 2), and the identity y^2 = [x, ay] of its proof | `alsq.py` (part 3) | `theta_x != 0 for 0` for gl, sl(m\|n), (m\|n) = (1\|1), (2\|1), (1\|2), (2\|2), (3\|1), (2\|3); `failures: 0` for the identity |
| Statement 1.7.4 (Sherman's symmetries for gl, sl, psl(n\|n), n = 2, 3, 4, p = 0, 2, 3, 5) | `sherman.py` | `= g_x x K^{0\|r}` in every block except psl(n\|n), r = n, where `dim t' = n-1`, `sdim g(x,t') = (0\|n+1)  DIFFERS` (12 such lines); `exists h with [h,x]=x: True` for gl always, and for sl, psl at r = n exactly when p divides n |
| Statement 1.7.4, item 1) for gl, psl(n\|n+pk) | `sherman.py npk` | every block `= g_x x K^{0\|r}`, `central r/r`, `exists h with [h,x]=x: True`; no `DIFFERS` |
| Proposition 1.7.8 (the two enumerations for gl(m\|n)), Example 1.7.9 (the four elements (22)), Remark 1.7.10 (iso-sets) | `aladj.py` | `=== 357 checks, 357 passed, 0 failed ===`; e.g. for gl(3\|3) over F_2: `262144 odd matrices, 2052 of them homological`, `the number of orbits is (k+1)(k+2)/2 = 10`, `the number of isomorphism classes of g_x is k+1 = 4` |
| Corollary 1.7.11, Proposition 1.7.12 (normal shapes as iso-sets with signs, W-orbits), Statement 1.7.13 (the cone of psl(n\|n), the family X') | `alweyl.py` | `=== 892 checks, 892 passed, 0 failed ===`; `224 homological x with x^2 = 0 in gl, 96 with x^2 in K*.1` over F_3 and `1824 ..., 1920 ...` over F_5, all with `sdim g_x=(0, 2)` |
| Tables (54), (55), (56) (gl, sl, psl(n\|n), n = 2, 3, 4, p = 0, 2, 3, 5) | `dscheck.py` (Section 3 part) | columns `rank` and `dim g_x`: gl rows `6, 8`; `10, 16, 18`; `14, 24, 30, 32`; sl rows `6, 7`; `10, 16, 17`; `14, 24, 30, 31`; psl rows `6, 6`; `10, 16, 16`; `14, 24, 30, 30`, the same for every p |
| The tables of Section 4 (gl, psl(n\|n+pk), p = 2, 3, 5) | `dscheck.py` (Section 4 part) | the ranks of the tables, e.g. `p=2  gl(3\|9):  r=1: rank 22 ... r=2: rank 40 ... r=3: rank 54`, with dim g_x equal to dim of the g_x printed |
| Subsecs. 6.3.1, 6.3.2: the columns (lambda+rho, mu) and (lambda+rho, mu)_p of the atypicality tables of brj(2;3) and brj(2;5) | `atcheck.py` | `8 rows checked, 0 mismatches`; `24 rows checked, 0 mismatches` |
| Statement 5.3.3 (Hilbert series of five small pairs) and Statement 5.3.4 (gl(m\|n), (m\|n) = (1\|1), (2\|1), (3\|1), (2\|2)) | `princ.py` (parts 3, 4) | `MATCH` on each of the nine lines |
| Lemma 5.6.1 (only sl(3) = (sl(2), L^4) is a Z/2-grading) and Statement 5.6.3 (the relative cohomology of the principal sl(2)) | `princ.py` (parts 1, 2) | `dim Hom(Lambda^2 L^4, L^4) = 0`, `... L^6, L^6) = 1`, `... L^10, L^10) = 1`; `H^.(g,k) = 1 + t^{7}` for o(5), `1 + t^{11}` for g(2), `1 + t^{5}` for sl(3) |
| Statement 5.3.5 (the queer series q, sq, pq, psq(n), n = 2, 3, 4) | `queer.py` | `OK` and `=> Krull dim = n` for gl(n), `n-1` for sl(n); the control gl(m\|n) gives min(m, n) |
| Statement 5.4.4 (the ranks of the small classical pairs) | `sym.py` (part 2) | `rank of the pair = ...  (abelian: certified maximal)` for each pair listed, e.g. `sl(5) / o(5) ... rank of the pair = 4` |
| Statement 5.5.2 (the Poincare polynomials of H(g, g_ev) for the 16 small pairs) | `sym.py` (part 2) | `differential ZERO   Poincare ...` with the polynomials of the table, e.g. `sl(5) / s(gl(2)+gl(3)) ... Poincare 1 + t^{2} + 2t^{4} + 2t^{6} + 2t^{8} + t^{10} + t^{12}` |
| Statement 5.5.3 (Borel's formula for the equal-rank exceptional pairs) | `sym.py` (part 4) | `chi = 3, 3, 12, 36, 27, 72, 63, 56, 120, 135` for g(2), f(4) (two), e(6) (two), e(7) (three), e(8) (two) |
| Statement 5.6.2 (the invariants of the exterior algebra for the principal sl(2)) | `sym.py` (part 1) | `1 + t^{3} + t^{4} + t^{7} + t^{8} + t^{11}` (g(2)), `1 + t^{3} + t^{4} + t^{7}` (o(5)), `1 + t^{5}` (sl(3)) |

## Computed with SuperLie, not deposited

The following claims rest on the *SuperLie* package (P. Grozman, A. Krutov), or on structure
constants of the exceptional algebras exported from it; neither the package nor the exported data is
part of this repository:

- the tables of Section 2 (the DS-homology of the exceptional g(A) for p = 2, 3, 5), the
  multiplication tables in g_x, and their recheck against formula (15);
- the verification of Hypothesis 1.7.1 for the exceptional g(A);
- the depth table (14);
- Statement 1.4.9 (theta_x for e(6,1), e(6,6), bgl(3;a), bgl(4;a)) and the witness y = x_8 + x_9
  after it;
- Subsec. 1.7.5 (Sherman's symmetries for bgl(3;a) and bgl(4;a), p = 2);
- Statements 1.7.14, 1.7.16, 1.7.17 and Remark 1.7.18 (the homological elements and the whole cone
  of brj(2;3), brj(2;5), bgl^(1)(3;a), bgl(4;a) over finite fields, and their orbits);
- the column sdim V^lambda of the tables of Subsecs. 6.3.1, 6.3.2, and Subsec. 6.3.4.

## Files

- `dscheck.py` — rank ad_x and dim g_x for gl, sl, psl in the alternating format (Sections 3, 4).
- `sherman.py` — Sherman's subalgebra g(x, t') for the same families (Statement 1.7.4).
- `alsq.py`, `gf2k.py` — Lie superalgebras with squaring over GF(2^k), the obstruction theta_x
  (Subsec. 1.4).
- `aladj.py` — homological elements of gl(m\|n), the pair (a, b), normal shapes (Subsec. 1.7.7);
  `alweyl.py` uses it (iso-sets, the Weyl group, psl(n\|n)).
- `atcheck.py` — the atypicality tables of brj(2;3), brj(2;5).
- `sym.py` — matrix models of the Z/2-graded classical Lie algebras, ranks of the pairs, the
  relative complex, Borel's formula; `princ.py` uses it (the principal sl(2), symmetric-algebra
  invariants); `queer.py` — the queer series.
