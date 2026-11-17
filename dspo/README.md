# dspo

**The Duflo--Serganova homology for the Poisson and queer versions of the general linear Lie algebra**

D. Leites, O. Lozhechnyk. Status: in preparation.

The scripts compute the DS-homology g_x = Ker ad_x / Im ad_x of the queer series q(n), sq(n),
pq(n), psq(n) for n <= 7 (Section 2) and of the Poisson Lie superalgebra po(0|m) for m = 5, 6, 7
(Section 3): ranks, superdimensions, the Z-grading of g_x, derived series, a search over
non-monomial homologic elements, and the semisimple quotients g_x/r_x of Statement 3.2.6.
`po.py`, `porun.py`, `qsq.py`, `sdim.py` work over Q with `fractions.Fraction` (exact); the radical
computations (`radical.py`, `radical2.py`, `probe.py`, `socle.py`, `ideal98.py`) work modulo a large
prime (2^31 - 1 for `radical.py`, 33554393 for the others), the structure constants being integers.

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r dspo/requirements.txt
cd dspo
../venv/bin/python qsq.py
```

Every script is run from inside this folder. Timings on one core: `sdim.py`, `po.py`,
`radical.py 5`, `radical2.py 5 6`, `probe.py 6` a few seconds each; `qsq.py` 10 s;
`porun.py 5 6 7` about 6 min (m = 7 takes most of it); `radical2.py 7` about 2.5 min;
`probe.py 7` about 7 min, the `socle.py` line below about 4 min, `ideal98.py` 20 s.
`radical.py` builds the whole associative envelope and is used for m = 5 only, as an independent
check of `radical2.py`.

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Lemma 2.2.2 (q(n)_{x_k} = q(n-2k), rank ad_{x_k} = 4k(n-k), the top storey of psq) and table (8), n = 3..7 | `qsq.py` | the four blocks `q(n)`, `sq(n)`, `pq(n)`, `psq(n)`, e.g. `6  72  3  36  0  0\|0` for q(6), `6  70  3  34  2  1\|1  <== psq(0) = 0` |
| Remark 2.2.3 (the commutator and anticommutator blocks of ad_{x_k}, each of rank 2k(n-k)) | `qsq.py` (last block) | columns `rk [B,-]` and `rk (BA+AB)` both equal to `2k(n-k)` |
| The second family (7): rank ad_{x'} = n^2-1 and pq(n)_{x'} = Pi(C), n = 3..7; ranks 14, 34 and sdim 1\|1 for psq(4), psq(6) | `qsq.py` (block "B^2 = c.Id") | `pq  4  15  1  0\|1`, `psq  4  14  2  1\|1`, `psq  6  34  2  1\|1` |
| Remark 3.1.2 (rank ad_x on homologic x: {0, 12, 16} for q(4); {0, 5, 10, 12, 13, 14, 16} for po(0\|5)) | `qsq.py`; `porun.py 5` | q(4): ranks `12`, `16`; po(0\|5): ranks 5, 10, 12, 16 on monomials and 13, 14 on non-monomials, with dim g_x 22, 12, 8, 0, 6, 4 |
| Formula (22) (sdim of po, h, h^(1)) and sdim h^(1)(0\|4) = sdim psl(2\|2) = 6\|8 | `sdim.py` | `h^(1)(0\|4) of sdim 6\|8   vs psl(2\|2) of sdim 6\|8`, `cross-check ... OK` |
| Tables of Subsecs. 3.1.3 (po(0\|5)), 3.1.4 (po(0\|6)), 3.1.5 (table (24), po(0\|7)), including the derived series of (24) | `porun.py 5 6 7` (`po.py 5 6` for a short version) | the rows by type, e.g. `5  (2, 0, 1)  5  16\|11  5\|0  11\|11` for m = 5 and `7  (3, 0, 1)  7  64\|57  7\|0  57\|57` for m = 7; derived series `[114, 112, 112]` etc. |
| Remark 3.1.6 (rank 3·2^{m-3}, dim g_x = 2^{m-2}; derived series 8, 6, 2, 0 and 8, 4, 0 for m = 5; m = 7 rank-48 g_x not solvable) | `porun.py 5 6 7` | m = 5: `derived [8, 6, 2, 0]`, `derived [8, 4, 0]`; m = 7: `derived [32, 30, 30]`, `derived [32, 28, 28]` |
| Theorem 3.2 and Remark 3.2.2 (graded duality n_i = Pi^m n_{m-4-i}^*; the centre [m/2]-2 fails for m odd; the grading of g_x for po(0\|5), x = xi1 xi2 theta) | `porun.py 5 6 7` | `(A) holds for every homogeneous x above` for each m; `(B) FAILS` for m = 5, 7; `n_-2=1\|0 n_-1=0\|2 n_0=3\|0 n_1=0\|3 n_2=2\|0 n_3=0\|1` |
| The table after Conjecture 3.2.3 (largest rank with g_x != 0 on non-monomial elements: 14, 30, 61; dim g_x = 4, 4, 6; g_x commutative) | `porun.py 5 6 7` (clique scan, seed fixed) | `max rank with g_x != 0 among these: 14`, `30`, `61`, each with `derived [4, 0]` or `[6, 0]` |
| Remark 3.2.4 (m = 6: derived series 16, 12, 4, 0; m = 7: perfect parts of superdimension (15\|15), (14\|14)) | `porun.py 6 7` | `derived [16, 12, 4, 0]`; `perfect part sdim 15\|15`, `perfect part sdim 14\|14` |
| Statement 3.2.6, columns rank ad_x and sdim s (all rows, m = 5, 6, 7) | `radical2.py 5 6 7`; `radical.py 5` (independent check) | `sdim g_x/r=(3, 0)`, `(10, 0)` (m = 5); `(8, 9)`, `(6, 7)` (m = 6); `(7, 8)`, `(17, 18)`, `(7, 8)`, `(3, 4)`, `(9, 8)`, `(21, 22)`, `(13, 14)`, `(21, 0)` (m = 7); `(0, 0)` for the solvable cases |
| Statement 3.2.6, column s (the structure: s', its centroid, the Killing form and toral rank of s_0) | `probe.py 6 7` | e.g. `xi1eta1theta g_x/r sdim=(7, 8) s'=14 sdim s'=(6, 8)` (psl(2\|2) of codimension 1), `xi1xi2xi3 g_x/r sdim=(8, 9) s'=17` (perfect, s_0 of Killing rank 8) |
| Statement 3.2.6 (the terminal derived term of s: sdim and centroid, i.e. the ideal S ⊗ Λ(n)) | `socle.py 6:xi1xi2xi3eta1eta2 7:xi1xi2eta1 7:xi1xi2theta 7:xi1xi2xi3 7:xi1xi2xi3eta1eta2 7:xi1xi2eta1eta2theta 7:xi1xi2xi3eta1theta` | `P=(6, 6)` (o(4) ⊗ Λ(1)) for m = 6; for m = 7: `P=(3, 3)`, `(6, 6)`, `(17, 17)`, `(12, 12)`, `(20, 20)`, `(9, 8)` (sl(2) ⊗ Λ(1), sl(2) ⊗ Λ(2), spe(3) ⊗ Λ(1), o(4) ⊗ Λ(2), o(5) ⊗ Λ(2), and s itself, perfect) |
| Statement 3.2.6, row m = 7, x = xi1 xi2 xi3 eta1 theta (s of sdim (9\|8) is perfect but not simple) | `ideal98.py` | `ideal generated: 12 of 17`, `ideal: sdim (6, 6)` (the ideal sl(2) ⊗ Λ(2)) |

Lemma 2.2.2, Proposition 2.7 and Theorem 3.2 are proved in the text; the scripts above are checks.
No claim of the text was computed with *SuperLie*.

## Files

- `po.py` — po(0|m) = Λ(m) with the Poisson bracket of formula (20), ad_x, Ker/Im, the quotient
  g_x with its bracket, derived and lower central series; checks Jacobi and super
  anti-commutativity.
- `porun.py` — the tables of Section 3 by automorphism type, the graded duality, the derived
  series, the search over commuting sets of monomials.
- `qsq.py` — q(n), sq(n), pq(n), psq(n) in the realization (6) and their DS-homology.
- `sdim.py` — the superdimensions of the superspaces of the text.
- `radical2.py` — the radical of g_x by iterated nilradicals (Dickson's trace criterion on a sample
  of the envelope, the result certified to be a solvable ideal) and the quotient s = g_x/r_x;
  `radical.py` does the same with the whole envelope (m = 5).
- `probe.py`, `socle.py`, `ideal98.py` — the structure of s (derived algebra, centroid, Killing form
  of s_0, terminal derived term, the ideal generated by the kernel of the Killing form of s_0).
