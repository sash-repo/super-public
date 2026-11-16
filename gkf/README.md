# gkf

**The Gelfand--Kalinin--Fuchs problem in a finite-dimensional model**

D. Leites, O. Lozhechnyk. Status: in preparation.

Two sides of one computation. The classical side: the Lie algebra h(2) of Hamiltonian formal vector
fields on the plane, its cochain complex relative to sl(2), the dimensions of its bigraded pieces that
the 1972 note of Gelfand, Kalinin and Fuchs printed, the cohomology in the internal degrees 5, 6, 7 and
Metoki's class. The model: the Lie superalgebra h^(1)(0|n) and the bigraded tables of
dim H^q_k(h^(1)(0|n); K) over Q and over F_p. Exact arithmetic throughout (integers,
`fractions.Fraction`, python-flint, through `../lib/exactlin.py`; the largest ranks are computed mod a
prime with FLINT's `nmod_mat`, which gives a lower bound for the rank over Q).

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r gkf/requirements.txt
cd gkf
../venv/bin/python test_ce.py
```

Every script is run from inside this folder. Some write a log to `runs/` (created on the first run).
Times are for a laptop running one script at a time.

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| (the engine itself) | `test_ce.py` | `all validations passed` (sl(2), sl(3), an odd abelian superalgebra, H^2(h^(1)(0\|4)) = 3) |
| Statement 1.4 (the 1972 table) | `h2.py table 4 8` | the rows `N=1  0 1 1 0 1 1 0 0`, ..., `N=4  0 0 5 13 17 18 14 4` |
| Statement 1.5 (the internal degree 5 is empty) | `h2.py coh 5 12` (about 15 minutes) | `dim C~  0 1 4 19 50 59 35 15 5 0 0 0` and `dim H` zero in every degree |
| Statement 1.6 (the degree 6 is empty, the degree 7 carries Metoki's class) | `cohN.py 6 12`, `cohN.py 7 12` (about 40 minutes; over an hour) | `cohN.py 6 12`: `dim C~  0 0 9 38 80 126 134 74 16 2 1 0` and `dim H` zero in every degree; `cohN.py 7 12`: `dim H  0 0 0 0 0 0 0 0 1 0 0 0` (the class in q = 9; output of the run behind the text) |
| Statement 1.8 (the Euler characteristics, N <= 20) | `wt.py dims 20` (about 10 minutes); `wt.py check` (about 7 minutes) checks the weight reduction | one row per N with its Euler characteristic and `CLASS` where it is nonzero; `defects: 0` |
| Statement 1.9 (Metoki's class: a cocycle of degree <= 5 and none of degree <= 4) | `metoki.py` (about 15 minutes) | the explicit cocycle, monomial by monomial |
| Statement 2.3 (the ceiling at q = p) | `run.py 4 8 7 11` (about 4 minutes) | the table over Q, then over F_7 the row `7 ... 37` (37 classes against 5 over Q), and `mod 11: IDENTICAL to Q` |
| Statements 4.1, 4.2, 4.3 (the tables for n = 4, 5, 6) | `run.py 4 10 11` (about 25 minutes), `run.py 5 6 --w0 101` (about 26 minutes), `run.py 6 4 --w0 101 13` (about 12 minutes) | the tables of the text, row by row, with the column of totals; `mod 11: IDENTICAL to Q` (n = 4) and `mod 13: IDENTICAL to Q` (n = 6) |
| Statement 4.4 (the column k = 0 for n = 6, one degree further) | `colbig.py 6 0 5 101 11` (about 5 minutes) | `dim H^4_{0} = 1`, `dim H^5_{0} = 1` (with `dim C = 5418, rank out = 4578, rank in = 839`) |
| Statement 5.3 (the model has its own class in the degree 7) | `column.py 5 0 6 11` (about 1 minute), `bigrank.py 5 0 7 11` (about 9 minutes) | `dim H^q_{0} = 0` for 1 <= q <= 6; `bigrank.py`: `C^7 = 12514, C^8 = 36485`, `rank = 9464` |
| Statement 5.4 (two classes in the degree 7) | `row.py 5 7 101 --also 11` (about 46 minutes, 8 GB); the same program at a smaller size: `row.py 6 5 101 --also 11` (about 19 minutes) | for `row.py 5 7`: `dim H^7_{-9} = dim H^7_0 = 1`, the other computed k giving 0 (output of the run behind the text); for `row.py 6 5`: `dim H` equal to 1 at k = 0 and to 0 at the other computed k |

## Files

- `po.py` — the Poisson superalgebra po(0\|n) and h^(1)(0\|n); `ce.py` — the Chevalley--Eilenberg
  complex as S(Pi g^*), with blocks by degree and torus weight; `run.py`, `column.py`, `colbig.py`,
  `bigrank.py`, `row.py` — tables, columns, single ranks and rows of H^q_k.
- `h2.py` — h(2), its relative complex and its cohomology; `cohN.py` — one internal degree with a log;
  `wt.py` — the sl(2)-weight reduction; `metoki.py` — Metoki's class and its cocycle.
- `exactlin_shim.py` — the import of `../lib/exactlin.py`.
