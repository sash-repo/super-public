# indecomp

**Supervarieties of indecomposable modules over Lie superalgebras and Grassmann superalgebras**

A. Krutov, D. Leites. Status: in preparation.

The scripts check the computations of the text: the first cohomology H^1(g; M), split by parity, of
q(n), sq(n), pe(n), spe(n) with coefficients in the tautological module gl(n|n), in the adjoint
module and in the trivial one; the cohomology rings H^*(g; C) and the relative cohomology
H^*(g, g_ev; C) behind formulas (14) and Statement 4.5.1; the odd deformations of the trivial
sl(1|1)-module (Subsection 3.3); the odd deformations of the irreducible osp(1|2)-modules in
characteristic p (Statement 4.6.1); and the indecomposability of the modules of type II over
Lambda(2). Exact arithmetic throughout (integers, `fractions.Fraction`, sympy; arithmetic mod p in
`hwmod.py`/`prol.py`); numpy only for integer bookkeeping.

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r indecomp/requirements.txt
cd indecomp
../venv/bin/python defcheck.py
```

Every script is run from inside this folder, with the arguments given below. Times are for a laptop
running one script at a time.

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Section 3, Lemmas 3.1, 3.2 (H^1 of sl(1\|1) and gl(1\|1)); Claims 4.2, 4.3, 4.4.2, 4.4.3 and the two tables of Subsection 4.5 (sdim H^1 of q(n), sq(n), pe(n), spe(n) with coefficients in gl(n\|n), in the adjoint module and in the trivial one) | `defcheck.py` (about 2 minutes) | one line per algebra and module, each ending `OK`, e.g. `H^1(sq(2); gl(2\|2)) = 1\|1  OK`, `H^1(spe(2); End(spe(2))) = 0\|1  OK`; then the column of H^1(g) with trivial coefficients |
| Subsection 3.3 and Lemma 2.7 (the odd deformations of the trivial sl(1\|1)-module and what the construction of Section 2 does to them) | `nice.py` | the two candidate tables `NOT a module`, the corrected table a module; and the conclusion that the construction sends the whole (0\|2)-dimensional family to the zero sheaf |
| Formulas (14) and Statement 4.5.1 (the cohomology rings of pe(n), spe(n), q(n); the transgression in the Hochschild--Serre spectral sequence) | `cohtriv.py pe 3 --deg 9`, `cohtriv.py spe 3 --deg 9`, `cohtriv.py q 3 --deg 6` (about half a minute; to degree 7 and the self-test, which includes q(3) to degree 7, over half an hour); `relinv.py pe 3`, `relinv.py spe 3` | `cohtriv.py pe 3`: `dim H^k: 1 1 0 0 0 1 1 0 0 0`; `cohtriv.py spe 3`: the line `dim H^k` for spe(3) (H^5 = H^6 = H^8 = 1); `cohtriv.py q 3 --deg 6`: `dim H^k: 1 1 1 2 1 1 1`; `relinv.py`: the relative cohomology and the E_2-page, e.g. for pe(3) `dim H^k(g,g_ev;C): 1 0 0 0 1 0 0 0 1 0 0` and `H^*(g_ev) = Lambda('xi_1', 'xi_3', 'xi_5')` |
| Statement 4.6.1 (the odd deformations of the p-dimensional irreducible osp(1\|2)-modules, p = 3, 5, 7) | `osph1.py`, `ospcoc.py 3`, `ospcoc.py 5` | `osph1.py`: `H^1 = 0\|2` exactly for the module of dimension p, `0\|0` for the others; `ospcoc.py`: `uniform formulas are cocycles: True True` |
| Subsection 2.3 (the type II modules over Lambda(2) are indecomposable) | `zncheck.py` | `indecomposable` for mu = 0 and for v = 1; the controls with v > 1, mu != 0 report `DECOMPOSABLE` |

## Files

- `defcheck.py` — Lie superalgebras q(n), sq(n), pe(n), spe(n), gl(1\|1), sl(1\|1) as matrix models and
  H^1(g; M) by exact linear algebra; `cohtriv.py` — H^*(g; C) with trivial coefficients, reduced by
  weight and by an extra Z-grading; `relinv.py` — H^*(g, g_ev; C) and the Hochschild--Serre E_2-page.
- `nice.py` — the deformations of the trivial sl(1\|1)-module.
- `hwmod.py`, `prol.py` — highest weight modules and representations of Lie superalgebras over F_p;
  `osph1.py`, `ospcoc.py` use them.
- `zncheck.py` — endomorphism algebras of the modules over Lambda(2).
