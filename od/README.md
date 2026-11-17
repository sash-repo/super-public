# od

**On odd parameters in geometry**

D. Leites. Status: in preparation.

The scripts check the computations of the text: the adjoint cohomology H^1(g; g) and H^2(g; g)
behind Section 7 ("Description of deformations") -- the Killing forms of osp(m|2n) and
osp_alpha(4|2), Statement 7.4.3 on osp(2n+2|2n), and the terms of the Hochschild--Serre spectral
sequence for psq(n) and spe(n) (Subsections 7.5 and 7.6) -- and the localization of a
supercommutative superalgebra at the complement of a prime ideal (Subsection 3.3). Exact rational
arithmetic throughout (`fractions.Fraction`); standard library only.

## How to run

```bash
cd od
python3 adjcoh.py
```

Every script is run from inside this folder, with no arguments. Times are for a laptop running one
script at a time.

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script | Expected output |
|---|---|---|
| Subsect. 7.4.1 (the Casimir argument dies exactly when the Killing form does) | `adjcoh.py` (about 1 minute), part 1 | the Killing form of osp(m\|2n) and of osp_alpha(4\|2) as a multiple of the supertrace form, e.g. `Killing form of osp(4\|2): rank 0 of 17   (identically zero)`, `Killing form of osp(6\|4): rank 0 of 49   (identically zero)`, `Killing form of osp(3\|2): rank 12 of 12` |
| Statement 7.4.3 (sdim H^1(g; g) = (0\|0), sdim H^2(g; g) = (1\|0) for g = osp_alpha(4\|2); both (0\|0) for osp(6\|4) and osp(8\|6)) | `adjcoh.py`, part 2 | for `osp_1(4\|2)`, `osp_2(4\|2)`, `osp_1/3(4\|2)`: `sdim H^1(g;g) = 0\|0`, `sdim H^2(g;g) = 1\|0`; for `osp(6\|4)`: `sdim H^1(g;g) = 0\|0`, `sdim H^2(g;g) = 0\|0` |
| Subsect. 7.5.1 (psq(n): the term E_1^{2,0} and the differential d_1) | `adjcoh.py`, parts 2 and 3 | for psq(3), ..., psq(6): `sdim H^1(g;g) = 0\|1`, `sdim H^2(g;g) = 0\|0`; the witness `(d mu_od)(W,W,W)` for W = diag(n-1,-1,...,-1); the cross-check of psq(3) by the full Chevalley--Eilenberg complex |
| Subsect. 7.6.1 (spe(n), and why n = 3 is exceptional) | `adjcoh.py`, part 2 | `spe(3)`: `sdim H^1(g;g) = 1\|0`, `sdim H^2(g;g) = 0\|1`; spe(4), spe(5), spe(6), and the dual realization spe(n)': `sdim H^1(g;g) = 1\|0`, `sdim H^2(g;g) = 0\|0`; the cross-check of spe(3) by the full complex |
| Subsect. 3.3 (Lemma 3.3.2, Proposition 3.3.3, Remark 3.3.4: the equivalence of fractions, the even part of a denominator, right fractions as left ones, cross-multiplication is not the equivalence) | `localiz.py` (seconds) | in both models U and T, every line `0 failures / 250` (or `/ ...`), and the table of the parity signs of the display of Subsect. 3.3.1: they differ only on rows with an odd denominator |
| Subsect. 3.3.1 (A. Lebedev's description of the equivalence relation via the even part of the denominator) | `localiz2.py` (seconds) | every check `[ok  ]`, ending with `TOTAL FAILURES: 0` |

`localiz2.py` imports the algebra models from `localiz.py`.

## Files

- `adjcoh.py` -- Lie superalgebras osp(m|2n), osp_alpha(4|2), psq(n), spe(n) in matrix form; their
  Killing forms; the relative (row q = 0) Hochschild--Serre complex C^p = Hom_{g_ev}(S^p(g_od), g)
  and the full Chevalley--Eilenberg complex in weight 0.
- `localiz.py` -- two models of a supercommutative superalgebra with a prime ideal, and the checks
  of the localization.
- `localiz2.py` -- A. Lebedev's relation, checked step by step on the same models.
