# oddcp1

**Non-split supermanifolds from the complex projective line, and the versal deformations of superstrings**

D. Leites, O. Lozhechnyk. Status: in preparation.

The scripts check the computations of the text. Sections 3–16: supermanifolds of dimension 1|m with
underlying manifold CP^1 and retract E = O(-k_1) + ... + O(-k_m), written by explicit transition
functions on the two standard charts, y = x^{-1}, eta_j = x^{-k_j} xi_j, composed with exp(z) or
exp(tau z) for a Cech cocycle z (tau an odd parameter of the base). From the transition functions the
engines compute the Lie superalgebra v of global vector fields (its superdimension, and homogeneity
via the evaluation v --> T_p, decided by a gcd), the Cech cohomology H^0, H^1 of the graded pieces T_q
of the tangent sheaf, the action of g = H^0(M; T_0) = aut(E) + sl(2) on H^1(M; T_q) for every q (odd
included), invariants of copies of sl(2), orbit dimensions, and brackets of classes. Sections 18–27:
the versal bases of superstrings with a structure over a curve of genus g, by Riemann–Roch and Serre
duality, symbolically in g (sympy), and the Schwarzian layer on the flat model k^L(1|n).
Everything is exact (`fractions.Fraction`, sympy; linear algebra through `../lib/exactlin.py`, which
uses python-flint when it is installed).

[BO] = Bunegina–Onishchik, J. Math. Sci. 82 (1996) 3503–3527 (the text's [BO2]); [LT] =
Leites–Tikhomirov, arXiv:2503.02416; [Ba1], [Ba2], [Ba5] = Bashkin's papers on CP^{1|4}, on the
retracts (k,k,2,0) and on CP^{1|5}. In `vaintrob.py`, `contact.py`, `riggings.py`, `srs.py` the key
[Va1] denotes Vaintrob's 1988 survey (Itogi Nauki 32), cited in the text as [Va2], and [Va2] there
is the 1984 note, cited in the text as [Va1].

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r oddcp1/requirements.txt
cd oddcp1
../venv/bin/python run.py
```

Every script is run from inside this folder, with no arguments except `runscan.py 4` (and the optional bound of `run.py`). Some scripts
(`run.py`, `runcd.py`, `evenhom.py`, `vaintrob.py`, `contact.py`, `riggings.py`, `runschw.py`) also
write a `.log` file repeating what they print. Running times, measured on a laptop running other
jobs at the same time, are given in the table; the scripts without a time take under 3 minutes.

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Fact 3.1, Remark 3.3, Example 4.1, Section 5 items 1) and 4): the 18 bundles (B--O Prop. 18 confirmed, the direct-sum reading wrong in 9); of the 45 retracts (k_1,k_2), 0 <= k_2 <= k_1 <= 8, 39 carry a non-split structure and exactly one, over (2,2), is homogeneous; sdim v = (7\|6) for the retract (2,2), (6\|6) for Q^{1\|2} | `run.py` (about 35 min; `run.py 5` restricts Part B to k_1 <= 5 and takes a few minutes) | Part A: `B--O's Prop. 18 reproduced by the engine in all 18 bundles; the naive direct-sum reading fails in 9 of them.`, row `(2, 2) (7, 8) (7, 6) (7, 6)`; Part B: row `(2, 2) 1 1 1 of them [(6, 6)]`, `homogeneous non-split 1\|2 supermanifolds found: [(2, 2, [(1,)])]` |
| Remark 4.2 ((1,1,1), (2,1,1), (3,1,1): sdim v = (12\|12), (12\|12), (14\|14), those of the retracts); Section 5, items 2) and 3) (the series (2,2,k) of [BO, Thm. 1]; (3,1,k), (4,0,k), (3,3,2)) | `runcd.py` (about 8 min) | Part C: `(2, 2, k)` homogeneous `True` for k = 0, 2, ..., 6 and `False` for k = 1; `(3, 1, k)`, `(4, 0, k)`, k = 0..3, and `(3, 3, 2)`: non-split `True`, homogeneous `False`; `(1, 1, 1) (2, 3) ... (12, 12) (12, 12)`, `(2, 1, 1) ... (12, 12) (12, 12)`, `(3, 1, 1) (2, 3) ... (14, 14) (14, 14)` |
| Statement 4.3, table (9) | `extra.py` (about 3 min) | rows `(2, 2, 3)`: `(12, 10)`, `(9, 8)`, `(11, 10)`, `(9, 8)`; `(2, 2, 5)`: `(16, 14)`, `(13, 12)`, `(15, 14)`, `(13, 12)`; `(2, 2, 0)`: `(14, 14)`, `(13, 13)`, `(13, 13)`, `(11, 10)`; transitive `True` exactly in the columns split and z_2 (and z_1 for (2,2,0)) |
| Proposition 7.1 (m = 1: orbit of a general class of dimension min(k-4,3), moduli max(k-7,0), aut E acts by scalars); Section 8, after table (15): for b <= 2, a <= 7 aut E acts on H^1(T_1) by scalars | `runorb.py` | m = 1: `max orbit: aut(E) 0, g` 0, 1, 2, 3, 3, ... and `moduli <=` 0, 0, 0, 0, 1, 2, 3, 4 for k = 4..11; m = 2: `aut(E) 0` for every (a,b) with b <= 2 |
| Proposition 8.1, formulas (13), (14), table (15) | `odd.py`, `t_mod.py` | `odd.py`: columns `[LT] (9)` and `whole H^1(T_1)`, e.g. `(4, 4) 2 ... 8`, `(6, 6) 6 ... 16`; `t_mod.py`: `0 mismatches out of 26` |
| Statement 9.1, 1) (the retracts (2,0), (3,0), (3,1), (3,2): one non-split family, not homogeneous, sdim v) and the absence of homogeneous families there (Statement 10.1, last paragraph); Statement 10.1, 2), 3) | `runscan.py 4` (about 25 min; the default, `runscan.py`, runs every a <= 6 and takes several hours) | `(2, 0)`: `(17, 17)` against `(18, 18)`; `(3, 0)`: `(16, 17)` against `(18, 18)`; `(3, 1)`: `(14, 14)` against `(16, 16)`; `(3, 2)`: `(12, 14)` against `(14, 14)`, each `homogeneous False <-- dense orbit`; `(4, 0)`: `orbit dim 0 sdim_C v = (19, 19) homogeneous True`; `(4, 2)`: `orbit dim 0 sdim_C v = (15, 16) homogeneous True` |
| Statement 9.1, 2), formulas (16)-(18); Statement 10.1, 1) | `run33.py` (about 1 min) | Part A: every line `OK`; Part B: orbit dimension 2 for Q = 0 and 3 for Q != 0; Part C: `(15, 15)` split, `(12, 15) False` for Q = 0, `(11, 15) True` for Q != 0 |
| Table (19) | `runjoint.py` | column `moduli <=`: `(2, 2)` 0, `(3, 1)` 0, `(3, 2)` 0, `(3, 3)` 0, `(4, 0)` 1, `(4, 1)` 1, `(4, 2)` 2, `(4, 3)` 3, `(4, 4)` 5, `(5, 5)` 11, `(6, 6)` 17 |
| Statement 10.1, 4), 5); Statements 11.1 (table (21)), 11.2, 11.3 | `evenhom.py` (about 3 min) | Part A: `dim H^1(T_1)^s = 0`, `dim H^1(T_1)^s' = 1`, invariant class `['1', '0', '3', '0']`, `Q = 3, rank = 2`; Part B: the rows of table (21); Part C: `dim H^1(T_3)^s' = 1`, orbit dim 7, the other copies 0; Part D: `(4, 1)` `(17, 14)` even-homogeneous `True`, homogeneous `False`; `(4, 3)` `(13, 16)`, `(4, 4)` `(15, 17)`, `(5, 4)` `(17, 18)`, `(5, 5)` `(15, 19)`, homogeneous `True`; Part E: `(3, 0)`, `(3, 1)`, `(3, 2)` even-homogeneous `False` |
| Proposition 12.1, formulas (22), (23) | `brack.py` | of the ten brackets only `[v0, v2]` and `[v1, v3]` are non-zero, both `1 x^(-2) xi_12 d/dx`, class `[0, 1, 0]` |
| Statement 13.1, table (24) and items 2), 3) | `run3.py` (about 20 min) | Part A: the rows of table (24); Part B: one non-zero bracket for (2,2,2), four for (2,2,3), seven for (2,2,4); Part C: `(2, 2, 0)`: split `(28, 28)`, families `(26, 27)`, `(24, 24)`, `(23, 22)`, `homogeneous False` |
| Statement 14.1, table (25), and the dimensions of H^0(T_q); Section 16: the numbers of [Ba2] (18, 32, 16k-16; 32, 54, 22k-12; 2k-1) | `runbash.py` (about 20 min) | rows `h^1` of `(1, 1, 1, 1)`: `0 0 0 10 8 1`, `h^0`: `8 19 8 0 0 0`; `(1, 1, 1, 1, 1)`: `0 0 0 40 50 20 2`; `(2, 2, 2, 0)`: `0 3 18 32 20 3`; `(3, 3, 2, 0)`: `0 5 32 54 32 5`; `(4, 4, 2, 0)`: `0 9 48 76 44 7` |
| Statements 15.1, 15.2, 15.3 | `run4.py` (about 30 min), `run4b.py` (about 50 min, the cutoff 12 being the slow part) | `run4.py`: H^1(T_3) = V (x) W, the image of aut(E) in gl(W) of dimension 16, orbit dimensions 4 and 7, sdim v = (35, 35), (30, 30), (27, 27); `run4b.py`: the counts identical at the cutoffs 8, 10, 12; rank 2: even part onto, odd part drops everywhere; rank 1: both parts drop at one point |
| Section 16, first paragraph ([Ba1]: the SL(2)-action on H^1(T_2), H^1(T_4) of CP^{1\|4} is trivial) | `t9.py` | `q=2: dim H^1 = 10, dim H^1^s = 10 -> action trivial: True`, `q=4: ... True` |
| Statement 16.1, formula (29) | `t2.py` (about 5 min) | `q=3  printed list: 10 cocycles -> (True, False, 10, 50)`, `q=3  + the omitted v-type: 50 -> (True, True, 50, 50)` |
| Statement 16.2, formula (30) | `t5.py`, `t7.py`, `t8.py`, `t10.py` | `t5.py`: `dim H^1(T_2)^{s'} for case 5 = 1`, `[f',z] reduces to zero mod B^1: True`; `t7.py`: `engine finds 6 ; the paper lists 5`; `t8.py`: `candidate is proportional to the engine's invariant: True`; `t10.py`: `kernel of ad_{e'} on span(v_1..v_6): 1 dimensional` |
| Statement 17.3, table (32), formulas (33), (34); Statement 17.4, formula (35) | `vaintrob.py` | `part A: 0 discrepancies with the computation` (sdim (11\|8), (60\|52), (231\|216), (756\|732)); `part B: 0 discrepancies`; part C: `12 discrepancies with d as printed, 0 with the corrected d`; `TOTAL discrepancies with the honest Cech computation: 0` |
| Theorem 21.1, table (47); Statement 22.2; Propositions 22.4, 22.6, 24.1, 24.2, 25.3 | `contact.py` | `[ok ]` throughout, e.g. `k(1\|3) (6*g - 3 \| 6*g + s_o - 6)`, `k(1\|4) (9*g - 2 \| 8*g + 4*s_o - 8)`, `m(1), g >= 2 (4*g - 3 \| 4*g - 4)`, genus 1 `(2^{n-1} \| 2^{n-1})` and `(2^{n-1} \| 0)`; `0 checks failed` |
| Section 23: Theorem 23.6 and its Corollaries, Propositions 23.11, 23.14, 23.15, 23.19, 23.21, Statement 23.17 (the tails) | `riggings.py` | Part D: odd tail `(0 \| 3)` at g=2, d=3, n=6, `(0 \| 21)` at d=9, `(0 \| 15)` at g=3, d=9; even tail `(3 \| 24)` at g=2, d=4, n=7; `0 checks failed` |
| Proposition 27.3, formula (59); Statement 27.5, formulas (64), (65); table (66) | `runschw.py` (about 25 min) | `[PASS]` throughout; `156 checks, 0 failed` |
| The engine | `test_ns.py` | `45 passed, 0 failed` |

## Files

Engines (imported, not run on their own):

- `ns.py` — Grassmann-valued Laurent polynomials, the supermanifolds (CP^1, O) of dimension 1|m from
  their transition functions (split, from a cocycle, families over Lambda(tau)), the Lie superalgebra
  v of global vector fields and the evaluation v --> T_p; `test_ns.py` is its self-test.
- `bashkin.py` — Cech H^0, H^1 of T_q of a split (CP^1, Lambda E) from the transition functions.
- `sl2inv.py`, `quot.py` — H^1(T_q) as an explicit quotient with the induced action of vector
  fields, invariants of copies of sl(2).
- `oddpar.py` — the action of g = aut(E) + sl(2) on H^1(T_q) for every q, weights, orbit dimensions.
- `famtools.py` — from a class in H^1(T_q) to the family exp(tau z) over Lambda(tau).
- `orbits.py` — orbits of aut(E) and of g on P(H^1(T_1)).
- `brack.py` — brackets of odd classes (also run directly: Proposition 12.1).
- `srs.py` — Riemann–Roch on a curve of genus g, symbolically in g; the sheaves of the structured
  superstrings. `orth.py` — the same for orthogonal (non-split) riggings.
- `schw.py`, `stringy.py` — the flat model of k^L(1|n), weight-0 cocycles, invariant operators, the
  Schwarzian sheaf.

Scripts run directly (see the table): `run.py`, `runcd.py`, `extra.py`, `runorb.py`, `odd.py`,
`t_mod.py`, `runscan.py`, `run33.py`, `runjoint.py`, `evenhom.py`, `brack.py`, `run3.py`,
`runbash.py`, `run4.py`, `run4b.py`, `t2.py`, `t5.py`, `t7.py`, `t8.py`, `t9.py`, `t10.py`,
`vaintrob.py`, `contact.py`, `riggings.py`, `runschw.py`, `test_ns.py`.
