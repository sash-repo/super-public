# freeslie

**Free Lie superalgebras in characteristics 2 and 3, and reflections of roots in characteristic p**

D. Leites, O. Lozhechnyk. Status: in preparation.

The scripts check the computations of the text: the numbers B_kj of the reflections of roots,
read off the root systems of nineteen Lie (super)algebras with Cartan matrix and compared with
Lebedev's formulas (arXiv:2311.02710) and with formula (7.4) of Bouarroudj-Grozman-Leites; the
free Lie superalgebras for p = 2 and p = 3, built from the identities and compared with the
tensor models, and Witt's formulas; the Grassmann envelope at p = 3 and the Even Rules at
p = 2 (the censuses of small superalgebras over F_2 with and without a squaring); and the
non-classical restricted structures (the algebra o^(1)(3), the series o^(1)(2n+1), the
additivity rule for the map [4]). Exact arithmetic throughout (integers, fractions, finite
fields, and `../lib/exactlin.py`, fast when python-flint is installed and correct without it).

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r freeslie/requirements.txt
cd freeslie
../venv/bin/python refl.py
```

Every script is run from inside this folder; it finds `../lib/exactlin.py` by itself. No
arguments are needed except where stated. Every run below finishes in under five minutes on a
laptop, except the full run of `testalg.py` (about 55 minutes, almost all of it in parts A'
and D; parts A, B, C, E take a few seconds). `ortho24.py` and `testalg.py` also write their output to `ortho24.log`, resp.
`testalg.log`.

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Statement 2.3.1 (B_kj on 154 ordered pairs of nodes of nineteen algebras: the recursion and formula (3) of Lebedev agree in all cases; formula (7.4) of [BGL] fails in four, the table) | `refl.py` | `pairs (k,j) tested: 154`, `Lebedev's recursion (Statement 2.1) disagrees with the root system: 0`, `Lebedev's closed formula (3) disagrees with the root system: 0`, `[BGL] (7.4) = [BGLL] (44) disagrees with the root system: 4`; the four rows marked `<== BGL (7.4) WRONG` are brj(2;5) (2,1) `4 ... 2`, brj(2;3) (2,1) `4 ... 2`, g(1,6) (2,1) and (2,3) `2 ... 1` |
| Statement 2.4.1 (61 nodes: rho_k(B) is a base, rho_k^2 = id; r_k(Delta) = Delta at all 21 nodes with A_kk != 0, fails at 27 of the 40 with A_kk = 0, the 13 exceptions listed) | `refl.py` | `nodes tested: 61`, `rule (2) fails to produce a system of simple roots: 0`, `the linear extension r_k does NOT preserve Delta: 27`, `non-isotropic nodes (A_kk != 0): 21, of them r_k(Delta) = Delta: 21`, `isotropic nodes (A_kk = 0): 40, of them r_k(Delta) = Delta: 13` (the 13 listed), `rho_k o rho_k = id on bases: 61 of 61 nodes` |
| Subsec. 2.5 (the Cartan matrix of bgl(4;alpha) printed in [BLLoS] 1.3.3: four of the twelve B_kj given by every formula differ from the root system; with the corrected matrix all agree) | `refl.py` | the block `bgl(4;a) [matrix as printed in BLLS 1.3.3]` has exactly four rows marked `<== BGL (7.4) WRONG <== Leb (3) WRONG <== recursion 2.1 WRONG`; the block `bgl(4;a)` has none |
| Statement 6.10 (3 alpha is never a root, alpha odd, in the sample of [BLLoS]; the characteristic-3 members listed) and the remark after it (all 20 entries of the sample of Section 2, characteristics 0, 2, 3, 5) | `cube3.py` | the column `3a a root?` is `NO` in every row; `entries: 20;  odd roots inspected: 246;  entries with 2a a root for some odd a: 8`; `The characteristic 3 members of the sample are: brj(2;3), el(5;3), g(1,6), g(2,3), g(4,3), g(8,3), br(3), br(2;a)` |
| Statement 6.5 (formula (15) for dim Free^-(V)_m: sdim V = (0\|1) in degrees <= 6, (0\|2) and (1\|1) in multidegrees <= (3,3)); Lemma 6.1 (the cubing map is additive, its values central); Theorem 5.2 and Corollary 5.3 (the p = 2 model: directness of the sum (10), closure, formula (11), PBW identity (13)) | `freelie.py` (`--quick`: smaller degree bound) | `formula  dim F_A(V)_m = dim L_m + dim (L_{m/3})_odd:  holds in all 6 multidegrees`, then `... in all 15 multidegrees` twice; `T(x+y) = T(x) + T(y) on the free object:  ok`, `[T(x), g] = 0:  ok`; for p = 2 every square sum `direct`, `PBW identity for U(F(V)) = T(V):  ok`, `p = 2 model verified: YES` |
| Subsec. 3.4 (Witt's formulas (3), (4), (5) and the PBW identity (6)) and Corollary 5.4 (dim Free(V)_m = W_s(m) at p = 2) | `wittsuper.py` | every line `ok`; `holds in all 2352 multidegrees tested (196 of them with a square term)`; `0 disagreements` for each of the six superdimensions; `all checks passed` |
| Proposition 7.2 and the example after it (the algebra A of sdim (1\|2) over F_3; its Grassmann envelope over Lambda(xi_1,xi_2,xi_3) has dimension 12 and is a Lie algebra, with [xi_1 y,[xi_2 y, xi_3 y]] = 2 xi_1 xi_2 xi_3 z; over Q super Jacobi fails at (y,y,y)) | `grass3.py` | `super Jacobi on all 27 triples: ok`, over Q `FAILS at [('y', 'y', 'y')]`, `dim G(A) = 12`, `ordinary Jacobi on all 1728 triples: ok`, `= 2*z@x1x2x3`, `VERDICT: all four checks pass` |
| Subsec. 7.3 (in U(A) over F_3 the element z = T(y) dies, y and h do not: the failure of condition (J)) | `nonclass.py` (part E) | `z lies in the ideal (i.e. dies in U(g)): True`, and `False` for y and h |
| Subsec. 9.8, items 1)-4) (o^(1)(3) is simple, the only simple 3-dimensional Lie algebra over F_2 and F_4: 512/120/28 and 262 144/8128/3024 brackets, one GL(3)-orbit each; no 2-structure; a (2,4)-structure) | `nonclass.py` (parts A, B) | `simple : True`, `x = X: (ad_x)^2 = ad_u with u = None` (also for Y), `q = 2: 512 brackets, 120 satisfy Jacobi, 28 are simple`, `q = 4: 262144 brackets, 8128 satisfy Jacobi, 3024 are simple`, `number of isomorphism classes: 1` twice |
| Lemma 9.4 (the additivity rule (21) for the map [4]) | `nonclass.py` (part C) | `verified on the generators x, y, z: True`, `the rule holds on o^(1)(3) for all 16 pairs (x,y) in g_- x g_-: True` |
| Proposition 9.7 (formula (22) for Jacobson's free restricted object; for sdim V = (1\|1), multidegree (4,0): dim Free(V) = 0, dim Free^[2](V) = 1) | `nonclass.py` (part D) | in every table the columns `S_2 dim` = `pred` and `2\|2 dim` = `pred`; row `(4, 0) \|      0 \|       0 \|    0 \|       1 \|    1` for (1\|1); `squares of L stay inside the 2\|2-model: True` |
| Proposition 9.10 (o^(1)(2n+1): grading, the square (24), no 2-structure, a (2,4)-structure with the rule (21), the centralizer of g in gl(2n+1) is 1-dimensional for n <= 3); the simplicity of o^(1)(3) and o^(1)(5) over F_2; Lemma 9.12 (the (2,-)-structure in the quotient of dimension 2N+2, N = 6) | `ortho24.py` (about 4 minutes) | for n = 1, 2, 3 over F_2 and n = 1, 2 over F_4: `YES` on every line, `dim of the centralizer of g in gl(2n+1): 1`, `x^2 (matrix square) outside g for` all nonzero elements of g_-, `additivity rule (Lemma 9.4) on all ... pairs of g_-: ... 0 fail`; `of the 7 nonzero elements, 0 generate a PROPER ideal ==> SIMPLE`, `of the 1023 ...`; `Jacobi identity in the quotient (N = 6, dim 14): YES`, `==>  b has no [4]: YES` |
| Corollary 8.5 and Proposition 8.4 (H^0 and H^1: q squarings over F_q, not isomorphic, one envelope of dimension 8); Lemma 8.7 (B has no squaring) and the algebra C (19); the remark after Proposition 8.2 (4 and 128 brackets, of them 1 and 6 with [x,x] != 0) | `evenrules.py` (about 2 minutes) | `squarings compatible with it: 2` (F_2), `4` (F_4); `parity-preserving isomorphisms H^0 -> H^1: none`; `envelope over Lambda(xi_1,xi_2,xi_3): dim 8, abelian: True`; for B `candidate squarings tried: 4; valid: 0` (F_2), `16; valid: 0` (F_4); `in 4 cases, differ in 0 ... 1 of them`, `in 128 cases, differ in 0 ... 6 of them` |
| Statement 8.8 (the census of superalgebras over F_2 without a squaring: none of dimension <= 2; in sdim (1\|2), 24 brackets satisfy the axioms and 7 admit no squaring) | `evenrules.py` | the table of part D: `(1\|2)        32             24             7`, `(2\|2) 4096 424 168`, `(1\|3) 4096 736 217`, and `0` in the last column for all formats of dimension <= 2 and for (2\|1), (3\|1), (0\|n) |
| Lemma 8.11 (D(v,w) = D(w,v) on every weak squaring in sdim (1\|2) over F_2 and F_4); Statement 8.12 (the weaker axiom first differs in sdim (1\|3), 21 of 736 objects, witness E); Subsec. 8.15 ([KLLS] Example 3.2.1; osp(1\|2) over F_2 and its derived algebra C, which has no squaring); Conjecture 8.13, its support in sdim (1\|2) and (2\|2) | `evenrules.py` | `0 violations`; in E5 the rows `(1\|2) 24 7 7`, `(2\|2) 424 168 168`, `(1\|3) 736 196 217`; `of the 736 objects of sdim (1\|3) over F_2, 21 carry a squaring in the sense of (11) but none in the sense of (14)`; the witness `<u even; x, y, z odd \| [u,z] = y, [x,z] = u>`; `ALL CHECKS PASSED` |
| Proposition 8.9 (a squaring exists if and only if every (ad_v)^2, v odd, is inner; checked on the census of Statement 8.8) | `sqcrit.py` (uses `evenrules.py`) | `F_2 (1, 2) objects 24 no squaring 7 mismatch 0`, `F_2 (2, 2) objects 424 no squaring 168 mismatch 0`, `F_2 (1, 3) objects 736 no squaring 217 mismatch 0`, `F_4 (1, 2) objects 24 no squaring 7 mismatch 0` |
| Proposition 8.21 (Lambda(3) suffices, Lambda(2) does not: the algebra of sdim (1\|3)) | `testalg.py A`; the sweep: `testalg.py "A'"` (slow, see above) | `[ok ] Lambda(2) sees nothing`, `[ok ] Lambda(3) sees the defect`, for p = 5 and p = 7; the sweep: `143 objects, of which 3 are Lie superalgebras`, `Lambda(2): agrees on 141, disagrees on 2`, `Lambda(3): agrees on 143, disagrees on 0`, `Lambda(4): agrees on 143, disagrees on 0` |
| Corollary 8.22 (at p = 3 the test algebras C do not detect T) | `testalg.py B` | every line `[ok ]`, `PASS` count with `FAIL 0` in the TOTAL |
| Proposition 8.23 (the quadratic map q_C; over K[xi]/(xi^3) it separates H^0 from H^1, over Lambda(1) it does not) and the sentence after it (B has no q_C over K[xi]/(xi^3)) | `testalg.py C` | `[ok ] over F_2[xi]/(xi^3) the enriched functor separates H^0, H^1`, `[ok ] over Lambda(1) it does not`, `[ok ] no q_C on g(C) for B either  -- 16 candidates tried` |
| Proposition 8.20 (the squaring as a lift along Gamma^2 -> Lambda^2; the lifts form a torsor over Hom(g_od^(2), z(g) cap g_ev)); Proposition 8.17, item 3) (definition 1) does not impose [a,a] = 0 for a even) | `testalg.py D` (slow, see above; most of it in sdim (2\|2) over F_4), `testalg.py E` | `#squarings = \|z\|^n on all 2, 17, 10, 256 brackets` for sdim (1\|1), (1\|2), (2\|1), (2\|2), over F_2 and over F_4; `[ok ] and the envelope is not a Lie algebra either` |

Running `testalg.py` with no arguments runs all parts and ends with `N checks, N pass, 0 fail`.

## Files

- `refl.py` — the root systems of the sample and the numbers B_kj; Section 2.
- `cube3.py` — is 3 alpha a root? Statement 6.10; imports the sample of `refl.py`.
- `freelie.py` — the free Lie superalgebras for p = 3 (variety model vs. tensor model) and
  p = 2 (L(V) with the squares); Sections 5 and 6.
- `wittsuper.py` — Witt's formulas and their super versions against the exact models;
  Subsec. 3.4 and Corollary 5.4.
- `grass3.py` — the Grassmann envelope at p = 3; Section 7.
- `evenrules.py` — the Even Rules at p = 2, the squarings and the censuses; Section 8.
- `sqcrit.py` — the criterion of Proposition 8.9 on the census.
- `testalg.py` — the test algebras C of the definition by points; Subsec. 8.20-8.23. Optional
  arguments `A`, `A'`, `B`, `C`, `D`, `E` select its parts.
- `nonclass.py` — o^(1)(3), the 3-dimensional simple Lie algebras over F_2 and F_4, the rule for
  [4], the three free objects at p = 2, and U(A) at p = 3; Subsec. 7.3 and Section 9.
- `ortho24.py` — the series o^(1)(2n+1) and the (2,-)-example; Section 9.
