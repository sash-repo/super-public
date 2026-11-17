# cartprol

**Cartan prolongs of small irreducible modules over exceptional modular Lie (super)algebras**

A. Krutov, D. Leites, O. Lozhechnyk. Status: in preparation.

The scripts compute: the Cartan prolongs (M, g)_* of the irreducible modules M of least dimension
(or the adjoint module) over the exceptional modular Lie (super)algebras g with indecomposable Cartan
matrix (Statement 3.1, Table (1)); and, for Shen's algebra gs(2) in characteristic 2, its irreducible
modules, the absence of non-trivial irreducible modules of dimension 2..6, the Cartan prolongs of its
modules V_7, V_8, the adjoint and the 64-dimensional one, and the simple cores of these prolongs
(Section 4). Exact arithmetic throughout: F_p and F_2(alpha) for the modules over g(A) (`hwmod.py`,
`prol.py`, `ratf.py`), F_2 and GF(2^K) on numpy integer arrays for gs(2) (`gs2/`).

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r cartprol/requirements.txt
cd cartprol
../venv/bin/python runprol.py prol brj23 2,0 2
```

The scripts of the top folder are run from it; those of `gs2/` from inside `gs2/` (and
`gs2/sbt7/sbt7_pair.py` from inside `gs2/sbt7/`). In `gs2/`, run first, in this order:
`shenL.py` (writes `shenL.npz`: gs(2) and its restricted closure L), `toral.py` (writes
`outerD.npy`, `toral.npy`), `superrep.py` (writes `restricted_irreps.pkl`); then `runalg.py`
before `analyze.py`, and `v8.py` before `v8alg.py`, `v8alg.py` before `count.py` and `geo.py` on the
`alg_V8_*_core.json` it writes. These files are not part of the deposit. numba compiles `count.py`
on the first call. Times are those of one run on a 10-core laptop.

## Claims of the text and the scripts behind them

Numbers are those of the text. "Table (n)" is the display numbered (n).

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Statement 3.1, Table (1): brj(2;3), p = 3, ad (10\|8) | `runprol.py prol brj23 2,0 2` | `M=(10, 8)`, `g_1: 0\|0` without and with `(+Id, +torus)` |
| Statement 3.1, Table (1): f(4), p = 3 (25) and p > 3 (26) | `runprol.py prol f4p3 0,0,0,1 2`; `runprol.py prol f4p0 0,0,0,1 2` (the second in characteristic 1000003, read as characteristic 0) | `M=(25, 0)` resp. `M=(26, 0)`, `g_1: 0\|0` in both runs |
| Statement 3.1, Table (1): ab(3), p = 3, (0,0,0,1) of sdim (21\|16) | `runprol.py prol ab3p3 0,0,0,1 2` | `M=(21, 16) g0=(24, 16)`, `g_1: 0\|0` twice |
| The remark after Table (1): ab(3) over C, ad (24\|16) | `runprol.py prol ab3 0,0,0,1 2` (characteristic 1000003, read as 0) | `M=(24, 16) g0=(24, 16)`, `g_1: 0\|0` twice |
| Statement 3.1, Table (1): el(5;5), p = 5, ad (55\|32) | `runprol.py prol el55 1,0,0,0,0 2` | `M=(55, 32)`, `g_1: 0\|0` twice |
| Statement 3.1, Table (1): el(5;3), (18\|16) | `runprol.py prol el53 0,0,0,0,1 2` | `M=(18, 16)`, `g_1: 0\|0` twice |
| Statement 3.1, Table (1): g(1,6), (13\|6) | `runprol.py prol g16 1,0,0 2` | `M=(13, 6)`, `g_1: 0\|0` twice |
| Statement 3.1, Table (1): g(2;3), (6\|6) | `runprol.py prol g23 1,0,0 2`; `runprol.py prol g23 0,1,0 2` | `M=(6, 6)`, `g_1: 0\|0` twice in each |
| Statement 3.1, Table (1): g(3;3), (8\|8) | `runprol.py prol g33 1,0,0,0 2` | `g_1: 0\|0`; with `(+Id, +torus)`: `g_1: 8\|8`, `g_2: 0\|0` |
| Statement 3.1, Table (1): g(2;6), (15\|6) | `runprol.py prol g26 1,0,0,0,0 2` | `g_1: 0\|0`; with `(+Id, +torus)`: `g_1: 15\|6`, `g_2: 0\|0` |
| Statement 3.1, Table (1): g(4;3), (12\|14) | `runprol.py prol g43 1,0,0,0 2` | `M=(12, 14)`, `g_1: 0\|0` twice |
| Statement 3.1, Table (1): g(3;6), ad (36\|40) | `runprol.py prol g36 2,0,0,0 2` | `M=(36, 40)`, `g_1: 0\|0` twice |
| Statement 3.1, Table (1): g(8;3), ad (55\|50) | `runprol.py prol g83 0,0,0,0,1 2` (seconds) | `M=(55, 50)`, `g_1: 0\|0` twice |
| Statement 3.1, Table (1): g(4;6), (32\|12) | `runprol.py prol g46 1,0,0,0,0,0 2` | `M=(32, 12)`, `g_1: 0\|0` twice |
| Statement 3.1, Table (1): wk(3;alpha), bgl(3;alpha), the six modules of dim 8 | `wkprol.py 3` | for each of the six weights: `wk(3) ... M=(8, 0)`, `bgl(3) ... M=(4, 4)`, `g_1: 0\|0`; with `+Id,+torus`: `g_1: 8\|0` resp. `g_1: 4\|4`, then `g_2: 0\|0` |
| Statement 3.1, Table (1): wk(4;alpha), bgl(4;alpha), dim 32 | `wkprol.py 4` | `M=(32, 0)` resp. `M=(16, 16)`, `g_1: 0\|0` in all eight runs |
| Statement 3.1, Table (1): wk^(1)(3;alpha)/c, bgl^(1)(3;alpha)/c, weights (1,0), (alpha,0), (0,1) | `wkq.py` | `M (14, 0)` resp. `M (6, 8)`/`(8, 6)`, `g_1: 0\|0` for all six |
| Remark after Statement 3.1: ag(2) over C, ad (17\|14) | `runprol.py prol ag2 4,0,0 2` | `M=(17, 14)`, `g_1: 0\|0` twice |
| Subsec. 4.1: gs(2) and its restricted closure L, dim 17 | `gs2/shenL.py` | the 17 basis elements of L with their weights and squares |
| Subsec. 4.1: the Chevalley generators generate 11 of 14 dims, for each of the 12 Borels | `gs2/chevgen.py` | twelve lines, each ending `whole algebra by the 4: 11 of 14` |
| Table (2), Proposition 4.3, 1): the irreducible highest-weight modules | `gs2/hwscan.py` (10 minutes, 10 processes) | `Counter({(64, (32, 32), ...): 211, (16, (8, 8), ...): 42, (1, (1, 0), ...): 1, (7, (4, 3), ...): 1, (14, (8, 6), ...): 1})` |
| Proposition 4.3, 1): the invariant form on V_7 is symmetric and not alternating | `gs2/props.py` | `L(1, 0) dim 7: ... gs(2)-invariant forms 1 ['sym rank 7']` |
| Proposition 4.3, 2): the induced module of dim 128 is irreducible | `gs2/chineg.py` (3 minutes) | `phi (1, 10) ... dim Z 128 sing 1 irr quotient 128` |
| Proposition 4.3, 2): its superdimension is (64\|64) for s_a and s_b | `gs2/supergen.py` | `dim Z 128  singular 1  sdim wrt D: (64, 64)`, twice for each of s_a, s_b |
| Proposition 4.3, 2): the modules V_8, defined over F_2 | `gs2/v8.py` | five lines `... dim 8 singular 1 rank rho(gs2) 14 ...` |
| Statement 4.4: no non-trivial irreducible module of dim 2..6 | `gs2/gen_eqs.py 1 1 1 1`, `2 1 1 1`, `3 1 1 1`, `0 2 2 2`, and `1 1 1 1 --generic` (the case mu not in F_2^2); then `gs2/gb.py eqs_1111.m eqs_1111g.m eqs_2111.m` (2 minutes); for `eqs_3111.m` and `eqs_0222.m`, `wolframscript -file gb.wl eqs_3111.m` (Mathematica; sympy does not finish them in 5 minutes); control: `gs2/calib_v7.py` | `GB = {1}? True` for all five; control: `V_7: 709 equations, 0 violated`, `perturbations detected: 10 of 10` |
| Table (3): the prolongs of V_7 and Pi(V_7) over gs(2), c(gs(2)), L, c(L), s_a, s_b, c(s_a), c(s_b) | `gs2/runprolong.py 7` | e.g. `g0 = gs(2)  g_{-1}=V: [14, 10, 1, 0]`, `g0 = L  g_{-1}=V: [17, 31, 49, 71, 97, 127, 161, 199]`, `g0 = s_b  g_{-1}=V (sdim (3\|4)): [17, 10, 1, 0]   Pi(V): [17, 17, 16, 16, 16, 16, 16, 16]`. The script's `V (sdim (3\|4))` is the text's Pi(V_7) and its `Pi(V)` the text's V_7 |
| Table (3): superdimensions of the finite super-prolongs | `gs2/sdimprol.py` | `s_a V7: ... [(3, 4), (8, 8), (6, 4), (1, 0)]  total (18, 16)` and so on |
| Subsec. 4.5: the adjoint module: prolong fg_{-1} + fg_0 + KD | `gs2/runprolong.py 7` (block `ad14`) | `g_{-1}=V: [14, 1, 0]` etc. for all g_0 |
| Subsec. 4.5: the 64-dimensional restricted module: trivial prolong | `gs2/prolong_big.py` (2.5 minutes) | four lines `dim g_1: V 0, Pi(V) 0` |
| Subsec. 4.5: the prolongs of V_8: 14 (15), 8, 1 | `gs2/v8.py` | `prolong V: [14, 8, 1, 0] Pi(V): [14, 8, 1, 0]  c: V [15, 8, 1, 0] ...` for the five V_8 |
| Theorem 4.6, 1)-3): cores of dim 28, (14\|14), (14\|16), (15\|16) | `gs2/runalg.py` (2 minutes), then `gs2/analyze.py`, `gs2/supercore.py`, `gs2/superseries.py` | `analyze.py`: `core dim 28  assoc(ad core) 784 = 28^2? True` for all six; `superseries.py`: `core sdim (14\|14)`, `(14\|16)`, `(15\|16)` |
| Proof of Theorem 4.6, 1): J_0 of sb~(7;1) is gs(2), 2-closure of dim 17 | `gs2/sbt7/sbt7_pair.py` (from inside `gs2/sbt7/`; rewrites `gs2/sbt7_J0.json`), then `gs2/isoJ0w.py` | `dim J_0 = 14  dim J_{-1} = 7  2-closure ... 17`; `graded isomorphisms gs(2) -> J_0 found: 6` |
| Theorem 4.6, 4): the derived algebra of (V_8, gs(2))_* is 30-dimensional and absolutely simple | `gs2/v8alg.py` (1 minute) | ten lines `... core 30 abs simple True` |
| Theorem 4.6, 4), Conjecture 4.7: 42752 and 12545; and 27584, 12545 for the 28-dimensional core | `gs2/count.py alg_V8_1_16_core.json alg_V8_1_24_core.json alg_V8_3_16_core.json alg_V8_4_1_core.json alg_V8_4_9_core.json alg_V7_gs2_core.json` | `dim 30 ... 2-nilp 42752  toral 12545` five times; `dim 28 ... 2-nilp 27584  toral 12545` |
| Theorem 4.6, 5), evidence for Conjecture 4.7: dim Der 36, 37, 38; 2-envelope 34; generic rank 28; minimal polynomial of degree 15 | `gs2/geo.py alg_V8_*_core.json lelo30/k30_42752_12545.json lelo30/k30_46080_28673.json lelo30/k30_67328_37121.json` | `Der 36 2-env 34 ... generic rk ad 28 ... minpoly deg 15` for the five and for `k30_42752_12545`; `Der 37` and `Der 38 2-env 31 ... deg 8` for the other two |

Not rerun for this deposit (longer than 30 minutes): `gs2/scan.py` (about 45 minutes, 10 processes;
writes `scan.json`) and then `gs2/scanverify.py` (about 16 minutes), which list the irreducible
quotients for chi(N+) = 0 != chi(N-) over all 12 Borels; their output, `irreducible (singular 1):
[(8, 5), (16, 14), (64, 238), (128, 100)]`, is behind "Irreducible modules of dimension 8, 16 and 64
occur as well" in Proposition 4.3, 2). `v8.py` imports the chamber list from `scanverify.py`
without `scan.json`.

Note on Table (1): the ab(3)-module L^(0,0,0,1) is the adjoint module (24|16) over C, and (21|16) for p = 3; the prolong is trivial in both cases.

The comparison with the list of Cushing, Stagg and Stewart ("No 30-dimensional algebra of the list of
[CSS] has these numbers", Theorem 4.6, 4)) and the three 30-dimensional algebras of [LeLo] are those
of the deposit `../cssei/` (`rand/css_inv.json`, `rand/ident.py`); the tables `gs2/lelo30/*.json` are
those three algebras (`../cssei/new11/export.py`).

## Files

- `hwmod.py`: irreducible highest-weight modules L^lambda over g(A) from the Cartan matrix, the
  parity vector and the weight (quotient of the Verma module by the radical); `prol.py`: the image of
  g(A) in gl(M) and the Cartan prolong, block by block in the Chevalley multidegree; `ratf.py`: the
  field F_2(alpha); drivers `runprol.py`, `wkprol.py`, `wkq.py`.
- `gs2/`: gs(2) in Brown's monomial basis and its restricted closure L (`shenL.py`); GF(2^K)
  arithmetic (`gfla.py`, `lalg.py`); induced modules and irreducible quotients (`induce.py`,
  `quot.py`, `hw.py`); the modules (`superrep.py`, `hwscan.py`, `props.py`, `chineg.py`,
  `supergen.py`, `scan.py`, `scanverify.py`, `v8.py`, `toral.py`, `chevgen.py`); modules of
  dimension <= 6 (`gen_eqs.py`, `gb.py`, `gb.wl`, `calib_v7.py`); prolongs (`cprolong.py`,
  `runprolong.py`, `sdimprol.py`, `prolong_big.py`), structure constants and cores (`buildalg.py`,
  `runalg.py`, `analyze.py`, `supercore.py`, `superseries.py`, `v8alg.py`), invariants (`count.py`
  with `tab2txt.py`, `geo.py`), the isomorphism with J_0 (`isoJ0w.py`, `sbt7_J0.json`); F_2 linear
  algebra `f2lin.py`, `f2vf.py`.
- `gs2/sbt7/`: the core of sb~(7;1) from the vector fields of Bouarroudj-Grozman-Lebedev-Leites-
  Shchepochkina (arXiv:1510.07255) (`sbt7_pair.py` on `der.py`, `prolong.py`, `run_row.py`,
  `rowtools.py`, `rows.py`, `blattner.py`, `absutil.py`, `f2vf.py`, copies of those of `../cssei/`).
- `gs2/lelo30/`: the three 30-dimensional simple Lie algebras of [LeLo] as structure tables.
