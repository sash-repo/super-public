# svodd

**Simple singularities of functions on supermanifolds with odd parameters**

D. Leites, O. Lozhechnyk. Status: in preparation.

The scripts classify the germs `f + theta g` over a base with one odd parameter `theta`, for the
normal forms `f` of the Serganova-Vaintrob tables of simple singularities of functions on
supermanifolds (Math. Scand. 64 (1989) 251-284). For each normal form they compute the local
superalgebra, the layer `Nt_f = m_odd / (m I_f)_odd`, the stabilizer algebra `s_f` and its action
on the layer (T_1, `stab.py`), the orbit separator (T_2, `orb.py`), and the certified enumeration
of the orbits (T_3, `t3.py`, with the affine certificate `affine.py` and the parametric
certificate `param.py` over the field `ratfun.py` of rational functions in the parameters of a
family). A second group of scripts audits Table 1.5 ("Regular odd elements; the dimensions of
their centralizers") of Ch. 1 of *Seminar on Supersymmetries*, vol. 2. Exact arithmetic over Q
or Q(mu) throughout; `../lib/exactlin.py` is fast when python-flint is installed and correct
without it.

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r svodd/requirements.txt
cd svodd
../venv/bin/python run_tables.py
```

Every script is run from inside this folder; the ones that need `../lib/exactlin.py` find it by
themselves. A type is named as in `probe.py`: `III_1 n m`, `III_2 n 1`, `III_3 3 1 9` (the last
number, optional, is the truncation degree). `run_b2.py one TYPE n m` runs T_3 on one type and
separates its representatives pairwise; `--nosep` skips the separation, which is the slow part.
The times in brackets were measured on a machine running many other jobs at once; an idle laptop
is about ten times faster.

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Subsec. 2.6.3, the table of `sdim O_F` (free rank, defect) | `found.py` [1 s] | part (B): the twelve rows of the table, e.g. `I_1 + theta.xi1  (0\|4)  7\|5  11\|11  4\|6  NOT free`; `SUMMARY: all checks passed` |
| Subsec. 2.7.3, the table of `dim N_f` and `dim Nt_f`; Examples 2.7.5 and 2.7.6 (the monomial bases) | `found.py`; the layer of every normal form of the tables: `run_tables.py` [3 s] | part (A): one row per type ending in `OK` (e.g. `I_2 ... 17\|20  20  26  6  6`), part (C): the bases; `run_tables.py` reproduces the Milnor numbers of A_n, D_n, E_6, E_7, E_8 (`OK`) and lists `sdim O_f` and the layer of every normal form |
| Theorem 3.2.3 (the two bounds coincide), Subsec. 3.4 (the stabilizer census), Theorems 4.3.1 and 4.3.3 (the modulus `a_1/a_3` over `VII_2`; the table of the types `VII`), Proposition 4.4.1 (`II(n,1)`) | `run_stab.py` [25 min] | the census rows with `dim S` = 31, 141, 191, 7, 7, 11, 62, 95, 99, each CERTIFIED; the `VII` rows with `dim S` = 51, 99, 163, 99, 99, 163 and modality 1; `SUMMARY: all checks passed` |
| Theorems 4.1.2 and 4.2.1, Corollaries 4.2.2 and 4.2.3 (corank 1\|2; proved in the text, checked here) | `run_orb.py` [12 min] | for every `II(n,m)`: `mu+1` classes, orbit dimensions `2(mu-j)`, `ok`; `SUMMARY: all checks passed` |
| Statement 3.7.6 (`T_3` on the types `II(n,m)`, 3 <= n <= 6) | `run_b2.py 0` [4 min] | for each of the 15 types `T_3: mu+1 classes CERTIFIED` with representatives `x^j xi_1` and `0`, all pairs separated by a T_2 certificate, `ok`; `SUMMARY: all checks passed` |
| Subsec. 3.9.1 and Lemma 3.9.3 (the degree filtration fails for `III_1(5,4)` and `III_1(6,5)`; the quasi-homogeneous weights repair it) | `v_filt.py` [6 min] | `III_1(5,4)`: `naive degree filtration: 5 offending matrix entries (4 distinct positions)`, the first of them `x^3xi_3 (degree 4) --> xi_1xi_2xi_3 (degree 3)`, `w(x) = 2, w(xi) = [4, 4, 1, 1]`, `weighted filtration: 0 offending entries`; `III_1(6,5)`: 10 entries, `[5, 5, 1, 1]`, 0 entries |
| Subsec. 3.11.2 (the volume pairing spans the invariants of degree 2 on `F^2` for `III_3`, `III_4`) | `inv.py vol` [30 min] | for each type `on F^2 ...: 1 invariant(s) of weight zero`, the invariant `-[xxi_4]*[xi_1xi_2xi_3] + [xxi_3]*[xi_1xi_2xi_4] - [xxi_2]*[xi_1xi_3xi_4] + [xxi_1]*[xi_2xi_3xi_4]`, `values at t = 0,1,2,3,5,-1: 1, 2, 3, 4, 6, 0` on the family `xxi_1 + (1+t).xi_2xi_3xi_4`; `all moduli explained by an explicit invariant` |
| Statement 3.11.4 (no semi-invariant of degree 2, 3, 4 on any `F^e` for `III_1(3,2)`) | `v_noabs.py` [18 min] | nine lines, each `0 invariant(s) of weight zero, 0 semi-invariant(s) of nonzero weight` |
| Theorem 3.15.2 and Subsec. 4.13.1 (the rank criterion for the exceptional members) | `sym.py B III_1 4 3` [1 min] | `mu = -1:  answer (1, (1,)) JUMP \| all ranks differ \| T_0 () COLLAPSES`, `criterion agrees: 1, fails: 0` |
| Theorem 4.5.2 (corank 1\|6: the degree-3 piece of dimension 14, the image of dimension 21, the unique invariant form, the generic orbit of codimension 1) | `v_freud2.py` [29 min] | for `V(3)` and `V(4)`: `dim of the image on the piece = 21`, `invariant bilinear forms: 1`, `one form: anti-symmetric, rank 14 of 14 (nondegenerate)`, `generic orbit dimension on the piece = 13  => codimension 1` |
| Statement 4.6.2 (the linear shortcut loses one class in `III_1(3,2)` and in `III_1(4,3)`) | `v_aff.py III_1,3,2` [under 30 min]; `v_aff.py III_1,4,3` [over 30 min]; the separation: `run_b2.py one III_1 3 2` [8 min], `run_b2.py one III_1 4 3` [28 min] | `shortcut: 9 classes, 2 families` against `affine  : 10 classes, 2 families`, and `++ MISSED BY THE SHORTCUT: xxi_1 + 1/3.xi_1xi_3xi_4 (orbit dim 2)`; for `III_1(4,3)`: `shortcut: 14 classes, 4 families` against `affine  : 15 classes, 4 families` (the counts of the generic member of the family; the Statement adds the exceptional member `mu = -1`, one more class in both lists, see Table 1); `separation of the 10 representatives: 45 pairs by a T_2 certificate`, `separation of the 15 representatives: 105 pairs by a T_2 certificate`, `0 unresolved` |
| Table 1 (corank 1\|4), column *generic* | `run_b2.py one TYPE n m --nosep` for the 18 types of the table [at most 7 min each]; without `--nosep` the pairs are separated as well (for `III_1(6,5)` and `III_2(6)` this takes about 20 min on an idle laptop) | `T_3: 3 classes` for `III_1(n,1)`; `10 classes, 2 of them a one-parameter family` for `III_1(n,2)`; `15 ..., 4` for `III_1(n,3)`; `21 ..., 6` for `III_1(5,4)`, `III_1(6,4)`; `27 ..., 8` for `III_1(6,5)`; `10`, `16`, `22`, `28` for `III_2(3)`-`III_2(6)`; `6 ..., 1`, `7 ..., 1`, `7` for `III_3`, `III_4`, `III_5`; every one `CERTIFIED` |
| Table 1, column *Exc.* | `v_par.py III_1,3,2 III_1,4,2 III_1,4,3 III_1,5,3 III_1,5,4 III_1,6,3 III_1,6,4 III_1,6,5 III_3,3,1 III_4,3,1` [21 min] | `SPECIAL mu = -1: DIFFERS  (1, (1,))` and `1 special members genuinely differ` exactly for `III_1(4,3)`, `III_1(5,3)`, `III_1(6,3)`, `III_1(6,4)`; `0 special members` for the others; `SUMMARY: 0 disagreements between the generic answer and an ordinary member` |
| Table 2 (corank 1\|5 after the affine certificate), rows `IV_1(3,2)` and `IV_2(3,1)` | `v_cmp.py IV_1,3,2` [12 min]; `v_cmp.py IV_2,3,1` [4 min] | `IV_1(3,2) WITHOUT the parametric certificate: 40 entries, NOT certified`; `IV_2(3,1) WITHOUT ...: 14 entries` |
| Subsec. 4.10.1, Subsec. 4.10.2 and Table 3 (the 52 classes of `IV_1(3,2)`, 26 of them families; the ten parametric nodes and their special members) | `run_b2.py one IV_1 3 2 --nosep` [8 min]; `v_par.py IV_1,3,2` [10 min] | `T_3: 52 classes, 26 of them a one-parameter family   CERTIFIED`; `v_par.py`: ten nodes with fibers of dimension 1, 3, ..., 3; generic answers (classes, family degrees) `(1,[0])` for nodes 0-2, 6, 7, `(1,[1])` for node 3, `(3,[0,0,1])` for nodes 4, 5, 8, 9; `SPECIAL mu = 0: DIFFERS (1, (1,))` at node 0, `SPECIAL mu = -1: DIFFERS (3, (0, 0, 1))` at nodes 1, 2, `(2, (0, 0))` at nodes 6, 7, no rational special member at nodes 4, 5, 8, 9; `5 special members genuinely differ`, `0 disagreements` |
| Table 4 (corank 1\|5 after the parametric certificate), rows `IV_1(3,2)`, `IV_2(3,1)`, `IV_2(3,2)` | `run_b2.py one IV_1 3 2 --nosep`; `run_b2.py one IV_2 3 1 --nosep` [3 min]; `run_b2.py one IV_2 3 2 --nosep` [16 min] | `52 classes ... CERTIFIED` (10 parametric nodes); `12 classes, 2 of them a one-parameter family   NOT CERTIFIED` (1 parametric node; the open node at degree 3); `24 classes, 10 of them a one-parameter family   NOT CERTIFIED` (9 parametric nodes) |
| Tables 5 and 6, Statement 5.3.4 (every diagram drawn is realized by a simple system, and every entry is its `sdim g_0`) | `corrected.py` [26 min]; `exc.py` (rows 17-19) [5 min] | `corrected.py`: `row k): ok` for rows 1-16, `0 row(s) not OK`; `exc.py`: the characteristics of `ag(2)`, `ab(3)`, `osp_a(2\|4)` with `sdim g_0 = (5, 0)`, `(6, 4)`, `(3, 0)` |
| Subsec. 5.3.3 (the rows of Table 1.5 of the book that are wrong as printed; the printed diagram of row 13 is realized by no simple system) | `drawn.py` [25 min]; `exc.py` | `SUMMARY by row`: rows 1, 7-10, 14-16 `ok`; row 2 `unrealizable`; row 3 `noshape, ok`; rows 4 `noshape, wrong`, 5, 6, 11, 12 `wrong`; row 13 `noshape`; `exc.py`, row 18: `total matches, splitting does not` |
| Statement 5.3.6 (the centralizer of a regular odd element in five series) | `centralizers.py` [12 s] | 26 lines ending in `OK` (sl(m\|n), m <= 5; osp(2\|2n), n <= 5; osp(1\|2n), n <= 4; osp(2m+1\|2m), osp(2m\|2m), m <= 3); `SUMMARY: all agree` |

The other rows of Tables 2 and 4 come from the same commands with the other types
(`v_cmp.py IV_1,4,2`, `run_b2.py one IV_4 3 2 --nosep`, `run_b2.py one IV_5 3 1 --nosep`, ...).
Each of these runs longer than half an hour on the loaded machine; with the pairwise separation,
`run_b2.py one IV_1 4 2` and `run_b2.py one IV_1 5 2` take about 3 and 5 hours on an idle laptop.

## Files

- `sv.py` — the ring `k[x] (x) Lambda(xi)` truncated in the x-degree; gradient ideal, local
  superalgebra, layer.
- `stab.py` — T_1: the stabilizer algebra `s_f`, its action on the layer, orbit dimensions,
  modality bounds.
- `orb.py` — T_2: the orbit separator, with certificates for "same orbit" and "different orbits".
- `t3.py` — T_3: the certified enumeration of the classes; `affine.py` the affine certificate;
  `param.py`, `ratfun.py` the parametric certificate over Q(mu); `probe.py` the normal forms
  (`TYPES`) and the graded anatomy of a type; `inv.py` explicit invariants.
- `found.py`, `run_tables.py`, `run_stab.py`, `run_orb.py`, `run_b2.py` — the drivers of the
  table above; `v_aff.py`, `v_cmp.py`, `v_par.py`, `v_filt.py`, `v_noabs.py`, `v_freud2.py`,
  `sym.py` — the checks of single claims; `test_ratfun.py` — self-tests of `ratfun.py`
  (`all checks pass`).
- `chars.py` (root systems, simple systems, `sdim g_0` of a marked diagram), `run_chars.py` (the
  sweep over every simple system; run alone, it checks every printed number of Table 1.5 and
  takes over half an hour), `corrected.py`, `drawn.py`, `exc.py` — Table 1.5;
  `regodd.py` (matrix realizations, centralizers of odd elements) and `centralizers.py`.
