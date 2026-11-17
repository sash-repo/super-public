# wk

**On maximal dimension of the irreps over simple Lie (super)algebras in characteristic 2**

D. Leites, O. Lozhechnyk. Status: in preparation.

The scripts check the computations of the text: for the series o^{(1)}_I(2n+1) at p = 2, which root
vectors square outside the algebra, the restricted closure L and its torus, the induced modules
M(lambda) (built by straightening in the enveloping algebra and checked to be representations),
their irreducibility by the n_+-singular vectors, the irreducibility criterion for n <= 3, the
superization oo^{(1)}_{IPi}(1|2n) by "method 2" and the superdimensions of the modules, the
reducible adjoint module of the superization, and the index of L (Sects. 2-6, 10); sl(3) and sl(5)
at p = 2 (Sect. 6); Shen's algebra gs(2), its restricted closure and the irreducible modules over it
and over its two superizations (Sect. 7, subfolder `gs2/`); the Hamiltonian algebras
h_I(2n+1; 1)' of the non-alternating symplectic form (Sect. 8); and Table (16): the dimension of
the minimal p-envelope L, its index and whether the generic stabilizer is a torus, for the
exceptional simple vectorial Lie algebras of characteristic 2, the eleven simple Lie algebras of
dimension at most 31 outside the list of Cushing-Stagg-Stewart, the Skryabin algebras and the
Bouarroudj superalgebra Bj(3; 1|2) for p = 3 (Sect. 9). Arithmetic is exact over the finite fields
F_{2^k} and F_3 throughout; "generic" elements are random points over F_{2^k} (k = 3, 4, 6, 16) or
F_{3^10}, and the minimum of the stabilizer dimension over a few such points is reported.

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r wk/requirements.txt
cd wk
../venv/bin/python run.py
```

Every script is run from inside this folder, those of `gs2/` from inside that subfolder (there,
`shenL.py` and then `toral.py` must be run first: they write `shenL.npz` and `outerD.npy`, which
the others read; `superrep.py` writes `restricted_irreps.pkl`, read by `verifyq.py`). Scripts of
this folder write a `.log` beside them as well. Running times on a laptop: `run.py` about 15
minutes, `crit3.py 3 3` about a minute and `crit3.py 3 4` about 15 minutes, `ham.py 2 4` 5 minutes, `gs2/hwscan.py` 8 minutes
(10 processes), `gs2/generic.py` 8 minutes, `kwind.py` on all the tables 2 minutes; everything
else under 2 minutes.

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Lemma 2.2.1: the torus t of o^{(1)}_I(2n+1) is a maximal torus of the restricted closure L (n = 1, 2, 3); Proposition 4.1.1: sdim s(g, gr) = (n(2n+1)\|2n) | `torus.py` | `=> t is a maximal torus of L: True` and `sdim s(g) = (3\|2)`, `(10\|4)`, `(21\|6)` for n = 1, 2, 3 |
| Lemma 3.1.1 and Theorem 3.2.1: dim g = n(2n+1), dim t = n, n of the n^2 negative root vectors are 4-elements, dim M(lambda) = 2^{n(n+1)} | `run.py` (part A) | the table rows `1 \| 3 \| 1 \| 1 \| 0 \| 1 \| 4 = 2^2` ... `5 \| 55 \| 5 \| 25 \| 20 \| 5 \| 1073741824 = 2^30`; `simple = True` for n = 1, 2 |
| Theorem 3.2.1, n = 1: Dolotkazin's dimensions 1, 3, 4 | `run.py` (part B) | `dimensions met: [1, 3, 4]` over F_4 and F_8 |
| Statement 3.2.2 (attainment, n = 1, 2, 3) and Statement 3.2.4 (the criterion and the counts (q-2), (q-2)(q-4), (q-2)(q-4)(q-6)) | `run.py` (part B), `crit.py` (n = 1, 2), `crit3.py 3 3` and `crit3.py 3 4` (n = 3) | `crit.py`: `irreducible for 2 of the 4 weights` ... `168 of the 256 weights, the criterion predicts 168; mismatches: none`; `crit3.py`: `512 weights tested: irreducible 48, predicted 48, mismatches none` (F_8), `4096 weights tested: irreducible 1680, predicted 1680, mismatches none` (F_16; 1680 = 14 * 12 * 10) |
| Proposition 4.1.1, n = 1: s(o^{(1)}_I(3)) is oo^{(1)}_{IPi}(1\|2), not oo^{(1)}_{II}(1\|2) | `run.py` (part D) | `s(o^{(1)}_I(3)) =~ oo^{(1)}_{IPi}(1\|2) : True`, `... oo^{(1)}_{II}(1\|2) : False` |
| Statement 4.2.2: superdimensions (1\|0), (2\|1), (2\|2) for n = 1, and (32\|32) for n = 2 | `run.py` (part C) | `lambda = 0 : ... irreducible quotient = (1\|0)`, `lambda = 1 : ... (2\|1)`, `lambda = 2 : ... (2\|2)`; maximal submodules `(1\|2)`, `(0\|1)`; `lambda = [2, 4] : M irreducible, sdim = (32\|32)` |
| Proposition 5.1.1 (g is a submodule of codimension 2n of the adjoint module of s(g, gr), not an ideal), n = 1, 2, 3 | `run.py` (part D) | `g is ad-invariant in s(g): True;  closed under the squaring: False` and `codimension 2`, `4`, `6` |
| Statement 6.1.1 (sl(3), sl(5) at p = 2) | `run.py` (part E) | `lambda = [1, 1] gives an irreducible M(lambda) of dimension 8`, `lambda = [1, 1, 1, 1] gives an irreducible M(lambda) of dimension 1024` |
| Subsec. 7.1: gs(2) of dim 14, the weights, L of dim 17, the three squares leaving gs(2) | `gs2/shenL.py` | `dim gs(2) = 14  dim L = 17` and `x^[2] = (u1)^2`, `(u2)^2`, `(u1u2)^2` for u1, u2, u1u2 |
| Subsec. 7.1, 2): tori of L of dim 3 = ind L; generic stabilizer = 2-dim torus + a 2-nilpotent line | `gs2/torus_index.py`, `gs2/stab.py` | `torus dim from generic x: 3`, `stabilizer dim 3`, `rank of {y^[2^k], k>=20}: 2  abelian: True`; `stab dim 3 p-closed True semisimple rank 2` |
| Subsec. 7.1, 1): dim N_- is 7 or 8 for every triangular decomposition | `gs2/generic.py` | `dim N- = 7` or `dim N- = 8` for the six Borels; for lambda generic the irreducible quotients are of dim 64 or 16 |
| Statement 7.1.1, 1): highest weight irreducibles of dim 1, 7, 14, 64 (restricted), 16 on three lines and 64 elsewhere | `gs2/hwscan.py`, `gs2/verifyq.py` | `Counter({(64, (32, 32), ...): 211, (16, (8, 8), ...): 42, (1, (1, 0), ...): 1, (7, (4, 3), ...): 1, (14, (8, 6), ...): 1})`; `verifyq.py`: every quotient has `singular in quotient: 1` |
| Statement 7.1.1, 2): with dim N_- = 7, chi(N_+) = 0 and chi\|N_- generic, the induced module of dim 128 is irreducible | `gs2/chineg.py` | `phi (1, 10) lam [10, 12] dim Z 128 sing 1 irr quotient 128` (sing 1 = irreducible; the lines for phi (-1, -10), with dim N_- = 8, are not used in the text) |
| Statement 7.1.1, 3): the superizations s_a, s_b (sdim L = (9\|8)); restricted sdims (1\|0), (4\|3), (8\|6), (32\|32) up to Pi; modules of sdim (64\|64) | `gs2/superrep.py`, `gs2/superidx.py`, `gs2/supergen.py` | `restricted L(0, 0) dim 1 ... (1, 0)`, `L(1, 0) dim 7 ... (3, 4)`, `L(0, 1) dim 14 ... (6, 8)`, `L(1, 1) dim 64 ... (32, 32)`; `sdim L = (9\|8)`; `dim Z 128  singular 1  sdim wrt D: (64, 64)` for s_a and s_b |
| Table (16), row gs(2): dim L = 17, ind L = 3, not a torus, 2^7 | `gs2/torus_index.py` | as above: `stabilizer dim 3`, semisimple rank 2 < 3 |
| Lemma 8.1.1, Lemma 8.2.1, Theorem 8.3.1, Remark 8.3.2 (n = 2: dim n_- = 9, dim n_+ = 19), Statement 8.3.3 | `ham.py 1 4`, `ham.py 2 3`, `ham.py 2 4` | `basis vectors squaring OUTSIDE g: ['xu1', 'xu2', 'xv1', 'xv2']`, `dim n_- = 9, dim n_+ = 19`, `F_2-points outside t with ad semisimple: none`, `L_f: dim 34`, `dim M_L(lambda) = 2048`, `256 weights: irreducible 168, predicted by the o_I criterion 168, mismatches none`; for n = 1: `16 weights: irreducible 0` |
| Theorem 8.3.1, 3): the u_{2,4}-induced module is reducible | `ham.py 1 4`; `python -c "import ham; ham.naive(2, 4, (14, 7))"` | `u_{2,4} module over g, lambda = (0,): dim 64, ... singular vectors 24`; `lambda = (14, 7): dim 4096, singular vectors 2` |
| Statement 9.2.1, Table (16), Remark 9.2.3 | `kwind.py tabs/*.json` (the files listed below) | one JSON line per algebra; its fields `dim`, `dimL`, `restricted`, `ind`, `toral_stab`, `bound` give the columns of the table, e.g. `"name": "kas8", ... "dimL": 59, "restricted": true, "ind": [3, 0], "toral_stab": true, "bound": "2^28"`, `"name": "Bj", ... "sdimL": [27, 27], ... "ind": [3, 9], "toral_stab": true, "bound": "3^12 * 2^9"` |
| Statement 10.0.1: ind L = n for n = 1, 2, 3, the generic stabilizer is a torus | `index.py` | `n = 3: dim L = 27, ind L = 3, (dim L - ind L)/2 = 12.0; ... abelian True, closed under squaring True, 2-map bijective True` (and the same for n = 1, 2) |

The rows of Table (16) and their files in `tabs/`: vle(9), vle(7) — `vle9.json`, `vle7.json`;
kas(8), kas(10), kas~(7) — `kas8.json`, `kas10.json`, `kast7.json`; vas(8) — `vas8.json`;
mb_3(11), mb_2(11) — `mb3_11.json`, `mb2_11.json`; kle(15) — `kle15.json`; sb~(7), sb~(15) —
`sbt7.json`, `sbt15.json`; the eleven algebras of the second block — `k<dim>_<nilp>_<toral>.json`;
my(6; 1), by(7; 1), smy(6; 1), sby(7; 1) — `my.json`, `by.json`, `smy.json`, `sby.json` (for the
last three `kwind.py` passes to the simple core, as it reports in `derived` and `centre`);
Bj(3; 1|2) — `Bj.json`. `o3.json` (o^{(1)}_I(3)) is a calibration: `dimL 5, ind 1, toral, 2^2`.
The row gs(2) is computed in `gs2/`.

## Files

- `wk.py` — the fields F_{2^k}, exact linear algebra, matrix Lie algebras, o^{(1)}_I(2n+1) with its
  Z-grading and the 2-/4-elements, the restricted closure, the induced module M(lambda) by
  straightening and its singular vectors, sl(m).
- `oo.py` — Lie superalgebras at p = 2 (bracket and squaring), "method 2", oo_{IPi} and oo_{II},
  the exhaustive isomorphism test over F_2.
- `run.py` (parts A-E), `crit.py`, `crit3.py`, `torus.py`, `index.py` — Sects. 2-6, 10.
- `ham.py` — the Hamiltonian algebras h_I(2n+1; 1)' as operators on O(2n+1; 1) (Sect. 8).
- `kwind.py` — for a simple Lie (super)algebra given by structure constants over F_2 or F_3: the
  derived algebra, the quotient by the center, the minimal p-envelope in Der, the index, whether
  the generic stabilizer is a torus, the number N(g) of formulas (14), (15) (Sect. 9).
- `tabs/` — structure constants of the algebras of Table (16), as JSON (`tab[i][j]` a bitmask of
  [e_i, e_j] over F_2, or a list `C` of `[i, j, k, c]` with [e_i, e_j] = sum c e_k over F_3). They
  were exported from the models of these algebras published in the folders `cssei/` (the
  algebras of BGLLS and the eleven new ones) and `sandall/` (Skryabin's my, by and their special
  subalgebras; Bj(3; 1|2)) of this repository.
- `gs2/` — Shen's gs(2): `shenL.py` (Brown's model, the grading, L), `f2lin.py`, `gfla.py`
  (F_{2^k} linear algebra), `lalg.py`, `induce.py` (induced modules by straightening), `quot.py`,
  `hw.py`, `generic.py`, `chineg.py`, `hwscan.py`, `verifyq.py`, `torus_index.py`, `stab.py`,
  `toral.py` (toral elements, the outer grading of s_b), `superrep.py`, `superidx.py`,
  `supergen.py`.
