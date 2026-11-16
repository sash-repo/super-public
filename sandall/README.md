# sandall

**Sandwiches, filtered deformations and normal shapes in modular Lie (super)algebras**

P. Grozman (deceased), A. Krutov, D. Leites, O. Lozhechnyk. Status: in preparation.

The scripts recompute, by exact arithmetic over F_p (and over Q where stated), the claims of the
text that rest on a computation. They are grouped by the part of the text they serve:

- `sandwiches_p3/` — Sections 1–11: sandwiches for p >= 3 (Zassenhaus algebras, the criterion
  (fs), the Bouarroudj superalgebra, odd sandwiches), the derived algebras and p-structures of the
  deformed Hamiltonian algebras, the 3-structures of the Skryabin algebras;
- `sandwiches_p2/` — Sections 12–15: p = 2 (the desuperized exceptional algebras, the algebra
  ir(9;1) and its superizations, the Kaplansky algebras, the W-gradings of F(kas) and F(vle), the
  Eick algebras and the simple Lie algebras of dimension 15 and 16 of the Cushing–Stagg–Stewart
  library, Eick's 16:6); `sandwiches_p2/ei166/` is Section 14;
- `filtered_deformations/` — Sections 16–21: H^2(g_-; g), H^2(g; g) and the filtered deformations
  of k^(1)(3;1), fr(3;1), svect(3;1), the Skryabin and Melikyan algebras, the Hamiltonian
  algebras H(2:n;omega);
- `normal_shapes/` — Sections 22–41: closed 2-forms over divided power (super)algebras, every p;
- `defrep/` — Sections 42–47, the part not computed with SuperLie: the Cartan prolongs of the
  modules marked "n", and the deformations of modules over sl(2), osp(1|2), sl(3).

Exact arithmetic throughout: integers mod p, `fractions.Fraction`, bit masks over F_2, and
`../lib/exactlin.py` (fast when python-flint is installed, correct without it).

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r sandall/requirements.txt
cd sandall/sandwiches_p3
../../venv/bin/python derall.py
```

Every script is run from inside its own subfolder (`sandall/<part>/`, or `sandall/sandwiches_p2/ei166/`);
it finds `../../lib/exactlin.py` by itself. No arguments are needed except where stated. Several
scripts also write their report to a log file in the current folder or in `runs/`. Scripts marked
"chain" read a file written by the script before them, so run them in the order given.

Times are those of a laptop with nothing else running, unless stated; most runs take seconds to a few
minutes.

## Claims of the text and the scripts behind them

Numbers are those of the text.

### Sections 1–11 (`sandwiches_p3/`)

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Statement 1.6.5 (sandwiches of small Lie superalgebras over F_3 and F_5) | `runodd.py` (uses `oddsand.py`) | `gl(1\|1) over F_3`: `odd : lines with (ad_s)^2=0: 4; of them satisfying (7)=(8): 2`, `lower central series [3, 1, 0]`; then the counts 8, 8, 160, 8, 14, 170, 8 (F_3) and 936, 32, 12 (F_5) of item 2), each with `0` satisfying (7)=(8) |
| Proposition 4.2 (the sandwiches of the Zassenhaus algebras vect(1;m) for small p, m) | `zass.py` | `all checks passed` |
| Corollary 5.2.1, Proposition 5.2.2 (the criterion (fs); dim s(vect(2;1\|0)) = 10) | `runfs.py` (about 5 min) | for n = 1, p = 3, 5, 7: `mismatches 0`; for n = 2, p = 3: `span of the F_p-points of the cone: dim 10, degrees [1, 2, 3]` |
| Remark 5.2.3 (the sandwich subalgebra of [GGRZ]) | `ggrz.py` (about 10 min) | `79 and 7 here` |
| Theorem 6.1 (the Bouarroudj superalgebra Bj(3;N\|3)), Theorem 6.2.1, Comment 6.2.2, item 3) of Statement 1.6.5 | `bj.py`; `runbj.py`, `runbj3.py` (about 10 min each); `runbj4.py`, `runbj5.py`, `runbj6.py`, `runbj7.py`, `sixs.py` | `bj.py`: self-checks; `runbj3.py`: `n_L(s): sdim (24\|17), codim 13`, `= L_0^A : False`, `= L_0^B : False`; `runbj6.py`: `s(Bj) under the repaired definition: sdim (8\|5), dim 13, lower central series [13, 0]  -- NILPOTENT`; `runbj5.py`: `dim of the intersection = 45` |
| Statement 7.4 1) (the derived algebras of h^O_{omega_1}(2k;N), every alpha) | `derall.py` (uses `derdim.py`) | `cases 100 mismatches 0` |
| Statement 7.4 2) (forms of the 2nd type) | `t2all.py` | `cases 46 mismatches 0 jacobi failures 0` |
| Statement 10.2 (the p-structure of h_omega(2k;N), 1st type, the listed (p, k, N)) | `pstrH.py`; `pstrH2.py` | `shape-cases 91 elements 445 mismatches 0`; `shape-cases 91 elements 883 mismatches 0` |
| Proposition 10.3, Remark 10.3.1 (p-th powers of the linear generating functions) | `plin.py` (uses `skrpb.py`) | section C: `translations with (ad_{u_i})^p != 0 : none` for h(4;1), p = 3, 5 and h(6;1), p = 3 |
| Statement 10.4 (h_{exp_j}(2k;1) is not restricted; the obstruction is one basis element, diagonal) | `pexp.py` (about 2 min) | `cases 10 all as claimed True` |
| Proposition 11.3 (cubes on g_-), Lemma 11.2.1 | `negcube.py` (about 5 min) | `my : 729 elements of g_-, nonzero cubes 0`, `by : 2187 elements of g_-, nonzero cubes 0`, Jacobson terms `0` |
| Theorem 11.5 (the Skryabin algebras my, smy, by, sby, dy and their derived algebras are restricted, the 3-map being the cube) | `test_skry3.py`; `runskry3.py` [kinds] (by default `my smy by sby`); `dy3.py`; `dyfull.py` | `17 checks, 0 failures`; for each algebra `==> RESTRICTED (Jacobson), the 3-map being the cube of the field`; `dy3.py`: `67 checks, 0 failures`; `dyfull.py`: the table `dim dy(10;One)_k: -4:3, -3:1, ..., 22:3` and every cube inside the algebra |
| Statement 11.9.2 (one raised height destroys the 3-structure of my, smy, by, dy; sby(7;N) = sby(7;1)) | `unskry_all.py`; `unby.py smy,by,sby`; `HT=3 ../../venv/bin/python unby.py sby,by 30`; `undy.py 14 7` (about 2 min); `dysplit.py`, `dyweights.py`, `sdy1.py` | `unskry_all.py`: `(ad d_y3)^3 nonzero on 10 fields of degree<=5; first (4, 'd_y1')`; `unby.py`: `smy ... nonzero on 39 fields`, `by ... nonzero on 29 fields`, `sby ... nonzero on 0 fields`; `undy.py`: `dy heights [1, 1, 1, 1, 1, 1, 1, 2, 1, 1]: ... (ad d_u8)^3 nonzero on 39 fields; first (8, 'd_u10')`, and `sdy ... nonzero on 0 fields` |

### Sections 12–15 (`sandwiches_p2/`)

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Statement 12.2.1 (the growth vectors of Table 25.5 of [BGLLS]) | `bglls25.py` | `of depth 1 ... : 4`, `contact (dim F(g_-2) = 1) : ['kas(1;N\|6)', 'mb(4;N\|5)', 'vle(5;N\|4)', 'kle(9;N\|6)']`, `MISMATCHES: 0` |
| Statement 12.3.1, Theorem 12.3.6 (the algebra ir(9;1): its grading, prolongs, restrictedness; the admissible parity functions) | `ir.py` (about 1 min) | `0 checks failed` |
| Theorem 12.3.3 (ir(9;1) is of type D_4), Theorem 12.3.17, Statement 12.3.20, Theorem 12.3.24 (Der ir = f(4)) | chain: `mkalgs.py` (writes the small cache `algs.pkl`), then `d4.py`, `torus2.py`, `eick.py`, `annfull.py` (10 to 30 min each); `f4.py` (about 1 min); `iso2.py`, then `verifyiso.py` | `d4.py`: the isomorphism with the D_4 algebra, verified; `eick.py`: `failures of 1 + d in Aut(L) over all 441 of them: 0`; `annfull.py`: `ann = 69616   (the zero derivation included)`; `f4.py`: `checks 28, failures 0`; `verifyiso.py`: `phi is an isomorphism of restricted Lie algebras: True` |
| Remarks 12.3.7, 12.3.8, Statement 12.3.9 (the mirrors of ir), Corollary 12.3.10, Theorem 12.3.12 (the superizations up to isomorphism) | `irfam.py`; `iriso.py` (about 2 min); chain: `supiso.py`, then `verifysup.py` | `irfam.py`: the superdimensions `(14\|12)`, `(26\|0)`, `(10\|16)` with the numbers of parity functions; `iriso.py`: `contact substitutions preserving ir: 48 of 384`, `dim Der(ir(9; \One)) = 52`; `verifysup.py`: both isomorphisms verified |
| Statement 12.3.15 (Der ir(9;1), Out, H^2) | `irder.py`, `irh2.py`, `irout.py` (about 1 min each) | `dim Der ir(9; \One) = 52`, `Out by degree: {-2: 6, 0: 14, 2: 6}`; `irh2.py`: `dim Hom_Out(Out, (H^2)^*) = 1`, `some element of it is invertible: True`; `irout.py`: `the squaring of Out ir is a p-structure: True` |
| Theorem 12.3.27 (ir(9;1) has no sandwiches, over F_2 and over K) | `irsand.py` (exhaustive, all 2^26 elements); `irK.py` | `elements with (ad_X)^2 = 0: 69615`, `sandwich elements (san) and (san2): 0`; `irK.py`: `the sandwich variety of ir(9; \One) is EMPTY over K` |
| Lemma 12.3.30, Theorem 12.3.31, Corollary 12.3.33 (all the W-gradings of ir and of its superizations) | `irw.py`; `irdiag.py` | `bad checks: 0`; `irdiag.py`: `agrees with "ir is d-graded" on 3888 of 3888` |
| Statement 12.6.2, Theorems 12.7, 12.8 (the Kaplansky algebras, their superizations and sandwiches) | `kap.py` (`kap.py 3` adds 2m = 6 and takes over 30 min); `kapK.py`; `kaptop.py` | `kap.py`: the dimensions (KapDim), the isomorphisms of item 4), and `KapS_{4,1,0}(4) sdim (10\|4) ... simple: yes` etc.; `kapK.py`: `the sandwich is inhomogeneous (parities [0, 1])`; `kaptop.py`: `it is a genuine sandwich` |
| Theorem 12.15.1, Lemma 12.16.1, Theorem 12.16.2, Corollaries 12.16.3–12.16.5 (the W-gradings of F(kas) and the mirror automorphisms) | `kas2.py`; `kasaut.py`; `kastau.py` | `kasaut.py`: `the two W-gradings are ISOMORPHIC as graded Lie algebras`; `kastau.py`: `(graded iso: YES)` for rows 2–5 |
| Statement 12.17.1 (the W-gradings of F(vle)) | `vle7.py`; `fvle2.py 2` (the lattice box [-2,2]^4; uses `fvle.py`, `prolong.py`, `f2vf.py`) | `vle7.py`: `restricted closure of J: 94 (46, 48)`; `fvle2.py 2`: `W-gradings: 29`, nine lines `codim 7 ... x2`, `codim 8 ... x3`, ..., `codim 54 ... x3` as in table (TabVleW) (`fvle2.py 1` finds 27 of them, missing the two of codimension 31) |
| Statement 12.17.3 (the weight sandwiches of F(vle)) | `fvlesand.py 1`; chain: `fvlewt2.py`, then `fvlecheck.py` | `dim s_T = 52`, the normalizer of codimension 29, contained in L_{>=0} of the two W-gradings of codimension 7 and the four of codimension 9 |
| Theorem 13.2.1 (the graded algebra underlying the Eick algebras) | `vect2.py`, `vect2gen.py`, `eick2.py` | the sl(3)-submodules of vect(3;1)_1 of dimensions 4 and 5 and the graded algebras g(W) |
| Lemma 13.3.1, Theorem 13.4, Theorem 13.6, Corollary 13.7, Theorem 15.2 (W-gradings and sandwiches of the simple Lie algebras of dimension 15 and 16) | `runall.py css-bases/dim15/*.txt css-bases/dim16/*.txt`; `wall.py css-bases/dim15/*.txt css-bases/dim16/*.txt`; `thin.py`, `maxsub.py`, `minmax.py`, `wgrad.py`, `sand4.py` (each on `css-bases/thin15/*.txt` by default) | the table (TabSand) of the text, algebra by algebra |
| Theorem 13.9 (the sandwiches over K) | `sandK.py`; `sandKcheck.py` | the sandwich variety over K for every algebra of the library, with the falsification controls |
| Statement 15.5, Remark 15.5.2 (the sandwiches single out a W-grading of depth 1; the search in codimension 6) | `wall.py` (as above); `prexh6.py css-bases/dim15/<file>.txt 6`, one algebra per run (6 to 12 min each) | the gradings (4\|6,4,1) and (3\|4,4,3,1) of formula (nsgr); `prexh6.py`: the rows of table (TabCod6), e.g. 4607 candidate functionals, 223 subalgebras of codimension 6, 36 of them maximal, for Ei(15:7) |
| Statement 14.3, Proposition 14.4, Corollary 14.4.1 (Eick's 16:6 is graded; the prolongs of A. Lebedev's modules) | in `ei166/`: `eick166.py`, `grade166.py`, `iso166.py`, `inv166.py`; `p166.py`, `allmin.py`, `conj166.py`, `conjtest.py` | `eick166.py`: 5 subalgebras of codimension 4, the graded algebra (4\|8\|4), `fg_0` with 10 elements with (ad_x)^2 = 0; `grade166.py`: the grading exists |

The structure constants of the 15- and 16-dimensional algebras in `css-bases/` are those of the
public library of Cushing, Stagg and Stewart, arXiv:2207.01094 (github.com/cushydom88/simple-lie-algebras),
with its LICENSE.

### Sections 16–21 (`filtered_deformations/`)

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Claim 16.1.2.1 (H^2(g_-;g) of the Hamiltonian algebras h(2k;N)) | `ham3.py p k N_1 ... N_2k`, e.g. `ham3.py 3 1 1 1`, `ham3.py 5 1 1 1`, `ham3.py 3 2 1 1 1 1` | e.g. `p=3 k=1 N=(1, 1) ch = (V,csp)_*: dims {-1: 2, 0: 4, 1: 2, 2: 1}, dim=9` and the invariants |
| Statement 16.1.2.2 (the row k = 1 of formula (DF:dimHam)) | `nongraded.py` | `dim expected ... : OK` and `Jacobi: holds` for p = 3, 5, 7, N = (1,1), and p = 2, 3, N = (1,2), (2,2) |
| Statement 16.6 (the opposite gradings) | `opposite.py` | the prolongs of the opposite gradings, finite and equal to g |
| Claim 17.1, Statement 17.2 (the five cocycles of H^2(k^(1)(3;1); k^(1)(3;1)), their integrability, the deforms and their Weisfeiler filtrations) | `deform.py`; `deform2.py` (uses `k3.py`, `coh.py`) | the table (DF:WF): `codim 3: dim L_0 = 23, filtration [23, 26]` etc. |
| Statement 17.3 (the deform of weight (-3;±3) is er^(1)(3;1)) | `er3.py`; `ermolaev.py`; `iddeform.py` | `sl(2) (+) hei(2) has dim 6, derived = 6, centre = 1, and its nilradical hei(2) is 3-dimensional with 1-dimensional centre` |
| Statement 17.4 (the deform of weight (-6;0) is k^(1)(3;1)) | chain: `iso6_tables.py`, `iso6.py`, `iso6_tori.py`, `iso6_wf.py 3 k1 E6_1`, `iso6_iso.py`; `iso6_run2.py k1 E6_1` | `explicit phi: k^(1)(3;1) -> E6_2: bijective True, homomorphism on all 676 pairs: True` |
| Statement 18.1, Lemma 18.2 (no filtered deformations of k^(1)(3;1); the g_0-module H^2(g_-;g)) | `run_k.py`; `modstr.py`; `check_ce.py` (cross-check of `ceh.py` against `coh.py`) | `dim H^2(g_-;g) = 5`, invariants `0`, the submodules of Statement 18.1 |
| Claim 18.4 (the Frank algebra, n = 1) | `fr3.py`; `modstr.py` | the degree-7 component, the same g_0-module as for k^(1)(3;1) |
| Claim 18.5, Statement 18.6 (svect(3;1) and Tyurin's list) | `vect3.py`; `svecth.py` | the 4-dimensional space of invariants; `fsvect_{1+x}(3;1)` of dimension 55 |
| Statement 18.7 (the degree-3 invariants of svect(3;1) are obstructed) | `svint.py` | `H^2(g;g)_3 = 0: no degree-3 infinitesimal filtered deformation exists in the full complex` |
| Claim 18.8 (er^(1)(3;1)) | `ermolaev.py` | H^2(g_-;g) and its invariants for er^(1)(3;1) |
| Claims 18.9–18.12 (smy^(1), my, sby^(1), by^(1)) | `prolong.py`; `run_skryabin.py my by`; `run_derived.py smy sby by` | `total invariants (H^2(g_-;g))^{g_0} =` `0` (my), `4` (smy^(1)), `7` (sby^(1)), `1` (by^(1)) |
| Claim 18.13, Claim 20.2 (the Melikyan algebra, p = 5, in its two gradings) | `run_mel.py`; `run_mel2.py`; `mel5.py`; `melcoh.py`; `regrade.py` | `Me (contact grading): dim (H^2(g_-;g))^{g_0} = 0`, `me (standard grading): dim (H^2(g_-;g))^{g_0} = 0` |
| Section 21 (the algebras H(2:n;omega) of [CM]: Der, H^2, the filtration) | `nongraded.py`; `altder.py`; `filt.py` | `altder.py`: `SkDer/ad = 3, AltDer/ad = 3   \|   H^2 from cochains = 3`; `filt.py`: `bracket mismatches off e: 0` |
| Lemma 21.5, Statement 21.6 (dim H^2 = n_1+n_2-1 at p = 2, not n_1+n_2) | `p2check.py`; `p2more.py` | `degree der (Rmk 4.4) derivation=True outer=True skew=False alternating=True`; the dimensions for more n |

### Sections 22–41 (`normal_shapes/`)

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Theorem 24.1 (the collapse at p = 2) | `collapse.py` | every check `True` |
| Theorems 25.2, 26.1, 26.2 (the diagonal is constant at p = 2; the parity trichotomy; non-degeneracy); Statement 25.3.2 (dim Z^2) | `run1.py`; `survey.py 5`; `strichot.py`; `perindeg.py` | `run1.py`: `dim Z^2 = 1`, `6`, `20`, `55` for M = 1..4, `10` for M = 2, N = (2,1), and `closed forms with non-constant diagonal : 0`; dim Z^2 = 139 for M = 5 is printed by `survey.py 5`; `strichot.py`: `TOTAL failures: 0`; `perindeg.py`: `mismatches: 0` |
| Theorem 27.1, Corollaries 27.2.1, 27.2.2 (the cohomology of the divided power de Rham complex, every p) | `sfactor.py`; `cohom.py` | `failures: 0`; `total mismatches: 0` |
| Proposition 30.1, Corollaries 30.1.1, 30.1.2 (the algebra preserving omega) | `shzed.py` | `failures: 0` |
| Theorems 31.1, 31.2, Corollary 31.2.1 (the brackets of the even and periplectic shapes) | `sbracket.py`; `peribr.py` | `TOTAL failures: 0`; `peribr.py`: the invariants of the two brackets |
| Theorems 32.1, 32.3, Statement 32.1.1, Statement 32.4 (le(n;1) against le_{omega_Xi}(n;1), p = 3: squares, square-zero odd elements, Der) | `leiso.py` (about 2 min); `lecount.py`, `leval.py` (about 5 min each); `lefib.py` | `leiso.py`: `dim Der = 8  (even 4, odd 4)` against `dim Der = 7  (even 3, odd 4)` (n = 1), `dim Der = 39  (even 19, odd 20)` against `dim Der = 38  (even 18, odd 20)` (n = 2), `#(x odd \| [x,x]=0) ... = 26001 of 129140163`; `lecount.py`: `exact 36207 of 129140163` (shape 0) and `exact 26001` (shape Xi), and the spans 18 and 108; `leval.py`, `lefib.py`: the controls, `agree = True` |
| Proposition 32.5 (conformal fields) | `conf.py` | `L_E_q omega_Xi = omega_0`, `L_E_xi omega_Xi = omega_0` |
| Statement 33.1 (the algebras preserving the twisted odd forms) | `perirun.py` (about 5 min), section E; `twist2nd.py` (both primitives, the auxiliary height raised by 1, 2, 3) | `sdim h_exp = (2\|3)   sdim h(omega_0) = (3\|3)` (p = 3), `(4\|5)`/`(5\|5)` (p = 5), `(6\|7)`/`(7\|7)` (p = 7), `(8\|9)`/`(9\|9)` (N = (2)), `(13\|14)`/`(18\|19)` (n = 2); `twist2nd.py`: `(13\|14)` for alpha_0 and `(15\|16)` for alpha_0' (n = 2), the same for r = 1, 2, 3 |
| Statement 33.3.1 (the odd forms over O(1\|1;1)) | `peri2nd.py` | `p = 3: 2 orbits, cohomology predicts 2 --- AGREE`, `p = 5: 1 orbits, cohomology predicts 1 --- AGREE` |
| Statement 34.1.1 (two classes for N = 1; the table (NS:tabshapes)) | `run1.py`; `run3.py` (3 s); `relcheck.py`; `survey.py 4`, `survey.py 5` (random samples); `extra.py`; `hamcheck.py` | `run3.py`: the two orbits, `|orbit| = 16384` and `49152`, with the derived series `[10, 7, 6, 6]` and `[10, 7, 7]`; `survey.py`: `[19, 15, 15]`, `[19, 15, 14, 14]` (M = 4) and `[36, 31, 31]`, `[36, 31, 30, 30]` (M = 5) |
| Statement 34.2.1 (thousands of classes under the graded group) | `scensus.py` | `458752` non-degenerate forms; `2456`, `1227`, `376` classes for (2\|1), (1\|2), (0\|3) |
| Statements 34.3.1, 34.4.1 (the classes of homogeneous forms, m+n <= 3; block types) | `shomog.py` | the table (NS:table) |
| Theorem 35.1, Statement 35.1.1 (twisted forms) | `stwist.py` | the superdimensions (3\|0), (2\|0), (1\|2) of Statement 35.1.1 |
| Statement 35.2.1 (the squaring separates the 34 classes) | `ssplit.py` (about 1 min; writes two LaTeX tables) | `O(0\|3;1): 4 classes; ...` for every superdimension |
| Theorem 36.5, Statements 36.6.2, 36.8.1, 36.9.1, 36.10.1 (restricted structures) | `srestr.py` (about 2 min); `srestr.py sigma` | the restricted cases of 36.6.2; `5` restricted spans of 34 and the ranks of sigma; the 5 classes with a (2,4)-type structure of table (NS:tab24) |
| Statements 37.2.1, 37.2.2, 37.2.4 (p = 2: the path formula, the 2-structure of the forms of the 1st type, the forms of the 2nd type) | `chk276.py`; `run276.py all`; `second276.py` (all use `sk2.py`) | `0` failures |
| Statement 37.3.1 (the exceptional forms of [KoKCh] in three indeterminates) | `kokuch.py`; `kokuch2.py`; `flagcheck.py` | `kokuch.py`: simple exactly where [KoKCh] claims, derived algebra `[7, 6, 6]` in the 6 remaining cases; `flagcheck.py`: `0 violations` |
| Statement 38.1.1 (the second normal shape) | `shape2.py` (uses `nonalt276.py`) | the second class for 2 <= M <= 5 |
| Theorem 38.2.1, Statement 38.3.1 (the dual bivector; the brackets of the normal shapes) | `bivec.py` | `A) det G is a constant : 0 failures`, `B) diag(G^{-1}) is constant : 0 failures`, `C) G alternating <=> G^{-1} is : 0 failures` |
| Theorem 39.1 (restrictedness for an arbitrary shearing vector, checked for M = 3 and M = 5) | `t24u.py [N ...]`, by default all twelve shearing vectors the text quotes; in three groups: `t24u.py 1,1,1 2,1,1 1,2,1 1,1,2 2,2,2`, `t24u.py 3,1,1 1,3,1 2,3,1 1,3,3`, `t24u.py 1,1,1,1,1 2,1,1,1,1 1,2,1,2,1` (the last two groups take the longest); `nonalt276.py`; `extra276.py` | for each N the restricted structures of h(omega) and Span{H_f} (`RESTRICTED` for h(omega) only at N = (1,1,1)), the obstructions, and the line `(2,4)-structure for the grading by deg x: ...; N = ..., so item 4) predicts ...: agree = True` |
| Controls of the conventions (no claim of their own) | `ssmoke.py`; `perismoke.py` | `TOTAL failures: 0  (must be 0)` |

### Sections 42–47 (`defrep/`)

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Statement 43.2 (no infinite-dimensional Cartan prolongs of the modules marked "n") | `runprol.py prol <name> <weight>` for `brj23 2,0`, `brj25 0,1`, `g16 1,0,0`, `g23 0,1,0`, `g23 1,0,0`, `g26 1,0,0,0,0`, `g33 1,0,0,0`, `g43 1,0,0,0`, `g2p5 0,1`, `f4p3 0,0,0,1`, `f4p0 0,0,0,1`, `sl3p3 0,2`, `sl3p3 1,1`; `wkprol.py 3`, `wkprol.py 4` (p = 2) | for g itself `g_1: 0\|0` in every case except `brj25 lam=[0, 1]` (the module Pi(ad), not marked "n"; see the paragraph after Statement 43.2): `g_1: 0\|1`, `g_2: 0\|0`; with the identity and the torus added (`(+Id, +torus)`), `g_1` is `0\|0` or of the superdimension of M (`g26`: `15\|6`, `g33`: `8\|8`, `sl3p3 lam=[0, 2]`: `6\|0`) and then `g_2: 0\|0`; `wkprol.py`: `g_1: 0\|0` for wk(3), bgl(3), wk(4), bgl(4) |
| Lemma 46.7.1.1, Claim 46.5.2 (the deformations of the sl(2)-modules) | `sl2h.py 3`, `sl2h.py 5`, `sl2h.py 7` | for each p the cocycles of weight 0 with the h-part equal to `(-1)^{\|u\|/2+1}` |
| Theorem 46.7.2.1, Lemma 46.7.3.1 (osp(1\|2)) | `osph1.py`; `ospcoc.py 3`, `ospcoc.py 5`, `ospcoc.py 7` | `osph1.py`: `H^1 = 0\|2` exactly for `p=3 V=L^1` and `p=5 V=L^2`, i.e. for L^((p-1)/2), and `H^1 = 0\|0` otherwise (p = 3, 5, 7); `ospcoc.py`: `uniform formulas are cocycles: True True` |
| The sl(3)-modules L^(0,1), L^(1,0), L^(0,2), L^(1,1) do not deform; Statement 46.7.4.1 (the Steinberg module L^(2,2), p = 3) | `steinberg.py` (uses `h1gen.py`) | `H^1 = 0\|0` for `lam=[0, 1]`, `[1, 0]`, `[0, 2]`, `[1, 1]`; `lam=[2, 2] target=end: H^1 = 7\|0` with one class of degree (0, 0) and six of degrees (±3, 0), (0, ±3), ±(3, 3), i.e. weights 0 and 3alpha |

## Claims not reproduced here

Theorem 45.6.4, items 4) and 5), Proposition 45.6.1 and Statement 45.6.2 (the irreducible
modules of Shen's gs(2) and the Cartan prolongs of V_7 and V_8) were computed with separate
engines, the Gröbner bases of Statement 45.6.2 with *Mathematica*; they are not part of this
folder.

### Claims resting on SuperLie

SuperLie (P. Grozman, A. Krutov, github.com/andrey-krutov/SuperLie) has no license and is only
cited. The following rest on SuperLie computations and are not reproduced by the scripts above:
the paragraphs of Theorem 5.1 for the Melikyan, Frank, Ermolaev and Skryabin algebras; the parts
of Theorem 6.1 on Me(3;N|3) and Bj(1;N|7); the cocycles of Statements 14.6.0.1 and 14.6.1.1;
the tables of dimensions and characters of Sections 43–47. The Claims of Sections 16–20 were
first obtained with SuperLie; the scripts in `filtered_deformations/` recompute them independently.

## Files

- `sandwiches_p3/` — `sandp.py` (sandwiches, s(g) and n_g(s(g)) of a graded Lie algebra over
  F_p), `vectp.py` (O(n;1) and vect(n;1|0) over F_p), `oddsand.py` (Lie superalgebras over F_p,
  odd sandwiches), `bj.py` (the Bouarroudj superalgebra), `skrpb.py` (Skryabin's normal shapes,
  Poisson brackets, p-structures), `gvect.py`, `dpvect.py` (vect(n;N) over F_3, prolongs),
  `prolong.py` (the Skryabin algebras as prolongs), `skry3.py` (3-structures), `dy3.py`,
  `css.py` (the library of [CSS]) and the drivers named in the table.
- `sandwiches_p2/` — `f2vf.py`, `prolong.py` (vector fields over F_2 and the Cartan prolong),
  `lin2.py`, `absirr.py` (F_2 linear algebra, absolute irreducibility), `css.py`, `thin.py`
  (the library of [CSS]), `torus2.py` (maximal tori over F_2), `ir.py`, `kap.py`, `kas2.py`,
  `vle7.py`, `vect2.py`, `vect2gen.py` and the drivers named in the table; `css-bases/` (data of
  [CSS], with its LICENSE); `runs/` (log files written by some scripts).
- `filtered_deformations/` — `k3.py` (k(3;1) over F_3), `coh.py`, `ceh.py` (Chevalley–Eilenberg
  cohomology by bigrading), `gvect.py`, `fr3.py`, `vect3.py`, `prolong.py`, `mel5.py`,
  `nongraded.py` and the drivers named in the table.
- `normal_shapes/` — `nalt2.py` (p = 2, the divided power de Rham complex), `peri.py` (odd forms,
  p > 2), `sform.py` (every p, the super case), `leiso.py`, `sk2.py` and the drivers named in the
  table.
- `defrep/` — `hwmod.py` (irreducible modules L^lambda over g(A) as Verma modules modulo the
  radical), `prol.py` (the graded Cartan prolong), `ratf.py`, `runprol.py`, `h1gen.py`
  (H^1(g; End M) by parity and degree), `steinberg.py`, `sl2h.py`, `osph1.py`, `ospcoc.py`,
  `wkprol.py`.
