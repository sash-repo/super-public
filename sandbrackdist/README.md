# sandbrackdist

**Explicit brackets and p-structures of divergence-free and Hamiltonian modular Lie algebras**

A. Krutov, D. Leites, O. Lozhechnyk. Status: in preparation.

The scripts recompute, over F_p, the claims of the text that rest on a computation: the derived
algebras of the deformed Hamiltonian algebras h_omega(2k;N) for forms omega of Skryabin's 1st and
2nd types (Statement 3.4); their p-structures on the quadratic and linear generating functions and
the obstruction to restrictedness of h_{exp_j}(2k;1) (Section 5); and the 3-structures of the
Skryabin algebras my(6;N), by(7;N), dy(10;N), of their special subalgebras and derived algebras,
together with what happens when a height is raised (Section 6). Every algebra is built from
scratch: functions are divided power polynomials over F_p, the brackets are the Poisson brackets of
the text or the commutators of vector fields, and the Skryabin algebras are the Cartan prolongs of
their non-positive parts. Exact arithmetic throughout: integers mod p, numpy integer matrices
reduced mod 3, sympy for small determinants and inverses mod p, and `../lib/exactlin.py` (fast when
python-flint is installed, correct without it).

The same scripts serve, in the larger text `../sandall/`, Sections 7, 10 and 11 of
`sandall/sandwiches_p3/`; here only those the present text needs are kept.

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r sandbrackdist/requirements.txt
cd sandbrackdist
../venv/bin/python derall.py
```

Every script is run from inside this folder, with no arguments except where stated; it finds
`../lib/exactlin.py` by itself. Several scripts also write their report to a log file in `runs/`.
Times are for a laptop running one script at a time; everything not marked takes under a minute.
`derall.py`, `t2all.py`, `pstrH.py`, `pstrH2.py`, `pexp.py` use 9 processes.

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Statement 3.4, heading 1) (dim of the derived algebra of h^O_{omega_1}(2k;N) for every alpha in Mat_k(F_p), the six listed (p,k,N)) | `derall.py` (uses `derdim.py`) | `cases 100 mismatches 0` |
| Statement 3.4, heading 2) (the derived algebra of h_{omega_{2,j}}(2k;N), every choice of the beta_ij) | `t2all.py` | `cases 46 mismatches 0 jacobi failures 0` |
| Statement 5.2 (the p-structure of h_omega(2k;N), 1st type: headings 1)-3), formula (36)) | `pstrH.py` (p = 3, 5, 7 with k <= 2, p = 3 with k = 3, and the listed N != 1); `pstrH2.py` (p = 5, 7 with k = 3, p = 3 with k = 4, p = 11 with k = 1, 2, p = 13 with k = 2) | `shape-cases 91 elements 445 mismatches 0`; `shape-cases 91 elements 883 mismatches 0` |
| Proposition 5.3 (the p-th powers of the linear generating functions, formula (37)), checked on every monomial | `plin.py` (about 3 minutes), section A | `--> 49 cases, all match: True` |
| The claim after Proposition 5.3 (for N = 1 the equation (ad_f)^p = ad_g has no solution g in O(F) exactly for the f = u_i of Proposition 5.3) | `pclosed.py` | `shape-cases 37 monomials without a p-th power in O(F) 118 mismatches 0` |
| Remark 5.3.1, formula (41) ((ad_{u_2})^p for N != 1 is a vector field minus d_4^{(p)}, the same d^{(p)}-part as in the undeformed h(2k;N)) | `plin.py`, sections B and C | for `om_{1,0}(4;un)  p=3 N=[2, 1, 1, 2]`: `(ad_{u_2})^3 : NOT in W(2k;un); = vect + sum lam_t d_t^{(p)}, lam=[0, 0, 0, 2] : verified` (and `lam=[0, 0, 0, 4]` for p = 5), i.e. the coefficient -1 of d_4^{(p)}; section C: `translations with (ad_{u_i})^p != 0 : none` for h(4;1), p = 3, 5, and h(6;1), p = 3 |
| Statement 5.4 (h_{exp_j}(2k;1) is not restricted, for (p,k) = (3,1), (5,1), (7,1), (3,2); the obstruction is one basis element, and (ad_{u_~j})^p is diagonal in the monomial basis) | `pexp.py` | ten lines `(p, k, j, True, [the exponent of u_~j], True)`, then `cases 10 all as claimed True` |
| Subsection 6.1 (the complete prolong of (g_-, sl(3)) for dy(10) has no component of positive degree) | `sdy1.py` | `G0[:8] {-4: 3, -3: 1, -2: 3, -1: 3, 0: 8}` (with g_0 = sl(3)), against `G0 {..., 0: 9, 1: 3, 2: 9, 3: 8}` (with g_0 = gl(3)) |
| Proposition 6.3 (the cubes of the non-positive part; for dy(10), (x+y)^3 = [x,[x,y]] is non-zero for some x in g_-1, y in g_-2) | `negcube.py` (about 1 minute); `dysplit.py`; `dyweights.py`; `dy3.py` | `negcube.py`: `my  : 729 elements of g_-, nonzero cubes 0; ...`, `by  : 2187 elements of g_-, nonzero cubes 0; ...`, `dy  : 59049 elements of g_-, nonzero cubes 37908; nonzero Jacobson terms on g_-1 basis pairs 0`; `dysplit.py`: `g_-1 only: (27, 0, 0)`, `g_-1 + g_-2: (729, 468, 468)` (all non-zero cubes lie in g_-4), `nonzero s(g_-1,g_-2) basis pairs: 3`; `dyweights.py`: the weights of the degree -3 fields, none of them 0; `dy3.py`: `non-zero cubes: ['h1', 'h2', 'h3']`, `cubes outside g_{<=0}: none`, `67 checks, 0 failures` |
| Theorem 6.4 and table (50) (my, smy, by, sby and the terms of their derived series are restricted, the 3-map being the cube; dimensions, degrees, derived series; the only non-zero cubes are those of the torus) | `test_skry3.py`; `runskry3.py` (about 35 minutes, `by` alone about 30; `runskry3.py my` etc. for one algebra); `prolong.py my smy by sby` (about 2 minutes); `dyfull.py` (dy(10;1) in full, about 40 minutes) | `test_skry3.py`: `17 checks, 0 failures`; `runskry3.py`, for each algebra: `my(6; One)` `dim = 162, degrees -2..11`, `cubes inside L: 162 of 162;  non-zero: 3`, `==> RESTRICTED (Jacobson), the 3-map being the cube of the field`; `smy(6; One)`: dim 84, `derived algebra L^(1): dim 81`, `L^(2): dim 77`; `by(7; One)`: dim 244, `derived algebra L^(1): dim 240`; `sby(7; One)`: dim 82, `derived algebra L^(1): dim 78`; in each term `non-zero` equals dim h; `prolong.py my`: the components `3, 3, 9, 9, 18, 18, 21, 21, 18, 18, 9, 9, 3, 3`; `dyfull.py`: the table `dim dy(10;One)_k` and every cube inside the algebra |
| Statement 6.6.2, item 1) (one raised height of a top indeterminate destroys the 3-structure of my, smy, by, dy; the example for my(6;(1,1,1,2,1,1))) | `unskry_all.py`; `unby.py smy,by,sby`; `undy.py 14 7` (about 6 minutes) | `unskry_all.py`: `heights [1, 1, 1, 2, 1, 1]: (ad d_y1)^3 nonzero on 8 fields of degree<=5; first (4, 'd_y1')` and likewise for the other two heights; `unby.py`: `smy ... (ad d_y1)^3 nonzero on 38 fields` (39 for y2, y3), `by heights [1, 1, 1, 1, 1, 1, 2]: ... (ad d_z)^3 nonzero on 29 fields; first (6, '2d_z')`; `undy.py`: `dy heights [1, 1, 1, 1, 1, 1, 1, 2, 1, 1]: ... (ad d_u8)^3 nonzero on 39 fields; first (8, 'd_u10')` |
| Statement 6.6.2, item 2) (sby(7;N) = sby(7;1) for N_z = 2 and N_z = 3) | `unby.py smy,by,sby`; `HT=3 ../venv/bin/python unby.py sby,by 30` | `sby heights [1, 1, 1, 1, 1, 1, 2]: dim through degree 8 = 81; (ad d_z)^3 nonzero on 0 fields; first None`; `sby heights [1, 1, 1, 1, 1, 1, 3]: dim through degree 9 = 82; (ad d_z)^3 nonzero on 0 fields; first None` (82 = dim sby(7;1), table (50)) |

## Files

- `skrpb.py` — divided power algebras O(2k;N) over F_p, Skryabin's normal shapes J_k(0), C_k,
  J_{k,r}(lambda), the deformed Poisson brackets of the 1st type (`Ham`) and of the 2nd type
  (`HamExp`), and their p-th powers; `pstrH.py`, `pstrH2.py`, `plin.py`, `pclosed.py`, `pexp.py`
  use it.
- `derdim.py` — the dimension of h^O_{omega_1}(2k;N) and of its derived algebra from the bracket;
  `derall.py`, `t2all.py` are the drivers of Statement 3.4.
- `gvect.py`, `dpvect.py` — vector fields over F_3 with divided power coefficients (heights 1, or
  arbitrary), and the F_3 linear algebra; `prolong.py` — my, smy, by, sby as Cartan prolongs;
  `dy3.py` — the non-positive part of dy(10) and its checks; `skry3.py` — the cube of a vector
  field, and the test whether it lies in the algebra; `unskry.py` — the prolong with raised heights.
- Drivers: `negcube.py`, `dysplit.py`, `dyweights.py`, `sdy1.py`, `test_skry3.py`, `runskry3.py`,
  `dyfull.py`, `unskry_all.py`, `unby.py`, `undy.py`.
- `runs/` — log files written by some of the scripts.
