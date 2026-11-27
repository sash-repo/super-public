# iljabook

**The algebraic language of differential equations**

I. R. Miklaszewski, edited by D. Leites (Seminar on Supersymmetries). Status: in preparation.

The scripts carry out the computations of the volume: the Statements of Chapter 4A
(Supersymmetries of differential equations) and of Section 5.A (odd metrics), and every other
claim that the text says is proved by computation. Exact arithmetic over Q throughout
(`fractions.Fraction`, sympy, and python-flint, also through `../lib/exactlin.py`); the odd
metrics use random rational coefficients with fixed seeds.

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r iljabook/requirements.txt
cd iljabook
../venv/bin/python table2.py
cd sdkp
../../venv/bin/python symm1.py 0 3 2 1     # writes sols_s0_c3_21.pkl (about 1 minute)
../../venv/bin/python symm1.py 1 3 2 1     # writes sols_s1_c3_21.pkl (about 1 minute)
../../venv/bin/python odd16.py
```

Run each script from inside its own folder (`iljabook/`, `sdkp/`, `skdv/`, `oddmetric/`). No
arguments are needed except where stated. The two runs of `symm1.py` must come before
`odd16.py`, `even17.py`, `joint.py` and `reparam.py`, which read the files they write; in
`oddmetric/`, `oddlaw.py psi2` must come before `lawcheck.py`, which reads the file it
writes. No other script writes a file. Run times on a laptop are given where they exceed a few
seconds.

## Claims of the text and the scripts behind them

Numbers are those of the volume. Lines marked "control" test a perturbed identity; they must
fail as stated.

### Chapter 4A

| Claim in the text | Script (folder) | Expected output |
|---|---|---|
| Statement 4A.6.1, Table (4A.11): for the four equations and the coefficients of degree at most 1 and at most 2, the dimensions "a of b" | `table2.py` (about 7 s) | under `degree <= 1`: `u''=0  0 of 4 \| 7 of 16 \| 12 of 12`, `u'''=0  0 of 5 \| 8 of 25 \| 20 of 20`, `u_x=u_y=u  0 of 8 \| 6 of 28 \| 19 of 32 \| 12 of 12`, `u_xx=u_xy=u_yy=0  0 of 12 \| 16 of 54 \| 46 of 72 \| 30 of 30`; under `degree <= 2`: `0 of 10 \| 14 of 40 \| 30 of 30`, `0 of 15 \| 21 of 75 \| 60 of 60`, `0 of 20 \| 15 of 70 \| 47 of 80 \| 30 of 30`, `0 of 42 \| 44 of 189 \| 151 of 252 \| 105 of 105` |
| Remark 4A.6.2: [X,Q] in K Q adds one dimension in degree 0; [X,Q] in O Q gives 0, 51, 193 in degrees -1, 0, 1; the three variants coincide in the top degree | `table2.py` | `line [0, 45, 151, 105]`, `module [0, 51, 193, 105]` |
| Proposition 4A.8.6 (the super KdV family is a coadjoint action): [K_F,K_G] = K_{{F,G}_K}, {F,G}_K = -1/2 {F,G}; items 2), 3); the coadjoint action at (U,1), not at (U,-1) | `coad.py` (`skdv/`) | 19 lines, every one `True` except the four controls and the line `... at (U,-1)`, which print `False`; `checks run: 19` |
| Proposition 4A.8.7, items 1)-3): system (4A.16); the numbers of odd factors; system (4A.17); item 4) (shown as well); item 5), first claim: for every (a,b) != (0,0), U -> D_i U is not a symmetry | `skdv_checks.py` (`skdv/`) | `N=3 item 1) ... True`; under `item 2)` the line `a {...}` has `'c12': [1]`, the line `b {...}` has `'c12': [1, 3]`, `'psi': [1]`, `'u': [0, 2]`, `'J1': [0, 2]`; `item 3) ... True`; `item 4) ... True`; `N=3 item 5) for every (a,b) != (0,0), U -> D_i U is not a symmetry (i = 1, 2, 3): True` |
| Remark 4A.8.8 (N = 2): the bosonic sector; the equations for f_1, f_2 linear in the fermions; the terms (2b-a)(f_1f_2)_x and 3a(f_1f_1xx + f_2f_2xx) - 12b(uf_1f_2)_x; Xi_1 on the unknowns; f_1 = u_x, f_2 = -m solve the equations for the fermions | `n2.py` (`skdv/`) | every line `True` |
| Theorem 4A.9.3 on KdV (a check of the formulas): d_psi, iota_s, L_s are symmetries of Pi T E; the brackets; Xi_s^2 = L_s | `oddtan_checks.py` (`skdv/`) | every line `True` |
| Statement 4A.10.2.1: for the flows t_2, t_4, t_6 of hierarchy (4A.20) (`r1`) and of the hierarchy of Manin and Radul (`MR`), the leading part of every equation for U_1, ..., U_5 has the order of its left-hand side (U_9 occurs in the flow t_6 of U_5); the leading parts for U_i^0 and U_i^1 are the components of one superfield equation | `limit.py` (`sdkp/`) | 30 lines, each with `OK ; ... OK superfield: True`; the line `r1 t6 U50: full idx [1, 2, 3, 4, 5, 6, 7, 8, 9] ...` |
| Statement 4A.10.2.2: system (4A.21) | `system.py` (`sdkp/`) | the four lines `U1_y = ...`, `U1_t - 3/2 U3_y = ...`, `U2_y = ...`, `U2_t - 3/2 U4_y = ...`, term by term the equations (4A.21) (`DUi` = D U_i, `'` = d_x) |
| Statement 4A.10.3.1, item 1): Xi_h, Xi'_h, A_a, B_b, T_f, S_f are symmetries for arbitrary functions of t; item 2): the three brackets (hence Corollaries 4A.10.3.2, 4A.10.3.3) | `verify_brackets.py` (`sdkp/`) | every line `True`, among them `[Xi_h,Xi_g] = T_{2hg} - 1/3 S_{(hg)'}: True`, `[Xi_h,Xi'_g] = T_{hg} - 1/3 S_{h'g}: True`, `[Xi'_h,Xi'_g] = 0: True`; `perturbed ... fails: True` |
| Statement 4A.10.3.1, item 3): the odd symmetries of the ansatz span a space of dimension 16, the sums of Xi_h, Xi'_h, A_a, B_b with h, a, b of degree at most 3 | `symm1.py 1 3 2 1`, then `odd16.py` (`sdkp/`) | `symm1.py`: `sigma=1: 14880 ansatz terms in 85 blocks, largest 947`, `16 solutions`; `odd16.py`: `census: odd solutions 16` and four lines `True` |
| Statement 4A.10.3.1, item 4): the even symmetries of the ansatz span a space of dimension 17: T_f, S_f, the families f(t) d_t + ... and g(t) d_y + 2/3 g'(t) y d_x + ... (f, g of degree at most 3), one more scaling | `symm1.py 0 3 2 1`, then `even17.py` (`sdkp/`) | `symm1.py`: `sigma=0: 14820 ansatz terms in 78 blocks, largest 929`, `17 solutions`; `even17.py`: `census: even solutions 17`, `rank of the map to (f, g, that constant): 9 of 9`, all other lines `True` |
| Proposition 4A.11.1.2, item 1): the conditions, the reduced system and its compatibility | `invariant.py`, case `Xi'_1` (`sdkp/`) | `U10_y = 2*U30x`, `U20_y = 2*U20*U20x + 2*U40x`, `U30_y = -2*U40*U10x + 2/3*U10t`, `U40_y = -2*U20*U40x + -2*U20^2*U20x + 2/3*U20t + -2*U20x*U40`, then three lines `True` |
| Proposition 4A.11.1.2, items 2), 3): the dispersionless modified KP equation, equation (4A.22), the linearization at v | `kzv.py` | five lines `True`, `checks run: 5` |
| Proposition 4A.11.2.1, items 1), 2): R_f is a symmetry for every f(t); [R_f,Xi_h] = -Xi_{fh'-f'h/6}, [R_f,Xi'_h] = -Xi'_{fh'-f'h/6}, [R_f,A_a] = -A_{fa'+f'a/6}, [R_f,B_b] = -B_{fb'+f'b/2}; for f of degree at most 3, R_f is in the family f(t) d_t + ... of Statement 4A.10.3.1, item 4) | `reparam.py` (`sdkp/`) | every line `True` (the controls with a wrong weight print `... fails: True`) |
| Proposition 4A.11.2.2, item 1): the conditions, the reduced system (as printed) and its compatibility | `invariant.py`, case `h^{-1} Xi'_h` (`sdkp/`) | `Theta vanishes on the conditions: True`, `the conditions are compatible with the system: True`, `the reduced system is the one printed ...: True` |
| Proposition 4A.11.2.2, item 2), first claim: the time reparametrization with the Schwarzian derivative, tau arbitrary | `timechange.py` (`sdkp/`) | four lines `residual = 0`, `the functions solve the reduced system for every tau: True`, `control (v without the Schwarzian term) fails: True` |
| Proposition 4A.11.2.3, items 2), 3): the solutions invariant under Xi'_1 + A_a + B_b, under Xi_1 - Xi'_1 and under Xi_1 - Xi'_1 + A_a + B_b | `invariant.py`, the last three cases (`sdkp/`) | for each case three lines `True`; last line `all checks passed: True` |
| Proposition 4A.11.2.4, item 2): modulo K Xi'_1, the brackets [E, Xi'_1] span the fields Xi'_h + A_a + B_b, h in span(t, t^2), deg a <= 2, deg b <= 3; the fields E with [E, Xi'_1] in K Xi'_1 are spanned by T_f - S_{f'/3}, d_y, d_t and the two scalings | `joint.py` (`sdkp/`) | `basis fields occurring modulo Xi'_1: ['A(0)', 'A(1)', 'A(2)', 'B(0)', 'B(1)', 'B(2)', 'B(3)', "Xi'(1)", "Xi'(2)"]`, `rank of the image modulo Xi'_1: 9 (expected 9)`, all other lines `True` |

### Section 5.A

| Claim in the text | Script (folder) | Expected output |
|---|---|---|
| Statement 5.A.7.4, item 1): in superdimension (2\|2), the Ricci tensor of an odd metric is anti-symmetric; checked for an odd metric whose coefficients are the constant normal form plus every quadratic monomial with a random rational coefficient (fixed seed) | `ricgen.py` (`oddmetric/`, about 5 min) | `generic odd metric: Ric anti-symmetric: True  symmetric part zero; Ric nonzero: True` |
| The same for the explicit odd metric of `oddexample.py` | `oddexample.py` (`oddmetric/`, about 40 s) | `Ric anti-symmetric: True  symmetric: False` |
| Statement 5.A.7.4, item 2), superdimension (2\|2): the Ricci tensor of a conformally flat odd metric g = lambda^2 g_0 vanishes | `ricflat.py` (`oddmetric/`, about 1 min) | `conformally flat Ric anti: True sym: True Ric nonzero: False ...` |
| Statement 5.A.7.4, item 2), superdimension (3\|3) | `ric33.py cflat` (`oddmetric/`, about 1 min) | `(3\|3) conformally flat: Ric zero: True` |
| Proposition 5.A.8.1 (the signs of connection (A.21) are the only ones of the form (-1)^{e p(a) + f p(s) + o} for which the connection preserves the scalar product of L_g; a check of the proof) and Statement 5.A.8.2: for the explicit odd metric and conformal factor lambda, psi is an isomorphism of bundles with connection if and only if law (A.22) holds, the odd function kappa staying free | `oddlaw.py psi2` (about 6 min; writes `oddlaw.pkl`), then `lawcheck.py` (`oddmetric/`) | `oddlaw.py`: `compatible sign variants: [(1, 0, 1, 1, 1, 0)]`, `solutions: 1`, `Q depends on beta: []`; `lawcheck.py`: `entries checked: 16 mismatches: 0` |
| Statement 5.A.8.3: no linear combination of the 73 contractions of the curvature tensor with g and g^{-1} (and a multiple of g) equals -Hess phi + d phi d phi for the conformally flat odd metrics on R^{2\|2}, in either sign convention | `schouten.py` (`oddmetric/`) | `contractions in the fit: 73`, `plain solvable: False []`, `convention (-1)^{p(a)} solvable: False []` |
| Control for the previous line: for an even metric the same fit succeeds | `schouten_even.py` (`oddmetric/`) | `even control: fit ... [{k1: -1, ...}]` (a solution exists) |
| Statement 5.A.8.4: for the grading of pe(3) (n = 2) by ad_h, in degrees k = 1 and k = 2 the image of the Spencer map has a cpe(2)-invariant complement; for k = 2 the map is injective, for k = 1 its kernel is 1-dimensional (the line g_2); control: o(4) | `normal.py 2 1`, `normal.py 2 2` (`oddmetric/`) | for `pe(3) [odd conformal]`, `degree k=1`: `dim C1=16, rank d=15, dim ker d=1`, `d L d = d solvable: True  (normalization EXISTS)`; `degree k=2`: `dim C1=4, rank d=4, dim ker d=0`, `d L d = d solvable: True  (normalization EXISTS)`; the control `o(4)` also `EXISTS` |

## Files

- `table2.py` — Statement 4A.6.1 and Remark 4A.6.2: the centralizer of Q, graded, computed
  from the Cartan supervariety of each equation (sympy for the algebra, python-flint for ranks).
- `sdkp/spdo.py` — super pseudodifferential operators in D = d_theta + theta d_x, D^2 = d_x,
  with superfield coefficients in components; self-tests (`python spdo.py`).
- `sdkp/mr.py`, `sdkp/flows.py`, `sdkp/disp.py` — the Lax operator, its powers, the flows and
  their leading parts under the scaling (`python disp.py 11` prints them for U_1, ..., U_8).
- `sdkp/limit.py`, `sdkp/system.py` — Statements 4A.10.2.1, 4A.10.2.2.
- `sdkp/jets.py` — jets of the components, total derivatives, system (4A.21) solved for the
  derivatives in y.
- `sdkp/symm.py`, `sdkp/symm1.py` — the symmetry condition and the ansatz of Statement
  4A.10.3.1, items 3), 4) (`symm1.py` arguments: parity, degree in x, y, t, degree of eta,
  degree of alpha and beta).
- `sdkp/brackets.py` — brackets of evolutionary fields; the census solutions as fields.
- `sdkp/fields.py` — Xi_h, Xi'_h, A_a, B_b, T_f, S_f for h, a, b, f = t^k.
- `sdkp/general.py` — Xi_h and Xi'_h for an arbitrary function h(t) (its jets are free
  variables); `sdkp/verify_brackets.py` — Statement 4A.10.3.1, items 1), 2).
- `sdkp/odd16.py`, `sdkp/even17.py` — Statement 4A.10.3.1, items 3), 4).
- `sdkp/invariant.py`, `sdkp/timechange.py`, `sdkp/reparam.py`, `sdkp/joint.py` — Section
  4A.11.
- `skdv/sf.py`, `skdv/model.py`, `skdv/evo.py` — N-extended superfields over jets, component
  systems, evolutionary fields; `skdv/coad.py`, `skdv/skdv_checks.py`, `skdv/n2.py`,
  `skdv/oddtan_checks.py` — Sections 4A.8, 4A.9.
- `kzv.py` — Proposition 4A.11.1.2, items 2), 3).
- `oddmetric/stractor.py` — functions on R^{n|k} with values in the Grassmann algebra (sympy),
  the Levi-Civita connection and the curvature of an even metric; run alone, it checks the sign
  conventions on an even metric in superdimension (1|2) (`signs B` gives `75/75`).
- `oddmetric/oddgeom.py` — the Levi-Civita connection, the curvature and the Ricci tensor of an
  odd metric, with the conventions of Section 5.A.
- `oddmetric/oddexample.py` — an explicit odd metric on R^{2|2} and the contractions of its
  curvature tensor.
- `oddmetric/ricgen.py`, `oddmetric/ricflat.py`, `oddmetric/ric33.py` — Statement 5.A.7.4.
- `oddmetric/oddlaw.py`, `oddmetric/lawcheck.py` — Proposition 5.A.8.1, Statement 5.A.8.2.
- `oddmetric/schouten.py`, `oddmetric/schouten_even.py` — Statement 5.A.8.3.
- `oddmetric/normal.py` — Statement 5.A.8.4 (arguments: n, then the degree k), exact over Q
  through `../../lib/exactlin.py`.
