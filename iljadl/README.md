# iljadl

**Supersymmetries of differential equations; semiholonomic jets**

D. Leites, I. R. Miklaszewski. Status: in preparation.

The scripts carry out the computations of the text: the coadjoint action behind the N=3 super
KdV family and its components (Proposition 4.1.4, Proposition 4.1.5); the super dispersionless
KP system, its dispersionless limit and its symmetries of order at most 1 (Statement 5.1.2,
Statement 5.1.4, Corollary 5.1.5); its solutions invariant under the odd symmetry Xi'_1
(Proposition 5.2.1). They also check the displayed formulas of the examples: the N=1 super KdV
equation (Example 4.1.2), supersymmetric quantum mechanics with the Darboux and Crum
transformations (Subsec. 4.2), the odd tangent covering of KdV (Theorem 4.2.2), the Frobenius
systems (Examples 6.4, 6.7), the curvature of an operator (Example 7.1.3) and of a derivation
(Theorem 7.2.1). Exact arithmetic over Q throughout (`fractions.Fraction`, sympy, and
python-flint through `../lib/exactlin.py`).

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r iljadl/requirements.txt
cd iljadl/sdkp
../../venv/bin/python symm1.py 0 3 2 1     # writes sols_s0_c3_21.pkl (about 1 minute)
../../venv/bin/python symm1.py 1 3 2 1     # writes sols_s1_c3_21.pkl (about 1 minute)
../../venv/bin/python system.py
```

Run each script from inside its own folder (`sdkp/`, `skdv/`, or `iljadl/` itself). No
arguments are needed except for `symm1.py`. The two runs of `symm1.py` must come before
`brackets.py` and `odd16.py`, which read the files they write; no other script writes a file.
Every run finishes in under 3 minutes on a laptop (the longest is `sym_checks.py`, about
3 minutes); the others take seconds.

## Claims of the text and the scripts behind them

Numbers are those of the text. Lines marked "control" are perturbed versions of an identity;
they must print `False`.

| Claim in the text | Script (folder) | Expected output |
|---|---|---|
| Proposition 4.1.4 (the N=3 super KdV family is a coadjoint action): the bracket [K_F,K_G] = K_{{F,G}_K} and {F,G}_K = -1/2 {F,G} (item 1); [V(F),V(G)] = V({F,G}) and the invariance of the pairing (item 2); omega anti-symmetric, <D_1D_2D_3F,G> = -omega(F,G) and identity (13) (item 3); the coadjoint action at (U,1), not at (U,-1); the sign of the sum in the convention with square +d_x | `coad.py` (`skdv/`) | 19 lines, every one `True` except the four controls and the line `... at (U,-1)`, which print `False`; `checks run: 19` |
| Proposition 4.1.5, items 1)-3): the bosonic sector; the numbers of odd factors (bilinear terms in the equations for u and J; cubic terms only in those for chi_ij, and only for b != 0); system (15) at J = 0; item 4): Xi_1 on the unknowns, Xi_i symmetries, [Xi_i,Xi_j] = -2 delta_ij d_x | `skdv_checks.py` (`skdv/`) | `N=3 item 1) ... True`; under `item 2)` the line `a {...}` has `'c12': [1]` and the line `b {...}` has `'c12': [1, 3]`, `'psi': [1]`, `'u': [0, 2]`, `'J1': [0, 2]`; `item 3) ... True`; `item 4) Xi_1 on the unknowns: {'psi': '1*J1', 'J1': '-1*psix', 'J2': '-1*c12', 'J3': '-1*c13', 'c12': '1*J2x', 'c13': '1*J3x', 'c23': '1*u', 'u': '-1*c23x'}`; all other lines `True` |
| Example 4.1.2: Xi is a symmetry of the N=1 super KdV equation and Xi^2 = T_x | `skdv_checks.py` (`skdv/`) | `N=1: Xi symmetry: True`, `N=1: Xi^2 = d_x: True`, `N=1: control (f->2u) fails: True` |
| Subsec. 4.2: A^dag A = H_-, A A^dag = H_+, H_+ A = A H_-; the one-soliton and the rational solution as Darboux images of the vacuum; the Darboux image of an eigenfunction; the 2-soliton (Crum) | `sym_checks.py` | the lines from `A^dag A = H_-: True` to `2-soliton (Crum) solves KdV: True`, all `True` |
| Theorem 4.2.2 on KdV: d_psi, iota_s, L_s are symmetries of Pi T E for s = u_x, K, K_5; d_psi^2 = 0; the brackets; Xi_s^2 = L_s | `oddtan_checks.py` (`skdv/`) | every line `True` |
| Statement 5.1.2, item 1): for the flows t_2, t_4, t_6 of hierarchy (17) (`r1`) and of the hierarchy of Manin and Radul (`MR`), the leading part of every equation for U_1, ..., U_5 has the order of its left-hand side, and the leading parts of the equations for U_i^0 and U_i^1 are the components of one superfield equation | `limit.py` (`sdkp/`) | 30 lines, each ending `OK ; ... OK superfield: True`; no `!!` |
| Statement 5.1.2, item 2): system (18) | `system.py` (`sdkp/`) | the four lines `U1_y = ...`, `U1_t - 3/2 U3_y = ...`, `U2_y = ...`, `U2_t - 3/2 U4_y = ...`, term by term the four equations (18) (`DUi` = cD U_i, `'` = d_x) |
| Statement 5.1.4, item 1): Xi_h, Xi'_h, A_a, B_b, T_f, S_f are symmetries for arbitrary functions h, a, b, f of t | `verify_brackets.py` (`sdkp/`) | `Xi_h is a symmetry for arbitrary h(t): True 0`, `Xi'_h ...: True 0`, `perturbed Xi'_h fails: True`, `perturbed Xi_h fails: True`, `A_a symmetry: True  B_b symmetry: True`, `T_f symmetry: True  S_f symmetry: True` |
| Statement 5.1.4, item 2): the three brackets; hence Corollary 5.1.5 | `verify_brackets.py` (`sdkp/`) | `[Xi_h,Xi_g] = T_{2hg} - 1/3 S_{(hg)'}: True`, `[Xi_h,Xi'_g] = T_{hg} - 1/3 S_{h'g}: True`, `[Xi'_h,Xi'_g] = 0: True` |
| Statement 5.1.4, item 3): the odd symmetries of the ansatz form a space of dimension 16, spanned by Xi_h, Xi'_g, A_a, B_b with h, g, a, b of degree at most 3 | `symm1.py 1 3 2 1`, then `odd16.py` (`sdkp/`) | `symm1.py`: `sigma=1: 14880 ansatz terms in 85 blocks, largest 947`, `16 solutions`; `odd16.py`: `census: odd solutions 16` and four lines `True` |
| Statement 5.1.4 (even part of the ansatz, for `brackets.py` and `odd16.py`) | `symm1.py 0 3 2 1` (`sdkp/`) | `sigma=0: 14820 ansatz terms in 78 blocks, largest 929`, `17 solutions` |
| Proposition 5.2.1, item 1): the reduced system for u, v, psi, chi and its compatibility with system (18) | `reduce.py` (`sdkp/`) | `U10_y = 2*U30x`, `U20_y = 2*U20*U20x + 2*U40x`, `U30_y = -2*U40*U10x + 2/3*U10t`, `U40_y = -2*U20*U40x + -2*U20^2*U20x + 2/3*U20t + -2*U20x*U40`; under `consistency of the constraints` four lines `0` |
| Proposition 5.2.1, items 2), 3): the dispersionless modified KP equation, equation (19), the linearization at v | `kzv.py`; `sym_checks.py` | `kzv.py`: five lines `True`, `checks run: 5`; `sym_checks.py`: the first four lines `True` |
| Examples 6.4, 6.7: [X,Z] = kappa_1 Z, [Y,Z] = kappa_2 Z; alpha ^ d alpha = -c dx ^ dy ^ du; d omega(X,Z) = -kappa_1, d omega(Y,Z) = -kappa_2; the three systems of item 4); the even fields commuting with X and Y for u_x = 0, u_y = x | `frob_kappa.py` | `... True`; `alpha^dalpha / (dx^dy^du) + c = 0`; `d(x al2 + al3)(X,Y) = 0`; `u_x=0,u_y=x: c=1, k1=0, k2=0`; `u_x=u,u_y=xu: c=u, k1=0, k2=0;  v=-x + log(u): X(v)=0, Y(v)=x`; `a=1,b=u^2: c=2*u, k1=1/u, k2=-u`; `general V commuting with X,Y (deg<=4 ansatz): [c65, c31, c62 + c65*y]`; `d(x al2+al3)(X,Z) + k = 0`, `d(x al2+al3)(Y,Z) + k = 0` |
| Subsec. 7.1: the curvature of a connection, kappa_nabla = d_i G_j - d_j G_i + [G_i,G_j], and kappa_nabla(u) = -c dx ^ dy for u_x = alpha u, u_y = beta u; Example 7.1.3 for n = 2, 3 | `sym_checks.py` | the lines `kappa_nabla ...` and `Riemann: ...`, `rank check ...`, all `True` |
| Theorem 7.2.1, items 3), 4): the curvature of a derivation for a generic quadratic field in 3 variables; the eigenvalues lambda_i + lambda_j on Lambda^2 | `sym_checks.py` | `curvature of a derivation, item 3) (n=3, generic quadratic field): True`, `eigenvalues of Lambda^2 T = lam_i+lam_j: True` |

## Files

- `sdkp/spdo.py` — super pseudodifferential operators in D = d_theta + theta d_x, D^2 = d_x,
  with superfield coefficients in components; supercommutative jet polynomials; self-tests
  (`python spdo.py`).
- `sdkp/mr.py`, `sdkp/flows.py` — the Lax operator L = D + sum U_i D^{1-i}, its powers and the
  flows of the hierarchy.
- `sdkp/disp.py` — the scaling weights of the dispersionless limit and the leading parts of the
  flows (`python disp.py 11` prints them for U_1, ..., U_8).
- `sdkp/limit.py` — Statement 5.1.2, item 1).
- `sdkp/system.py` — Statement 5.1.2, item 2): the leading parts fitted to superfield
  expressions in cD, d_x and products.
- `sdkp/jets.py` — jets of the components, total derivatives, system (18) solved for the
  derivatives in y.
- `sdkp/symm.py` — the symmetry condition D_y(phi) = phi(R) and the ansatz solved block by
  block in the two scaling weights; `sdkp/symm1.py` — the ansatz of Statement 5.1.4, item 3)
  (arguments: parity, degree in x, y, t, degree of eta, degree of alpha and beta).
- `sdkp/brackets.py` — the brackets of evolutionary fields; the census solutions as fields.
- `sdkp/general.py` — Xi_h and Xi'_h for an arbitrary function h(t) (h and its derivatives are
  free jet variables); `sdkp/verify_brackets.py` — Statement 5.1.4, items 1), 2).
- `sdkp/odd16.py` — Statement 5.1.4, item 3).
- `sdkp/reduce.py` — Proposition 5.2.1, item 1).
- `skdv/sf.py` — N-extended superfields over jets of the components; `skdv/evo.py` —
  evolutionary fields on supercommutative jet polynomials.
- `skdv/coad.py`, `skdv/skdv_checks.py`, `skdv/oddtan_checks.py` — Section 4.
- `kzv.py`, `sym_checks.py`, `frob_kappa.py` — sympy checks of Sections 4-7.
