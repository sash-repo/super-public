# stl

**SUSY between Schrödinger and KdV operators**

D. Leites, I. Shchepochkina. Status: in preparation.

The scripts check the computations of the text: the cocycles of the contact (Neveu–Schwarz) and
Möbius (Ramond) stringy superalgebras built from the invariant operators D_theta and the
anti-self-adjointness of the operators L_n (Sect. 2.8, Lemma 5.8, Remark 5.9); the central
extensions of k^L(1|4), k^L'(1|4) and k^M(1|4) by the window method (Statement 2.8.11); the super
Miura transformations (Subsec. 5.10); Radul's Bott cocycle on the group (Statement 6.2.6); the
invariant of the coadjoint orbits of k^L(1|1) and k^M(1|1) (Statement 7.1.1); the Euler equation
on ns(1)^* (Statement 9.4.1); the sl(lambda)-hierarchy and its Adler bracket (Sect. 10); the
generation of Feigin's Lie algebra A_c of the Virasoro vertex algebra (Subsec. 11.7); and the
eigenvalue problem of St_1 = D_1 D_theta (Sect. 12). Exact arithmetic throughout (integers,
`fractions.Fraction`, sympy, and python-flint, directly or through `../lib/exactlin.py`).

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r stl/requirements.txt
cd stl
../venv/bin/python selfadjeq.py
```

Every script is run from inside this folder. No arguments are needed except where stated. Every
run below finishes in under ten minutes on a laptop (the longest are `selfadj.py`, 3 to 9
minutes, and `nsgrass.py` and `h2k14.py 3 7`, about 2 to 5 minutes each); none writes a file.

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Statement 2.8.3, Neveu–Schwarz series: identity (75) for L_n = D_1^{3-n} D_theta1 ... D_thetan of formula (70), n <= 3 | `selfadjeq.py` | for n = 0, 1, 2, 3: `121 / 484 / 1936 / 7744 pairs, 6 / 14 / 32 / 70 non-zero values, 0 failures`; `TOTAL failures: 0` |
| Statement 2.8.3, Ramond series n = 1 (realization (i), the operator K_zeta K_1^2, zeta = x^{1/2} theta, Res^M of (153)) | `selfadj.py M` (about 2 minutes; `selfadj.py` runs both series, about 4-9 minutes) | `--> n=1: M works for m in [2]` (the only power m in -1..4 giving a super anti-symmetric 2-cocycle) |
| Lemma 5.8 (the sign of H_f, [K_f,K_g] = K_{f,g}, eps = (-1)^n); formula (154): the D_theta, not the K_theta, enter the cocycles (69) | `kdops.py` | `the sign in H_f is FORCED ...`; for n = 1, 2, 3 `... exactly one eps does it: eps = -1 / 1 / -1` and `and it is not a cocycle` (the K-form); `18 checks, 18 pass, 0 fail` |
| Statement 2.8.11 (central extensions of k^L(1\|4) and k^M(1\|4)): (2), (3) of Table (77) vanish on k^M(1\|4), are independent non-trivial cocycles on k^L(1\|4), dim H^2(k^L(1\|4)) = 2; and dim H^2(k^L'(1\|4)) = 3 | `h2k14.py 3 7` (2 to 5 minutes) | `k^L'(1\|4) {... 'H2(inner)': 3, '(2)': ('cocycle', 'nontrivial'), '(3)': ('cocycle', 'nontrivial') ...}`; `k^L(1\|4) {... 'H2(inner)': 2, ... 'rank of tests mod B (inner)': 2 ...}`; `k^M(1\|4) {... 'tests identically zero on window': [True, True]}` |
| Table (77), the functionals (1), (2), (3) on k^L'(1\|4) and k^L(1\|4) | `k14.py 1` | `Jacobi failures ... : 0`; `(2)` and `(3)`: 0 failures on both; `(1) on k^L'(1\|4): (triples, failures, undefined) = (11857, 432, 0)` (see the note below the table) |
| Proposition 5.10.1 (the cocycles c_n = St_n with values in F_{1-n/2}, n = 0, 1, 2) and the classical Hamiltonian property of the Miura map (n = 0, after Statement 5.10.2) | `nsgrass.py` | six lines `PASS`, among them `n=3: D1D2D3(H) is a 1-cocycle in F_{1/2}` |
| Statement 5.10.2 (the super Miura map is a Poisson map for n = 1, c = -1/2) | `miura1.py` | `even h diff zero: True`, `odd h diff zero: True` |
| Statement 5.10.3 (no non-trivial weight-0 1-cocycle of k^L(1\|3) with values in F_{-1/2}; D_theta1 D_theta2 D_theta3 is one with values in F_{1/2}; d/dx : F_{-1/2} -> F_{1/2} is not invariant) | `h1kn.py`; `miura3.py`; `nsgrass.py` | `n=3 lam=-1/2: ... [(0, 0), (3, 3)]` (3 cocycles, all 3 coboundaries), `n=3 lam=1/2: ... [(0, 0), (1, 0)]`; `parity 0 / 1 d intertwines ...: False`, `n=0, d: F_0->F_1: True` |
| Statement 6.2.6 (Radul's contact criterion: conformal linear part, beta_1 = -sigma_2', beta_2 = sigma_1'; the frame matrix, its inverse, the integral in 2 pi Z; the matrix logarithm) | `bott2.py` | `the solution is unique and is be_1 = -al_2', be_2 = +al_1'`; `14 checks, 0 failures` |
| Statement 7.1.1 (the invariant I(u, xi) of the coadjoint orbits) | `casimir.py` | `signs 1 1 variation is a total derivative: True` |
| Statement 9.4.1 (the Euler equation (268) on ns(1)^* for H = Res(u^2/2 + 2 xi_x xi)) | `kupereuler.py`; cross-check `kupereuler2.py` | under `odd cocycle coefficient 1`: `u_t = -c*u''' + 3*u*u' + e1e2*(6*(-a*b'' + b*a''))`, `xi_t(e1-part) = -4*c*a''' + 3*a*u'/2 + 3*u*a'`; `flow with phi=-4xi' conserves H~=int(u^2/2+2xi'xi): True`; `kupereuler2.py` prints the same system with t -> -t |
| Statement 10.4.2 (the flows (274) of the sl(lambda)-hierarchy) | `hier.py 7` | `d u2/dt2 = -lam*u2_2 + 2*u2_2 + 2*u3_1`, `d u3/dt2 = -lam**2*u2_3/3 + lam*u2_3 - 2*u2_0*u2_1 - 2*u2_3/3 + u3_2 + 2*u4_1 + 4*u2_0*u2_1/lam`, `d u2/dt3 = lam**2*u2_3/4 + lam*(-3*u2_3/2 - 3*u3_2/2) - 3*u2_0*u2_1 + 9*u2_3/4 + 9*u3_2/2 + 3*u4_1 + 9*u2_0*u2_1/lam` |
| Statement 10.6.2 (the entries P_11, P_12, P_21, P_22 of the Adler bracket (277) and the Virasoro bracket (278)) | `adler.py`; `checks.py` (item 7) | `(1, 1) {1: -lam}`, `(1, 2) {... 2: lam*(lam - 1)/2}`, `(2, 2) {... 1: ... + 2*u2_0, 3: lam*(lam - 1)*(2*lam - 1)/6}`; `7. Dirac-reduced {u2,u2} = ...;  c = lam*(lam - 1)*(lam + 1)/12` |
| Theorem 10.4.3 (the Gelfand–Dickey locus u_k = 0, k > n, is invariant at lambda = n; KdV (275) at n = 2) and the KP flows | `checks.py` | `2. u_k=0 (k>n) invariant at lam=n, n=2..5: 33`, `3a. lam=2: t2 trivial, t3 = KdV ...`, `4. q = L^{1/lam} evolves by KP (m=2,3): 7 coefficients` |
| Theorem 11.7.3, the "if" parts of headings 1)-3) (gcd of the maximal minors over Q(c) for K_6, K_8, K_9) | `w6sym.py`; `wnsym.py 8 4 8`; `wnsym.py 9 4 9` | `gcd of 2x2 minors: (c + 2)*(5*c + 22)/5`; `dim K_8 = 3  rank over Q(c): 3`, `gcd ... (sampled): (c + 2)**2*(2*c - 1)*(5*c + 22)**2/50`; `dim K_9 = 1  rank over Q(c): 1`, `gcd ... (sampled): c + 2` |
| Problem 11.7.6 (products of length <= 4 reach only a 2-dimensional subspace of the 4-dimensional K_10) | `wnsym.py 10 4 10` | `dim K_10 = 4  rank over Q(c): 2` |
| Statement 11.7.4, heading 1) (c = 7/3, -64/5, weights <= 12) | `conf.py 7/3 12`, `conf.py -64/5 12` (about a minute each) | `... 10:4/4 11:2/2 12:7/7` (reached = dim K_n for every n) |
| Statement 11.7.4, heading 2) (nine central charges, weights <= 10) | `conf.py c 10` for c in 0, 1, 25, 4/5, 7/10, -3/5, -46/3, -68/7, -114/7 | each line `... 6:2/2 7:0/0 8:3/3 9:1/1 10:4/4` |
| Statement 11.7.4, heading 3) (the fiber c = infinity, weights <= 10) | `run89c.py 4 3` | `classical K_6 (dim 2) reached: 2`, `K_8 (dim 3) reached: 3`, `K_9 (dim 1) reached: 1`, `K_10 (dim 4) reached: 4` |
| Statement 12.3.2 (dim 3, 6, 12 of the even solutions of St_1 f = zeta_1 f; all in Ker D_1^3) | `eig.py` (part 2) | `dim_Q = 3 ...`, `6 ...`, `12 (deg_x <= 4), 12 (deg_x <= 6), 12 (deg_x <= 8)`; `all 6 solutions lie in ker(d_x^3) = ker(St_1)^2` |
| Statement 12.9 and Corollary 12.9.2 (D_theta, D_1 anti-self-adjoint, St_1 self-adjoint, L_1 = St_1 D_1 and St_1^2 = D_1^3 anti-self-adjoint) | `eig.py` (part 7) | `sign -1`, `sign -1`, `sign 1`, `sign -1`, `sign -1`; `==== 18/18 checks pass ====` |

**Note on Table (77), row (1).** In the model of `k14.py` (bracket of Lemma 5.8, theta-basis,
D_theta = theta d_x + d_theta, D_1^{-1} x^a = x^{a+1}/(a+1) applied last as in Proposition
2.8.10), the functional (1), (-1)^{p(f)} Res(f Delta_IS(g)), fails the cocycle identity on
432 of the 11857 weight-0 triples of the window on k^L'(1|4); none of the eight sign variants
(prefactor (-1)^{p(f)}, (-1)^{p(g)}, sign of the theta d_x term) passes either. The space
H^2(k^L'(1|4)) is nevertheless 3-dimensional (`h2k14.py`), so a third cocycle exists; it is not
of this shape in this convention.

## Files

- `selfadjeq.py` — identity (75) for the Neveu–Schwarz L_n (uses the Grassmann algebra of `kdops.py`).
- `kdops.py` — the contact fields K_f and the invariant operators D on the superline; Lemma 5.8, cocycles (69).
- `selfadj.py` — the scan of the family K_theta1 ... K_thetan K_1^m, series L and M (in the series
  L it works with the contact fields K_theta, not with the D_theta of formula (70); only its M
  output is cited above).
- `stringy.py`, `k14.py`, `h2k14.py` — k^L(1|4), k^M(1|4): Table (77), Statement 2.8.11.
- `nsgrass.py` — superfields over C-points on the (1|n)-dimensional superline; `miura1.py`,
  `miura3.py`, `casimir.py`, `kupereuler.py` use it; `h1kn.py` — 1-cocycles of k(1|n) in F_lambda.
- `grass2.py`, `kupereuler2.py` — an independent computation of Statement 9.4.1.
- `bott2.py` — Statement 6.2.6.
- `psdo.py` (pseudodifferential symbols of complex order), `hier.py`, `adler.py`, `checks.py` — Sect. 10.
- `va.py` (the Virasoro vertex algebra at a numeric c), `gensym.py` (the same over Q[c]),
  `conf.py`, `w6sym.py`, `wnsym.py`, `run89c.py` — Subsec. 11.7.
- `eig.py` — Sect. 12.
