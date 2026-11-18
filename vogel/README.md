# vogel

**Vogel's approach to modular Lie (super)algebras**

D. Leites, O. Lozhechnyk. Status: in preparation.

The scripts recompute the claims of the text that rest on a computation: Vogel's point of
psl(n|n); Vogel's parameters of the exceptional modular Lie (super)algebras with Cartan matrix
in characteristics 3 and 5 (Table (9)); the fake Casimir operator of pe(n) and the odd mixed
Casimir operator of psq(n); the simplicity of m(n;1); the absence of an invariant form on
spe(n); and the invariant forms and the split Casimir operator of the vectorial Lie algebras
(Table (28)). Arithmetic is exact (integers mod p, `fractions.Fraction`, sympy, python-flint
through `../lib/exactlin.py`), except in `pslnn.py`, which works in floating point over R and
reads off eigenvalues that come out as integers.

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r vogel/requirements.txt
cd vogel
../venv/bin/python pslnn.py 2
```

Every script is run from inside this folder. `vogele.py` and `vogvec.py` also append their
output to `vogele.log` and `vogvec.log` in this folder. Running times are those of one core of an
Apple M-series laptop.

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Statement 1.1.2 (Vogel's point of psl(n\|n)), n = 2: on gl(2\|2) the Casimir operator of the supertrace form is nilpotent on the adjoint module (rank 1, square 0); on psl(2\|2) it vanishes; on S^2 it is semisimple with eigenvalues 0 (trivial submodule only), 4, -4, the last two of superdimension (24\|24) | `pslnn.py 2` (< 1 s) | `gl(2\|2): [C,ad] max 0.0  C on adjoint: rank 1  C^2 rank 0`; `S^2 dim 97  eigenvalues of C:` -4.0, 0.0, 4.0 with multiplicities 48, 1, 48 (printed as numpy scalars); `ev -4.0 dim 48 ddim = dim_ev - dim_od 0` and the same for 4.0, i.e., (24\|24) |
| Statement 1.1.2, n = 3: the same, superdimensions (144\|144) | `pslnn.py 3` (3 s) | `S^2 dim 577  eigenvalues of C:` -4.0, 0.0, 4.0 with multiplicities 288, 1, 288; `ev -4.0 dim 288 ddim = dim_ev - dim_od 0` and the same for 4.0, i.e., (144\|144) |
| Statement 2.8 and Table (9), the rows br(3), brj(2;3), g(1;6), g(3;6), g(4;3), g(8;3), g(4;6), g(6;6), el(5;3) (p = 3) and brj(2;5), el(5;5), g(2), ag(2) (p = 5): Casimir value 2t on g and on the highest weight vectors V_1, V_2 of S^2(g), whence t and (alpha, beta, gamma) mod p | `vogelp.py` (< 1 s; data in `vogelp-data.json`) | 13 blocks; every line `V1 wt ... OK; V2 wt ... OK`; e.g. `g(2)  p=5 ... t=4  (alpha,beta,gamma)=(0,3,1)`, `ag(2)  p=5 ... t=1  (alpha,beta,gamma)=(4,3,4)`, `brj(2;5) ... t=0 (alpha,beta,gamma)=(4,1,0)`, `el(5;5) ... t=0 (alpha,beta,gamma)=(2,3,0)`, and `t=0` for every p = 3 algebra |
| Table (9), row br(2;eps), p = 3, and Subsec. 2.7.1a: (alpha, beta, gamma) = (1, -eps, eps-1), t = 0 | `br2eps.py` (< 10 s) | `t -3*eps (Poly(0, ...))`, `alpha -3*eps - 2 (Poly(1, ...))`, `beta 2*eps (Poly(-eps, ...))`, `gamma 2 - 2*eps (Poly(eps - 1, ...))` |
| Table (9), rows ab(3), e(6), e(7), e(8) and the simple subquotient e(6)/z (Subsecs. 2.7.2e, 2.7.3a): the invariant forms solved for (one), the Casimir operator on g, the singular vectors of S^2(g) and Vogel's parameters | `vogele.py ab3:5 e6:5 e6/z:3 e7:5 e7:3 e8:5 e8:3` (40 s) | every algebra: `super-Jacobi ... failures 0`, `even supersymmetric invariant forms: 1`; then `ab3: t = 1, (alpha,beta,gamma) = [1, 1, -1]`, `e6 p=5: t = -2, [2, -1, 2]`, `e6/z p=3: center: dim 1 ...; t = 0, ... [-1, 0]`, `e7 p=5: t = 2, [2, 2, -2]`, `e7 p=3: t = 0, ... [-1, 1]`, `e8 p=5: t = 0, ... [2, -2]`, `e8 p=3: t = 0, ... [-1, 0]` (projective: the table prints them up to a common factor and a permutation; for t = 0 the third parameter is minus the sum of the two printed) |
| Statement 3.1.1a, 1) and 2), n = 2, 3: the fake Casimir operator Omega on pe(n) ⊗ V has the eigenvalues 0, 2, -1 with superdimensions (n\|n), (n(2n^2+1)/3 \| same), (4n(n^2-1)/3 \| same); on pe(n) ⊗ gl(n\|n) the eigenvalues 0 and ±2√2 | `fakecas2.py pe 2 3` (100 s) | `Omega on g(x)V, parity 0: [0]^2 [2]^6 [-1]^8` (same for parity 1); `Omega on g(x)gl(n\|n), parity 0: [0]^40 (-8x^0 + 1x^2)^12`; for p(3): `[0]^3 [2]^19 [-1]^32` and `[0]^180 (-8x^0 + 1x^2)^72` |
| Statement 3.1.1a, 1), n = 4 | `VONLY=1 fakecas2.py pe 4` (11 s) | `p(4): sdim g = (16\|16)`, `Omega on g(x)V, parity 0: [0]^4 [2]^44 [-1]^80` (same for parity 1) |
| Statement 3.1.2a, n = 3: the odd mixed Casimir operator of psq(3) vanishes on g; D^2 = 0 and Ker D = Im D on g ⊗ g; D(E^2) has superdimension sdim g = (8\|8), D(S^2) superdimension (56\|56) | `psqD.py 3` (6 s) | `psq(3) sdim (8\|8): odd Casimir on g: rank 0, square zero True; D on g(x)g: rank 128, D^2=0: True; dim S^2=128, E^2=128; rank D\|S^2 = 112, rank D\|E^2 = 16`; `rank D on S2_even = 56, on S2_odd = 56`, `rank D on E2_even = 8, on E2_odd = 8` |
| Statement 3.1.2a, n = 4: the same, superdimensions (15\|15) and (210\|210) | `psqD.py 4` (8 min) | `psq(4) sdim (15\|15): odd Casimir on g: rank 0, square zero True; D on g(x)g: rank 450, D^2=0: True; dim S^2=450, E^2=450; rank D\|S^2 = 420, rank D\|E^2 = 30`; `rank D on S2_even = 210, on S2_odd = 210`, `rank D on E2_even = 15, on E2_odd = 15` |
| Statement 3.2.2b: m(n;1) is simple for (n, p) = (1, 3), (1, 5), (1, 7), (2, 3), (2, 5), (3, 3) (transitive, and the ideal generated by M_1 is the whole algebra) | `msimple.py` (25 s) | six lines `p=.. n=.. dim=.. grading-element defects=0 transitive=True dim ideal(M_1)=..` with `dim ideal(M_1)` equal to `dim`: 12, 20, 28, 72, 200, 432 |
| Proposition 3.5, the case n = p = 3 proved by computation (and the cases n = 3..6, p = 3, 5, 7 as a check): spe(n) has no invariant bilinear form, even or odd | `spe.py` (3 s) | three `CONTROL sl(..) inside ...: dim inv forms=1` lines, then `n=3 p=3 spe sdim=(8\|9) dim inv forms=0 []` and `dim inv forms=0` for every other (n, p) |
| Table (28), the rows vect(1;1) ≅ sl(2) and h'(2;1) ≅ psl(3) (p = 3), and the control g(2), p = 5: the NIS (unique up to a factor), 2t on g, Omega on S^2 | `vogvec.py vect1:3:1 ham:3:1,1 g2p5` (< 1 s) | `vect(1;1) over F_3: ... invariant symmetric forms 1, NIS yes`, `C on g = 2t = 2`, `x + 2: dim 6, Jordan block 2`; `h'(2;1,1) over F_3: ... C on g = 2t = 0`, `x: dim 1`, `x + 2: dim 27`; `g(2) [control ...] over F_5: ... x: dim 27`, `x + 2: dim 78` |
| Table (28), the columns sdim, NIS and d, and Statement 3.6: for vect(1;2), vect(1;3) (p = 3), h'(2;(1,1)) (p = 5, 7), h'(2;(2,1)) (p = 3), smy''(6;1), sby'(7;1)/c, my(6;1) (p = 3), me(5;1) (p = 5): a NIS unique up to a factor, of degree d (first coordinate of `sector`); 2t = 0 on g; the split Casimir operator Omega on S^2(g) is non-zero and Omega^2 = 0 | `vogvec.py vect1:3:2 vect1:3:3 ham:5:1,1 ham:7:1,1 ham:3:2,1 sm2 sby my mel` (4 min; my(6;1) 80 s, me(5;1) 110 s) | each algebra: `invariant symmetric forms 1, NIS yes, sector (d, ...)` with dim and d = 9, 6; 27, 24; 23, 4; 47, 8; 25, 6; 77, 6; 77, 6 (after `center of dim 1 factored out`); 162, 9; 125, 10; then `C on g = 2t = 0 (t = 0)` and `x: dim N, Jordan block 2`, N = dim S^2(g) (the only factor, nilpotent with Jordan blocks of size at most 2: Omega ≠ 0, Omega^2 = 0) |

Not deposited: the highest weight vectors V_1, V_2 of S^2(g) for the algebras of `vogelp.py`, and
the rows of Table (9) for the simple subquotients of g(2;3), g(3;3), g(2;6), were computed with the
*SuperLie* package (see [GK] in the text), which is not distributed here. `vogelp.py` takes the
highest weight vectors as data, as printed in the text, and checks that each is homogeneous of the
printed weight.

## Files

- `pslnn.py` — gl(n\|n), psl(n\|n), the Casimir operator of the supertrace form on the adjoint
  module and on S^2 (Statement 1.1.2). The optional second argument selects the convention for
  the dual basis (default 2; the values 0 and 3 are non-invariant controls).
- `vogelp.py`, `vogelp-data.json` — Casimir values on highest weight vectors from the Cartan matrix
  and the positive roots, Vogel's law C|_g = 2t, C|_{Y_2(x)} = 4t - 2x, t = alpha + beta + gamma.
- `br2eps.py` — the same for br(2;eps), symbolically in eps (sympy).
- `vogele.py` — e(6), e(7), e(8) (Chevalley basis by the Frenkel–Kac sign rule) and ab(3) (the
  weight-basis model of `exc3.py`, with `lin.py`) over F_p: invariant forms, the Casimir operator,
  singular vectors of S^2(g). Arguments `name:p`, with `/z` for the quotient by the center.
- `fakecas.py` (the matrix models of pe(n), psq(n) and the dual bases), `fakecas2.py` (the fast
  driver: `pe n ...` or `psq n ...`; `VONLY=1` restricts `pe` to pe(n) ⊗ V), `psqD.py` (psq(n)).
- `msimple.py` — the pericontact algebra m(n;1); uses `nisvect.py` (sparse echelon), `mvect.py`,
  `dervect.py`, `laplace.py` (generating functions and brackets of the vectorial superalgebras).
- `spe.py` — invariant bilinear forms on spe(n) and pspe(n) over F_p.
- `vogvec.py` — the vectorial Lie algebras of Table (28): the NIS solved for, the Casimir operator
  on g, the characteristic and minimal polynomials of Omega on S^2(g) block by block. The Skryabin
  algebras come from `prolong.py`, `run_derived.py`, `ceh.py`, `gvect.py`, the Melikyan algebra
  from `mel5.py`, the quotient by the center from `vogele.py`.
