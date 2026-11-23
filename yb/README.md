# yb

**On the Yang--Baxter equation and Lie superalgebras**

D. Leites. Status: accepted by Theoretical and Mathematical Physics.

The scripts check the computations of the text: the non-degeneracy of the even 2-cocycle
omega = d(ell) on vect(0|3) (Claim 2.3) and the census of non-degenerate even 2-cocycles on the
small vectorial Lie superalgebras (Remark 2.4); the constant CYBE for the trigonometric
r-matrices with values in sl(2|1) (Proposition 3.9); the restriction of the 2-cocycles of the
distinguished stringy superalgebras to their abelian subalgebras (Table (19), Theorem 5.9); and
the recursion for H^1(g; g (x) g) over windows of Laurent exponents for vect^L(1|2)
(Remark 5.22.7) and svect^L_lambda(1|2) (Remark 5.22.10). Exact arithmetic over Q throughout
(integers, `fractions.Fraction`, sympy, and python-flint through `../lib/exactlin.py`), except
`cocyc.py`, `census.py` and `frob.py`, which work modulo the prime P = 2147483629 (see the note
below the table).

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r yb/requirements.txt
cd yb
../venv/bin/python vect03.py
```

Every script is run from inside this folder. No arguments are needed except where stated. Each
run below finishes in under five minutes on a laptop (the longest are `census.py` with the
arguments of the table, about 4.5 minutes, 2.5 of them for `vect:5`; `svect2.py`, about 3
minutes; `liftforms.py`, about 2 minutes; `vect2b.py`, about 1.5 minutes); none writes a file.

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Claim 2.3, proof: the four ranks (Q and R of rank 3; P and S non-degenerate on the 6-dimensional annihilators), hence omega(x,y) = ell([x,y]) is non-degenerate on vect(0\|3) | `vect03.py` | `entries of omega pairing g_a with g_b, a+b not in {0,2}: 0`; `rank Q (g_0 x g_2) = 3`; `rank R (g_-1 x g_1) = 3`; `dim of the annihilator of g_2 in g_0 = 6 ; rank of P on it = 6`; `dim of the annihilator of g_-1 in g_1 = 6 ; rank of S on it = 6`; `dim g = 24 ; rank of omega = 24`; `super Jacobi identity on 13824 triples: True` |
| Claim 2.3: r = omega^{-1} is a constant solution of the CYBE (a cross-check over Q of what the text deduces from [St]) | `vect3exact.py` | `basis 24 nonzero entries of r: 38 super-anti-symmetric: True nonzero CYBE components over Q: 0 of 747` |
| Remark 2.4: X in gl(3) acts on the ideal I spanned by the xi_1 xi_2 xi_3 d_i by -X^t + tr(X) 1_3 | `vect03.py` | `action of gl(3) on I is X -> -X^t + tr(X) 1_3: 9 of 9 matrix units checked` |
| Remark 2.4, first fact: among vect(0\|n) n <= 5, svect(0\|n) n <= 5, the deformed svect(0\|4), h'(0\|n) 4 <= n <= 7, only vect(0\|3) has a non-degenerate even 2-cocycle (all even 2-cocycles) | `census.py vect:2 vect:3 vect:4 vect:5 svect:3 svect:4 svect:5 svectdef:4 h:4 h:5 h:6 h:7` (about 4.5 minutes) | `vect(0\|3): ... dim Z^2_even=12 rank(random cocycle)=24 of 24 -> NON-DEGENERATE EXISTS`; every other line `... -> all degenerate`, with (dim Z^2_even, rank, dim g) = vect: (4, 6, 8), (32, 60, 64), (80, 145, 160); svect: (8, 12, 17), (25, 46, 49), (64, 103, 129); deformed svect(0\|4): (25, 46, 49); h': (9, 12, 14), (16, 27, 30), (31, 56, 62), (64, 119, 126) |
| Remark 2.4, second fact: among vect(0\|n) n <= 7 and svect(0\|n) n <= 6, only vect(0\|3) has a non-degenerate 2-cocycle of the form ell([x,y]), ell even (coboundaries only; also run on the cases of the first fact) | `frob.py vect:2 vect:3 vect:4 vect:5 vect:6 vect:7 svect:3 svect:4 svect:5 svect:6 svectdef:4 h:4 h:5 h:6 h:7` (about 30 s) | `vect(0\|3): ... 24 of 24 -> FROBENIUS`; every other line `-> not (for coboundaries)`, among them `vect(0\|6): ... 378 of 384`, `vect(0\|7): ... 776 of 896`, `svect(0\|6) sign0: ... 314 of 321` |
| Proposition 3.9: P + P^21 = Omega and the CYBE with no spectral parameter for P(s) (symbolically in s, i.e., for all s at once), P_A, P_B; the CYBE with spectral parameter and the unitarity of r(z) = P + Omega/(z-1) | `sl21.py` | `P(s) P+tau(P)=Omega: True CYBE: True`, the same for `P_A`, `P_B`; `P(s) spectral CYBE: True`, `P(s) unitarity: True`, the same for `P_A`, `P_B` |
| Remark 3.6 (Karaali's r), Example 3.4 (the toral part r_0), formula (10) (det r(z)) | `sl21.py` | `Remark 3.6, Karaali's r: r+tau r=Om: True CYBE: True`; `Example 3.4, r0 solving the CYBE: [{a: 0, b: 0, c: -1, d: 0}]`; `det: -z**3*(s*z - s - 1)*(s*z - s + z)/(z - 1)**8  vs text: (the same)` |
| Table (19), Theorem 5.9: the restriction map c -> c\|_a has rank 1 for k^L(1\|3), {k^L}'(1\|4), k^M(1\|3), k^M(1\|4), with c(K_{t^a theta_S}, K_{t^b theta_T}) non-zero only for S = T and a + b = -1 (Neveu--Schwarz), a + b = 0 (Ramond); rank 0 for k^L(1\|2), vect^L(1\|2), svect^L_lambda(1\|2), {svect^L}'(1\|2); for {k^L}'(1\|4), kappa vanishes on a 2-dimensional subspace of the 3-dimensional space of classes | `liftforms.py` (about 2 minutes) | for every row `coboundaries (must be 0): c\|_a = 0 ...`; `k^L(1\|3)`: `H^2_0 = 1`, `rank 1`, the pairs `(-3,2), (-2,1), (-1,0)` of exponents; `{k^L}'(1\|4)`: `[(-1, (1, 2, 3, 4))]` missing from the derived algebra, `H^2_0 = 3`, `rank 1`, `12 non-zero pairs`, all with S = T and a + b = -1; `k^M(1\|3)`: `H^2_0 = 1`, `rank 1`, pairs with a + b = 0; `k^M(1\|4)`: `none (k^M(1\|4) is perfect there)`, `H^2_0 = 1`, `rank 1`, `12 non-zero pairs`, S = T, a + b = 0; `k^L(1\|2)`: `all cocycles: rank of the restriction map = 0`; `vect^L(1\|2)`, `svect^L_lambda(1\|2)` (lambda = 1/3, 1/2, -1/3) and `{svect^L}'(1\|2)` (lambda = 0, 1): `all cocycles: c\|_a = 0 for EVERY cocycle` |
| Remark 5.22.7: for vect^L(1\|2), over the windows \|a\| <= 1 and \|a\| <= 2 the recursion has no non-trivial solution | `vect2b.py` (about 1.5 minutes) | `window \|a\| <= 1: ... unknowns = 1007`, `dim B^1 = 15 <= dim Z^1 <= 15   ***  EQUAL`; `window \|a\| <= 2: ... unknowns = 3041`, `dim B^1 = 15 <= dim Z^1 <= 15   ***  EQUAL` |
| Proposition 5.22.5, Lemma 5.22.6 (in the window): the torus (40), g_0 = t, root spaces of dim <= 3, no invariants of weight 0 in g (x) g; the window \|a\| <= 1 with all equations | `vect2.py` (about 30 s) | `g_0 = t, of dim 3 ... ok`; `every root space is of dim <= 3 ... ok`; `(g (x) g)^g has no element of weight 0 in the window ... ok`; `window \|a\| <= 1: ... dim Z^1 ... = 15`, `dim B^1 (window) = 15`; `every structural check passed` |
| Remark 5.22.10: for svect^L_lambda(1\|2), lambda = 1/3, 1/2, -1/3, over the windows \|a\| <= 1 and \|a\| <= 2 the recursion has no non-trivial solution; Proposition 5.22.9 (the centralizer of t, the integer locus) | `svect2.py` (about 3 minutes) | in part 6), for each of `lambda = 1/3`, `1/2`, `-1/3` and each window `\|a\| <= 1`, `\|a\| <= 2`: `dim Z^1 - dim B^1 (window) = 0` (with dim Z^1 = dim B^1 = 4; also for lambda = 1/3, `\|a\| <= 3`); for lambda = 1 (not self-centralizing torus, shown for contrast) the difference is `30` and `249`; `every structural check passed` |

**Note on Remark 2.4.** `census.py` computes the whole space Z^2_even of even 2-cocycles (the
cocycle identity on every triple of basis elements) and the rank of one random element of it;
`frob.py` tests only the coboundaries ell([x,y]). Both work modulo the prime P, with fixed
random seeds. A non-degenerate rational even cocycle would reduce mod P to a non-degenerate
element of the computed space (unless P divides every value of its determinant on the integral
cocycles), and a random element of a space with a non-degenerate element is degenerate with
probability at most dim g / P (Schwartz--Zippel). So "all degenerate" is a computation mod one
large prime with a random element, not an exact proof over Q; the positive verdict for
vect(0\|3) is proved over Q by `vect03.py`. For vect(0\|n),
n <= 5, and svect(0\|n), n <= 5, the computed dim Z^2_even equals dim g_ev, i.e., every even
2-cocycle is a coboundary; for h'(0\|n) it is larger (there are non-trivial classes), so for
h' only `census.py` is conclusive, and it is run for all four n. For vect(0\|6), vect(0\|7)
and svect(0\|6) the whole space Z^2_even is not computed by these scripts (the cubic loop over
triples is too slow); for them only the coboundaries are tested (`frob.py`).

## Files

- `vect03.py` — Claim 2.3 (the four ranks) and the gl(3)-action of Remark 2.4, over Q.
- `cocyc.py` — Grassmann algebra and vector fields mod P; all even 2-cocycles of vect(0|n),
  svect(0|n) and the deformed svect(0|n) (`python3 cocyc.py vect:3 ...`); used by the next three.
- `census.py` — the census of Remark 2.4 (adds h'(0|n) to `cocyc.py`).
- `frob.py` — the coboundary test of Remark 2.4; `ham_fields` builds h'(0|n).
- `vect3exact.py` — Claim 2.3: r = omega^{-1} satisfies the CYBE, over Q.
- `sl21.py` — Section 3, sl(2|1), with sympy.
- `stringy.py`, `yb.py` — functions on the superstring and the stringy superalgebras
  k^L(1|n), k^M(1|n), vect^L(1|m), svect^L_lambda(1|2), m^L(1), witt as bracket oracles.
- `coc.py` — degree-0 2-cocycles of a stringy superalgebra in a Laurent window, as the Jacobi
  identity of the central extension; calibrated on witt (Gelfand--Fuks cocycle).
- `coc2.py` — derived algebras, the non-monomial basis of svect^L_lambda(1|2), the family
  X^(i)_a of formula (28), the restriction map.
- `liftforms.py` — Table (19), all eight rows.
- `vect2.py`, `vect2b.py` — vect^L(1|2): Proposition 5.22.5, Remark 5.22.7.
- `svect2.py` — svect^L_lambda(1|2): Proposition 5.22.9, Remark 5.22.10.
