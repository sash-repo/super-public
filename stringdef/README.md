# stringdef

**Deformations of the distinguished stringy Lie superalgebras: what the coefficients decide**

D. Leites, O. Lozhechnyk. Status: in preparation.

The scripts check the computations of the text. For the distinguished stringy Lie superalgebras
with Laurent coefficients they compute H^2(g; g) by the window method: the formal rigidity of witt,
k^L(1|n) for n <= 3, k^M(1|n) for n <= 2, vect^L(1|1), m^L(1) and of their polynomial counterparts;
the weights of the cohomology; and the non-rigidity of the family svect^L_lambda(1|2), the derivative
d mu/d lambda not being a coboundary. They also compute the central extensions, used as a
calibration, and the charges C_g(a) of the bundles of semi-infinite forms (the analogs of Mumford's
exponent). Appendix A is covered too: H^q(witt_-; witt) by truncations, the proof of H^1 = 0, the
differential-operator cochains and the controls, and the cohomology of the negative parts of the
Neveu-Schwarz and Ramond superalgebras with trivial coefficients (the super Goncharova theorem, the
super analog of L_2, the super Weinstein relations). They compute the structure functions of the
Ramond distribution where it degenerates, and H^q(L_2; witt). Exact arithmetic throughout
(integers, `fractions.Fraction`, sympy, and python-flint through `../lib/exactlin.py`).

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r stringdef/requirements.txt
cd stringdef
../venv/bin/python test_defco.py
```

Every script is run from inside this folder. No arguments are needed except where stated.
`stringy.py`, `yb.py`, `defco.py`, `family.py`, `ghost.py`, `xieta.py`, `witt2.py`, `polycoh.py`,
`nsgonch.py`, `rc.py`, `real.py` are modules imported by the others (`family.py`, `ghost.py`,
`witt2.py`, `polycoh.py`, `nsgonch.py` can also be run directly). Most runs take seconds to a few
minutes on a laptop. The table gives the times of the longer ones, measured with up to nine scripts
running at once, so they are upper bounds. `gonch_par.py` uses 9 processes. The names given to
`census.py` select algebras; quote them in the shell (`census.py 'k^L(1|3)'`).

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Subsec. 2.3 (the grading element is inner for every algebra of the list but svect^L_lambda(1\|2), lambda != 0); Subsec. 2.4 (the models and the window method: Jacobi, grading, coboundaries inside the cocycles for both parities, the bracket = delta(id)) | `test_defco.py` | `72/72 checks pass` |
| Subsec. 2.6 (dim H^2(g; C) = 1 for the Laurent algebras, 0 for the polynomial ones; a calibration of the method) | `central.py` | `h` = 1 for each Laurent algebra and 0 for each polynomial one, at N = 3, 4, 5 |
| Statement 2.7.1, list (5) (formal rigidity in weight 0); Remark of Subsec. 3.3 (the polynomial list is rigid) | `census.py 'witt' 'k^L(1\|1)' 'k^L(1\|2)' 'k^M(1\|1)' 'k^M(1\|2)' 'vect^L(1\|1)' 'vect(1)' 'k(1\|1)' 'k(1\|2)' 'm(1)'` (about 2 min); `census.py 'k^L(1\|3)'` (about 37 min; 21 min of it the window N = 6); `census.py 'm^L(1)'` (about 14 min) | in sections 1 and 2, `h=0` at the largest windows N for every algebra (for witt, k^L(1\|2), k^M(1\|n), vect^L(1\|1) the nonzero h at N = 3, 4 are truncation artifacts that disappear at N = 5) |
| Lemma 2.2 (only weight 0 for an inner grading element); Subsec. 3.8 (h(N,M) at the integer lambda: 2 at lambda = 0 in weight 0; 1 in weight -2 at lambda = 1 and in weight 2 at lambda = -1) | `weights.py` (about 52 min) | `[0, 0, 0, 0, 0]` for the seven algebras with `E inner`; for svect^L_lambda(1\|2): `[0, 0, 2, 0, 0]` (lambda = 0), `[1, 0, 0, 0, 0]` (1), `[0, 0, 1, 0, 0]` (1/3, 2/5), `[0, 0, 0, 0, 1]` (-1) |
| Subsec. 3.2, eq. (8) (d mu/d lambda is not a coboundary at lambda = 1/3) | `family.py` | `super Jacobi identically in lambda: OK`; `controls: delta(id) recognised as a coboundary: YES; delta(phi): YES`; `d mu / d lambda is a coboundary on the core \|deg\| <= 1: NO` |
| Statement 3.5, item 1) (weight 0, even cochains: exactly one class at lambda_0 = 1/3, 2/5) | `census.py svect` (about 50 min) | section 3: `h=1` for svect^L_{1/3} and svect^L_{2/5} at N = 3, 4, 5 (the lines for lambda = 0, 1 belong to Subsec. 3.8) |
| Statement 3.5, item 2) (no cohomology in the weights 0 < \|w\| <= 4, both parities) | `svect_wide.py 5 1/3,2/5 -4,-3,3,4` and `svect_wide.py 5 1/3,2/5 -2,-1,1,2` (each of the 16 nonzero systems takes 4-13 min; run one weight per process to parallelize); `others.py` section C (lambda = 1/3, \|w\| <= 2, N = 3, 4) | every line `'h': 0` |
| Statement 3.5, item 3) and Subsec. 3.6 (no odd 2-cochain of weight 0 for svect^L_lambda(1\|2); the odd cochains of k^M(1\|1), k^M(1\|2) carry no cohomology) | `others.py` (about 27 min); also `census.py` section 4 | section B: `unk=0 ... h=0`; section A: `h=0` for k^M(1\|1), k^M(1\|2) (and m^L(1)) at N = 3, 4, 5 |
| Statement 4.12, list (18) (not a coboundary at lambda_0 = 1/2, 1/3, 2/3, 3/2, 5/7, 22/7) | `lamscan.py` | the six lines with `dims OK ... yes yes NO`; `NOT a coboundary (so H^2 != 0) at lambda_0 = 2, 1/2, 1/3, 2/3, 3/2, 5/7, 22/7` (the integers 0, 1, 2, -1 are not claims: there the frame does not span the algebra, Subsec. 3.8; the dimension guard of the script compares only the degrees of the window N = 3, so it does not see the missing element of degree -4 at lambda = 2) |
| Statement 5.3, table (24), Subsecs. 5.4, 5.5, 5.8 (the charges C_g(a); -26, -15, -6, 0 at a = -1) | `ghost.py` (about 2 min), `charges34.py` (about 2 min) | `fit: C(a) = -12 a^2 + 12 a + -2` (witt), `0 a^2 + 12 a + -3` (k^L(1\|1), k^M(1\|1)), `0 a^2 + 0 a + -6` (k^L(1\|2), k^M(1\|2)), `0 a^2 + 0 a + 0` (k^L(1\|3)); `C=0` for k^M(1\|3), k^M(1\|4), k^L(1\|4) (the last block of `ghost.py`, the b-family in the theta frame, ends with `neither sign gives a representation`: the second divergence is written in the xi-eta frame, which is what `mumford.py` uses) |
| Subsec. 5.7 (the charge of k^L(1\|2) does not depend on the second weight b) | `mumford.py` | for sigma = +1 and both signs of Div_2: every `C` is `-6` |
| Statement A.6.2, item 1), \|w\| <= 120 at N = 20, 24 | `witt_wide.py` | `nonzero found: NONE` |
| Statement A.6.2, items 1)-2), \|w\| <= 40 at N = 22, 28, degrees 2 and 3 | `witt2.py --q 2 3 --N 22 28 --wlo -40 --whi 40` | headers only (the script prints only the weights with a nonzero entry) |
| Statement A.6.2, items 1) and 3), the table of Subsec. A.6 | `sweepN.py` | the table of dim H^2(Q_N), w = -8, ..., 8, N = 6, ..., 38; then `dimZ^2 = dimB^2 = N-1` at w = 0 for N >= 12 |
| Theorem A.7.1, eqs. (40), (41) | `h1proof.py` | `P(w) = -(w - 1)*(w**2 + w + 6)/6`, `integer roots of P : [1]`; `H^1=0` at w = 1; `dim Z^1=1` in every weight |
| Conjecture A.8.1 (the evidence) | `z2struct.py` | `iso` for N >= 14, except `w=1 N=14: dimZ2=14 ... NO` |
| Subsec. A.9 (the densities: H^2 != 0 only for lambda = 0, 1, 2) | `z2struct.py` (second part) | nonzero lines only for `lambda=0`, `lambda=1`, `lambda=2` |
| Subsec. A.9 (the adjoint module of witt_-) | `witt2.py --module adjneg --N 16 20 24 --q 1 2 --wlo -12 --whi 12` | H^1: one class in weight 0; H^2: one class in each of the weights -4, -3, -2 |
| Statement A.9.2, eq. (44) | `polycoh.py`, `polyscan.py` | `Z^1=1`, `Z^2 = B^2`, `H^2 = 0` for D = 3, 5, 7; `polyscan.py`: weight 0 only, `H^2=(1, 1)` (lambda = 0), `(2, 2)` (1), `(2, 2)` (2) |
| Statement A.12.1 (super Goncharova theorem) | `gonch_par.py ns 28 1 5`, `gonch_par.py ram 26 1 5` (about 2 min each) | `ns q 2 [...]` with the weights q(2q-1)/2, q(2q+1)/2; `ram q 2 [('q^2', 2)]`, for q = 1, ..., 5 |
| Remark after Statement A.12.1, eq. (47) (Gauss's identities); Statement A.13.3 and eq. (48) (the super partitions) | `superpartitions.py` | `q 2q+1 True` for q = 1, ..., 6; `NS Gauss True`, `Ramond Gauss True` |
| Statement A.13.3 (the cohomology of NS_{>=3/2}) | `gonch_par.py nsd3 36 1 5` (about 5 min) | `nsd3 q 2q+1 [...]`, one class in each weight q(2q+1)/2 + j/2, j = 0, ..., 2q |
| Subsec. A.13.5 (NS_{>=1}: the dimensions) | `gonch_par.py nsd2 26 1 4`, `gonch_par.py nsd2 30 5 5` | `nsd2 1 3`, `nsd2 2 4`, `nsd2 3 4`, `nsd2 4 4`, `nsd2 5 4`; see the note below |
| Statement A.14.1 (the super relations) | `relations.py witt1 20`, `relations.py ns 20`, `relations.py ram 16` | first kind: `relations 3` (witt), `1` (ns), `2` (ram), and second kind `relations 2` (witt), from weight 8 (witt), 11/2 (ns), 6 (ram) on; see the note below |
| Statement A.15.2 (holomorphic structure functions of the Ramond symbol, n <= 4) | `ramond_hol.py n R0 4 0`, n = 1, 2, 3, 4 | `d=0: even H2=1 ... odd H2=0`, and `H2=0` for d = 1, ..., 4 |
| Lemma A.15.6, item 1); Statement A.15.7; the direct computation after Theorem A.15.8 | `ramond_real.py n R0 3 0` (n = 1, 2, 3), `ramond_real.py 4 R0 2 0` | `hol H^0_-2={0: 1, ...}`, `hol H^0_-1={..., 1: 1}`, `hol H^1_-1={0: 1, 1: 0}`; real H^2 in degrees 0, 1, 2: `3, 3, 1` (n = 1), `5, 2` (n = 2), `9, 5` (n = 3), `14, 9` (n = 4), zero beyond |
| Statement A.16.4, items 1), 2); H^2(L_2; K) = 5 | `wittk.py 2 adj -12 12 0 16 20 24`, `wittk.py 2 adj -12 12 1 16 20 24`, `wittk.py 2 adj -30 30 2 20 28`, `wittk.py 2 triv -14 0 2 20 24` | every row `[0, 0, 0]` or `[0, 0]`; trivial module: `[1, 1]` in five consecutive weights (-11, ..., -7 in the sign convention of `witt2.py`) |

**Note on Subsec. A.13.5.** The dimensions of H^q(NS_{>=1}) for q = 1, ..., 5 are 3, 4, 4, 4, 4
(with H^0 = 1); the sequence 1, 3, 4, 4, 4 printed in the text is that of q = 0, ..., 4. For q = 1 the
weights are 1, 3/2, 2: the weight 1/2 of Statement A.12.1 is absent, F_{1/2} not being in NS_{>=1}.
The window W must exceed the largest weight sought: with W = 26 the run for q = 5 misses the two
classes of weights 55/2 and 28, hence the separate run with W = 30.

**Note on Statement A.14.1.** `relations.py` reports the counts weight by weight. The values stated
in the text are reached from weight 11/2 (Neveu-Schwarz) and 6 (Ramond) on. Below that, the first
kind has 0 relations in the weights 3/2, 2 and 2 relations in weight 5 (Neveu-Schwarz), and 2, 4, 4
relations in the weights 3, 4, 5 (Ramond); for witt the stable 3 is reached from weight 7 on. In the
second kind, the products of the form delta(F_r) delta(F_r) vanish identically, since delta(F_r) is
odd in the cochain superalgebra; the column `of them zero` counts them. Among the nonzero products
there are no relations.

## Files

- `stringy.py` — functions and vector fields on superstrings; `yb.py` — the distinguished stringy
  Lie superalgebras as exact bracket oracles, each checked against the super Jacobi identity.
- `defco.py` — H^2(g; g) by the window method (the cocycle condition is the Jacobi identity of
  [x,y] + eps c(x,y) modulo eps^2); `test_defco.py`; `central.py` — H^2(g; C).
- `census.py`, `weights.py`, `others.py`, `svect_wide.py` — the censuses of Sects. 2 and 3.
- `family.py` — svect^L_lambda(1|2) over Q(lambda) and its cocycle d mu/d lambda; `lamscan.py`.
- `ghost.py` — semi-infinite forms, the cocycle (20), the charges; `xieta.py` — k^L(1|2) in the
  xi-eta frame; `mumford.py`, `charges34.py`.
- `witt2.py` — H^q(witt_-; M) by truncations (M = witt, the densities F_lambda, K, witt_-);
  `witt_wide.py`, `sweepN.py`, `z2struct.py`, `h1proof.py`, `wittk.py` (L_k in place of witt_-).
- `polycoh.py`, `polyscan.py` — the differential-operator cochains.
- `nsgonch.py` — H^*(g; K) of the negative parts of witt, NS(1|1), R(1|1); `gonch_par.py`,
  `gonch_q.py` (one q at a time), `superpartitions.py`, `relations.py`.
- `rc.py` — polynomial vector fields on C^{1|n} and the structure functions of a distribution;
  `real.py` — the real form; `ramond_hol.py`, `ramond_real.py`.
