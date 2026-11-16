# Gordan-Rankin-Cohen brackets on 1|n-dimensional superstrings with a contact structure for n ≤ 6

S. Bouarroudj, D. Leites, O. Lozhechnyk, I. Shchepochkina.

**Status:** in preparation.

This folder holds the programs behind the computed claims of the paper: the
Statements, the dimension counts in the classification theorems, and the checks of
printed formulas. Every program uses exact arithmetic (Python `fractions` and
`sympy`). Nothing here uses Mathematica or SuperLie.

## How to run

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
cd grcsof
../venv/bin/python <script>.py <arguments>
```

Run every script from inside this folder, since the scripts import one another.
`osp22gen.py` also imports `../lib/exactlin.py`, which uses FLINT when
`python-flint` is installed. Without FLINT it falls back to pure Python and is slower,
but the result is the same.

Running times are for one core of an Apple M-series machine under moderate load. The
classifications run order by order. The table gives the small cases, which finish in
minutes. The larger cases in the paper use the same command with larger arguments,
for example a larger maximal order or more weights.

## The engines

| File | What it is |
|---|---|
| `super12.py`, `k12.py`, `invar.py` | Superfunctions on R^{1|2}, k(1|2) in the ξη realization, the weighted densities F_{λ,ν}, and the solver for i(osp(2|2))-invariant unary/binary operators |
| `superN.py`, `knclass.py`, `verify_kn.py`, `run_kn.py` | k(1|n) for an arbitrary Gram matrix of the odd coordinates; the binary/unary classifier; an independent invariance test against all the generators |
| `blo.py` | k(1|N) and osp(N|2) in the conventions of [BLO] = Ben Fraj–Laraiedh–Omri, J. Math. Phys. 54 (2013) 023501 |
| `k14family.py`, `run_n4.py`, `bol_n4.py` | The family h(a) ≅ osp_a(4|2) ⊂ k(1|4), its binary and unary (Bol) operators |
| `n5.py`, `n5bin.py` | i(osp(5|2)) ⊂ k(1|5) and its binary operators, with the torus filter |
| `kas6.py`, `kasops.py`, `kasbin.py`, `kastens.py`, `kastypes.py`, `kasV.py` | as ⊂ kas ⊂ k(1|6), the operators between densities and between tensor fields of type V |
| `m1.py`, `classify_m1.py` | m(1), i(pe(2)), and the solver for its invariant operators |

## Claims and the scripts behind them

Numbers are those of the paper as printed. "Run" is the command, run from this folder.

| Claim in the text | Run | Expected output |
|---|---|---|
| Framework check for k(1\|2): [BLS1, Thm 1.5] reproduced | `check_k12_setup.py` | ends `ALL CHECKS PASSED` (≈10 s) |
| The Koszul-sign convention for binary operators | `check_binary_conv.py` | ends `CONVENTION VALIDATED` (≈8 s) |
| Statement 2.7 (two odd operators of order 1); Theorem 2.9 (two operators in every order, all four weights symbolic) | `classify_k12.py 4` | for orders 1–4: `dim 1` at charges ±1 (odd orders), `dim 2` at charge 0 and `dim 0` at charges ±2 (even orders) (≈2.5 min). The paper's range n ≤ 6 is `classify_k12.py 6` |
| Theorem 2.9, generic weights to order 12 | `osp22gen.py 8` | `totals per order: k=0: 1, k=1: 2, …, k=8: 2` (≈2 min). Order 12 is `osp22gen.py 12` |
| Theorem 2.9 with all eight generators; the formulas (17), (21), (22) | `verify_k12.py` | part A: the dimensions of `classify_k12.py` at the five non-zero weight points (columns 2–6). The first column is the point where all four weights are 0; there the order-1 conditions are empty, and the script prints 0 where every operator of the ansatz is invariant; part B: six `OK` lines (≈14 min) |
| Statement 2.10 (the explicit operators of every order) | `check_k12_explicit.py 2` | ten `OK … 3/3 weights` lines, orders 1–5 (≈6 min). Orders ≤ 7 is `check_k12_explicit.py 3` |
| Corollary 2.11 (the ansatz of Theorem 2.6(i) has an invariant operator only if ν₁ = ν₂ = 0) | `probe_ansatz.py` | general weights: only `alpha = beta = 0`; at ν = 0: `beta = 0`, α free (≈9 s) |
| §3: [BLO]'s conventions read correctly (their N = 1 and N = 2 formulas are invariant) | `check_blo.py`; `check_blo2.py` | every line `invariant` (≈5 s, ≈10 s) |
| §3: [BLO]'s printed formulas (4.1), (4.2) are not invariant | `check_blo3.py form` | `… failures` for every k (≈1.3 min) |
| Statement 3.3.2 (the signs in (4.2) of [BLO]) and Statement 3.4.2 (the Levi-Civita sign in (4.1)) | `fix_blo.py` | every corrected operator `nonzero, invariant`, k = 1,2,3,4 and 3/2,5/2,7/2 (≈2 min) |
| Statement 3.4.2, term by term | `compare_blo41.py 3/2 5/2` | the terms differing from (4.1) are exactly the ones with the missing sign (≈45 s) |
| Statement 3.5.2 (the exceptional weights for osp(3\|2)) | `check_blo3.py dim 1/2,1` | `generic dim = 0` (order 1), `generic dim = 1` (order 2), no exceptions (≈5 min) |
| Statement 3.5.2, the four exceptional pairs at order 3 | `verify_blo_exc.py 3` | dim 2 at (−1,−1), (−1,−½), (−1,0), (−½,−½) and (0,−1), stable for tmax = 3,4,5; dim 1 at (0,0), (0,−½) and at generic pairs (≈9 min). The full grid for orders 3, 4 is `check_blo3.py dim 3/2,2` |
| k(1\|n) machinery, n = 1,…,4 | `check_superN.py` | every line `ok` (≈1 s) |
| Statement 4.2.1 (the family h(a) ≅ osp_a(4\|2), α = (1+a)/(1−a)) | `k14family.py`; `k14iso.py` | one free parameter in the intertwiners, `ok`; `alpha = (-a - 1)/(a - 1)`, `j = (a**2 + 3)**3/(4*(a - 1)**2*(a + 1)**2)` (≈1 s) |
| Theorem 4.3.1 (dims 0, 1, 0, 2 in orders 1–4) | `run_n4.py 1,2,3,4 0,2` | `dim = 0, 1, 0, 2` for a = 0 and a = 2 (≈4 min) |
| Theorem 4.3.1, the order-4 space depends on a | `binbol_n4.py span 4 0,2` | `dim=2` for each a, `rank of the joint span: 4` (≈6 min) |
| Theorem 4.3.1, the case a = ∞ | `binf_n4.py` | dims 0, 3, 0, 5 at both weight pairs (≈3 min) |
| Statement 4.4.1 (the Bol operators T₄ + (a/2)∂_t² at λ = −3) | `bol_n4.py scan 0,2 1,2,3,4` | nothing in orders 1–3; order 4 only at `lam = -3`, `dim = 1`, verified (≈1 min) |
| §4.4.3, [BLS1, Thm 1.6(ii)] at a = ∞ | `ainf_n4.py inf` | ∂_t^k for every λ; `symbolic lam, order 2/4: dim = 1` (≈45 s) |
| Statement 4.5.1 (ξ₁ ↔ η₁ sends h(a) to h(−a)) | `ainf_n4.py swap` | five `True` (≈1 s) |
| Statement 4.6.1 (B₀(a) ⊗ 1 lies in the binary space at λ₁ = −3) | `binbol_n4.py bolweight 4 0,2` | `dim=2`, `in the span: True`, `invariant (direct check): True` for each a (≈6 min) |
| Statement 5.2.1 (rigidity; n = 4 is the only n with a family) | `n5family.py` | `dim Hom_o(n)(V, Lambda^3 V) = 0, 1, 0, 0` for n = 3,4,5,6 (≈2 min) |
| Theorem 5.3.1 (the Bol operators T₅∘∂_t^k) | `n5unary.py t5 1`; `n5unary.py scan 1,2,3,4,5` | T₅ and T₅∂_t invariant, at no other weight of the grid (≈10 s); the scan finds only order 5 at `lam = -4` (≈45 s) |
| Theorem 5.4.1 (generic dims 1, 0, 1, 0 in orders 1–4) | `n5bin.py 1,2,3,4` | `dim = 0, 1, 0, 1`, each solution invariant under the 23 generators (≈9 min). Orders 5–7 are `n5bin.py 5,6,7` |
| Statement 5.4.2 (the order-2 bracket is k(1\|5)-invariant) | `n5contact.py 2` | invariant under all 43 generating functions (≈3 min) |
| Statement 5.4.5 (the ratio of the extreme blocks of the order-5 bracket) | `n5fit.py` | five `MATCH` (≈7 min) |
| The printed formulas (36), (37) and the ratio (39) | `n5verify.py 1` | `PASS` for J2 and for J4, and `ratio = 9985162/1529437, conjecture = 9985162/1529437` at (3/7, −5/11) (≈8.5 min). All five weight pairs: `n5verify.py` |
| §5, no exceptional weights in order 2 | `n5exc.py 2 1 -2,2` | `generic dim = 1; 0 weight pairs with a different dim` (≈2.5 min) |
| Statements 6.2.1 and 6.2.2 (as is 16\|16, closes, K₁ central) | `kas6.py` | `dim as = 32 (even\|odd) = (16, 16)`, `as closes: True`, `K_1 central in as: True` (≈1 s) |
| Theorem 6.4.1 (unary as-invariant operators: the powers of ∂_t) | `kasops.py unary as 1,2,3,4` | dim 0 in odd orders, dim 1 in even orders, at every λ tried (≈15 s) |
| Statement 6.4.2 ([BLS1, Thm 1.4.2] at n = 6: T₆ for i(osp(6\|2))) | `kasops.py unary osp 4,5,6 -5,-6,-7,0` | the only nonzero answer is order 6 at `lam = -5`, dim 1 (≈20 s) |
| Theorem 6.5.1 (binary operators over as, kas, i(osp(6\|2))) | `kasbin.py as 1,2,3,4`; `kasbin.py kas 2,4`; `kasbin.py osp 2,4` | as: dims 0, 3, 0, 5; kas: 1, 0; osp: 1, 1 (≈10 min each) |
| Theorem 6.5.1, the order-2 bracket is k(1\|6)-invariant | `check_k16.py` | invariant under all of k(1\|6) (≈2 min) |
| Theorem 6.5.1, the closed formula of the order-4 osp bracket | `fit_osp4.py`; `verify_osp4.py` | coefficients fitted as polynomials of degree ≤ 2 (≈8 min); `invariant, verified against all generators of i(osp(6|2))` at (1, 2) and (1/2, 1/3) (≈14.5 min) |
| §6.6.1 (a(f) ∈ co(6); the modules T(V) satisfy the module axiom) | `kastens.py`; `check_types.py` | `0 failures`; `module axiom OK` for every type (≈3 s; ≈4.5 min) |
| Theorems 6.6.5–6.6.7 (as-invariant operators between tensor fields) | `batchV.py a-unary`; `batchV.py b-unary`; `batchV.py c-unary` | the dimensions of the three Theorems, order by order (≈1, 1, 3.5 min). Binary cases: `batchV.py a-binary`, `b-binary`, `c-binary` |
| §6.6.4 (grad, div, and grad∘div as the second order-2 operator on T(Λ²)) | `idops.py` | `Hom(T(Lam1),T(Lam1)) in order 2: dim 2`, `grad o div = (0)*e0 + (1)*e1 -- exact: True` (≈25 s) |
| Theorem 6.7.3, Statements 6.7.4, 6.7.5 (the four operators of [GLS, §10]; what k(1\|6) sees) | `gls.py all kas 1`; `glscheck.py` | grad at (0, 2), div at (−4, −4), Dirac at (−5/2, −3/2); controls FAIL at neighboring weights; Dirac is not k(1\|6)-invariant, grad and div are (≈1.5 min; ≈50 s) |
| §6.7.6–6.7.11 (the list of [GLS, §10] is not closed under †; the two missing families) | `dagger.py`; `symtype.py`; `gap.py 2` | `0 lie in the list` for both missing families; S^m_0 of dims 20, 50 (≈1 s each); `gap.py 2` recomputes degree 1 between the fibers F, V₆ (= `L1`) and S²₀: the gradients F → V₆, V₆ → S²₀ are family 1a) of [GLS, §10]; the divergences V₆ → F and S²₀ → V₆, both at λ = −4, are `IN NO ENTRY OF THE LIST`, kas- and k(1\|6)-invariant, and fail at the control weights (≈9.5 min). The fiber S³₀ is `gap.py 3` |
| §6.7.17 (the two regradings: sdim g₋₁ = (4\|4), (4\|3); g₀ = (13\|10), (13\|12)) | `regrade.py struct` | the components with these superdimensions, depth 1 (≈1 s) |
| Theorem 6.7.19 (singular vectors over 1-dimensional fibers) | `regrade.py ops 3`; `regcheck.py deg1`; `regcheck.py axioms 1` | 3ξ: one vector of degree 1, none of degree 2, one of degree 3; 3η: (η₁η₃)^m m₁ at λ = −(m−1)/2; module axiom `ALL HOLD` (≈4 s; ≈1 s; ≈11 s). Degree 4 is `regrade.py ops 4` |
| §6.7.20 (the comparison with [GLS, §§11, 12]) | `glsreg.py` | the dictionary checked entry by entry; `2a) deg 2: SURVIVES at lam1 = -1` (≈1 s) |
| §7: the realization of pe(2) and m(1) | `check_m1.py` | `ALL CHECKS PASSED` (≈1 s) |
| Theorem 7.3 (the GRC brackets over pe(2) are the Bol operators in one argument) | `check_bj.py 4`; `verify_sec3_vol.py` | `VERDICT: Theorem 7.3 CONFIRMED as printed`; `INVARIANT` at λᵢ = (k−1)/2, and the control weights `NOT invariant` (≈5 s; ≈20 s) |
| Statement 7.4 (three operators of order 2) | `classify_m1.py 3`; `verify_m1.py` | unary part reproduces [BLS1, Thm 5.1]; binary `s= -2 w=+0 parity 1: … generic dim 3` (≈2 min); dims `[3, 3, 3, 3]` with all eight generators (≈2 min) |
| Statement 7.5.1 (the family D_ξD_q + 2a∂_τ, a correction to [BLS1, Thm 5.1] at k = 2) | `blsk2.py` | `k=2: family`, `k=1,3,4: as printed` (≈2 s) |
| §8 (the five sectors; generic dimensions) | `bjtypes.py generic 3`; `bjtypes.py num 6` | generic dims per (k, v); at λ = (7/3, 11/5): even B dims 2, 3, 2, 2 for k = 3,4,5,6, even A and even C dim 0 (≈1 min; ≈2.5 min) |
| Theorem 8.4 (the closed formula for odd Type A) | `check_typeA.py 3,4,5` | `dim = 1`, `all … coefficients agree` for k = 3,4,5 (≈35 s) |

The resonance scans of §8 are `bjtypes.py grid <k-list> <sector v> <2λ_min> <2λ_max>`,
for example `bjtypes.py grid 3,4 2 -1 2` for the even Type B sector.

## Runs

Every command in the table was run from scratch, in a fresh virtual environment built
from `requirements.txt` (Python 3.14, sympy 1.14.0, python-flint 0.9.0), and printed
the output given there. Four of them take longer than five minutes:
`verify_k12.py`, `verify_blo_exc.py 3`, `verify_osp4.py` and `gap.py 2`, each between
9 and 15 minutes; the binary runs `run_n4.py`, `binbol_n4.py`, `n5bin.py` and
`kasbin.py` take 4 to 10 minutes each.
