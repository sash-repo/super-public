# domvera

**Dominant weights, systems of simple roots, and reflections**

D. Leites. Status: in preparation.

The scripts check the computations of the text: Serganova's criterion of dominance run by chains of
odd reflections for ag(2), osp_alpha(4|2), ab(3) and osp(2m|2n); the roots, systems of simple roots
and reflections of pe(n), vect(0|n) and the deform svect~(0|n); and, for the distinguished stringy
Lie superalgebras, the central symmetry of the roots of k^L(1|n) and k^M(1|n), the absence of an
invariant form on k^M(1|7) (Statement 8.3.2), the systems of simple roots and reflections of witt/vir,
k^L(1|2), k^L(1|3), k^L(1|4), svect^L_lambda(1|2) and vect^L(1|2). Exact arithmetic throughout
(integers, `fractions.Fraction`, sympy, and python-flint through `../lib/exactlin.py`); numpy is used
only by `pe_borels.py`, on integer matrices. [S2] is V. Serganova's text on dominant weights in
*Seminar on Supersymmetries* (the source the text expounds).

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r domvera/requirements.txt
cd domvera
../venv/bin/python kl_symmetry.py
```

Every script is run from inside this folder (those of `osp2m2n/` from inside that subfolder). No
arguments are needed except where stated. `run2.py T` writes `dom_T.json`, which `check_T.py` reads.
Every run below finishes in under four minutes on a laptop (the longest, `nis.py`, about 3 minutes;
`osp2m2n/exhaust.py` uses 10 processes).

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Propositions 3.1 (ag(2)) and 4.1 (osp_alpha(4\|2)), Fig. 2 | `domrefl.py` | for every alpha and base `0 disagree`; `shape check alpha=...: 2744 weights, 0 disagree` |
| Section 5: the base of the printed Cartan matrix of ab(3) | `run1.py` | `6 bases`, then a line `MATCH [...]` |
| Proposition 5.1 (dominance conditions for ab(3)) | `run2.py T` (about 1 minute), then `check_T.py` | `T 1078`; `weights 11466 dominant 1078 mismatches 0` |
| Theorem 6.4 (dominance conditions for osp(2m\|2n), the three end types) | `osp2m2n/exhaust.py` | one line per base, then `TOTAL cases, dominant, mismatches: [1169168, 3461, 0]` |
| Formula (14), Subsec. 8.1 (pe(n): the non-essential roots in a Borel subalgebra form a final segment) | `pe_borels.py` | for n = 2, ..., the patterns listed are exactly the final segments |
| Subsec. 8.2, formulas (19)-(21) (roots of vect(0\|n), multiplicities, check-Delta) | `vect0n_roots.py` | `n roots ... checkDelta ... True True True True` for n = 2, ..., 6 |
| Proposition 8.2.1 and Remark 8.2.2 (the deform svect~(0\|n) has the roots of svect(0\|n)) | `tsvect.py`, `tsvect_b.py`, `tsvect_c.py` | `tsvect.py`: `weights whose multiplicity differs (svect, svect~): {}` for n = 4, 6; `tsvect_c.py`: `4 elements D with [E,D] outside svect~` (n = 4) and `6 ...` (n = 6) |
| Remark 8.3.1 (the roots of k^L(1\|n) and k^M(1\|n) are centrally symmetric) | `kl_symmetry.py` | every line `asymmetric: 0` |
| Statement 8.3.2 (k^M(1\|7) has no non-degenerate invariant form; controls: k^L(1\|6) has exactly one, Res int fg) | `nis_forms.py "(7,True,3,0)" "(7,True,3,1)"` (about 1.5 minutes each); controls `nis_forms.py "(6,False,2,0)" "(4,False,2,0)" "(5,True,2,1)" "(6,True,2,0)" "(6,True,2,1)"` (under a minute) | `k^M(1\|7) window 3, even forms: 132 unknowns, ... nullity 0` and `odd forms: 132 unknowns, ... nullity 0`; controls: `k^L(1\|6) ... nullity 1`, all others `nullity 0` |
| Statement 8.3.2: the model of k^M(1\|7) used is a Lie superalgebra | `ramond_jacobi.py` | `k^M(1\|7) Jacobi: 0 failures among 3000 triples` |
| Subsec. 8.3.3 (witt and vir: two systems of positive roots, simple roots eps and 2eps) | `witt_roots.py` | `2788 checks, 0 failures` |
| Subsecs. 8.3.4-8.3.12 (k^L(1\|2), k^L(1\|3): roots, simple roots, odd reflections, Lemmas 8.3.6, 8.3.11, Corollaries 8.3.7, 8.3.12) | `kl12_refl.py` | the brackets printed, then the count of checks and an empty list of failures |
| Proposition 8.3.9 (the odd reflections of k^L(1\|2) are induced by the automorphism phi) | `kl12_flow.py` | `c= -2 (True, 300)`, and `False` for the other values of c |
| Subsec. 8.3.13 (k^L(1\|3): the choice (36), the rules (37) and (38) for the root vectors) | `kl13_refl.py`, `kl13_beta.py` | `342 checks, 0 failures`; `131 checks, 0 failures`, `9 of the even n mismatch` (for n even the image of X_gamma is not the choice (36)), `adapted choice for s_beta: 114 checks, 0 failures` |
| Subsecs. 8.3.15-8.3.18 (svect^L_lambda(1\|2): root vectors, generation of n_+, reflections, Cartan matrices) | `svect_refl.py`, `svect_check.py` (part E; part C: the root vector t^{k+1} xi1 xi2 d_t) | `checks: 3795 reflections: 450`; `checks: 8315` |
| Subsecs. 8.3.19-8.3.21, table (46), Statement 8.3.21 (vect^L(1\|2): bases of Delta(delta), systems of simple roots) | `vectL12_roots.py` | `multiplicities True ; bases realized 4 of 6 ; 4006 checks, 0 failures` (the six rows give four triples) |
| Subsecs. 8.3.22-8.3.24, Lemma 8.3.23 and the Remark after it (k^L(1\|4): roots, H not in the derived algebra; k^L(1\|5)) | `k14.py`, `k14aut.py`, `svect_check.py` (part A) | `part 1: 2148 checks, 0 failures`; `90 automorphism checks, 0 failures` |
| Lemma 8.3.26, Statement 8.3.27, Proposition 8.3.29 (k^L(1\|4): brackets, systems of simple roots, reflections) | `k14b.py`, `k14gen.py`, `svect_check.py` (part B) | `brackets: 1215 checks, 0 failures`, `base property: 1999 checks, 0 failures`, `converse: 3499 checks, 0 failures`; `400 systems, height <= 6: 0 fail` |

## Files

- `ab3dom.py` — the roots of ab(3), odd reflections, Serganova's criterion; `run1.py`, `run2.py`,
  `check_T.py` use it.
- `domrefl.py` — Serganova's criterion for ag(2) and osp_alpha(4\|2).
- `osp2m2n/` — `dom.py` (the criterion by odd reflections for osp(2m\|2n)), `crit.py`, `typeII.py`,
  `samp.py` (the explicit conditions on the labels), `exhaust.py` (the comparison).
- `pe_borels.py`, `vect0n_roots.py`, `tsvect.py`, `tsvect_b.py`, `tsvect_c.py` — Sects. 8.1, 8.2.
- `nis.py` — models of k^L(1\|n), k^M(1\|n) on a window of powers of t; run directly, it checks
  witt, vir, k^L(1\|n), k^M(1\|n), q(n) and sl(2)^(1) for invariant forms.
- `nis_forms.py`, `ramond_jacobi.py`, `kl_symmetry.py` — Remark 8.3.1, Statement 8.3.2.
- `kL.py` — the contact bracket on k^L(1\|N), N = 2, 3, 4; `kl12_refl.py`, `kl12_flow.py`,
  `kl13_refl.py`, `kl13_beta.py`, `k14.py`, `k14aut.py`, `k14b.py`, `k14gen.py` use it.
- `svect.py` — svect^L_lambda(1\|2); `svect_refl.py`, `svect_check.py` use it.
- `vectL12_roots.py`, `witt_roots.py`.
