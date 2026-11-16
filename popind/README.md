# popind

**Replacing the positive-definite form with a nondegenerate one in Popov's *Discrete complex
reflection groups***

D. Leites, O. Lozhechnyk. Status: in preparation.

The scripts check the computations of the text: the reflection groups of the indefinite plane
(null coordinates, the crystallographic groups and their generalized Cartan matrices, a group
without a special point); the bases, chambers, wall-crossings and rank-2 residues of Serganova's
generalized root systems of small rank; the signed graphs of real reflection groups with a
nondegenerate form; the cocycle classes of the crystallographic groups on the indefinite plane;
and, in the Appendix, the proofs that V. L. Popov, *Discrete complex reflection groups*,
arXiv:2304.08941 (cited as [P] in the scripts), leaves to the reader: the invariant lattices of
G(4,2,2), K_12, K_31 and the "additional lattices" of its Sec. 4.7.7. Exact arithmetic
throughout (integers, `fractions.Fraction`, sympy, and exact arithmetic in the cyclotomic fields
Q(zeta_8) and Q(zeta_24)); numpy and floating point are used only for sanity checks (the order of a
group, a numerical fundamental domain), and are labeled as such in the output.

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r popind/requirements.txt
cd popind
../venv/bin/python check_vi.py
```

Every script is run from inside this folder. No arguments are needed except where stated. `check_477.py`, `check_477mod.py`, `enum_roots.py` and `run_lat.py` also write what they print to a file `*.out` in this folder (`run_lat.py` to `check_lat.out`). Every
run below finishes in under five minutes on a laptop (the longest, `check_app.py`,
`check_iv_b.py` and `check_iv_a.py`, take 2-4 minutes).

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Subsec. 7.3 and Lemma 7.4 (the plane in null coordinates: reflections are the S_mu, boosts T_lambda, perpendicular reflections) | `check_iii_a.py` | every `check ...` line `True` |
| Theorem 7.10 (crystallographic groups of row 8 and generalized Cartan matrices, both sign patterns; the index-2 subgroup with Cartan matrix (2,3;3,2), trace 7) | `check_iii_b.py` | `isometries=True` on each of the four lines; `tr(Q1Q2)=7  vs  tr((R1R2)^2)=7`, `Q1 = R1 ? True   Q2 = R2 R1 R2 ? True` |
| Proposition 7.13 (an r-group with Tran W = 3Z+Z and no special point) | `check_iii_f.py` | block `A=(2,-3;-2,2), u = 1/2*[[3, 2]]`: `Tran W basis (columns): [[3, 0], [0, 1]]` and three congruences with no common solution (the first two force w2 in Z, the third w2 in 1/2+Z) |
| Proposition 7.15.1 (the affine reflection groups preserving the root lattice of sl(2\|1)) and Proposition 7.15.2 (its six bases, the six sectors in cyclic order, the walls that carry reflections) | `check_iii_c.py`, `check_iii_d.py` | `alpha^perp = [[1, -2]] = alpha-2beta, square -2 ; is it a root? False`, `-R_alpha = R_(alpha-2beta) ? True`, `number of bases: 6`, `6 sectors`; the six positive systems in the cyclic order of Proposition 7.15.2, `wall a ... ANISOTROPIC`, `wall b`, `wall a-b ... isotropic -> carries no reflection` |
| Theorem 8.7 and the table of Subsec. 8.8 (n, \|R\|, \|R_re\|, anisotropic and isotropic walls, \|W\|, bases = chambers, iota(R) for the fifteen rows) | `check_iv_a.py` | for each system, e.g. `=== F(4) ===`: `walls: 18 total = 10 anisotropic + 8 isotropic`, `\|W\| = 96`, `chambers of A(R) = 576 (Zaslavsky)`, `enumerated regions = 576 (complete)`, `bases by brute force over 4-subsets = 576`, `refinement index ... = 6` |
| Theorem 8.11 (wall-crossings are the simple and the odd reflections; an odd crossing is not induced by an isometry) | `check_iv_b.py` | for each system `all crossings equal Serganova's s_gamma: True` and `odd crossings whose base map is NOT an isometry: N/N  (all)`; for A(1,1) `number of bases = 8` |
| Theorem 8.11, the input "no imaginary root is multipliable"; Theorem 8.18 2) (rank of span R_re); Subsec. 8.20 (for psl(2\|2), none of the 16 triples given by Serganova's Remark 6.3 is a base) | `check_iv_c.py` | `defect = 1` exactly for sl(m\|n) and osp(2\|2n), `0` otherwise; `sets from the 24 linear orderings, deduplicated: 16`, `of them satisfying Definition 6.1 verbatim (unique representation): 0`; `multipliable imaginary: 0` on every line |
| Warning 8.15 and the table of Subsec. 8.16 (the twelve types of rank-2 residues, with the column "occurs in") | `check_iv_b.py census` (a few seconds) | twelve lines `k=... a=... \|R_X\|=... nondeg=...  cycle ...  in: ...`, then `number of types: 12`; `nondeg=False` exactly on the three degenerate rows |
| Lemma 9.3 (trace formula), Corollary 9.4 (the signed graph is balanced), Theorem 9.8 (rank 2: trichotomy, mixed edges give infinite groups), Subsec. 9.14 (the (3,3,4)-triangle group), Theorem 9.16 and Remark 9.17 (the five irrational entries satisfy palindromic quadratics) | `check_v.py` | sections (1)-(6), `FAILURES: none` |
| Lemma 9.11 (a generalized root system has no mixed edge) | `check_v_grs.py` | `no real roots of opposite sign are ever joined: True` |
| Section 10: \|H^1(K; V/T)\|, the class of Proposition 7.13, Theorem 10.20 and Theorem 10.23 for \|a_12\|, \|a_21\| <= 6, H^1(K; V) = 0 when -id is in K, row 4 of Theorem 7.8 | `check_vi.py` (about 1 min) | `exactly one of a12,a21 odd: \|H^1\| = [4], #r-groups = [2, 3]`, `both even: \|H^1\| = [16], #r-groups = [5]`, `Gamma_0 = T for every hyperbolic pair: T is a root lattice   128 pairs tested`, `ALL CHECKS PASSED` |
| Theorem 10.20 (the numbers 1, 2, 3, 5 of classes realized by r-groups) and item 3) of Theorem 10.23 (Tran W = Gamma_0) over the whole range \|a_12\|, \|a_21\| <= 10 | `check_vi.py probe` (about 2 min) | one line per pair, then `pairs tested: 384;  count 1/2/3/5 and Gamma_0 = T violated in: none` |
| Appendix: Example A.3, Proposition A.5, Lemma A.8, Theorems A.9, A.11, A.12, A.13 | `check_app.py` | one block per item, every line `ok`, then `ALL CHECKS PASS` (A.11 c) is a numerical check, so labeled) |
| Warning A.17 (for G(4,2,2) exactly seven triples satisfy a), b) and discreteness; for three of them Gamma_j is larger than Lambda_j; all seven are K-invariant root lattices in the four similarity classes) | `enum_roots.py` | `tuples (Delta_1,...,Delta_s) satisfying a), b) and discreteness: 7`, three of the seven with `Gamma_j = Delta_j: False`, `similarity classes among them: 4  (sizes [4, 1, 1, 1])`, `checks: 11/11 passed` |
| Theorem A.19 and Remark A.20 (G(4,2,2): four invariant lattices of full rank up to similarity), Theorem A.22 (K_12), Theorem A.24 (K_31) | `run_lat.py` | the three blocks `K = K_2 of type G(4,2,2)`, `K = K_12`, `K = K_31`, each identification with Table 2 of [P] `[ok]`, then `checks: 98/98 passed` |
| Theorem A.27 (the lattices of the Case-2 list of Sec. 4.7.6 of [P]), Theorems A.29, A.30 (the reversal Q of a palindromic chain, the refinement (9) in Q(zeta_24)), misprint 1) of Subsec. A.33 (the row [F_4]^beta_2 of Table 2 of [P]) | `check_477.py` (about 2 min) | the SUMMARY table (`extra` = 2, 0, 0, 0, 2, 1, 0, 2, palindromic exactly where `extra` > 0), `as printed ... K-invariant = False`, `corrected ... K-invariant = True`, `checks passed: 69 / 69` |
| Subsec. A.31 and Corollary A.32 (Q is the Fricke involution of Y_0(c); the rows of Table 3 of [P]), misprint 3) of Subsec. A.33 (the non-equivalence) | `check_477mod.py` | `the printed left entry [G(2,1,2)]^g_2 is NOT equivalent to [G(2,1,2)]^{(g-1)/(g+1)}_3`, `checks passed: 18 / 18` |

## Files

- `check_iii_a.py`, `check_iii_b.py`, `check_iii_c.py`, `check_iii_d.py`, `check_iii_f.py` —
  Section 7, the indefinite plane (sympy).
- `check_iv_a.py` — generalized root systems of small rank: axioms, walls, \|W\|, chambers by
  Zaslavsky's theorem and by enumeration, bases; the root systems used by the next three.
- `check_iv_b.py` — wall-crossings and rank-2 residues; `census` prints the residue table.
- `check_iv_c.py` — rank of span R_re, psl(2\|2), multipliable roots.
- `check_v.py`, `check_v_grs.py` — Section 9, signed graphs.
- `check_vi.py` — Section 10, cocycles; `probe` covers \|a_ij\| <= 10.
- `check_app.py` — the Appendix items A.3-A.13.
- `lat_core.py` — exact arithmetic in Q(zeta_8) and Z-lattices in C^n.
- `check_lat.py` — the invariant-lattice machinery of Sec. 4.7-4.8 of [P] (a library; run
  `run_lat.py`).
- `run_lat.py`, `enum_roots.py` — G(4,2,2), K_12, K_31.
- `lat24.py` — exact arithmetic in Q(zeta_24) with a transcendental modulus.
- `check_477.py`, `check_477mod.py` — Sec. 4.7.7 of [P]: the additional lattices.
