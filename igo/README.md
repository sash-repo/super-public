# igo

**Homogeneous supermanifolds whose base is a Hermitian symmetric space**

D. Leites, I. Shchepochkina. Status: in preparation.

The scripts check the computations of the text. In Section 4: Bott's theorem on the compact
Hermitian symmetric spaces X = G/P, applied to the layers of the tangent sheaf of a split
supermanifold (X, Lambda(E)) — the cotangent retract E = Omega^1 (Statement 4.4.3), the
obstructions to a retraction over the Lagrangian Grassmannians LGr_n (Statement 4.5.1), the
non-split even-homogeneous supermanifolds over LGr_n, 3 <= n <= 7 (Theorem 4.5.4), and the comparison
with Onishchik's classification (Statement 4.6.2). In Appendix A: Berezin's two families of
obstructions against the Green–Manin class over CP^1, by Čech cohomology with explicit transition
functions (Lemma A.4.1, Lemma A.5.2, Proposition A.5.3, Statements A.5.4–A.5.6). Everything is exact
(integers and `fractions.Fraction`; linear algebra through `../lib/exactlin.py`, which uses
python-flint when it is installed). Weights, roots, Weyl groups and the inverse Cartan matrices are
derived from the Cartan matrices; no table of roots is typed in.

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r igo/requirements.txt
cd igo
../venv/bin/python cot.py
```

Every script is run from inside this folder. No arguments are needed except where stated.
`retract.py --lgr` also writes `lgr.log`, and `onish.py` writes `onish.log`; both repeat what is
printed. Every run below finishes in under two minutes on a laptop.

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Statement 4.4.3, items 1)–4), over the 41 spaces of the census (cases I, II, III of Theorem 4.4.1), including Q_3 = LGr_2 | `cot.py` (about 30 s) | `H^0 = C always, H^1 = g in the cases I, II and 0 in the case III, H^p = 0 for p >= 2: verified for all 41 spaces`; `Q_3 ... ok 6`, `LGr_2 ... ok 6`; item 4): `k = 1` for every `Q_m` and for `LGr_2`, `k = 2` for `Q_6`, `LGr_3`–`LGr_5`, `OGr_4`, `OGr_5`; `H^1(X; Omega^3 (x) Theta) = 0 throughout`; `All tests passed.` |
| Statement 4.4.3, item 3) (the exception Q_3 = LGr_2, dim 6 = 1 + 5) | `cot.py --census` | table with `Q_3 ... 6 1 0` and `LGr_2 ... 6 1 0` (dim H^1, invariant part, H^1 of Omega^3 (x) Theta); `spaces where H^1(Omega^2 (x) Theta) exceeds its invariant part: Q_3, LGr_2` |
| Statement 4.4.3, items 1)–3), for E = E_7/P_7 | `e7.py` (about 15 s) | `p=1 dims {0: 1, 1: 133} inv {0: 1}`, `p=2 dims {1: 1, 2: 133} inv {1: 1}`, `p=3 dims {2: 1, 3: 133} inv {2: 1}` |
| Statement 4.5.1 (the obstructions to a retraction over LGr_n, 2 <= n <= 7) | `retract.py --lgr` (about 30 s) | `LGr_2`: `id, a = 1: ... j=-2 (dim 4)`, `id, a = 2: ... j=-3 (dim 1)`, `2*omega_1, a = 2: ... j=-4 (dim 6)`; `LGr_4`: `L^2(id), a = 3: non-zero H^1 for j=-2 (dim 36)` and no other layer; `LGr_3`, `LGr_5`, `LGr_6`, `LGr_7`: `H^1(X; Theta (x) Lambda^p(E)) = 0 for every such retract and every p >= 2` |
| The engine behind Statements 4.4.3 and 4.5.1, calibrated against Bott's theorem on line bundles and against `cot.py` | `bott.py`, `retract.py` | `Tests passed.`; every line `ok` |
| Theorem 4.5.4, items 1) and 2) | `sub4.py` (a few seconds) | for n = 3, 4, 5, 6, 7 and p_0 = 2, 4, 4, 6, 6: at p = p_0 `quot H0,1,2=[0, 0, 0]  sub H0,1,2=[0, 1, 0]  sub H^1=[(1, (0, ..., 0), 1)]`; at every other even p, H^1 = 0 in both halves, and H^2 = 0 in both for p > p_0 |
| Theorem 4.5.4: the blocks of the sub half E^* (x) Lambda^{p+1}(E) with H^1 =/= 0 over LGr_n | `sub.py` (about 1 minute) | for n = 3, ..., 7, the line `B(a=n-1), V=id: [(-1, 1)]` (the summand (17)); over LGr_4 also the blocks of `L^2(id)` |
| Remark 4.5.6, last sentence (over Gr_s^n, n <= 7, Igonin's range: every space (16) vanishes) | `subgr.py` (about 40 s) | `Gr_s^n: no block with H^1` for every 2 <= s <= n-2, 4 <= n <= 7 |
| Statement 4.6.2, items 1)–3) (only Omega^1 satisfies both hypotheses; 14 irreducible rank-2 bundles, Lambda^2 of each is O(odd); over LGr_2 = Q_3, S^2(id) (x) d^{-2} = Omega^1 in both models) | `onish.py` (under 2 minutes) | `=> the only retract of ours inside his hypotheses is the cotangent one   ok`; `=> 14 irreducible rank-2 bundles in the census, 0 of which can have Lambda^2 = O(-2)`; `dim H^1(X; Theta (x) Lambda^2(Omega^1)) = 6` for `LGr_2` and `Q_3`; `all checks passed` |
| Appendix A: exactness of the sequence (22), via Euler characteristics; Theorem A.5.1 (the sub vanishes for odd dimension <= 2); Lemma A.5.2; Proposition A.5.3 (dims 9, 3, 12); Statement A.5.4 ((12\|9) against (8\|6)); Lemma A.4.1 (45 pairs); Statements A.5.5 and A.5.6 (one representative of each class) | `runber.py` (under 2 minutes) | `sdim v(split) = (12, 9),   sdim v(X) = (8, 6)`; `(1, 'deg') sdim v = (8, 6)`, `(1, 'nondeg') ... (7, 6)`, `(2, 'deg') ... (5, 4)`, `(2, 'nondeg') ... (4, 4)`, `(3, '-') ... (3, 0)`; `67/67 checks pass` |
| Statement A.5.5 (the superdimension of the vector fields is constant on each of the five classes and tells them apart) | `orbit3.py` (under 2 minutes) | `60 supermanifolds computed`; the five lines `rank 1, image deg : sdim v in [(8, 6)]` ... `rank 3, image - : sdim v in [(3, 0)]`, `(12 samples)` each; `[PASS] five classes, told apart by sdim v, constant on each orbit` |

## Files

- `bott.py` — root systems from the Cartan matrices, the compact Hermitian symmetric spaces, Bott's
  theorem on G/P; run directly, its tests and the table of the bound D_m.
- `cot.py` — H^q(X; Omega^p (x) Theta_X) by Klimyk's formula; `e7.py` runs it over E_7/P_7.
- `retract.py` — H^q(X; Theta_X (x) Lambda^p(E)) for an arbitrary completely reducible retract E.
- `sub.py`, `sub2.py`, `sub4.py` — the sub half E^* (x) Lambda^{p+1}(E) of the layers of T_p;
  `subgr.py` — the same over the Grassmannians.
- `onish.py` — the comparison with Onishchik's hypotheses.
- `ns.py`, `bashkin.py` — vector fields and Čech cohomology of (1|m)-dimensional supermanifolds over
  CP^1 from the transition functions; `berobs.py` — the sub, quotient and total cohomology of T_q;
  `sl2inv.py`, `quot.py` — the sl(2)-action on H^1; `orbit3.py`, `runber.py` — Appendix A.
