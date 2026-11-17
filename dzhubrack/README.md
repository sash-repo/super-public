# dzhubrack

**The superalgebra of antisymmetrizors, the periplectic case, and what the exponents really govern**

D. Leites, O. Lozhechnyk. Status: in preparation.

The scripts check the computations of the text: the antisymmetrizors (N-commutators)
a_k(X_1,...,X_k) = sum over S_k of sign(s) sign(s,P) X_{s(1)}...X_{s(k)}, evaluated by a subset
recursion costing k 2^(k-1) products instead of k! ones. They check the composition rule (8) of
arXiv:1906.01380 in the free associative superalgebra; which a_k close the Lie superalgebras
pe(n), spe(n), osp(m|2n), q(n), sq(n), sl(m|n) (Section 4); the algebras A(g) of the classical
Lie algebras and of g(2) (Section 5); the trace forms tr a_k and the exponents (Section 6); A(f4)
and A(e6), with f4 = Der J built from the octonions and the Albert algebra J (Section 7); the
weighted densities F_lambda over vect(1) (Section 8); and the contact vector fields k(3), with
calibration on vect(1), vect(2), h(2) (Section 9). Arithmetic: exact over Q (`fractions.Fraction`)
in `sant.py`, `salg.py`, `starprod.py`, `closure.py`, `densities.py`, `a3formula.py`, `albert.py`
(with `exactlin.py`, which uses python-flint when present); modulo a prime in `expo.py` and the
scripts built on it (p = 251, or 65521 for f4/e6, carried by float64 matrix products with all
intermediate entries below 2^53) and in `kcomm.py` (p = 1000003). A nonzero result or a witness of
"leaves g" modulo p is one over Q as well; a zero modulo p is the verdict of independent random
tests.

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r dzhubrack/requirements.txt
cd dzhubrack
python3 starprod.py
../venv/bin/python runexpo.py
```

Every script is run from inside this folder, with the arguments given below. Scripts that import
numpy (`expo.py`, `superclose.py`, `runexpo.py`, `traces.py`, `runexc.py`, `f4e6traces.py`) need
the venv; the others run with plain `python3`. Times are for a laptop running several scripts at
once.

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Theorem 3.2 (the composition rule (8) is unchanged by superization) | `starprod.py` (about 25 s) | one line per pair, e.g. `a_3 * a_5: all 128 parity vectors checked  (OK)`; last line `total: 776 agreements, 0 mismatches` |
| Statement 4.2 (a_5 maps spe(n) into pe(n) but not into spe(n); the 16 of 32 parity patterns), the computational note after Conjecture 4.3 (n = 3, 4, k <= 7), and Lemmas 1.1.2, 1.1.3 re-run | `superclose.py` (about 25 s) | `pe(3)     a_5   CLOSED (all parity patterns)`, `spe(3)    a_5   leaves g on 16/32 parity patterns, e.g. (0, 0, 0, 0, 0)`, the same two lines for pe(4), spe(4); `spe(3)    a_6   CLOSED (all parity patterns)`, `spe(3)    a_7   leaves g on 127/128 ...`, `spe(4)    a_7   leaves g on 128/128 ...` |
| Statement 4.4 (small n: spe(2) closed under a_5; a_k = 0 on pe(1) for k >= 3 and on spe(1) for k >= 2; sl(1\|1) closed under every a_k), exhaustive over all multisets of basis elements | `closure.py sanity` (about 15 s); the pe/spe part: see the note below the table (about 20 minutes) | `sl(1\|1)      a_5   CLOSED , nonzero   (11 multisets)` (and a_2, a_3, a_4 likewise); `spe(2)       a_5   CLOSED , nonzero   (231 multisets)`; `pe(1)        a_3   CLOSED , VANISHES identically` (and a_4, a_5, a_6); `spe(1)       a_2   CLOSED , VANISHES identically` (and up to a_6); `spe(3)       a_5   not closed, nonzero` |
| Lemma 1.1.3 for osp (exhaustive re-check) | `closure.py osp` (about 10 minutes) | `osp(1\|2)     a_6   CLOSED , VANISHES identically`; `osp(m\|2)     a_5   CLOSED , nonzero`, `a_6   CLOSED , nonzero` for m = 2, 3; a_3, a_4 `not closed` |
| Statement 5.1 (formulas (11), (12)) and Proposition 5.2 (formula (13): a_{4n-2} vanishes on o(2n)) | `runexpo.py` (about 3 minutes) | `sl(n)      ==> A(g) = {a_k : k in [2, 4, ..., 2n-2]}` for n = 2..5; `o(3)  ... [2]`, `o(4) ... [2, 5]`, `o(5) ... [2, 5, 6]`, `o(6) ... [2, 5, 6, 9]`, `o(7) ... [2, 5, 6, 9, 10]`, `o(8) ... [2, 5, 6, 9, 10, 13]`, `sp(4) ... [2, 5, 6]`, `sp(6) ... [2, 5, 6, 9, 10]`, `g2 ... [2, 10]`; and `o(8)       a_14  vanishes identically` |
| Statement 6.2, formula (15), classical lines (o(6): omega_5 vanishes) | `traces.py` (about 2 minutes) | `sl(n)    tr a_k =/= 0 for k in [3, 5, ..., 2n-1]`; `o(5)` and `sp(4)`: `[3, 7]`; `o(6)    tr a_k =/= 0 for k in [3, 7]` against `2m+1 over the exponents m: [3, 5, 7]`; `o(7)`, `o(8)`, `sp(6)`: `[3, 7, 11]`; `g2       tr a_k =/= 0 for k in [3, 11]` |
| Statement 6.2, formula (15), the f4 and e6 lines | `f4e6traces.py 17` (about 13 minutes) | `f4: tr a_k =/= 0 for k in [3, 11, 15]`, `e6: tr a_k =/= 0 for k in [3, 9, 11, 15, 17]` |
| Statement 6.4 | comparison of the outputs of `runexpo.py` and `traces.py` above | — |
| Subsection 7.1 (the models: f4 = Der J of dimension 52 on J_0) | `albert.py` (about 1 minute) | `octonions: composition and alternativity OK`, `Albert algebra: dim = 27, commutative: True`, `dim Der(J) = 52`, `f4 on J_0: 52 matrices of size 26` |
| Statement 7.2, formula (17) (A(f4) and A(e6) meet {a_k \| k <= 18} in {a_2}) | `runexc.py 18 both` (about 18 minutes) | `f4     ==> A(g) contains exactly [2] in the range 2..18`, `e6     ==> A(g) contains exactly [2] in the range 2..18`; every a_k with 3 <= k <= 18 `leaves g` |
| Statement 8.1, item 1) (formula (19)) | `a3formula.py` (seconds) | `a_3 = -lambda(lambda-1) W * Id: 280 cases (m_1<m_2<m_3<=7, 5 values of lambda), 0 mismatches` |
| Statement 8.1, items 2)--4) | `densities.py` (seconds) | for lambda = 0 and 1: `a_3  a_k = 0 identically`; for lambda = -1, 1/2, -1/2, 2, -2, 3/7: `a_3  leaves the algebra`; for every lambda: `a_2  CRITICAL` and `a_k = 0 identically` for k = 4, 5, 6 |
| Statement 9.1, item 1) (orders 2 or 3 for 3 <= N <= 9) | `runk3.py N 3 1` for N = 3, ..., 9 (the largest, N = 9, about 2 minutes; the script uses 10 processes) | `k3 N=3 deg=3 seed=1 order2 ord 2`, `N=4 ... order2`, `N=5`, `6`, `7 ... order3 ord 3`, `N=8`, `9 ... order2 ord 2` |
| Statement 9.1, item 2) (a_10 = 0 on k(3)) | `runk3.py 10 3 1` (about 2.5 minutes), `runk3.py 10 4 3` (about 20 to 30 minutes) | `k3 N=10 deg=3 seed=1 zero ord -1`, `k3 N=10 deg=4 seed=3 zero ord -1` |
| Remark 9.2 (calibration: critical numbers 2 of vect(1), 2 and 6 of vect(2), 5 of h(2)) | `kcomm.py vect1 3 4 1 1`, `kcomm.py vect2 6 3 1 1`, `kcomm.py vect2 7 3 1 1`, `kcomm.py h2 5 4 1 1`, `kcomm.py h2 6 4 1 1` (seconds each) | `vect1 N=3 deg=4 zero ord -1`, `vect2 N=6 deg=3 field ord 1`, `vect2 N=7 deg=3 zero ord -1`, `h2 N=5 deg=4 field ord 1`, `h2 N=6 deg=4 zero ord -1` |

`closure.py pe` (no other argument) runs the exhaustive test for pe(n), spe(n), n = 1, 2, 3, and
k = 2, ..., 6; it did not finish within 40 minutes (the case pe(3), a_6 is the long one). The part the text needs runs in
about 14 minutes:

```bash
python3 -c '
import salg
from closure import survey
for nm,(b,pv) in [("pe(1)",salg.pe(1)),("spe(1)",salg.spe(1)),("pe(2)",salg.pe(2)),("spe(2)",salg.spe(2))]:
    survey(nm,b,pv,[2,3,4,5,6])
for nm,(b,pv) in [("pe(3)",salg.pe(3)),("spe(3)",salg.spe(3))]:
    survey(nm,b,pv,[2,3,4,5])
'
```

The full statement for n = 3, 4 and k <= 7 is covered by `superclose.py` (random homogeneous
arguments of every parity pattern), which is cross-checked against this exhaustive engine on
pe(1), spe(1), pe(2), spe(2).

## Files

- `sant.py` — the antisymmetrizors by the subset recursion, over any associative product.
- `salg.py` — matrix models of sl(m|n), q(n), sq(n), pe(n), spe(n), osp(m|2n), o(N), sp(N), g(2),
  each built from its defining linear system and checked for closure (`python3 salg.py` prints
  `closed, sdim = ...` for twelve of them).
- `starprod.py` — the composition rule in the free associative superalgebra.
- `closure.py` — the exhaustive closure test over multisets of basis elements.
- `expo.py` — a_k modulo p by matrix products; `superclose.py`, `runexpo.py`, `traces.py` use it.
- `albert.py` — octonions, the Albert algebra, f4 = Der J, the operators of e6; `exactlin.py` —
  exact linear algebra (python-flint if installed, pure Python otherwise); `runexc.py`,
  `f4e6traces.py` — the f4 and e6 tests.
- `densities.py`, `a3formula.py` — vect(1) on the weighted densities.
- `kcomm.py` — N-commutators of vector fields in n variables (vect(n), h(2), k(3)) modulo a prime;
  `runk3.py` — its parallel driver for k(3).
