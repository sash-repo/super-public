# egorov

**How to superize gl(infinity)**

George Egorov (deceased), D. Leites (in loco authoris). Status: in preparation (an earlier
version is arXiv:2205.07046).

The scripts check the finite computations of the text: the cocycle (17) on the band
superalgebra gl_{p,c} and the steps of its uniqueness proof; the classification of parity
functions p: Z -> Z/2 under the permutation groups S_c, S_n, S_o, S_m; the odd 2-cocycles on the
periodic subalgebras q(k)^{l(2)} and q(n)^{l(1)}; Serganova's universal super Weyl group UW,
acting on the weight sets of typical Kac modules, against the Coxeter group C_g of its diagram;
and the derivations of gl(m|n). Exact arithmetic throughout (integers, fractions, and
`../lib/exactlin.py`, fast when python-flint is installed and correct without it).

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r egorov/requirements.txt
cd egorov
../venv/bin/python h2c.py
```

Every script is run from inside this folder; it finds `../lib/exactlin.py` by itself. No
arguments are needed except where stated. Every run below finishes in under a minute on a laptop.

## Claims of the text and the scripts behind them

Numbers are those printed in the current version of the text.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Theorem 2.2.11 (uniqueness of the central extension of gl_{p,c}), steps 2) locality, 5) the values on the matrix units (the relation hat g(a,b)+hat g(b,t)+hat g(t,a)=0 for all eight parity patterns), 6) normalization; the cocycle (17) and its values G_k(i) | `h2c.py` | `0 failures` on each of the lines (1)-(6), and `(3') window [-N,N]: solution space dim = 2N, expected 2N (= u mod constants)` for N = 2, 3, 4 |
| Subsec. 4.3 (Remark 4.3.6): Lemma 4.3.1 (the monotone bijection moves the fewest points), Theorem 4.3.2 (classification under S_c, S_n, S_o), Example 4.3.3 (the density spectrum is not complete), Proposition 4.3.4 and Remark 4.3.5 (S_m does not see the density) | `sceq.py` | seven groups of checks, each line `[ok ]`, then `failures: 0` |
| Remark 1.4.1h: for sl(m\|n), m+n <= 5, m > n, and for osp(2\|4), osp(2\|6), the kernel of C_g -> UW_g has no non-trivial element of length <= 12 | `cox.py`; `coxosp.py 12 9` | `sl(2|1)`, `sl(3|1)`, `sl(4|1)`, `sl(3|2)` (and the four with m < n): `ball of radius L of G has ... elements, all images distinct` with L = 30, 9, 7, 6; `osp(2|4): 11 modules, ball of radius 12 of C_g has 776 elements, all images distinct`, `osp(2|6): 11 modules, ball of radius 9 of C_g has 775 elements, all images distinct` (a ball of radius L mapped injectively excludes kernel elements of length <= 2L) |
| Remark 1.4.1j: for sl(3\|2) and x = r_2 r_4 r_3, x^4 acts trivially on S^k(V) + Lambda^k(V), trivially on the weights of K(lambda_0), lambda_0 = 2eps_1+eps_2+delta_1, and moves 470 of the 840 weights of K(2 lambda_0) | `kerword.py` | `kernel element u = a b^{-1} = 423423243243 length 12` (this is x^4, since r_2 r_4 = r_4 r_2), `k=1..8: u trivial on S^kV+Lambda^kV: True`, `Kac K(2, 1, 0, 1, 0): u moves 0 of 291 weights`, `Kac K(4, 2, 0, 2, 0): u moves 470 of 840 weights` |
| Statement 6.1.1 (no odd 2-cocycle on q(k)^{l(2)} for k=2, pi=(0,1), R=3, \|d\|<=2; k=2, pi=(0,0) and k=3, pi=(0,1,1), R=2, \|d\|<=1); Remark 6.1.4 (k=1: the cocycle (55)); Theorem 6.2.1 (the odd cocycle (56) on q(n)^{l(1)}); Theorem 6.2.2 (its corner realization) | `oddcoc.py` | (A) control `[OK ]` twice; (B) `dim H^2_odd` equal to `0` in every listed t-degree for k = 2, 3, and `1` in the odd degrees for k = 1; (B') and (C) `[OK ] cocycle`, `[OK ] not a coboundary` for n = 1, 2, 3; (D) `the P-centralizer in Mat(4) has dimension 8`; `ALL CHECKS PASSED` |
| Subsec. 7.5: the derivation census, no outer derivations of gl(m\|n) beyond the central ones except one even derivation for gl(1\|1) (also: Theorem 7.3.1, e_00 as a sum of two supercommutators of banded matrices, and Lemma 7.4.3, the supertranspose) | `sherst.py` | `gl(1|1): sdim gl = (2|2),  sdim Der = (3|2),   Der - gl = (+1|+0)` and `Der - gl = (+0|+0)` for (2\|1), (1\|2), (2\|2), (3\|1), (1\|3), (3\|2), (2\|0), (3\|0); `=> all OK` for parts 1) and 2) |

## Files

- `h2c.py` — the cocycle (17) on band matrices; Theorem 2.2.11.
- `sceq.py` — parity functions and their enumerations; Subsec. 4.3.
- `uw.py` — weights of sl(m|n)-modules (adjoint, typical Kac modules) and the reflections
  r_alpha on weight sets; run alone, it prints the orders of the products r_alpha r_beta on the
  distinguished base of sl(2|1), sl(3|1), sl(2|3) (the sl(m|n) row of Table 1).
- `cox.py` — C_g in its Tits representation and the ball test for sl(m|n).
- `coxosp.py` — the same for osp(2|2n); arguments: the radii for n = 2 and n = 3.
- `sep.py` — the ball test with one module family (weights of S^k(V) used by `kerword.py`).
- `kerword.py` — the element x^4 of Remark 1.4.1j.
- `oddcoc.py` — odd 2-cocycles on the periodic subalgebras; Section 6.
- `sherst.py` — the Appendix: e_00 as supercommutators, the supertranspose, Der gl(m|n);
  `--log FILE` also writes the output to a file.
