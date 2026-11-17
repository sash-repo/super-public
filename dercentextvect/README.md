# dercentextvect

**Derivations and central extensions of simple vectorial Lie (super)algebras**

P. Grozman, A. Krutov, A. Lebedev, D. Leites. Status: in preparation.

The scripts check the computations of the text: Skryabin's identity Der L = N_{Der O}(L) for the
superizations of the series vect, svect, h, k (and its failure for le, for the pericontact series m
and the family b_lambda, and for some derived algebras), the squaring condition for p = 2, the
central extensions H^2(L; K) (Table (87)), the Poisson and Buttin classes, and the derivations of the
simple relatives of o_Pi(2n) for p = 2. Everything is exact arithmetic over the prime field F_p
(integers mod p; F_2 matrices as bitmasks in `opi.py`), with python-flint through `../lib/exactlin.py`
for the ranks and null spaces. Nothing here is random: every run gives the same output.

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r dercentextvect/requirements.txt
cd dercentextvect
../venv/bin/python census.py
```

Every script is run from inside this folder. Those writing a `.log` beside themselves also print it.
Running times on a laptop (single core): `census.py` about 30 minutes, `bcheck.py` 16, `cecensus.py`
10, `p2census.py` 6, `mvect.py` 6, `xcheck.py` 5, `gaps.py` 5, `bmp.py` 3,
`sweep6.py 3 4 5 6 7 8 9` about 8 (n = 9 alone 6), `ambient.py 3 4 5 6 7` 1.5, `mxcheck.py` 1,
`bmpE.py 1 2 3` 0.5; every other script under a minute.

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Subsec. 3.3: the correspondence (44) is an isomorphism psl(4) -> o^(2)_Pi(6) over F_2; dim out der o^(2)_Pi(6) = 7; dim out der(o^(2)_Pi(8)/c) = 26 | `run_opi.py` | `phi([x,y]) = [phi x, phi y] on all 105 pairs got 0`, `dim out der o^{(2)}_Pi(6) got 7`, `dim out der (o^{(2)}_Pi(8)/c) got 26`; `summary: 22/22 checks passed` |
| Statement 3.4, formula (45) (dim Der g_n = 2n^2+n, dim H^1 = 2n+1 / 2n+2; n = 4: 52 and 26), 3 <= n <= 9 | `sweep6.py 3 4 5 6 7 8 9` | `n=3: ... dim Der=21, dim out=7`, `n=4: ... dim Der=52, dim out=26`, `n=5: 55, 11`, `n=6: 78, 14`, `n=7: 105, 15`, `n=8: 136, 18`, `n=9: 171, 19` |
| Statement 3.4: the kernel of ad: o_Pi(2n) -> Der g_n is the unit matrix; one derivation (17 for n = 4) not from the ambient | `ambient.py 3 4 5 6 7` (reduced: n = 8, 9 not run, see below) | `not from the ambient=1` for n = 3, 5, 6, 7 and `=17` for n = 4; `dim ad(o_Pi)\|_g = dim o_Pi(2n) - 1` |
| Statement 5.9 (p = 3, 5), formula (66), Table 1, the exception k(1\|3) for p = 3 | `census.py` | one line per algebra: `dim Der`, `dim N`, `dim C=0`, `gap`; `k(1\|3)` at p = 3: `dim Der=25 ... dim N=24 ... gap=1`, neg block `(-3, 1)`; the `out` column agrees with Table 1, except the row `p=3 svect(2\|1;[1, 1])` (and its derived algebra): `dim Der=39 out=2 \| dim N=38 dim C=0 gap=1` |
| Statement 5.9 (p = 2): the exceptions svect(0\|3), svect(1\|2), svect(2\|1); the squaring condition cuts nothing out | `p2census.py` | `svect(1\|2;[1])`, `svect(2\|1;[1, 1])`, `svect(0\|3;None)`: `dimL=17 ... dimN=18 Der(bracket)=21 Der(+squaring)=21 ... gap=3`; every other line `gap=0`; in every line `Der(bracket) = Der(+squaring)` |
| Statement 5.11, second equality of (67), for N = (1,...,1) only (m = 2, n <= 3 for p = 3, n <= 1 for p = 5; m = 4, n = 0 for p = 3) | `census.py` (rows h(2\|n), h(4\|0)) | `dim N - dimL = 1 + m`: h(2\|n) `dim N = dimL + 3`, h(4\|0) `dimL=80 ... dim N=85` |
| Paragraph after Remark 5.11.1: H(2;1)^(2) at p = 3 (dim 7, Der 14, N 11) | `census.py` (row `p=3 h(2\|0)^(1)`) | `dimL=7 ... dim Der=14 ... dim N=11 ... gap=3` |
| Statement 5.13 (svect(1\|2;1)^(1), svect(1\|1;2)^(1), h(0\|4)^(1), p = 3, 5), formulas (69), (70); dim Der L obtained twice | `gaps.py` | `p=3 svect(1\|2;[1])^(1): dim L=24 dim Der L=27 dim N_DerO(L)=26 dim C_DerO(L)=0 gap=1`, a representative D in block `(deg -1, par 0)` matching (69); `p=3 svect(1\|1;[2])^(1): dim L=18 dim Der L=22 ... gap=1`, block `(deg -3, par 1)`, `D(x0^(k)y0.e0+x0^(k+1).d0) = x0^(k-2).e0`, i.e. (70); h(0\|4)^(1) gap 1 in block `(deg -2, par 0)`; the same for p = 5; each followed by `ungraded recomputation: dim Der L = ... (graded: ...)` with equal numbers |
| Proposition 5.16 (H(2;(1,n))^(2), p = 3: dim Der = 3^{n+1}+n+2, dim N = 3^{n+1}+n+1; no gap for N = (2,2)) | `bmp.py` | `N=(1, 2): dim H^(2)=25 ... dim Der=31 ... dim N_DerO=30 gap=1`; `N=(1, 3): 79, 86, 85, gap=1`; `N=(2, 2): 79, 85, 85, gap=0` |
| Proposition 5.16: E is a derivation not induced by Der O, F is induced | `bmpE.py 1 2 3` | for each n: `E derivation? bad pairs=0 ... in N_DerO\|_g? NO`, `F derivation? bad pairs=0 ... in N_DerO\|_g? yes` |
| Statement 5.18, formula (73) (le(r\|r), p = 3, r = 1, 2; p = 5, r = 1); open problem 2) of Subsec. 7.1 (the map K_f -> K_{d xi1 d xi2 d xi3 f} on k(1\|3) is not a derivation) | `laplace.py` | `le(...), Delta: ... well defined=True is a derivation=True not induced by Der O=True ... gap=1` (three lines); `k(1\|3), d_xi1 d_xi2 d_xi3: ... is a derivation=False` (p = 3, 5) |
| Statements 5.20, 5.21 (m(n;1): Der = ad = N, C = 0; S^m is not a derivation), and the gap column of Table (78) | `mvect.py` | `p=3 m(1)`, `p=5 m(1)`, `p=3 m(2)` and their derived algebras: `dim Der = dim ad = dim N`, `gap=0`; `Delta^m: ... is a derivation=False` (three lines) |
| Statement 5.23, formula (77), Table (78) | `bcheck.py` | one line per (p, n, lambda), with `dimL`, `centre`, `Der`, `N`, `C`, `gap` as in Table (78); `p=3 n=2 b_(0:1) lam=0 : dimL=36 ... centre=1 ... Der=39 N=37 C=1 gap=3` |
| Table (78): every case of dim <= 40 computed twice | `mxcheck.py` | `graded=... ungraded=... ok` on every line, `mismatches: 0` |
| Statement 5.24 (the odd Laplacian on b(n)) | `bdelta.py` | `b_0(n) = b(n), Delta: ... is a derivation=True not induced by Der O=True` for (p, n) = (3, 1), (5, 1), (3, 2) |
| Statement 6.3 (the Poisson and Buttin classes) | `named.py` | `weight=-2 parity=even cocycle=YES coboundary=NO \| dim H^2_(-2,0)=1` for h; `parity=odd ... dim H^2_(-2,1)=1` for le |
| Statement 6.5, formulas (85), (86), Table (87) | `cecensus.py` | `H^2=(even\|odd)` per algebra as in Table (87); `SKIPPED` for the dashes; the vect(1\|0;N) rows carry one even class of weight p^{N_1} |
| Statement 6.5: the cocycle identity (cocyc) gives d^2 = 0 | `dcheck.py` | `failures=0` on every line |
| Subsec. 6.4: every case with dim L <= 60 (69 of them) computed twice | `xcheck.py` | 69 lines `graded=... ungraded=... OK` |

`brute.py` (dim Der L with no grading used), `extra.py` (the derivations not induced by Der O, with a
representative), `p2.py` (the squaring condition at p = 2), `centext.py` (the 2-cochains, the cocycle
condition, the weight blocks, H^2 with no grading) are library modules used by the scripts above.

`ambient.py` is run for n <= 7 (about 1.5 minutes); n = 8, 9 take much longer and are not part of the
deposited run. The values of dim Der g_n for n = 8, 9 are checked by `sweep6.py`.

## Claims without a deposited script

- Statement 5.11: the first equality of (67), dim h(omega)/L = m, and the case N = (2,1); no script
  here computes the stabilizer h(omega). The second equality for N = (1,...,1) is in `census.py`.

## Computed with SuperLie, not deposited

- Subsec. 2.7 (Melikyan's first construction): the constraint Par N = 2 ("we verify this by
  computer").

## Files

- `dervect.py` — the divided power superalgebra O(m\|n;N), Der O, vect/svect/h/k/le, the normalizer
  N_{Der O}(L), dim Der L block by block in the (degree, parity) bigrading.
- `brute.py`, `extra.py`, `p2.py`, `centext.py` — see above.
- `census.py`, `p2census.py`, `gaps.py`, `laplace.py`, `bmp.py`, `bmpE.py` — Section 5.
- `mvect.py`, `bcheck.py`, `bdelta.py`, `mxcheck.py` — Subsecs. 5.19-5.24.
- `cecensus.py`, `named.py`, `dcheck.py`, `xcheck.py` — Section 6.
- `opi.py`, `run_opi.py`, `sweep6.py`, `ambient.py` — Subsecs. 3.3, 3.4.
