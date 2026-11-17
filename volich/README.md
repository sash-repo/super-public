# volich

**A natural generalization of supersymmetry**

D. Leites. Status: in preparation.

The scripts check the computations of the text: the derivations of a supercommutative background
with and without even variables (Subsection 1.4); the identities of Volichenko algebras -- the
multilinear identities in degree <= 5 and their S_n-module structure, the even elements, the
identity of Bremner and Peresi, the computations in a fixed multidegree up to degree 10 and the free
Volichenko algebra on one generator (Subsections 2.5--2.7); the comparison with the Duflo--Serganova
functor (Remark 2.14.3); the isotropic odd root vectors of osp_a(4|2), ag(2), ab(3) (Remark 3.2.1);
the descriptions of vq, vspe, vosp (Subsection 3.4); the two corrections to Iyer's paper and the
vector fields tangent to the metaspace (Subsections 3.6--3.9); and the absence of one-sided ideals
(Subsection 4.13 and Statements 4.13.3, 4.15, 4.16, Remark 4.17.1). Exact arithmetic throughout:
`fractions.Fraction` and FLINT rationals (`exactlin.py`, which uses python-flint when present and a
pure-Python reference otherwise); ranks modulo a large prime p where a script says so -- a rank mod p
never exceeds the rank over Q, so a full rank mod p, or the equality dim I = dim C of an upper and a
lower bound, carries over to Q (for p > n, Lemma 8 of Bremner--Peresi). numpy is used for the
streaming echelon mod p of `mdeg.py`, `gens.py`, `hilb3.py`.

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r volich/requirements.txt
cd volich
../venv/bin/python dereven.py
```

Every script is run from inside this folder, with the arguments given below. Times are for a laptop
running several scripts at once. `gens.py` (half a minute) writes the cache `gens.pkl` that `run.py`
reads (`run.py` builds it itself if it is absent); `vlucid.py`, `vshort.py` and `veventest.py` write
and reuse the cache `C5.cache` of the consequences in degree 5. Run `gens.py` first, and do not
start two scripts that would build the same cache at the same moment.

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Subsection 1.4, the table after Theorem 1.4.2 (rank of d_{n\|m}, of (vect(n\|m) (x) Lambda)_ev, of a^x_n and of a_m per x-monomial, for Lambda = Lambda(N); a^x_n = 0 exactly when m+N is even); headings 1)--4) of Theorem 1.4.2 | `dereven.py` (seconds) | part 0): the twelve columns of the table, one line each, e.g. `   1  2  1 \|    14 \|                 12 \|            1 \|     1  OK`, `   2  2  1 \|    19 \|                 16 \|            2 \|     1  OK`; parts 1)--5) every line `OK`; last line `ALL CHECKS PASSED` |
| Subsection 1.4, the case n = 0: dim a_m = 2^{m-1} - eps against dim (vect(0\|m) (x) Lambda(N))_ev = m 2^{m+N-1}; headings 1), 3), 4) of Theorem 1.4.1 | `dermeta.py` (seconds) | e.g. `m=3 N=0 :  dim d_m =  15 =  12 (parity-preserving) +   3 (reversing)`, `m=4 N=0 :  dim d_m =  40 =  32 (parity-preserving) +   8 (reversing)`, every `predicted: ... OK`; last line `ALL CHECKS PASSED` |
| Statement 2.5.1, table (19): dim M_n, dim V(Z)_n, dim I_n, dim I_n/C_n for n <= 5 | `vnew.py 5 2147483647` (2 minutes); `run.py --degree 3`, `run.py --degree 4`, `run.py --degree 5` (seconds) | `vnew.py`: `degree 3:  dim I_n = 4 ... NEW: 4`, `degree 4:  dim I_n = 73     consequences of lower degrees: 64     NEW: 9`, `degree 5:  dim I_n = 1311   consequences of lower degrees: 1296   NEW: 15`; `run.py`, the rows mu = (1, ..., 1): `dim M_mu = 12` / `dim V_mu = 8`, `dim M_mu = 120` / `dim V_mu = 47`, `dim M_mu = 1680` / `dim V_mu = 369` |
| Statement 2.5.1, formula (20): the new identities as S_n-modules | `vchar.py 4`, `vchar.py 5` (2 minutes); degree 3: `brp.py`, part (A) | `vchar.py 4`: `NEW  I_n / C_n               dim 9      ...   [2,2] + 2.[3,1] + [4]`; `vchar.py 5`: `NEW  I_n / C_n               dim 15     ...   [2,2,1] + [3,2] + [4,1] + [5]`; degree 3 (vchar.py has the character tables of S_4 and S_5 only): `new multiplicities  {'3': 1, '21': 1, '1^3': 1}` in part (A) of `brp.py` |
| Remark 2.5.2 (the count agrees with Bremner--Peresi entry by entry); Remark 2.6.3 (identity (v) alone gives 1306); Statement 2.6.6 (identity (23) holds, 52 monomials after linearization, its S_5-orbit and C_5 span I_5; (v) has 144 terms); table (24) of Kostka numbers | `brp.py` (6 minutes) | (A) three lines ending `OK` for n = 3, 4, 5 and `total irreducible identities in degrees 3,4,5: 11`; (B) `13 terms; value in the free algebra: 0  -- it IS an identity`; (C) `52 terms (their Remark 5 says 52)`, `dim (C_5 + its S_5-orbit) = 1311   (dim I_5 = 1311)`; (D) `identity (5) has 144 terms; lies in I_5: yes`, `dim (C_5 + its S_5-orbit) = 1306`; (E) `K = 0`, `0`, `0`, `0`, `1`, `2`, `5`; last line ends `A OK   B OK   C OK   D OK   E OK` |
| Statement 2.6.1 (the even elements span dimensions 1, 4, 23 in degrees 2, 3, 4, and A^(JJ) spans them) | `veven.py 2 4` (seconds) | `even elements: dim Z = 1     from A^(JJ): dim E = 1     new = 0`, the same with `4` and with `23` |
| Statements 2.6.2 and 2.6.4 (dim (C_5+E_5) = 1306, the quotient is [2,2,1]; the alternating part of I_5 has dimension 108, of which 107 are old) | `veventest.py` (1 minute) | `C_5 + (all even-element identities) has dim 1306  ->  they contribute 10 of the 15 new`, `what they give, as an S_5-module: [3,2] + [4,1] + [5]`, `alternating in {1,2,3} and in {4,5}:  dim of that part of I_5 = 108,  new beyond C_5: 1,  beyond the even-element ones: 1` |
| Statement 2.6.5 (no alternated [,]/o-monomial is a new identity; a representative of Baranov's identity with 40 monomials) | `vlucid.py` (about 20 minutes); `vshort.py` (1 minute) | `vlucid.py`: `[,]/o-monomials of degree 5: 1680`, then `Alt(a,b,c) Alt(d,e) :  0 monomials give a NEW identity` and the same `0` for `Alt(a,b,c)`, `Alt(d,e)`, `Alt(a,b,c,d)`; `vshort.py`: `shortest found (random 0): 40 monomials in the [,]/o-basis`, followed by the 40 monomials |
| Statement 2.7.2 (for each of the twelve multidegrees of 4 and 5, dim(I_mu/C_mu) is the Kostka prediction; 9, 15 for (1^n); 6 for (2,2,1)) | `run.py --degree 4`, `run.py --degree 5` (seconds) | every multidegree: `NEW in this multidegree: k   (Kostka prediction k)` with equal numbers, among them `mu = (2, 2, 1)` ... `NEW in this multidegree: 6   (Kostka prediction 6)`, `mu = (1, 1, 1, 1)` ... `9`, `mu = (1, 1, 1, 1, 1)` ... `15` |
| Statement 2.7.3 and table (27) (dim I_mu = dim C_mu in the 24 multidegrees run) | `run.py 6 5,1 4,2 3,3 4,1,1 3,2,1 2,2,2` (1 minute); `run.py 3,1,1,1 2,2,1,1` (5 minutes); `run.py 7 6,1 5,2 4,3 5,1,1` (3 minutes); `run.py 8 7,1` (half a minute); `run.py 9 10` (1 minute) | for every mu `NEW in this multidegree: 0`; the rows of table (27): `mu = (2, 2, 1, 1)   n = 6   dim M_mu = 7560`, `dim V_mu = 897,  dim I_mu = 6663`, `dim C_mu = 6663`; `(4, 3)`: `4620`, `279`, `4341`; `(5, 1, 1)`: `5544`, `320`, `5224`; `(7, 1)`: `3432`, `92`, `3340`; `(9,)`: `1430`, `17`, `1413`; `(10,)`: `4862`, `30`, `4832`. The six further runs of the table, `run.py 2,1,1,1,1 1,1,1,1,1,1`, `run.py 4,2,1 3,3,1`, `run.py 8,1` and `run.py 11`, take from one to about eight hours each and print `(1, 1, 1, 1, 1, 1)`: `30240`, `3649`, `26591`; `(3, 3, 1)`: `18480`, `1152`, `17328`; `(8, 1)`: `12870`, `178`, `12692`; `(11,)`: `16796`, `54`, `16742`, each with `NEW in this multidegree: 0` |
| Statement 2.7.6, table (28) (the free Volichenko algebra on one generator) | `hilb.py 10` (half a minute) for n <= 10; `hilb3.py 12` (6 minutes) for n = 11, 12 | `hilb.py`: columns `dim M_(n)`, `dim V over Q` and two primes, e.g. `  9       1430             17           17           17   OK`, ` 10       4862             30           30           30   OK`; `hilb3.py`: ` 11      16796           54           54   OK`, ` 12      58786           96            -   OK`. For n = 11, 12 the numbers are ranks modulo the primes shown (lower bounds for the rank over Q; for n = 11 two primes agree, for n = 12 one prime) |
| Remark 2.14.3, the table of sdim g, sdim g_x and nontriviality for q(2), q(3), q(4), gl(p\|q), gl(2\|2), h'(0\|4)--h'(0\|7) | `ds.py` (seconds) | e.g. `q(4)  sdim g = (16\|16) ... ==>  sdim g_x = (4\|4)` / `x ensures nontriviality: yes`; `gl(2\|2), twoblock sdim g = (8\|8) ... sdim g_x = (0\|0)` / `yes`; `h'(0\|6), x = H_{th_1}  sdim g = (30\|32) ... sdim g_x = (0\|2)` / `nontriviality: yes`; `h'(0\|4)`: `(6\|8)`, `(0\|2)`, `no`; last line `ALL CHECKS: OK` |
| Remark 3.2.1 (every isotropic odd root vector of osp_a(4\|2), ab(3), ag(2) ensures nontriviality: 8 of 8, 16 of 16, 12 of 14; the witness S(a,b) = -2E_{65} in gl(4\|2)) | `nontriv.py` (9 minutes; `--quick` skips ab(3)); `ospcheck.py` (seconds) | `nontriv.py`, for five values of a and for ag(2), ab(3): `-> 8 odd root vectors, 8 isotropic, 8 of those ensure nontriviality`, `-> 14 odd root vectors, 12 isotropic, 12 of those ensure nontriviality`, `-> 16 odd root vectors, 16 isotropic, 16 of those ensure nontriviality`, each followed by `-> Table 2.12.1, last row: 'always' -- confirmed`; `ospcheck.py`: `[x,x] = 0 : True`, the matrix `[[a,x],[b,x]]` with the single entry `-2` in row 6, column 5, `=> x ENSURES NONTRIVIALITY in osp(4\|2).` |
| Statement 3.4.1 (the dimension formulas (59) for vq(n\|k), vspe(n\|p,q), vosp(m\|n\|p)); Remark 3.4.2 (the only exception is psq(2), k = 1) | `vdescr.py` (1.5 minutes) | one line per case, e.g. `psq(4), k = 2  dim h = 15  dim h_0 = 7  dim h_1 = 8  nontrivial: yes`, `spe(5), p=2, q=1  ... dim h_0 = 12`, `osp(9\|10), p = 2  ... dim h_1 = 44`; the only `nontrivial: NO` is `psq(2), k = 1`; last line `all assertions passed` |
| Statements 3.6.1 and 3.6.2 (the operator y_i d/dy_j - y_i d/dx_j is not a ghost; psi_1 does not preserve K[x+y]) | `iyer.py` (seconds) | the formulas of Iyer's Subsec. 4.2.3 `OK`; the claims of [Iy2] refuted appear as `FAIL`: `ghost_11: acts as 0 on k[Z]   at z-word (1, 0) \|-> {((0, 0, 0), (0, 1)): Fraction(-2, 1)}` (this is (y_1 d/dy_1 - y_1 d/dx_1)(z_2 z_1) = -2 y_1 y_2), the same for all nine (i, j), and `psi_1 keeps k[Z] invariant   at z-word (0, 0) \|-> {((1, 0, 0), (0,)): Fraction(2, 1), ((2, 0, 0), ()): Fraction(2, 1)}` (this is psi_1(z^2) = 2x^2 + 2xy); last line `10 FAILURES: [...]`, exit code 0 |
| Statements 3.8 and 3.9 (d ensures nontriviality iff n >= 2, with the witness (19eqWit); the ghost ideal; v(M) = V + Ghost(M), direct for n even, Ghost(M) in V for n odd; inhomogeneous, epimorphic, closed under the superbracket) | `vdmeta.py` (1 minute); `vdcheck.py` (seconds) | `vdmeta.py`: `n=1: NO -- S(a,b) = [[a,d],[b,d]] = 0 ...`, `n=2: YES` with `S(a,b) = (-2*y1*y2) d/dx_1` for `a = x^(0, 1)y() d/dx_1`, `b = x^(2, 0)y() d/dx_1`; for n = 1, 2, 3: `superbracket closure ...: OK`, `-> epimorphic`, `v(M) = V + ghosts; dim(V n ghosts) = 2`, `= 0`, `= 12`, and `ghost model ... -> MATCHES`; `vdcheck.py`: `OK: all 24 of the a + [d,a] preserve M` |
| Subsection 4.2, after Lemma 4.2.1 (f o S = 0, i.e. eq. (32), for the finite-dimensional examples of Tables 2.12.1 and 2.12.2) | `nontriv.py`, `onesided2.py`, `onesided3.py`, `psl22.py`, `ds.py` | `nontriv.py`: `h closed in g (eq. (32)) ............... True`; `psl22.py`: `closed: True` in every line; `onesided2.py`, `onesided3.py` assert it before every case (an assertion error otherwise); `ds.py`: `D0-D3: OK` (S takes values in Im ad_x) |
| Subsection 4.13, proof of Proposition 2.13 (dim L = 13, 73, 241 against 16, 81, 256 for vgl_mu(p\|q), p+q = 2, 3, 4; 256 for vpsq_mu(3); 18 = 9 + 9 of 36 for vpsq_mu(2); 81, 289, 576 for the three exceptional algebras) | `ideals.py` (1 minute); `onesided.py` (1 minute) | `ideals.py`: `L:  13 R:  13  (End=16, cap=13)`, `L:  73 R:  73  (End=81, cap=73)`, `L: 241 R: 241  (End=256, cap=241)`, `vpsq_(1:1)(3)  dim = 16 ... L: 256 R: 256 of 256   -- no one-sided ideals`, `vpsq_(1:1)(2)  dim =  6 ... L:  18 R:  18 of 36`; last line `ALL CHECKS PASSED`; `onesided.py`: `generate dim 81 of End(h) = 81`, `289 of ... 289`, `576 of ... 576`, last line `Proposition 2.13 holds for all of them: True` |
| Statement 4.13.3 (h'(0\|n): no one-sided ideals for n = 5, 6, 7; for n = 4, dim L = 18 of 36, two two-sided ideals of dimension 3) | `hamvol.py 4 5 6` (half a minute); `hamvol.py 7` (6 minutes); `ham4.py` (seconds) | `hamvol.py`, for both x (`ANISOTROPIC`, `ISOTROPIC`): n = 4: `dim <R_a> = 18,  dim <L_a> = 18,  dim End h = 36   ==> NOT absolutely irreducible`, `dim <L_a, R_a> = 18 of 36  ==> h has proper TWO-sided ideals too as a Volichenko algebra`, two lines `a proper right ideal of dimension 3`; n = 5, 6, 7: `dim <R_a> = 225,  dim <L_a> = 225,  dim End h = 225   ==> NO one-sided ideals (Burnside)`, the same with `900` and `3969`; `ham4.py`: the two ideals spanned by theta_i theta_j + (Hodge dual) and by theta_i theta_j - (Hodge dual), `two-sided ideals generated by a single  t_i t_j +- dual : [3]` |
| Statement 4.15 (no one-sided ideals for the ten pairs (g, x) with g = sl(2\|3), sl(2\|4), sl(3\|4), psl(3\|3), psq(3), psq(4)) | `onesided2.py` (2 minutes) | ten lines ending `-> no one-sided ideals`, e.g. `sl(3\|4) x^2_1  dim h =  24, End(h) =   576; left   576, right   576`; `10 of 10 cases run give A = End(h), i.e. no one-sided ideals.` |
| Statement 4.16 (spe(n), 3 <= n <= 5, osp(2m\|2n), osp(2m+1\|2n), 2 <= m, n <= 3: fifteen pairs) | `onesided3.py` (15 minutes) | fourteen lines ending `A = End h  -- no one-sided ideals`: spe(3), spe(4) (two), spe(5) (four), osp(4\|4), osp(4\|6), osp(6\|4), osp(6\|6), osp(5\|4), osp(5\|6), osp(7\|4), e.g. `osp(6\|6) p=1  dim h =  36   dim End h =  1296   dim A_left =  1296   dim A_right =  1296`; last line `14 pairs (g, x) run; all give A = End h`. The fifteenth pair, osp(7\|6), is not in the list of `onesided3.py`; it is checked by `python3 -c "import onesided3 as O, vdescr as V; print(O.run('osp(7\|6) p=1', V.osp(3, 3, 1), V.x_osp(3, 3, 1, 1)))"` (12 minutes): `osp(7\|6) p=1  dim h =  42   dim End h =  1764   dim A_left =  1764   dim A_right =  1764   A = End h  -- no one-sided ideals`, then `True` |
| Remark 4.17.1 (x ensures nontriviality for psl(3\|3) with p+q = 3, psq(4) with 2k = 4, and antidiag(1_n, 1_n) in psl(n\|n), n = 3, 4; in sl(2\|2) and sq(2) it does too) | `psl22.py` (1 minute); `onesided2.py` | `psl22.py`: `psl(3\|3) antidiag(1,1) ... ensures nontriviality: True`, `psl(4\|4) antidiag(1,1) ... True`, `psl(3\|3) x^1_2 ... True`, `psl(3\|3) x^2_1 ... True`, against `psl(2\|2) ... False` for both kinds of x; `onesided2.py`, boundary cases: `sl(2\|2)  x^1_1  dim h =   7, ensures nontriviality: True`, `sq(2)    x_1  dim h =   4, ensures nontriviality: True`, `psl(2\|2) ... False`, `psq(2) ... False` |

## Files

- `exactlin.py` -- exact linear algebra (rank, row echelon form, nullspace, solve) over Q and mod p;
  python-flint when present, pure Python otherwise.
- `dermeta.py`, `dereven.py` -- the derivations of Lambda(m) (x) Lambda, resp.
  K[x] (x) Lambda(m) (x) Lambda, over Lambda = Lambda(N) (Subsection 1.4).
- `volid.py` -- the free Volichenko algebra inside the free associative superalgebra; `videal.py` --
  the multilinear part M_n of the free nonassociative algebra, the identities and their
  consequences (the T-ideal); `vnew.py`, `vchar.py`, `veven.py`, `veventest.py`, `vcand.py`,
  `vfind.py`, `vlucid.py`, `vshort.py` -- the identities in degree <= 5 (new ones, S_n-characters,
  even elements, the alternating constituent, the [,]/o-monomials); `brp.py` -- the comparison with
  Bremner--Peresi, J. Algebra 322 (2009) 2000--2026.
- `mdeg.py` -- identities in a fixed multidegree, Kostka numbers, the streaming echelon mod p;
  `gens.py` -- the new identities of degrees 3, 4, 5 (cached in `gens.pkl`); `run.py` -- one
  multidegree at a time; `hilb.py`, `hilb3.py` -- the free Volichenko algebra on one generator.
- `ds.py` -- h_{ad_x} against the Duflo--Serganova functor g_x.
- `exc3.py`, `d21a.py`, `lin.py` -- rational models of ag(2), ab(3) and osp_a(4|2) (and the linear
  algebra they use); `nontriv.py` -- the Volichenko algebra h_{ad_x} (closure, Baranov's identities,
  simplicity, the Burnside closure) for these; `ospcheck.py` -- osp(4|2) rebuilt independently
  inside gl(4|2).
- `vdescr.py` -- osp(2m+e|2n), spe(n), psq(n) as matrix superalgebras and the descriptions of
  Subsection 3.4.
- `iyer.py`, `vdmeta.py`, `vdcheck.py` -- the metabelian algebra K[Z] in K[x] (x) Lambda[y] and the
  derivations of Subsections 3.6--3.9.
- `ideals.py`, `onesided.py`, `onesided2.py`, `onesided3.py`, `psl22.py` -- one-sided ideals by the
  Burnside closure of the left and right multiplications (mod 2^61 - 1); `hamvol.py`, `ham4.py` --
  the Hamiltonian algebras h'(0|n).

Every module the scripts import is in this folder.
