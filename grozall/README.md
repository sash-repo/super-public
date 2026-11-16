# grozall

**P. Grozman operators on symplectic, periplectic, contact and pericontact (super)manifolds**

D. Leites, O. Lozhechnyk. Status: in preparation.

The scripts classify the invariant bilinear (and linear) differential operators between
tensor fields over the Lie (super)algebras of vector fields preserving a symplectic form
(h(2m)), an even symplectic or a periplectic form on a supermanifold (h(2m|n), le(k)), a
contact structure (k(2n+1), k(2n+1|m)) and a pericontact structure (m(n) and the Buttin
family b_{a,b}(n)). In each case translation invariance makes the coefficients constant,
the first fundamental theorem for the zero part reduces the ansatz to complete contraction
schemes (in the contact cases: to monomials in an invariant frame), and invariance under
the generators of degree 1 is then a finite linear system. Everything is exact rational
arithmetic (Python `fractions` and `../lib/exactlin.py`, fast when python-flint is
installed and correct without it; sympy only for the gcd of minors in `contact/`).

The code is in five subfolders, one per group of Sections of the text:

| Folder | Sections | Subject |
|---|---|---|
| `symplectic/` | 2-9 | h(2m) on metrics and on fibers of every type; the operator hGz, hGz_{1,0}, Grozman's case P4) |
| `supersymplectic/` | 10-11 | h(2m\|n) (even form) and le(k) (periplectic form) on supermanifolds |
| `contact/` | 13-21 and 22-29 | k(2n+1) and k(2n+1\|m): densities, metrics, the operator kGz, the contact complex, the descent |
| `pericontact/` | 30-34 | m(n) and b_{a,b}(n) |
| `gmo/` | 36-39 | symmetries of the modules of differential operators; the tables of Gargoubi-Mathonet-Ovsienko |

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r grozall/requirements.txt
cd grozall/symplectic
../../venv/bin/python final.py 1
```

Every script is run from inside its own subfolder; it finds `../../lib/exactlin.py` by
itself. Some scripts save their kernel to a small `.pkl` file in the same folder and others
read it back; the order is stated in the table ("after ..."). `contact/runn.py`, `contact/descend.py`, `pericontact/runm.py`, `runb.py`,
`unarym.py` also write their output to a `.log` file beside them, and
`contact/mkformula.py` to `kgzform.tex`. The times are those of a
laptop under heavy load; on an idle machine they are several times shorter. Where the
full arguments take hours, the table gives the arguments that were run and states the
full ones.

## Claims of the text and the scripts behind them

Numbers are those of the text.

### `symplectic/` (Sections 2-9)

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Statement 4.1 (hGz for m=1) and Remark 4.1.1 (the printed formula differs in four places, one of them the sign of the dxdy Poisson term) | `m1form.py`; `verify.py 1` | `corrected m=1 formula matches the universal one: True`, `printed sign of the dxdy Poisson term matches: False`; `Gz  ==  1/2 * (H + E1 + E2 - P)   [Grozman's m=1 shape]` |
| Statement 5.1 (for m=2 no combination of the four plane blocks is invariant) | `explicit.py 1`; `explicit.py 2` (4 min) | m=1: `invariant combinations among ['H', 'E1', 'E2', 'P']: 1`; m=2: `... : 0` |
| Statement 5.2 (formula (Gzany) for every m, symmetric) | `final.py 1`; `final.py 2 6`; `verify.py 2 6` (8 min each) | `the printed formula equals the engine's answer (x2): True`, `it is not the zero operator: True`, `invariant under ALL Hamiltonians of degree 1` ... `4: OK`; `verify.py`: `Gz(A,A') = Gz(A',A)?  True`. The second argument bounds the number of terms of the random test fields; `final.py 2` (full fields) and m = 3 take more than two hours |
| Statement 6.1 (order 2: dimension 1 for m=1, 2; order 4, m=1: zero) | `run_gz.py 1 2` (3 min); `run_gz.py 2 2 2 1` (10 min; the fourth argument tests one cubic Hamiltonian, which suffices by Lemma 3.1 3); with all 20 cubics the run takes over two hours); `run_gz.py 1 4` (45 min) | `dimension of the space of INVARIANT OPERATORS of order 2: 1` (m=1 and m=2); `... of order 4: 0` |
| Statement 6.2 (the algebra of metrics: commutative, not associative, not Jordan, not alternative) | `algebra.py 1` | `commutative:  True`, `associative:  False`, `Jordan:       False`, `alternative:  False` |
| Statement 7.4 (the fiber model reproduces Sections 3-6) | `rungen.py --m 1 --lam 2 --mu 2 --nu 2 --order N` for N = 0, 1, 2, 3, 4; `rungen.py --m 2 --lam 2 --mu 2 --nu 2 --order 2`; `memberrank.py --m 2 --lam 2 --mu 2 --nu 2 --order 2`; `canon.py 2` (50 min) | `DIMENSION ... of order 0: 1`; orders 1, 3: `no schemes: the slot count is odd`; order 2: `1` (m=1 and m=2); order 4: `0`; `146 schemes; 22 independent operators landing in the fiber`; `canon.py`: `independent operations: 22` |
| Statement 8.1 (hGz_{1,0} exists and is unique, m=2, 3; the 1086 schemes span 31) | `rungen.py --m 2 --lam 2,1 --mu 2 --nu 2,1 --order 2` (25 min); the same with `--m 3` (30 min); `memberrank.py --m 2 --lam 2,1 --mu 2 --nu 2,1 --order 2` (13 min); `verifygen.py gen_m2_2-1_2_2-1_N2.pkl` (after the first, 50 min) | `ansatz schemes: 1086`, `DIMENSION of the space of invariant operators of order 2: 1` (m=2 and m=3); `1086 schemes; 31 independent operators landing in the fiber`; `VERDICT: PASS` |
| Statement 8.2 (order 0: dimension 2; order 4: zero) | `rungen.py --m 2 --lam 2,1 --mu 2 --nu 2,1 --order 0`; the same with `--order 4` (20 min) | `... of order 0: 2`; `ansatz schemes: 3534`, `... of order 4: 0` |
| Statement 8.8 (formula (Gz10) = pi(Theta), one expression for all m, values in V_(2,1), nonzero) | `proj.py`; `gz10check.py --m 2` (after `rungen.py --m 2 ... --order 2`, 5 min); `gz10check.py --m 3` (14 min) | `projector self-test (m=2, lambda=(2, 1)): PASS`; `formula = 1/5 x (the engine's operator)`, `VERDICT: PASS` (m=2); `formula with 20 terms, m = 3`, `VERDICT: PASS` (m=3) |
| Statement 9.1 (the counts of P4): 2, 2, 2, 4) | `rungen.py --m 2 --lam 2 --mu 2 --nu 2,1 --order 1`; `--m 2 --lam 2,1 --mu 2 --nu 2 --order 1`; `--m 2 --lam 2,1 --mu 2,1 --nu 2,1 --order 1` (25 min); `--m 3 --lam 2,1 --mu 2,1 --nu 2,1 --order 1` (45 min) | `DIMENSION ... of order 1: 2`, `2`, `2`, `4` |

### `supersymplectic/` (Sections 10-11)

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Statement 10.3 (the reductions superized; h(2m\|n) generated by degree <= 3) and the conventions behind every count below (the sign of the bracket and of the Lie derivative, the contraction by omega^{ij}) | `test_ssuper.py`; `test_speri.py` (3 min); `liesign.py` (3 min); `sbrack.py` | `classical limit: PASS`, `LIE_SIGN = (0, 0, 0, 0, 0, 0, 0, 0)`; `k=3: image 66 of dim h_2 = 66  OK`, `sdim S^2(W^*) = (k^2|k^2)`; `PASS: exactly eps = 0`; `odd form k=2, omega sign +1: invariant (pT=1) = True` |
| Statement 10.5 (for n=0 the superized computation reproduces Sections 2-6) | `srun.py 1 0 0`; `srun.py 1 0 2`; `srun.py 1 0 4` (3 min); `srun.py 2 0 2` (15 min); `sspan.py 2 0 2` | `DIMENSION of the space of invariant operators of order 0: 1`; `... of order 2: 1`; `... of order 4: 0`; `... of order 2: 1`; `span of dimension 22 as operators into the fiber` |
| Statement 10.7 (hGz superizes: dimension 1 in sdim (2\|1)...(4\|2); one combination of 13 schemes in these and in (2\|0), (4\|0), (6\|0), (4\|3)) | `srun.py m n 2` for (m,n) = (1,1), (1,2), (1,3), (1,4), (2,1), (2,2) (2-25 min each); `sverify.py 1 3 2` (after `srun.py 1 3 2`); `suniv.py` (after `srun.py 1 3 2`, 20 min) | `DIMENSION of the space of invariant operators of order 2: 1` in each; `VERDICT: PASS`; `UNIFORM IN THE SUPERDIMENSION: YES` |
| Statement 10.9 (formula (Gzsuper), = -2 times (Gzany)) | `sblocks.py` (20 min); `sblockprint.py`; `sgz9r.py` (after `srun.py 1 3 2`) | `invariant: YES   non-zero: YES` for every superdimension; the display `\Gz(2m|n)=\mathsf p-\mathsf g_1'-...+\frac{1}{2}\,\mathsf m.`; `formula (9) = -1/2 x the read-off operator` in each superdimension |
| Statement 10.11 (commutative; no order 4 in (2\|1), (2\|2)) | `ssym.py` (after `srun.py 1 3 2`); `srun.py 1 1 4`; `srun.py 1 2 4` (10 min) | `symmetric on 3/3 random pairs, antisymmetric on 0/3` in each superdimension; `DIMENSION ... of order 4: 0` twice |
| Statement 11.2 (periplectic: order 2 dim 3 for k=2, 3, split 1+2; order 0 dim 3; order 4 zero) | `srun.py 2 2 2 2 4 3 odd`; `srun.py 3 3 2 2 4 3 odd`; `srun.py 2 2 0 2 3 3 odd`; `srun.py 3 3 0 2 3 3 odd`; `srun.py 2 2 4 2 6 3 odd`; `sverify.py 2 2 2 odd`; `soddsym.py`; `soddinfo.py` (5 min) (all after the `srun.py` runs) | `... of order 2: 3` (k=2, 3); `... of order 0: 3` (k=2, 3); `... of order 4: 0`; `VERDICT: PASS`; `order-2 space of dimension 3 = 1 symmetric + 2 antisymmetric` (k=2, 3); `3 of 3 still invariant, 3 non-zero` at k = 2, 3, 4 |
| Statement 11.4 (the unary invariants: id, ptr, d(ptr A); nothing of order 2) and the even contrast | `soddcensus.py 2`; `soddcensus.py 3`; `soddshapes.py 2`; `soddshapes.py 3`; `sevencensus.py 2 0`; `sevencensus.py 1 3` | `order 0 -> metrics ... DIMENSION 1`, `order 0 -> functions ... DIMENSION 1 (parity odd)`, `order 1 -> 1-forms ... DIMENSION 1`, `order 2 -> metrics: 3 schemes, kernel 0, DIMENSION 0`, `order 2 -> functions: 4 schemes, kernel 3, DIMENSION 0`; `d(ptr A) proportional to dv: YES`; even form: only `order 0 -> metrics ... DIMENSION 1` |
| Statement 11.6 (the three are the compositions; c_2 = -64, c_3 = -576) and Corollary 11.7 | `sodddecomp.py 2`; `sodddecomp.py 3`; `soddfactor.py 2`; `soddfactor.py 3` | `C1 = ... = (1/2) e3`, `C2 = ... = (1/2) e1`, `C3 = dvA . dvB = (-64) e2` (k=2), `(-576) e2` (k=3), `DIMENSION SPANNED BY THE COMPOSITES: 3 of 3`; `every invariant of order 2 vanishes on them: YES` |
| Statement 11.10 (the three operators of order 0; formula (odd03) with 16 fitted coefficients) | `sodd0.py`; `sodd00.py 2`; `sodd00.py 3`; `soddsign.py` | `=> the named operators SPAN it`, `on ptr-free metrics: T1 = T2 = 0 by construction, T3 SURVIVES`; over an even form `the three named operators span a space of dimension 1`; `soddsign.py`: `against the closed rules: AGREE everywhere`, `held-out check: 324 components, 0 mismatch` |
| Statement 11.14 (formulas (hamodd), (C1form), (C3form)) | `soddexplicit.py 2`; `soddform.py` | `nullity 0`, `0 mismatches  PASS` at k=2 and, with the same coefficients, at k=3 |
| Statement 11.16 and Corollary 11.17 (the traces: {pi,pi'}, -{pi,pi'}, -{pi,pi'}) | `soddtrace.py` | `ptr C1 = c * {pi,pi'} with c = 1`, `ptr C2 = ... c = -1`, `ptr C3 = ... c = -1`, `ptr(C1+C2) = 0 : True`, `ptr(C1-C2+2C3) = 0 : True` |
| Statement 11.18 (order 0: split 2+1; traces -pi pi', pi pi', 0) | `soddnames.py` | `oGz_1 + oGz_2  SYMMETRIC`, `oGz_1 - oGz_2  ANTIsymmetric`, `ptr oGz_1 = (ptr A) A' = (-1) pi pi'`, `ptr oGz_2 = ... = (1) pi pi'`, `ptr oGz_3 = ... identically 0` |
| Statement 11.23 (the two fibers: no invariant S^2 <-> Lambda^2, unary invariants) | `test_sfib.py` (17 min); `sunfib.py 2`; `sunfib.py 3` | `ALL GOOD`; `S^2 --> Lam^2, order 0 ... DIMENSION 0`, `order 2 ... DIMENSION 0`, `Lam^2 --> S^2 ... DIMENSION 0` (both orders), `S^2 --> S^2, order 0 ... DIMENSION 1`, `Lam^2 --> Lam^2, order 0 ... DIMENSION 1`, order 2: `DIMENSION 0` |
| Statement 11.27 (the census of the eight configurations) | `runfib.py odd 2 0`; `runfib.py odd 2 2` (8 min); `runfib.py odd 3 2` (60 min); `runfib.py odd 2 4` (45 min) | one line per configuration: order 0: `3 1 0 1 0 1 0 0`, order 2: `3 4 0 0 0 0 0 0` in the order SSS, SS-Lam, S-Lam-S, S-Lam-Lam, Lam-SS, Lam-S-Lam, Lam-Lam-S, Lam-Lam-Lam; order 4, SS-Lam: `dim  0` |
| Statement 11.28 (S^2 x S^2 -> Lambda^2: order 2 dim 4, split 3+1, d(ptr A)^d(ptr A') among them) | `fibnamed.py 2 2` (2 min); `fibnamed.py 3 2` (15 min) | `symmetric: 3`, `antisymmetric: 1`, `d(ptr A)^d(ptr B) lies in the space: True` |

### `contact/` (Sections 13-29)

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Sections 14-16: the algebra k(3), the frame, the modules T(k,lambda); Propositions 15.1 and 16.1 (the scheme calculus) | `test_contact.py` (3 min); `test_gen.py` (25 min); `test_contactn.py`; `calib.py` (20 min) | `27 checks, 0 failed`; `8 checks, 0 failed`; `0 checks failed`; `23 of 23 calibration cases passed` |
| Statement 17.1 (the densities, n=1: dim 1 at W=0, 2, else 0, no jump for W <= 6, generic for W = 8, 10; n = 2, 3) | `runden.py` (75 min); `runn.py dens 2 4` (7 min); `runn.py dens 3 3` (11 min); `namedn.py` (10 min) | `W= 0 ... generic dim=1`, `W= 2 ... generic dim=1`, `W= 4`, `6`, `8`, `10`: `generic dim=0`, odd W: `no sp(2)-invariant scheme at all`, `grid: dim is constant` and `gcd of 24 random maximal minors: 1`; `dims [1, 1, 1, 1, 1]` at W = 0, 2 and `[0, ...]` otherwise; `the bracket family is invariant`, `at lambda = mu = -1 it is the contact bracket` |
| Statement 18.1 (the linear census: identity, d: F_0 -> T(1,0), T(1,n) -> F_{n+1}; for n=1 also T(1,0) -> T(1,1)) | `runlocus.py` (n=1, 15 min); `runn.py linear 2` (8 min); `runn.py linear 3` (45 min); `namedn.py` | n=1: `T(0,lam) -> T(1,lam+0)  W=1 : only at lam = ['0']`, `T(1,lam) -> T(0,lam+1)  W=1 : only at lam = ['1']`, `T(1,lam) -> T(1,lam+1)  W=2 : only at lam = ['0']`, all other: identity; `T(0,l) -> T(1,l+0)  W=1 : [('0', 1)]`, `T(1,l) -> T(0,l+1)  W=1 : [('2', 1)]` (n=2), `[('3', 1)]` (n=3); `the second-order arrow T(1,l) -> T(1,l+1): does NOT exist` (n >= 2) |
| Statement 19.1 (metrics, n=1: the table, the jump locus lambda+mu=0) | `runmetric.py` (26 min); `runlocus.py` | `T(2)xT(2)->T(k3) W=0 ... generic dim=1`, W = 2, 4: `generic dim=0`, odd W: `dim V_0=0`; `T(2)xT(2)->T(0) W=2 : ... gcd = lam + mu`, `T(2)xT(2)->T(4) W=2 : ... gcd = 1` |
| Statement 19.2 (kGz: unique, on lambda+mu=0, antisymmetric at lambda=0, order 2, formula (kgzinv), not a composition); Subsec. 19.3 (in coordinates) | `runexc.py` (17 min); `props.py`; `fitexc.py`; `formula.py`; `mkformula.py` (also writes `kgzform.tex`) | `lam=... mu=... : dim = 1` on the line; `B(F,G) = -B(G,F) on 9 pairs: yes`, `classical order = 2`; `fit: -3*OmOmPi + -2*OmOmRF + 2*OmOmRG + 4*OmThFThG + -2*OmThFdvF + 2*OmThGdvG` (minus formula (kgzinv)); the 26-term display |
| Statement 19.4 (n=2: nothing at W=2 for k3 = 2, 4; for k3 = 0 dimension 1 exactly on lambda+mu = n-1) | `runn.py metric 2` (40 min); `runn.py line 2` (30 min) | `k3=2 W=2 ... [... 0 ...]`, `k3=4 W=2 ... 0`; the line scan has its single `1` where lambda+mu = 1 |
| Proposition 20.2, Lemma 20.4, Remark 20.7 (the descent) | `restrict.py`; `descend.py equiv 2`; `descend.py iota 2`; `descend.py rho 2 1 1 0 2`; `descend.py rho 3 0 0 0 6`; `descend.py where 2 1 1 0 2`; `descend.py direct 3 1 1 0 2` (each writes `descend-<job>-<n>.log` too) | `ALL CHECKS PASS (66147 bracket pairs)`; `pi^* IS equivariant on every test`; `0 of 57 (generator, field) pairs fail`; `unitriangular in the number of frame derivatives: True` on the densities; `k_2 ( 9 generators): 4 fail   t*p1*q1, t*p1^2, t*q1^2, t^2`; `dim K(3)=1, of them 1 descend to a nonzero operator, ... FAIL invariance` |
| Statement 25.2 (the frame is a weight basis in the symmetric shape, not in the orthonormal one) and Corollary 25.3 | `torus.py` (5 min); `test_khyp.py`; `test_kmix.py` | `NOT weight vectors: D1, D2` in the orthonormal shape, `every frame operator is a weight vector of t` in the symmetric one, the weights of the frame as in the table; `W = 2: ansatz 17 terms, of t-weight 0: 7` etc. (the numbers of Corollary 25.3); `0 checks failed` |
| Statement 25.5 (the number of the weights: 1, except 2 for k(1\|2), second multiplier D_xi D_eta f) | `weights.py` (k(1\|n), n <= 3); `mixchecks.py` | `n=1: ... dim = 1`, `n=2: ... dim = 2` (the second `rho(f) = -1*d1.d2 + ...`), `n=3: ... dim = 1`; `the space of weights is 1-dimensional (got 1)` for six algebras and `2-dimensional (got 2)` for k(1\|2); `0 checks failed` |
| Statement 25.6 (dim V_0, W = 0..3) | `runhyp.py v0` | the rows `[1, 0, 3, 0]` (k(3) and n >= 1), `[1, 2, 3, 4]` (k(1\|1)), `[1, 0, 6, 0]` (k(1\|2)), `[1, 0, 3, 4]` (k(1\|3)), both shapes `AGREE` |
| Statement 26.1 (k(1\|1): the census and its loci) | `test_super.py` (`47 checks, 0 failed`); `runsuper.py 1` | `generic dim = 1` at W = 0, 1, 2 and `0` at W >= 3; `gcd of the maximal minors: 4*lam**2*mu + 4*lam*mu**2 + 4*lam*mu` at W=3, `mu` at W=5 |
| Statement 26.2 (the two families) | `namedsuper.py`; `mixchecks.py` | `the formula is invariant: True` at every weight; `B_2 at lam = mu = -1 is the contact bracket: ... yes` for n = 1, 2, 3; `all checks passed` |
| Statement 26.3 (k(1\|3), W <= 4; k(1\|2), W <= 3, second weights add) | `runsuper.py 3` (3 min); `runhyp.py k13`; `runhyp.py k12`; `run2.py` (7 min) | `generic dim = 1` only at W = 0, 2, `no jump on the grid`; in `runhyp.py` only W = 0, 2 carry `1`; for k(1\|2): `nu = (a, b -> a+b):  dim = 1`, nothing at W = 1, 3 |
| Statement 26.4 (k(1\|1): linear operators id and D: F_0 -> F_{1/2}; the exceptional ones are compositions) | `unarysuper.py 1`; `comp1.py` | `W=1: lam=0 dim=1 1 D_1`, `W=2..5: none`; `D o B_2 spans it: True`, `3 candidate compositions span it: True`, `both are (D x id) and (id x D) followed by the multiplication: True` |
| Statement 27.1 (k(3\|1), W <= 5) | `runhyp.py k31` (26 min) | each line: parity 0 has dimension `1` at W = 0, 2 and `0` otherwise; parity 1 is `0` throughout |
| Proposition 28.2 (both maps are equivariant on the densities, odd direction) | `descendsuper.py`; `restrict.py` | `ALL PASS: True` |

### `pericontact/` (Sections 30-34)

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Sections 31-32: m(n), the frame, the densities F_{a,b} | `test_mperi.py` | `57/57 checks pass` |
| Proposition 31.3.1 (two weights) | `weightsm.py` | `dim m_0/[m_0,m_0] = 2` for n = 1, ..., 4, `they kill [m_0,m_0]: True` |
| Statement 33.1.1 (the bilinear census over m(n): W <= 6, 4, 3 for n = 1, 2, 3) | `runm.py` (one hour; also writes `runm.log`) | at `W=0 kappa=+0` and `W=2 kappa=+0` dimension `1` at every sample weight for each n; for n=1 also `W=1 kappa=+1`, `W=3 kappa=+1` and `W=2 kappa=+2` at the weights of item 3), 4); every other (W, kappa): `0` |
| Statement 33.1.2 (linear operators: identity, and for n=1 the odd T_1 on a=0) | `unarym.py` (8 min; also writes `unarym.log`) | m(1): `W=1 kappa=+1 ... + T_1` at the sample weights with a=0 only; m(2), m(3): only `W=0 ... + 1` |
| Statement 34.1.1 (one weight over b_{a,b}(n)); Statement 34.2.1 (the operator Delta_lambda = X T + c R with c = 2/lambda + 2 - 2n); Statement 34.2.2 (the bilinear census over b_{a,b}(2)) | `runb.py` (12 min; also writes `runb.log`); `unarym.py` | `quotient = 1 (m(n) has 2)` for n = 1, 2, 3; for b_{1,3}(2) (lambda = -1/2): `W=2 kappa=+0 ... + X_1T_1 + X_2T_2 - 6R`; for b_{1,3}(1) (lambda = -1): `+ X_1T_1 - 2R`; the bilinear census over b_{a,b}(2) at (W,kappa)=(2,0) has the three operators listed |

### `gmo/` (Sections 37-39)

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Proposition 38.2 (T(2,lambda) occurs in gr^W M only at W=2, twice, for every lambda) and Statement 38.4 (the layers W = 0, 2, 4, 6 with multiplicities 1, 3, 2, 1 for m=1 and 1, 6, 3, 1 for m = 2, 3) | `layers.py` | `layers.py: 86 checks pass`, `W=  2   r=2 s=0 c=1   multiplicity 1`, `W=  2   r=0 s=1 c=0   multiplicity 1`; `m=1:  W=0 (mult 1), W=2 (mult 3), W=4 (mult 2), W=6 (mult 1)`, `m=2:` and `m=3:  W=0 (mult 1), W=2 (mult 6), W=4 (mult 3), W=6 (mult 1)` |
| Statement 39.3 (the tables of [GMO] reproduced; short by 1 at (0,1); 2 at (0,4/5) and 3 at (0,5/4) for k=4) | `gmo.py`; `scan_gmo.py` (14 min) | `k=4  (lam,mu)=(0,4/5): dim=2`, `k=4  (lam,mu)=(0,1): dim=5   [GMO]: b + R^2  (dim 6)`; `(lam,mu)=(0,5/4)  dim=3`, `k=4 (lam,mu)=(0,4/5) rmax=6: dim=2` |

## Notes

- `gmo/layers.py` identifies S^k D with S^k D^* (x) L^{-k}, from D = D^* (x) L^{-1}; with
  this sign the layer of kGz is W = 2, as Proposition 38.2 states.
- `contact/weights.py` imposes the cocycle condition on k_{-2} + ... + k_2; its count is
  reliable for k(1|n) with n <= 3, the range used above (for n = 4 a candidate supported in
  higher degrees passes vacuously).
- `pericontact/unarym.py` computes the linear invariants over b_{1,3}(n) for n = 1, 2, i.e.
  lambda = -1 and -1/2, where formula (PM-bvc) gives c = -2 and -6, as printed. For n = 1 it
  also finds the odd operators T_1 (W = 1) and T_1 R (W = 3), invariant at every weight;
  Statement 34.2.1 is therefore borne out by these scripts for n = 2.

## Files

- `symplectic/`: `symp.py` (polynomials, omega, Hamiltonian fields, the Lie derivative on
  metrics, the contraction schemes), `run_gz.py`, `explicit.py`, `canon.py`, `final.py`,
  `verify.py`, `m1form.py`, `algebra.py` (the metrics); `fiber.py` (irreducible
  sp(2m)-fibers), `gen.py` (the scheme ansatz for any three fibers), `rungen.py`,
  `verifygen.py`, `memberrank.py`, `proj.py` (the projector onto V_(2,1)), `iexpr.py` and
  `gz10terms.py` (the printed index expressions, evaluated independently), `gz10check.py`.
- `supersymplectic/`: `ssuper.py` (supercommutative polynomials, both forms, the
  u-encoding of metrics), `test_ssuper.py`, `test_speri.py`, `liesign.py`, `sbrack.py`,
  `sgen.py`, `srun.py`, `sverify.py`, `sspan.py`, `suniv.py`, `ssym.py`, `sblocks.py`,
  `sblockprint.py`, `sgz9r.py`; the periplectic case: `sunary.py` (module used by
  `soddcensus.py`, `sevencensus.py`), `sodd0.py`, `sodd00.py`, `soddinfo.py`, `soddsym.py`,
  `sodddecomp.py`, `soddfactor.py`, `soddshapes.py`, `soddexplicit.py`, `soddform.py`,
  `soddtrace.py`, `soddsign.py`, `soddnames.py`; two fibers: `sfib.py`, `srunfib.py`,
  `runfib.py`, `sunfib.py`, `fibnamed.py`, `test_sfib.py`; `symp.py` (copy, for the
  comparison at n=0).
- `contact/`: k(3): `contact.py`, `test_contact.py`, `gen.py`, `test_gen.py`, `fast.py`,
  `affine.py` (the weights as a matrix pencil), `sweep.py` (grid + gcd of minors),
  `unary.py`, `named.py`, `runden.py`, `runmetric.py`, `runlocus.py`, `runexc.py`,
  `props.py`, `formula.py`, `fitexc.py`, `mkformula.py`; k(2n+1): `contactn.py`,
  `test_contactn.py`, `genn.py`, `calib.py`, `linn.py`, `namedn.py`, `runn.py`,
  `restrict.py`, `descend.py`; k(1\|n) and k(2n+1\|m): `super.py`, `test_super.py`,
  `weights.py`, `gensuper.py`, `runsuper.py`, `unarysuper.py`, `namedsuper.py`,
  `comp1.py`, `run2.py`, `kmix.py`, `test_kmix.py`, `test_khyp.py`, `genmix.py`,
  `mixchecks.py`, `torus.py`, `runhyp.py`, `descendsuper.py`.
- `pericontact/`: `mperi.py` (m(n), the frame solved for, the densities, b_{a,b}(n)),
  `test_mperi.py`, `weightsm.py`, `genm.py` (the ansatz and the two gradings), `runm.py`,
  `runb.py`, `unarym.py`.
- `gmo/`: `gmo.py` (the local symmetries of D^k_{lam,mu} over vect(1)), `scan_gmo.py`,
  `layers.py` (sp(2n)-multiplicities and the layer condition).
