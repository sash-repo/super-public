# dpcohom

**Cohomology of Lie (super)algebras in characteristic 2 as a derived functor**

D. Leites, O. Lozhechnyk. Status: in preparation.

The scripts build, over F_2 (and, for Section 10, over Z reduced mod p > 2), the enveloping
algebra U(g) of a Lie superalgebra with the squaring relations x^2 = s(x) for odd x, the free
resolution V = U(g) ⊗ Λ(g_ev) ⊗ Γ(g_od) of the trivial module, its dual (the divided power
cochain complex), and the restricted enveloping algebras of all restrictednesses of
characteristic 2 (2-, 2|4-, (2,−)-, (2,4)- and (2,4|4)-structures), whose Ext is computed
independently from free resolutions over the finite-dimensional algebra. Section 11 uses
bit-mask models of the Hamiltonian Lie algebras over the divided power algebras O(2;N) and
O(4;N) at p = 2. All arithmetic is exact (bit vectors over F_2, `../lib/exactlin.py` for ranks
mod p and over Q).

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r dpcohom/requirements.txt
cd dpcohom
../venv/bin/python run.py
```

Every script is run from inside this folder; it finds `../lib/exactlin.py` by itself. The
scripts take no arguments (except `ham4.py`, see below) and most of them also write their output
to a log file of the same name (`run.log`, `runrest.log`, ...), created by the run; `reps.py`
reads the `ham4.log` written by `ham4.py`. Each line of a check ends with
`OK`/`AGREE`/`EQUAL` or prints the numbers quoted in the text.

## Claims of the text and the scripts behind them

Numbers are those printed in the current version of the text. Times are wall-clock on a laptop.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Theorem 3.3 (the resolution V = U(g) ⊗ C is free and exact, so dpH = Ext over U(g)); Theorem 2.3 (PBW, by associativity of the rewriting); Example 2.5 | `smoke.py`, `run.py` (each < 1 s) | `U(g) assoc: OK  d^2=0: OK  exact to degree 4: OK` (degree 3 for the larger samples) for every sample satisfying the axioms; for `K[f]: <e\|f>, s(f)=e` (Example 2.5) `dpH^n(g;K) = [1, 1, 0, 0, 0]` |
| Theorem 4.2 (Lebedev's cochains of [BGL, Sec. 4.1] are the divided power cochains, n <= 3: his d^1, d^2 coincide with the dual of the resolution) | `run.py` (part 1) | `d^1: agree   d^2: agree` for every sample, `total mismatches over 25 random cochains per case: 0` |
| Theorem 6.4 (u_{2,4}(g) well defined: centrality, basis independence, dim = 2^{dim g_+} 4^{dim g_-}); Remark 6.6 (Hochschild's sequence gives the kernel of H^2_res -> H^2) | `runres.py` (< 1 s) | part C: `dim u_{2,4} = 2048, expected 2^3 4^4 = 2048  OK`, `(iii) ... for all 15 non-zero elements of g_-: OK`; parts A and B end in `OK`; part D prints H^n_res, H^n and the kernel dimension predicted by Hochschild's sequence |
| Theorem 7.3 (the restricted cochains are divided power cochains, all degrees), Theorem 7.6 (the restricted closure of a (2,4)-structure), Remark 7.4 (the complex of [EM] is incomplete in degrees >= 4), Corollary 9.2 (u_{2\|2} = u_{2\|4}) | `runrest.py` (about 1 minute on an idle machine) | `H^n = [...]   Ext^n_u = [...] ... ==> AGREE` for every sample of parts A-D (see the note on sample A6 below); part E: `dim C^n([EM])` falls below `Ext^n` in degrees >= 4, e.g. `[1, 1, 1, 1, 0, 0, 0]` against `[1, 1, 1, 1, 1, 1, 1]` |
| Example 7.9 (h_I(3;1): H^n_res(g;K) = 1,1,7 and H^n_res(g;g) = 0,3,3) | `runrest.py` (part C4) | `M = K   H^n_res(h_I(3;One);M) = [1, 1, 7]`, `M = g   H^n_res(h_I(3;One);M) = [0, 3, 3]` |
| Example 7.10 and Corollary 7.7 (oo^{(1)}_{IΠ}(3\|2): ten Z/2-gradings, two with a (2,4)-structure, dim L = 14, dim u_{2,4\|4} = 2^14; H_res(g;K) = 1,0,10, H_res(g;g) = 0,2,0, H(g;K) = 1,0,4, H(g;g) = 0,2,2) | `ex244.py` (about 15 minutes on an idle machine, up to an hour under load; `ex244.py quick` skips the M = g runs) | `Z/2-gradings over F_2 (D^2=D): 10`, two lines `(2,4) on g_ev exists: True`, `dim u_2(L) = 2^14 = 16384 vs dim u_{2,4\|4} = 16384: EQUAL`, `H^n_res ... [1, 0, 10]`, `[0, 2, 0]`, `ordinary ... [1, 0, 4]`, `[0, 2, 2]` |
| Theorems 8.1 and 8.4 (the comparison sequence, also for (2,4)-structures; θ value by value) | `runseq.py` (a few seconds) | for every sample S1-S7: `exact at ... : OK` at the four inner spots and `the connecting map = theta on all ... ordinary 1-cocycles: OK` |
| Theorem 8.2 (E_1 of the square-weight spectral sequence, the components (1,0), (−1,1), (−3,2)); Remark 8.3 (Heisenberg algebra, zero 2-structure, trivial coefficients: d_2: E_2^{3,0} -> E_2^{0,2} is nonzero) | `ssN.py` (a few seconds; the details go to `ssN.log`) | in `ssN.log`, `(2) E_1^{i,j} = ...: OK, all 12 spots` for every sample; for `S3 2-str Heisenberg, [2]=0`, `M = K`: `n=3: (3,0):1 1 0 0 0` and `n=4: ... (0,2):6 6 5 5 5`, i.e., `full SS: E_r = E_inf from r = 3` |
| Theorem 10.1 (two lattices in one complex over Z: integrality, Φ = diag(c_X), agreement for n <= p−2); item 4): AH^n = 1,1,1,2,1,1,2 (p=3) and 1,1,0,0,1,2,1 (p=5) for sdim (1\|1), [f,f] = 2e | `runodd.py` (a few seconds) | parts A-D end in `OK`; part E: `p = 3   dpH^n = [1, 1, 0, 0, 0, 0, 0] ... AGREE;   AH^n = [1, 1, 1, 2, 1, 1, 2]` and `p = 5 ... AH^n = [1, 1, 0, 0, 1, 2, 1]` |
| Theorem 10.6 and Example 10.7 (polynomial functions on Πg = divided power lattice, Grassmann-neighborhood functions = anti-symmetric lattice; the tables for osp(1\|2), the Heisenberg superalgebra of sdim (1\|2) and gl(1\|1)) | `sshcheck.py` (a few seconds) | `poly=dp grass=AH` on every line; e.g. for osp(1\|2), `p=3  poly H^n=[1, 0, 0, 1, 0, 0, 0]  grass H^n=[1, 0, 2, 3, 1, 6, 5]`; for gl(1\|1), `p=3 ... poly H^n=[1, 1, 0, 2, 2, 0, 2]  grass H^n=[1, 1, 0, 2, 2, 1, 4]` |
| Subsec. 11.3, item 6) (dim Der g/ad g = n1+n2+2 for n1,n2 >= 2, and 2n2+2 for n1 = 1, n2 <= 4); Proposition 11.6 and the comment after it (dim H^2 = n1+n2−1, also for n1 = 1); Theorem 11.8 (no 2-structure); Remark 11.9 ((2,4)-structures; H_res = 1,0,3,0 and 0,2,0,4 for h^{(1)}(2;(1,1);ω_2)) | `ham7.py` (about 10 minutes) | `SUMMARY` lines: `('g0(1,2)', 6, ..., [1, 0, 2], [0, 6, 6])`, `('g0(1,3)', 14, ..., [1, 0, 3], [0, 8, 22])`, `('g0(1,4)', 30, ..., [1, 0, 4], [0, 10, 54])`, `('g0(2,2)', 14, ..., [1, 0, 3], [0, 6, 9])`, `('g0(2,3)', 30, ..., [1, 0, 4], [0, 7, 19])` (the last two lists are H^0,H^1,H^2 with coefficients K and g; H^1(g;g) = Der/ad); `RESTRICTABLE (2-structure): False` for every algebra; in the `(2,4)-STRUCTURES` part, only g0(1,1), g2(1,1), g0(1,2), g0(2,1) carry one, and for g2(1,1) `H^n_res` = `[1, 0, 3, 0]`, `[0, 2, 0, 4]` |
| Statement 11.12 (four indeterminates, N = (1,1,1,1) and (2,1,1,1)): items 1)-3) | `ham4.py N1,N2a,N2b` (up to an hour; `ham4.py N1` alone takes a few minutes), then `reps.py` (which reads `ham4.log`) | for N=(1,1,1,1): `omega0 ... 'h1': 14, ..., 'outer': 7, 'H2': 7`; the classes printed by `reps.py`: `Pf,h2,outer,H2 (0, 14, 6, 3) 35`, `(1, 15, 4, 0) 27`, `(1, 15, 5, 0) 1` (the exception B_13 = B_24 = 1); for N=(2,1,1,1) the `omega2 classes by invariants` of `ham4.log`: `'outer': '7'` (7 choices of B), `'outer': '6'` (28), both with `'h2': '30'`, `'H2': '4'`, and `'h2': '31'`, `'outer': '5'`, `'H2': '0'` (28) |
| Statement 11.12, item 4) (ω_1 has the invariants of ω_0) | `om1.py` (a few minutes) | `(1, 1, 1, 1) j= ... 'h1': 14, ..., 'outer': 7, 'H2': 7` and `(2, 1, 1, 1) j= ... 'h1': 30, ..., 'outer': 8, 'H2': 8` |
| The remark after Statement 11.12 (the ad-rank distribution splits item 2) and item 3) in two for N = (1,1,1,1)) | `adall.py` (a few minutes) | five classes `(Pf, ad-rank distribution) count [B's]`: two with Pf = 0 (15 and 20 choices), three with Pf = 1 (15, 12, 1) |

The numbers of the text that are proved in the text (Example 2.5, Proposition 6.5, Lemma 11.5, Proposition 11.6,
Lemma 11.11, Section 12) are not listed; the scripts above also re-check several of them.

**Two samples are deliberately outside the hypotheses**: in `run.py` the sample
`(2|1) solvable, s(f)=e0` and in `runrest.py` sample A6 do not satisfy the axioms of the
squaring (the scripts print `axioms fail`/`FAILED` for them first), so the theorems do not apply,
and the `FAIL`/`MISMATCH` lines that follow (including a negative "dimension") are the expected
output: they show that the axioms are necessary for the checks to pass.

## Files

- `dpco.py` — U(g) by rewriting (PBW), the coalgebra C = Λ(g_ev) ⊗ Γ(g_od), the resolution
  V = U ⊗ C, the dual complex; F_2 linear algebra on bit vectors.
- `lebedev.py` — the cochains of [BGL, Sec. 4.1] implemented verbatim and compared with `dpco.py`.
- `smoke.py`, `run.py` — Sections 2-5.
- `resco.py` — finite-dimensional restricted enveloping algebras, free resolutions, Ext.
- `nalt2.py`, `hamcheck.py`, `run1.py`, `hI.py` — Hamiltonian algebras of non-alternating forms
  at p = 2; `hI.py` gives the structure constants of h_I(3;1).
- `runres.py` — Section 6 (u_{2,4}) and Hochschild's sequence in low degrees.
- `restcx.py`, `runrest.py` — Section 7 (restricted cochains in all degrees, the restricted closure).
- `oo.py`, `wk.py`, `ex244.py` — the ortho-orthogonal Lie superalgebras and Example 7.10.
- `seq5.py`, `runseq.py`, `ssN.py` — Section 8.
- `podd.py`, `runodd.py`, `sshcheck.py` — Section 10 (p > 2).
- `ham7.py`, `grade24.py` — Subsections 11.1-11.9 (two indeterminates).
- `ham4.py`, `reps.py`, `om1.py`, `adall.py` — Subsection 11.10 (four indeterminates).
