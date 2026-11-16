# max3exc

**Maximal subalgebras in the exceptional simple finite-dimensional Lie superalgebras ag(2), ab(3)
and osp_a(4|2)**

D. Leites, O. Lozhechnyk. Status: in preparation.

The scripts build exact models of osp_a(4|2) = D(2,1;a), ag(2) = G(3) and ab(3) = F(4) (root
vectors as a basis, rational structure constants, the super Jacobi identity checked on all
triples), enumerate their Z-gradings, and run, grading by grading, the search for maximal graded
subalgebras described in the text: the cases 2°-6° over all torus-invariant subspaces V of g_-1,
over lines and planes spanned by sums of two weight vectors, and over random subspaces. Every
candidate is certified maximal by the criteria of Lemmas 2.7 and 2.8 of the text. Further scripts
classify the maximal subalgebras of maximal rank (closed sets of roots), read the regular ones off
the extended Dynkin diagrams, and treat the subalgebras containing a principal sl(2). Exact
arithmetic over Q throughout (fractions, and `../lib/exactlin.py`, fast when python-flint is
installed and correct without it); `casmodp.py` works modulo a prime, see below.

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r max3exc/requirements.txt
cd max3exc
../venv/bin/python gradings3exc.py
```

Every script is run from inside this folder; it finds `../lib/exactlin.py` by itself. Output goes
to stdout. The models of ag(2) and ab(3) are cached in `.exc3cache.G3`, `.exc3cache.F4` on first
use (a few minutes).

**The searches write their results to `runs/`.** The scripts marked "two steps" read these
files, so run first

```bash
mkdir -p runs
for h in 1,0,0 0,0,1 1,0,1 0,1,1 1,1,1; do ../venv/bin/python run3.py G3 $h full > runs/G3-${h//,/}.txt; done
for h in 1,0,0,1 2,0,0,0 0,0,1,1 0,0,0,2 1,1,1,1 2,0,1,1 0,1,1,2 3,1,1,1; do ../venv/bin/python run3.py F4 $h full > runs/F4-${h//,/}.txt; done
../venv/bin/python check_d21a.py > runs/D21a-check.txt
```

Each `run3.py` call takes from 1 to about 25 minutes on a laptop (the longest is `F4 2,0,0,0`,
the contact grading of ab(3)); the calls are independent and can be run in parallel. Together
they write about 6 MB of `runs/*.pkl`.

## The gradings

The h of `run3.py` are given in the coordinates (delta; eps_1, eps_2) for ag(2) and
(delta; eps_1, eps_2, eps_3) for ab(3), as in tables (17) and (18) of the text; the files are named
after them:

| Text | ag(2), N = 1 (contact) | N = 2 | N = 3 | N = 4 | N = 5 |
|---|---|---|---|---|---|
| `run3.py G3 h` | `1,0,0` | `0,0,1` | `1,0,1` | `0,1,1` | `1,1,1` |

| Text | ab(3), N = 1 (depth 1) | N = 2 (contact) | N = 3 | N = 4 | N = 5 | N = 6 | N = 7 | N = 8 |
|---|---|---|---|---|---|---|---|---|
| `run3.py F4 h` | `1,0,0,1` | `2,0,0,0` | `0,0,1,1` | `0,0,0,2` | `1,1,1,1` | `2,0,1,1` | `0,1,1,2` | `3,1,1,1` |

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| The models: ag(2) and ab(3) satisfy the super Jacobi identity and are simple | `exc3.py` (about 25 min) | `ag(2) = G(3)    N=31  sdim 17\|14`, `ab(3) = F(4)    N=40  sdim 24\|16`, each with `super-Jacobi failures: 0` and `simple: yes` |
| Statement 3.1, tables (17)-(19): the gradings satisfying conditions (15) are 2 + 5 + 8 | `gradings3exc.py` | 6, 7 and 18 grading elements h (`total: ...` after each algebra) with their components; up to the permutations of the coordinates eps_i (the Weyl group of g_ev) they are 2, 5 and 8, the rows of tables (19), (17), (18) |
| Theorem 5.1, tables (22), (23): the maximal graded subalgebras of osp_a(4\|2), cross-check of the search against the theorem | `check_d21a.py`; `analyse2.py` (about 50 min); `osppar.py`; two steps: `ospss.py` | `check_d21a.py`: for a = 5, 1, 1/3 and both gradings the table `MAXIMAL GRADED SUBALGEBRAS`, the semi-simple rows being g_ev (9\|0), sl(2)+osp(1\|2) (6\|2) and, for a = 1 only, osp(3\|2) (6\|6); `analyse2.py`: for a = 5, 1, 1/3, both gradings, the candidates of the cases 2°-6° with `MAXIMAL (Burnside)` or `MAXIMAL (no intermediate found)`, among them `6\|2 ... non-isotr. even line` for every a and, for a = 1 only, `6\|6 ... (1\|2) non-isotr. even line`; `osppar.py`: every line `closed True MAX`; `ospss.py`: the (6\|2) answers `-> MAX` |
| Statement 1.8: the non-parabolic maximal subalgebras of maximal rank, their orders 2 and 3, the values j(a) | `regmax.py`; `classify.py`; `ident.py` (a few minutes each) | `classify.py`: `osp : 1 symmetric maximal closed sets` (9\|0); `ag : 10 ... 5 W(g_ev)-orbits` of sdim (17\|0), (11\|2), (9\|6), (9\|8), (9\|6); `ab : 13 ... 4 W(g_ev)-orbits` of sdim (24\|0), (12\|4), (12\|8), (16\|8); exhaustive for osp: `'OTHER': 0`. `ident.py`: for each class an h with the class as the fixed points of exp(2 pi i ad_h), and `order 3` for (11\|2), the A2 type (9\|6) and (12\|4), `order 2` for the others (for g_ev of ab(3) it prints `order 1` with h = (0, 0, 0, 1), integral on the even roots and half-integral on the odd ones, so exp(2 pi i ad_h) is the parity automorphism, of order 2) and `a = c2/c1 = 3 ... j(a) = 2197/144`, `a = c2/c1 = 2 ... j(a) = 343/36` |
| Proposition 4.3 (the example in ag(2), N = 3), Statements 7.1-7.5 and 8.1-8.8, the lists (25) and (31): the semi-simple maximal graded subalgebras, grading by grading, with their components and the case 2° or 6° | `run3.py G3 h full`, `run3.py F4 h full` (see above); two steps: `post3.py runs/G3-*.pkl runs/F4-*.pkl`, `tables3.py runs/G3-*.pkl runs/F4-*.pkl` | each run ends with the table `MAXIMAL GRADED SUBALGEBRAS`; its rows marked `PERFECT` are the semi-simple answers, of the sdims of the lists (25), (31) in the columns of the tables above (e.g., `G3 1,0,0`: (9\|8), (17\|0) as `g_ev`, (9\|6) twice, (11\|2) twice, and (7\|4), which is not maximal, see Remark 9.2); `post3.py runs/<file>.pkl` names every answer (`g(2) + sl(2)`, `osp(3\|2) + sl(2)`, `osp_a(4\|2)=D(2,1;a)`, ...), gives a for each copy of osp_a(4\|2), always in the S_3-orbit of 3 for ag(2) and of 2 for ab(3) (`==> D(2,1;a) with a = ...  (S_3-orbit ...)`), and marks the others `NOT semisimple`; the number of those is the number in the Statement: 137, 57, 39, 29, 29 for ag(2), N = 1, ..., 5, and 172, 324, 128, 66, 95, 88, 55, 48 for ab(3), N = 1, ..., 8 (from 1.5 to 40 min per file); `tables3.py` prints the LaTeX tables of the Statements from the same files (about 1 min for `runs/G3-111.pkl`), with the certificate `max(b)` where the search alone does not decide; these are settled by the next row |
| Lemma 2.8 applied to the answers (all of them are maximal) | two steps: `maxball.py` (about 8 min); `casmax.py`; `casmodp.py` (about 3 min) | `maxball.py`: the summary lines `('F4', (9, 4)) {'UNDECIDED', 'MAX'}` ... `('G3', (12, 10)) {'UNDECIDED'}`; `casmax.py` (40 min): the `SUMMARY`, one line per answer flagged `MAX(b)` by the search, all `MAX` (two with `irrational components`, their components being defined over a quadratic field only) except the (7\|4) of ag(2), `u_ev not perfect`; `casmodp.py`: the components modulo 10007 of the copies of (9\|4) in `F4-1001` whose components are defined over a quadratic field, each generating everything (`/ 40`) |
| Remark 9.1: the parameter a of each copy of osp_a(4\|2), computed from the invariant form and again from the Killing form of the ambient | two steps: `checka.py runs/G3-*.pkl runs/F4-*.pkl` | `nonzero entries of its own Killing form: 0` for D(2,1;5), then one line per copy of osp_a(4\|2) of sdim (9\|8) in ag(2): `ambient-Killing a = 3 ;  invariant-form a = 3` or the same with 1/3 (18 lines; the script looks at the (9\|8) answers only, so the copies in ab(3) are identified through the invariant form alone, by `post3.py`) (about 30 min) |
| Remark 9.2: g(2)+osp(1\|2) is maximal in ab(3); the (7\|4) answers sl(2)+sl(2\|1) of ag(2) are not maximal | `g2osp.py`; `check74.py`; two steps: `rank74.py` | `g2osp.py`: `u closed: True  sdim (17\|2)`, `u + 7 closed: False`, `u + 2(x)7 closed: False`; `check74.py`: `3 symmetric closed sets of sdim (7\|4)`, `contained in maximal ones of sdim [(9, 6)]`, `is_max: False`; `rank74.py`: every copy `ad z semisimple: True`, `dim span(u cap h, z) = 3` (or 2 for the random V) |
| Statement 9.4: the regular maximal subalgebras read off the extended Dynkin diagrams | `extdiag.py` (about 5 min) | for every simple system the deleted nodes with mark > 1 and the verdict `maximal True orbit k` (the W(g_ev)-orbit of Statement 1.8) or `maximal False`, the latter only for mark 4: (7\|4) in ag(2), (18\|0) in ab(3); then `orbits reached: [0] of 1`, `[0, 1, 2, 3, 4] of 5`, `[0, 1, 2, 3] of 4` |
| Statement 9.5: no super analog of the principally embedded sl(2) is maximal | `principal.py` (about 6 min) | for every even part u_ev containing the principal sl(2) and every generator x: the subalgebra A generated by u_ev and x, either `generates g` or contained in a proper subalgebra; the only `MAX` is `A of sdim (17, 2)` in ab(3) (g(2)+osp(1\|2)) |
| Remark 9.6: the irreducible branch of the case 3° in the contact gradings | `irrbranch.py` | `dim ker(E) on g_{-1} = 1  ==> g_{-1} is IRREDUCIBLE over the principal sl(2)` (ag(2)), and for u_0 = sl(2) and c(sl(2)) in both algebras: `dim u_1 = 0 ==> ... NOT maximal` |

## Files

- `lin.py` — exact linear algebra over Q (fractions).
- `d21a.py` — the model of osp_a(4|2) and its two gradings; `check_d21a.py` runs the search on it
  for a = 5, 1, 1/3 and writes `runs/D21a-*.pkl`; `analyse2.py` — the direct analysis of its
  graded subalgebras.
- `exc3.py` — the models of ag(2) and ab(3) (Chevalley basis, structure constants, checks).
- `gradings3exc.py` — the Z-gradings satisfying conditions (15).
- `sweep3.py` — the search: candidates of the cases 2°-6°, prolongs, the maximality certificate.
- `run3.py` — the driver (arguments: `G3|F4 h [full|hom]`); writes `runs/<name>-<h>.pkl`.
- `ident3.py`, `post3.py`, `tables3.py` — identification of the answers and the tables.
- `maxb.py`, `maxball.py`, `casmax.py`, `casmodp.py`, `ospss.py` — the maximality tests of
  Lemma 2.8 on the results in `runs/`.
- `regmax.py`, `classify.py`, `ident.py` — maximal rank (Statement 1.8); `check74.py`,
  `rank74.py`, `g2osp.py`, `osppar.py` — single cases; `checka.py` — the parameter a.
- `extdiag.py`, `principal.py`, `irrbranch.py` — Section 9.
