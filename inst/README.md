# inst

**On superinstantons**

D. Leites. Status: in preparation.

The scripts check the computations of the text: the count of the odd moduli of the N-extended
SU(2)-instantons of charge k (Statement 7.2.1: the linear system (9) for the odd matrices c_j has a
solution space of real dimension 4k), both at 't Hooft's data for k = 1, ..., 5 and at general
data of charge 2, together with the defect delta of the system read off from the dual pairs (P, Q)
of Subsection 7.2; and the dictionary of Subsection 3.6 between the linear algebraic data of type I
and Ia and the construction of Atiyah--Drinfeld--Hitchin--Manin (the quaternions in the guise of
the tau_mu, the conditions (3) as a reality condition, and the self-duality of the resulting
curvature with the orientation eps_{0123} = 1). Exact arithmetic throughout: `fractions.Fraction`
with ranks through `../lib/exactlin.py` (python-flint when installed, pure Python otherwise) in
`oddmod.py`/`oddgen.py`; sympy over the Gaussian rationals in `adhm.py`.

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r inst/requirements.txt
cd inst
../venv/bin/python oddmod.py
```

Every script is run from inside this folder, with no arguments; it finds `../lib/exactlin.py` by
itself. `oddmod.py` and `oddgen.py` also run under plain `python3` (standard library only, slower
ranks). Times are for a laptop running one script at a time.

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Statement 7.2.1 (for every j, the solutions of system (9) form a real space of dimension 4k; hence the tangent superspace of the moduli superspace is of superdimension (8k-3\|4kN)), at 't Hooft's data a = (lambda; Y), b = (0; -1), k = 1, ..., 5 | `oddmod.py` (about 5 seconds) | `k=1: real dim of c per odd index, sign variants [4, 2, 6, 4]; expected 4k=4`, ..., `k=5: real dim of c per odd index, sign variants [20, 10, 30, 20]; expected 4k=20`. The first entry of each list is system (9) as printed (signs - and +); the other three are the sign-swapped variants, shown for contrast |
| Statement 7.2.1 at general (not only 't Hooft's) data of charge 2; Subsection 7.2: the defect delta is the dimension of the space of pairs (P Hermitian, Q anti-Hermitian) with bP + aQ = 0, it vanishes at non-degenerate data, and delta = 3 for the degenerate data of charge 2 with Y = 0 | `oddgen.py` (about 7 seconds) | first the five lines of `oddmod.py` (it imports that module), then three times `2 a^+a real: True [8, 4, 12, 8] expected 8`; three times `k=2 general: coker dim 0 kernel 8`; and `degenerate (coincident centres at 0): coker 3 kernel 11` (i.e., 4k + delta = 8 + 3). The three lines `3 a^+a real: False ...` are not claims: for k = 3 the random generator does not produce data satisfying the reality condition, so those lines test nothing |
| Lemma 3.6.1 (tau_j^2 = -1, tau_1 tau_2 = tau_3 etc., (x^mu tau_mu)^+(x^mu tau_mu) = \|x\|^2 1_2); Lemma 3.6.2 (the conditions (3) amount to condition 1) of Subsection 3.5, i.e., M^+M = diag(D, D) commutes with every tau'_mu), checked on 't Hooft's data for k = 1, 2 and on data violating them; the end of Subsection 3.6 (for m = 2, n = 0 the curvature is self-dual with the orientation eps_{0123} = 1, not anti-self-dual) | `adhm.py` (several minutes; writes `adhm.log`) | 31 lines, each ending `OK`, e.g. `A. tau_1 tau_2 = tau_3   OK`, `C. 't Hooft k=2: M^+M = diag(D,D) ('type Ia' cond. 1)   OK`, `D. k=1 't Hooft: F is SELF-dual (F = *F), i.e. F^- = 0, identically in x OK`, `D. k=2 't Hooft: F is SELF-dual at 3 rational points   OK`; last line `31 checks, 0 failures` |

## Files

- `oddmod.py` -- quaternionic matrices over Q, the linear system (9) for one odd matrix c_j at
  't Hooft's data, and its real solution space by exact rank.
- `oddgen.py` -- general data of charge 2 (lambda solved for so that a^+a is real), the same count,
  and the dual space of pairs (P, Q) that measures the defect; imports `oddmod.py`.
- `adhm.py` -- the matrices tau_mu, the 't Hooft data written as M(x) = B + C x^mu tau'_mu, the
  conditions (3) and their type Ia form, and the curvature of A_mu = N^+ d_mu N, symbolically in x
  for k = 1 and exactly at rational points (first-order Taylor data) for k = 2.
