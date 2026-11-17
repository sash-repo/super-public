# ivan

**Simple Lie superalgebras of anti-symmetric elements in associative superalgebras**

D. Leites. Status: in preparation.

The scripts check the computations of the text: the census of the anti-automorphisms of Mat(a|b)
and Q(n) listed in Subsection 1.4 (which are anti-automorphisms, their squares, their
anti-symmetric elements K, and whether SL(K) is simple), with the classification of the gap
between conditions (1) and (2) (Propositions 1.4.5, 1.4.6, 1.4.7, Theorem 1.4.8); the
supercentralizer of the odd operator J on the osp(1|2)-module of Subsection 2.2 and the
queerification with its queertrace; the analogs of q and pe on the modules I(V) of
Subsection 2.3; the irreducibility of the homogeneous components of po(2|m); the real forms of
gl(lambda) (Proposition 4.1.1); the osp(1|2) realization (22) and its Casimir; the expansion
formulas (37) and (39) of Subsection 2.5; and Vasiliev's deformed oscillator as U_lambda
(Proposition 3.5.1). Exact arithmetic throughout (sympy over Q(i), Q(lambda), Q(nu);
`fractions.Fraction` in `po22.py`); no numerics.

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r ivan/requirements.txt
cd ivan
../venv/bin/python gap.py
```

Every script is run from inside this folder, with no arguments. Times are for a laptop running
one script at a time. `gapclass.py` and `realf.py` also write their output to `gapclass.log`,
`realf.log` in this folder.

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Statement 1.4.10 (the census of the maps of Subsect. 1.4: st, the osp map G^{-1}X^{st}G, Pi, delta_{-1}, -q), with Proposition 1.4.5, Corollary 1.4.4 and Proposition 1.4.9 (st on Mat(3\|3): K = o(3)+o(3), not simple) | `gap.py` (about 20 seconds) | `st  (as printed)  anti-auto=True  auto=False *^2=delta_{-1}`; for Mat(3\|2): `G^{-1}(.)^{st}G  [osp]  anti-auto=True  auto=False *^2=id  sdim K=(6\|6)`; `-q  (eq. (5))  anti-auto=True  auto=False *^2=delta_{-1}`; `K_odd = 0 ? True` in every line of part 2; `K = o(3) + o(3) (a direct sum of two ideals), not simple.`; in part 5 `Pi is an assoc. automorphism      : True`, `Pi is an assoc. anti-automorphism : False` |
| Proposition 1.4.6, Proposition 1.4.7, Theorem 1.4.8 (every even anti-automorphism of Mat(a\|b) and Q(n) satisfying condition (1): which lie in the gap, and K and L(K) for each), checked for (a\|b) = (2\|2), (3\|2), (3\|3), (4\|2), (4\|1), (5\|1) and n = 2, 3, 4 | `gapclass.py` (about 15 minutes) | for each B or (g, c): `anti=True  (antiInv)=True  *^2=delta_{-1}` or `*^2=id`, with `sdim K`, `dim L(K)` and the number of simple ideals; e.g. for Mat(3\|3), `B=antidiag(1,i) [odd] ... *^2=delta_{-1}  sdim K=(9\|0) ... dim L(K)=8 ... 1 simple ideal(s)`; for Q(n) only `c=i`, `c=-i` pass, with `K = o(n)` (g=1) or `K = sp(n)` (g=J); the SUMMARY lists the four cases of Theorem 1.4.8 |
| Subsect. 1.4 (center vs supercenter of Q(n), the hypothesis dim_Z R > 16 read as n >= 5; formula (6), q^2 = delta_{-1}; the anti-symmetric elements of -q are o(n)) and Subsect. 1.4.12 (condition (9) read as u^2 in Z\{0}) | `qn.py` (seconds) | `dim(ordinary center) = 2, dim(supercenter) = 1`; `n=5: ordinary center -> dim_Z R = 25   PASS`; `q^2 = id ? False    q^4 = id ? True    q^2 = delta_{-1} ? True`; `-> o(n)` for n = 2, 3, 4 |
| Statement 2.2.3 (J = -(x d_theta + theta d_x), J^2 = E; J in U_lambda; the supercentralizer z_J = C[J] in E-degree 0; [E^mJ, E^nJ] = 2E^{m+n+1}) and Statement 2.2.6, item 2) (Q(Mat(m\|n)) = Q(m+n), qtr kills brackets) | `qla.py` (seconds) | `J = -(x d_theta + theta d_x)                      : True`, `J^2 = E  (E = x d_x + theta d_theta)              : True`; the degree table with `7` even and `7` odd solutions in degree `0` (one per V_k) and `0` in degrees -3, ..., 3 otherwise; `[E^m J, E^n J] = 2 E^{m+n+1}   : True`; `J = nabla^- P_ev + nabla^+ P_od  in U_lambda        : True`; in part F every row `True      True` and `Q(m+n) ... : True` |
| Statement 2.2.6, item 1) (the supercentralizer of id (x) j in gl(cL^n (x) C^{1\|1}) is of sdim ((2n+1)^2\|(2n+1)^2)); Statement 2.3.3 (the analogs of q and pe on I(V), M = 2, 4) | `pei.py` (about 45 seconds) | `n = 1: ... supercentralizer = (9\|9),  Q(3) = (9\|9) : True`, `n = 2: ... (25\|25) ... : True`; for dim V = 1, 2: `Theta^2 = 1 ... True`, `supercentralizer of Theta = q(M)`, `it singles out osp(M\|M) ... : True`, the control `B0 = (0 I / I 0)` with `pe(M)=...: True`, `its derived algebra = spe(M) ... : True`, and the three compositions `Theta^t H`, `H Theta`, `Theta H` with `False False` and the preserving algebra `(2\|2)` resp. `(8\|8)`; the last table `Fix(Ad_Theta o st_h)` = `(2\|2)`, `(8\|8)` |
| Statement 2.3.5 (each component po(2\|m)_{k-2} = S^k(V) is an irreducible osp(m\|2)-module, sdim (k+1\|0), (k+1\|k), (2k\|2k); highest weight vector p^k), checked for m <= 2, k <= 8 | `po22.py` (seconds) | for each m and k: `dim ker n^+` = `1` and `irreducible?` = `YES`; e.g. for po(2\|2): `3   (6\|6)   1   12   YES`; the highest weight vectors are multiples of `p^k`, `[even]` |
| Proposition 4.1.1 (omega is an anti-automorphism, -omega an involutive anti-linear automorphism with fixed points u(n); -omega o phi is A -> conj(A) with fixed points gl(n;R)), checked at lambda = n, 2 <= n <= 4 | `realf.py` (seconds) | for gl(2), gl(3), gl(4): `omega_0 is an automorphism / anti-automorphism      : False / True`, `-omega_0 is an automorphism, and involutive         : True, True`, `Fix = anti-Hermitian = u(n): True`, `Fix = real matrices = gl(n;R): True` |
| Formula (22) (the realization of osp(1\|2) on C[x,theta]; with the constant -lambda the Casimir acts by lambda^2+lambda and at lambda = n there is a submodule of sdim (n+1\|n)) | `ivan.py` (seconds) | `--> nabla^+ = -x d_theta + ...  (sign of eq. (22)): all osp(1\|2) relations hold`; `c = -lambda    (eq. (22)): C = lamda**2 + lamda  and mu = lambda`; `finite-dimensional submodule of sdim (n+1\|n)` for n = 1, ..., 5. The lines marked `earlier draft` are controls (the opposite sign of x d_theta, and the constant lambda-1/2) and are expected to FAIL resp. differ |
| Formulas (37), (38), (39) of Subsect. 2.5 (D^j o f = sum_k binom(j,k) f^{(k)} D^{j-k}, also for negative j; the Manin--Radul expansion of Delta^j o a) | `sdo.py` (seconds) | `without the (-1)^k      = ...   --> AGREES` for j = 1, 2, 3; `D o (D^{-1} o g) = (g(x))D^0 ... --> OK`; `p(a)=0,1, j=1..4 :  OK`, `--> the MR formula, ... : verified`. The lines marked `earlier draft` are controls and are expected to DISAGREE |
| Proposition 3.5.1 (Vasiliev's algebra A_nu as U_lambda, lambda = -(1+nu)/2; S = Omega + 1/2 anticommutes with the odd elements, S^2 = C + 1/4) | `vas.py` (seconds) | `osp(1\|2) brackets: [True, True, True, True, True, True, True]`, `C - (lam^2+lam): True`, `Omega = -(1+nu K)/2: True`, `S anticommutes with odd: True S^2 = C+1/4: True` |

`gapclass.py` imports the matrix models from `gap.py`; `po22.py` imports `exactlin.py` (which
uses python-flint when installed and falls back to pure Python otherwise).

In the output of `qla.py` the queer algebra of Subsection 2.2 is written `q(lambda|lambda+1)`;
the text writes `q(lambda+1|lambda)`. In the output of `gapclass.py`, the line `K = ...` under
each map names the candidate whose dimension equals dim K_ev; it is meaningful in the gap
(`*^2=delta_{-1}`), where K is purely even.

## Files

- `gap.py` -- the superalgebras Mat(a|b) and Q(n) as sympy matrix models over Q(i); the maps of
  Subsection 1.4 and the tests: anti-automorphism, square, K, K', the ideal generated by an
  element.
- `gapclass.py` -- all even anti-automorphisms X -> B^{-1} X^{st} B of Mat(a|b) for
  (a|b) = (2|2), (3|2), (3|3), (4|2), (4|1), (5|1), and of Q(n) for n = 2, 3, 4, satisfying condition (1); L(K) and its centroid.
- `qn.py` -- center and supercenter of Q(n), the map -q.
- `qla.py` -- the module cL^lambda = C[x, theta] over U_lambda, the operator J, its
  supercentralizer degree by degree; the queerification Q(A) and the queertrace.
- `pei.py` -- odd forms and supercentralizers on cL^n (x) C^{1|1} and on I(V).
- `po22.py` -- po(2|m), m = 0, 1, 2, from the Poisson bracket (33); kernels of n^+ by exact
  linear algebra; `exactlin.py` -- exact linear algebra over Q.
- `realf.py` -- the anti-linear involutions of gl(n) and their fixed points.
- `ivan.py` -- the realization (22) and the Casimir; `sdo.py` -- (pseudo)differential symbols
  and the operators Delta^j on C[x, theta]; `vas.py` -- Vasiliev's deformed oscillator on its
  Fock module.
