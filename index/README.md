# index

**The Index Theorem for Homogeneous Differential Operators on Supermanifolds, and Chern Classes**

D. Leites. Status: in preparation.

The scripts check the computations of the text: the invariant (Haar) Berezin volume of the compact
Lie supergroups and the Killing form of gl(a|b) (Subsecs. 2.3-2.5); the worked example
G = U(2|1) of the superized Bott theorem, with the Kac modules built from scratch and the
symbol-to-character map i_* computed for typical and atypical weights (Subsecs. 2.6-2.7); the
super holomorphic Lefschetz formula on the flag supermanifolds of GL(1|1), GL(2|1) and on Q-type
fixed points (Subsec. 2.9); the queer Chern-Weil theory of q(n)-valued superconnections, the
transgression and the smallest example (Sect. 5); the Berezin integral of po(0|n) quantized into
the Clifford algebra and the parity dichotomy of its characteristic classes (Subsec. 5.9); and the
uniqueness of the supertrace as an invariant functional of gl(a|b) together with the two
evaluations of Z[eps]/(eps^2-1) (Sect. 6). Exact arithmetic throughout (integers,
`fractions.Fraction`, sympy, and python-flint through `../lib/exactlin.py`); the only non-linear
step, the existence of a homological element over C for osp(2|2), osp(3|2), osp(4|2), is decided
by a Groebner basis in sympy.

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r index/requirements.txt
cd index
../venv/bin/python haar.py
```

Every script is run from inside this folder, with no arguments. `bott21.py` takes about two
minutes; every other script finishes in under ten seconds.

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Subsec. 2.3, formula (6): the convention "coframe T dx carries the density Ber(T) D(x)", checked on the Borel subgroup of GL(1\|1) | `haar.py` (part 0) | `ok  Ber(T) = 1/b, the density b^{-1} da db D(theta) is invariant` |
| Statement 2.4: vol U(a\|b) = 0 whether or not a = b | `haar.py` (part 1) | `top coeff = 0` and `the volume vanishes` for gl(1\|1), gl(2\|1), gl(1\|2), gl(2\|2), gl(3\|1) |
| Statement 2.5: the volume of osp(1\|2n) is ±(2n-1)!! for n = 1, 2, 3, and among the listed algebras it is non-zero exactly for osp(1\|2n) | `haar.py` (part 2) | `top coeff = 1`, `-3`, `-15` for osp(1\|2), osp(1\|4), osp(1\|6); `ok  the volume is non-zero exactly for osp(1\|2n) among those tested` |
| Remarks 2.5.1, 1): every algebra of the list except q(1) has a homological element; for q(1), Ber(T) is even in xi | `haar.py` (parts 2, 2a) | `homological element: yes` for all but osp(1\|2n) and q(1); `ok  q(1): dim g_od = 1 is odd, so the top coefficient vanishes for parity` |
| Remarks 2.5.1, 2), formula (7): str(ad_X ad_Y) = 2(a-b) str(XY) - 2 str X str Y | `haar.py` (part 3) | `ok` for gl(1\|1), gl(2\|1), gl(2\|2), gl(3\|1); the Killing form of sl(a\|a) vanishes, of sl(2\|1), sl(3\|1) does not; last line `35/35 checks pass` |
| Subsec. 2.6.1 and Theorem 2.6.4: the data of gl(2\|1), the atypicality of eps_1, K(eps_1) of sdim (4\|4) with quotient C^{2\|1}; for typical nu, <ch L_nu, s(D_mu)> = delta | `bott21.py` (parts 1-6) | `ok  K(eps_1): the representation property on all 81 brackets`, `ok  sdim K(eps_1) = (4\|4)`, `ok  typical nu: <ch L_nu, s(D_mu)> = delta_{nu, mu}` |
| Statement 2.7, 1): for typical mu, i_*(s(D_mu)) = [L_mu] | `bott21.py` (part 7) | `mu = (2, 0, 5) (typical), range 8: 1[L(2, 0, 5)]` and the same for (1, 1, 0) |
| Statement 2.7, 2), formula (11): for mu = eps_1 and for mu = 0 infinitely many terms | `bott21.py` (parts 7, 8) | `mu = (1, 0, 0) (atypical), range 8: 1[L(1, 0, 0)] + -eps[L(1, 1, -1)] + eps[L(2, 2, -3)] + -1[L(3, 2, -4)] + ...`; `the survivors grow with the range`; last line `24/24 checks pass` |
| Example 2.9.2: the single fixed point of the GL(1\|1) flag supermanifold gives e^lambda(1-e^{-beta}) | `lefschetz.py` | `GL(1\|1): local term at the single point: (1 - exp(-beta))*exp(lam) = sch K(lam)` |
| Corollary 2.9.3: the fixed-point sum is the supercharacter of the Kac module (GL(2\|1), four weights of gl(2), three values on delta) | `lefschetz.py` | `GL(2\|1): fixed-point sum = sch of the Kac module for lam in {(0,0),(1,0),(2,1),(3,0)}, mu in {0,1,2}` |
| Remark 2.9.4: Ber(1-X) = 1 for a Q-type matrix X, since tr(Y^{2k}) = 0 for the odd Y | `lefschetz.py` | `Q-type, n=1 (2, 3): ... Ber(1-X) = 1: True` (the last two lines, on Q(2), are a bookkeeping check of the fixed-point sum of the odd characters) |
| Lemma 5.5, formulas (81), (82): the shape of F, F^k = (F_0 + iF_1)^k, Bianchi's identity, closedness and odd degree of qtr(F^k) | `queer.py` | `[ok ] F = (F_omega + phi^2) + (nabla_omega phi) eps`, `[ok ] Bianchi: ...`, `[ok ] d qtr(F^k) = 0  for k = 1..4`, `[ok ] every qtr(F^k) has odd form degree` |
| Lemmas 5.2, 5.3 (the splitting of qtr dies once d enters; eps behaves as sqrt(-1) on the even part), Theorem 5.7 (vanishing for an ordinary connection) | `queer.py` | `[ok ] rho_+ is NOT multiplicative on Omega (x)^ q(n)`, `[ok ] (P,Q) multiplies as P + iQ on the even part`, `[ok ] qtr(F^k) = 0 identically for a q(n)_ev-valued connection` |
| Theorem 5.11, formula (85), and Example 5.11.1, formula (87) | `queer.py` | `[ok ] qtr(F^k) = d beta_k  with the transgression primitive   k=1:ok ... k=4:ok`; `d beta_k = qtr(F^k)` for k = 1, 2, 3; last line `ALL CHECKS PASSED` |
| Subsec. 5.9, item 3): the Berezin integral of po(0\|n) is a supertrace of parity n mod 2; the primary classes vanish for n odd and not for n even | `po.py` | `int_B(F^k) = 0 identically  ->  True` for n = 1, 3, 5 and `-> False` for n = 2, 4; last line `ALL CHECKS PASSED` |
| Subsec. 6.1 and Lemma 6.2: str is the only invariant functional of gl(a\|b); tr[E_12, E_21] = 2 in gl(1\|1); tr is invariant on gl(a) + gl(b); the two evaluations of Z[eps]/(eps^2-1) give ind D_ev ± ind D_od | `bridge.py` | `invariant linear functionals = C.str   [dim = 1]` for six (a\|b); `tr[e_12,e_21] = 2 != 0`; last line `45 checks, 45 passed, 0 failed` |

## Files

- `haar.py` — Grassmann algebra, the Maurer-Cartan matrix T in the chart (g_0, xi) -> g_0 exp(xi),
  Ber(T), homological elements, the Killing form (Subsecs. 2.3-2.5).
- `bott21.py` — gl(2\|1), Kac modules, irreducible characters over Z[eps]/(eps^2-1), the symbol
  and i_* (Subsecs. 2.6-2.7).
- `lefschetz.py` — the fixed-point checks of Subsec. 2.9.
- `poly.py` — exact polynomials over Q; `queer.py` (Sect. 5) and `po.py` (Subsec. 5.9) use it,
  `po.py` also uses `queer.py`.
- `bridge.py` — Sect. 6; imports `../lib/exactlin.py`.
