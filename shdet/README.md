# shdet

**The Shapovalov determinant for stringy Lie superalgebras**

A. Lebedev, D. Leites. Status: in preparation.

The scripts compute, from scratch and in exact arithmetic, the Verma modules and Shapovalov forms of the
distinguished stringy Lie superalgebras (vir, ns(1), r(1), ns(2), the centrally extended
svect^L_lambda(1|2)), their anti-involutions and determinants; the cocycle tables of k^L(1|4) and
k^M(1|4) in the theta and xi-eta bases; the Harish-Chandra image of the odd Casimir of k^M(1|7); the
Shapovalov determinants of the Poisson superalgebras po(0|2n); the critical exponents; and the Lie
algebra K_lambda of Sect. 11. Exact arithmetic throughout (integers, `fractions.Fraction`, sympy, and
python-flint through `../lib/exactlin.py`).

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r shdet/requirements.txt
cd shdet
../venv/bin/python runvir.py
```

Every script is run from inside this folder (those of `osh/` from inside `osh/`), with no arguments
except where stated. Times are for a laptop running one script at a time. For the two runs of
over half an hour (`signrule.py`, `shsvect.py`) the expected output is that of the runs behind the
text; the main block of `kM4.py` (a scan of conventions, not needed by the claims) takes longer still.

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script and arguments | Expected output |
|---|---|---|
| Statement 1.4.1 (the Sign Rule) | `signrule.py` (over half an hour), `parcheck.py` | `signrule.py`: on every weight space `ratio 1 (predicted 1) ok`; `parcheck.py`: for vir, ns(1), ns(2): `parity constant on every one of them` |
| Statement 3.2.1 (the index in the determinant of vir), Statements 3.3.1, 3.3.2 (ns(1), ns(2)) | `runvir.py` (about 3.5 minutes), `runns2.py` (about 28 minutes), `ns2exp.py` (about 28 minutes) | `runvir.py`: `det/prod_i ... -> AGREES (up to a scalar)`; `ns2exp.py`: `mismatches: 0` |
| Statement 3.3.5, Statement 5.0.1 (the determinant of r(1)) | `r1.py` (about 18 minutes) | `det B / its square = 65536  -> AGREES (a scalar)` |
| Statements 4.1.1, 4.2.1, 4.2.2 (svect^L_lambda(1\|2): the anti-involution, the factors, the first values) | `svectlam.py`, `sigmalam.py` (about 8 minutes), `oddfact.py`, `shsvect.py` (over half an hour) | `sigmalam.py`: the unique anti-involution, generator by generator; `oddfact.py`: the kappa-coefficient of each odd factor `(fits all 4 values: True)`; `shsvect.py`: the determinant on each weight space, factored, e.g. `(N,B,Q) = (1, 0, 2) dim 2 ... det = -(ell - kappa + 1)*(ell*lamb + ell + 2*h)*(-ell*lamb + ell + 2*h + 2*kappa*lamb)/4` |
| Statement 6.3.3 (the table of cocycles of k^L(1\|4), theta basis) | `nsTheta.py` (about 2 minutes), `fitTheta.py`, `fitTheta2.py`, `solveTheta.py`, `nsdim.py`, `betaTheta.py`; xi-eta basis: `kL4.py`, `solveL.py` | `solveTheta.py`: `dim H^2 (weight 0) = 3`; `nsdim.py`: `each is a cocycle: True`, `independent modulo coboundaries: True`, `so they span H^2 (weight 0): True`; `nsTheta.py`, `kL4.py`: the failing triples of the printed table |
| Statement 6.3.5 (the table of cocycles of k^M(1\|4), exact and unique) | `ramscan.py` (about 6 minutes), `ramrun.py`, `ramflip.py`, `ramond.py` (about 22 minutes), `solveM.py` | `ramscan.py`: `sigma=-1 k=-1 kz=-1 : Jacobi bad 0, cocycle fails 0` (and nonzero for the other conventions); `ramond.py`: `sigma=-1 k=-1 kz=-1 : 0 failing triples`; `solveM.py`: `dim H^2(k^M(1\|4); C)_ev = 1` |
| Statements 9.2.1, 9.3.1, 9.3.2 (the minimal and superminimal models as zero sets) | `crit.py` | `all checks passed` |
| Subsec. 5.3, eq. (48) (the Harish-Chandra image of the odd Casimir of k^M(1\|7)) | `hc.py` | `with H_100,H_010,H_001 replaced by H_011,H_101,H_110: True` |
| Statements 10.2.1, 10.2.3, 10.2.4 (po(0\|2n): the Casimir, the two determinants, roots not quasiroots) | `run_poi.py`, `run_cas.py`, `run_shap.py` | `run_cas.py`: `7/7 checks pass`; `run_shap.py`: `14/14 checks pass` |
| Subsec. 11.4, Proposition 11.4.1 (osp(1\|2): the anti-involution and the Casimir eigenvalues) | `osp12.py` | the pairs `(0, x^k) (0, -lambda x^k - x^k)` |
| Subsec. 11.6, Proposition 11.6.1, Statement 11.6.4 (K_lambda: the determinant on M_{-1}, the next components) | `osh/test1.py`, `osh/deg1.py` (about 3.5 minutes), `osh/deg2.py 7 3 2 5`, `osh/deg3.py 7 3 2 5 3 4 5`, `osh/derived.py` | `checks 41 ok`; `= prod(59)*Vandermonde^2: True`; `deg -3 N=3 size=22 rank=11`, `deg -3 N=5 size=65 rank=11`; `h in span: True` for each lambda |

## Files

- `verma.py` — Verma modules and Shapovalov forms over the stringy superalgebras; `runvir.py`,
  `runns2.py`, `ns2exp.py`, `r1.py`, `signrule.py`, `parcheck.py`, `crit.py` use it.
- `stringy.py` — functions on the superstrings; `kL4.py`, `kM4.py`, `nsTheta.py` build the contact
  brackets on it; `fitTheta*.py`, `solveTheta.py`, `nsdim.py`, `betaTheta.py`, `solveL.py`, `solveM.py`,
  `ramond.py`, `ramrun.py`, `ramscan.py`, `ramflip.py` use them.
- `svectlam.py` — svect^L_lambda(1\|2) and its central extension; `sigmalam.py`, `shsvect.py`,
  `oddfact.py` use it.
- `poi.py` — the Poisson superalgebras po(0\|2n); `run_poi.py`, `run_cas.py`, `run_shap.py` use it.
- `hc.py`, `osp12.py`; `osh/osh.py` — the algebra K_lambda, used by the other scripts of `osh/`.
