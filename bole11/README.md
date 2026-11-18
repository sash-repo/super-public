# bole11

**Gordan-Rankin-Cohen operators on the spaces of weighted densities in superdimension 1|1**

V. Bovdi, D. Leites. Status: in preparation.

The scripts recompute the claims of the text that rest on a computation: the dimensions of the
spaces of osp(1|2)-singular vectors and of the osp(1|2)-invariant bilinear (GRC) operators on the
weighted densities over the (1|1)-dimensional superstring with a contact structure; the
superdimensions of the spaces of pgl(2|1)-singular vectors over the general (1|1)-dimensional
superstring; the indexing of the odd brackets by 1/2, 3/2, 5/2, ...; and the test of the
normalizations that could make the product of the brackets associative. Arithmetic is exact
(`fractions.Fraction`); no third-party packages are needed.

## How to run

```bash
cd bole11
python3 check_formulas.py
```

Every script is run from inside this folder with the system `python3` (3.10 or later).
Running times are those of one core of an Apple M-series laptop.

## Claims of the text and the scripts behind them

Numbers are those of the text.

| Claim in the text | Script | Expected output |
|---|---|---|
| Theorem 2.2.2 (the number of osp(1\|2)-singular vectors), for n <= 3 and the weights of the sample list | `check_formulas.py` (< 1 s), Section 2 | `mismatches with the corrected criteria: 0`, then the sample dimensions |
| Theorem 3.6.1 (the superdimension of the space of pgl(2\|1)-singular vectors), n >= 1 | `check_formulas.py`, Section 3 | `mismatches with the Theorem of the text: 0`; the line `n=0 check: (1, 0) (the Theorem is stated for n >= 1)` |
| Theorem 2.2.4a (the dimension of the space of GRC operators) and the sign convention of the odd brackets | `grc_ops.py` (about 3.5 min) | `mismatches: 0`, then the invariant operators for generic weights |
| Subsec. 2.3: the shift of the bracket of order k is k, the bracket is odd if and only if k is odd, and the odd brackets are indexed by 1/2, 3/2, 5/2, ... | `halfint.py` (5 s) | `51 checks, 0 failures` |
| Remark 2.3.1a, item 1): the normalizations making the product of the even brackets associative through order 6 | `check_assoc.py` (35 s) | for total orders 4 and 6, `unique solution; equals (u)_k/k!  -> True` |
| Remark 2.3.1a, item 2): s_0 = s_1 = s_2 = 0, since the odd brackets of orders 1, 3, 5 are not associative up to normalization, already on even f, g, h | `assoc_odd.py` (3 s); the order-1 case also in `check_assoc.py` | six lines `... proportional: False` and `brackets of orders 1, 3, 5 associative up to normalization: none` |
