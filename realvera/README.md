# realvera

**Automorphisms and real forms of simple Lie superalgebras (after V. Serganova)**

D. Leites. Status: in preparation.

The scripts check the computations behind the chapters on the automorphisms and real forms of the
distinguished stringy Lie superalgebras and of the loop superalgebras: the automorphisms of the
superstrings and the tables of their actions, the map c_h induced by complexification, the three types of
real forms in the Laurent model, the structure each real form preserves, and the realization of psl(2|2)
by supermatrices. Exact arithmetic throughout (sympy over Q(i, sqrt 2) and over Grassmann algebras,
`fractions.Fraction`, and python-flint through `../lib/exactlin.py`).

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r realvera/requirements.txt
cd realvera
../venv/bin/python psl22.py
```

Every script is run from inside this folder, with no arguments; each takes under a minute.

## Claims of the text and the scripts behind them

Numbers are those of the text (tables and displayed formulas share one numbering).

| Claim in the text | Script | Expected output |
|---|---|---|
| Tables (231), (232) and formula (196): the automorphisms of the superstrings, the filtration-1 equation, and the sign of the odd part of the divergence | `aut372.py` | `ALL CHECKS PASSED`; part (E) compares the cells of the table as they stood in an earlier draft with the derivation, and the three it marks `DIFFERS` are the cells corrected in the text |
| The map c_h from Aut h/Aut°h to Aut g/Aut°g, g = h (x) C, row by row against Table (185) | `cmap.py` | `All checks passed.` |
| The three types of real forms of the loop superalgebras in the Laurent model | `loop374.py` (uses `lau.py`) | `=== all checks passed ===` |
| Which structure each real form of Table (219) preserves | `struct373.py` (uses `lau.py`) | `failures: 0` |
| The realization of psl(2\|2) by supermatrices from sl(2\|2) and its corrected bracket | `psl22.py` | `ALL CHECKS PASSED` (the corrected bracket satisfies the super Jacobi identity on 2744 triples) |

## Files

- `lau.py` — Laurent polynomials in z = e^{ia/2} over Q(i): the functions on the supercircle.
- `aut372.py`, `cmap.py`, `loop374.py`, `struct373.py`, `psl22.py` — the checks listed above.
