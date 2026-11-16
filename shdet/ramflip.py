"""
(cocycR1) holds for the bracket whose odd part carries +(-1)^{p(f)}, while the
paper's own eq. (kb) writes -(-1)^{p(f)}.  The two brackets are carried into
each other by the substitution (all odd indeterminates) x -> i*x, under which
K_{t^a M} -> i^{|M|} K_{t^a M}; so the cocycle entries pick up i^{|M|+|N|},
i.e. exactly the entries with |M|+|N| = 2 mod 4 change sign.  Test that.
"""
from fractions import Fraction as Q
import sys
from kM4 import (monomials, check_cocycle, check_skew, name_of, table_raw,
                 Z, XI, ETA, TH)

ODDNAME = {"Gx", "Ge", "Gt", "Gz", "Ft", "Fxez", "Fxtz", "Fetz"}
# |M| for each named basis vector
SIZE = {"L": 0, "Txe": 2, "Txt": 2, "Tet": 2, "Txz": 2, "Tez": 2, "Ttz": 2,
        "S": 4, "Gx": 1, "Ge": 1, "Gt": 1, "Gz": 1,
        "Ft": 3, "Fxez": 3, "Fxtz": 3, "Fetz": 3}


def table_kb(nx, ny):
    """(cocycR1) transported to the convention of eq. (kb)."""
    v = table_raw(nx, ny)
    if v is None:
        return None
    s = (SIZE[nx[0]] + SIZE[ny[0]]) % 4
    return -v if s == 2 else v


B = int(sys.argv[1]) if len(sys.argv) > 1 else 3
small = monomials(range(-B, B + 1))
KB = dict(sigma=-1, k=Q(1), kz=Q(1))       # eq. (kb)
OTHER = dict(sigma=1, k=Q(1), kz=Q(1))

print("super-skew-symmetry of the transported table:",
      "OK" if not check_skew(small, table_kb) else "violations")
for nm, tab, kw in (("(cocycR1) as in the text on spinor-oscillator representations", table_raw, OTHER),
                    ("the same, in the convention of eq. (kb)", table_kb, KB)):
    f = check_cocycle(small, tab, **kw)
    tot = sum(len(v) for v in f.values())
    print("%-42s : %d failing triples (|a| <= %d, %d monomials)"
          % (nm, tot, B, len(small)), flush=True)
    for sig in sorted(f, key=lambda s: -len(f[s]))[:4]:
        ex = f[sig][0]
        print("      %-22s %5d  e.g. %s %s %s -> %s"
              % ("/".join(sig), len(f[sig]), name_of(ex[0])[:2],
                 name_of(ex[1])[:2], name_of(ex[2])[:2], ex[3]))
print()
print("entries that change sign between the two conventions:")
NAMES = sorted(SIZE)
seen = set()
for a in NAMES:
    for b in NAMES:
        if table_raw((a, 1), (b, -1)) is not None or \
           table_raw((a, 1), (b, 0)) is not None or \
           table_raw((a, 0), (b, 0)) is not None:
            if (b, a) in seen:
                continue
            seen.add((a, b))
            if (SIZE[a] + SIZE[b]) % 4 == 2:
                print("   c(%s, %s)" % (a, b))
