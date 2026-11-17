"""Where the rank-2 family over CP^{1|4} fails to be homogeneous, and at which
degree cutoff the answer settles.

The class of rank $2$ has a twisted copy of sl(2) in its stabilizer, so by the
mechanism of [BO, Prop. 15] the base is moved transitively; Remark 10.2 and
Statement 15.3 ask whether that suffices.  This script answers it: it prints, chart by
chart and parity by parity, where the evaluation v --> T_p drops rank, and it
repeats every count at three cutoffs, so that no conclusion rests on a
truncation.
"""
from fractions import Fraction as Fr
import sys

import oddpar as O
import famtools as F
from ns import SM, _drop_locus, _pstr

KS = [1, 1, 1, 1]
REPS = {"rank 2": [1, 0, 0, 0, 1, 0, 0, 0],
        "rank 1": [1, 0, 0, 0, 0, 0, 0, 0]}


def main():
    print(__doc__, flush=True)
    L = O.Layer(KS, 3)
    models = [("split", SM.split(KS, npar=1))]
    for nm, s in REPS.items():
        v = [Fr(c) for c in s]
        models.append((nm, F.cocycle_to_family(KS, F.vec_to_coc(L.Q, v),
                                               npar=1)))

    print("1.  The counts at three cutoffs (a truncation would show as drift):")
    print(f"    {'model':>8}" + "".join(f"{'deg ' + str(d):>18}"
                                        for d in (8, 10, 12)))
    for nm, M in models:
        row = []
        for d in (8, 10, 12):
            row.append(f"{str(M.sdim_vect(d)) + ' ' + str(M.transitive(d)):>18}")
        print(f"    {nm:>8}" + "".join(row), flush=True)

    print("\n2.  Where the evaluation drops (chart by chart, parity by parity):")
    for nm, M in models:
        print(f"\n    {nm}:")
        ev = M.evaluations(12)
        for chart in ("U", "V"):
            e = _drop_locus(ev[chart], M.m)
            for tag in ("even", "odd"):
                g = e[tag]
                print(f"       chart {chart}, {tag:>4} part: "
                      + ("onto everywhere" if g is None
                         else f"drops on {_pstr(g)}"), flush=True)

    print("""
3.  Reading.  For the rank-2 class the EVEN part is onto -- the twisted sl(2)
    does lift and does move the base, exactly as [BO, Prop. 15] predicts --
    and it is the ODD part that drops, on the whole of CP^1: the fields of
    degree -1 do not lift.  So a lifted sl(2) does NOT suffice
    (Statement 15.3).""")


if __name__ == "__main__":
    sys.exit(main())
