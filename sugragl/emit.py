#!/usr/bin/env python3
"""
emit.py -- compute one model and write its structure functions as JSON, so that
the LaTeX of the Table is generated (mktable.py) and never transcribed.

    ../../tools/venv/bin/python emit.py <Par>   ->  out/<Par>.json
"""

import json
import os
import sys
import time

from models import Prolong
from cohom import h2, irreps

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def compute(par):
    rec = {"par": par, "kinds": {}}
    for kind in ("riem", "full"):
        t0 = time.time()
        P = Prolong(par, kind)
        H, c2d = h2(P)
        M = P.M
        rec["blocks"] = ["%d%s" % (s, "o" if p else "e") for s, p in M.blocks]
        rec["nsl2"] = len(M.sl2)
        rec["depth"] = P.depth
        nev = sum(1 for a in range(P.ngm) if P.gm_par[a] == 0)
        rec["sdim_gminus"] = [nev, P.ngm - nev]
        rec["kinds"][kind] = {
            "prolong": {str(m): P.dim(m) for m in sorted(P.basis)},
            "c2dim": {str(d): n for d, n in sorted(c2d.items())},
            "H": {str(d): [[list(w), p, m] for (w, p, m) in
                           irreps(P, H.get(d, {}))]
                  for d in sorted(set(list(H) + list(c2d)))},
            "seconds": round(time.time() - t0, 1),
        }
    return rec


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for par in sys.argv[1:]:
        rec = compute(par)
        with open(os.path.join(OUT, par + ".json"), "w") as f:
            json.dump(rec, f, indent=1)
        print(f"wrote out/{par}.json")
