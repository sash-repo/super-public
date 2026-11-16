#!/usr/bin/env python3
"""Driver: build (and cache) the models of ag(2), ab(3) and run the case
analysis of the Leites--Shchepochkina method on the gradings of the census.

  python3 run3.py G3            -- all 5 gradings of ag(2)
  python3 run3.py F4            -- all 8 gradings of ab(3)
  python3 run3.py G3 1,0,0      -- one grading, full sweep
  python3 run3.py G3 1,0,0 hom  -- one grading, torus-invariant V only
"""
import os, pickle, sys
from fractions import Fraction as F
import exc3
from sweep3 import run_grading, Grading, sdim_of

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".exc3cache")

# one representative of each Weyl-orbit of gradings (from gradings3exc.py)
GRADINGS = {
    "G3": [((0, 0, 1), "ag(2): depth 2, g_-1 = 4|4"),
           ((1, 0, 0), "ag(2): CONTACT, ag(2) < k(1|7)"),
           ((1, 0, 1), "ag(2): depth 2, g_-2 = 2|2"),
           ((0, 1, 1), "ag(2): depth 3"),
           ((1, 1, 1), "ag(2): depth 3, g_-3 = 2|1")],
    "F4": [((1, 0, 0, 1), "ab(3): DEPTH 1 (Kac's K_10), g_-1 = 6|4"),
           ((0, 0, 0, 2), "ab(3): depth 2, g_0 = c(sl(2)+sp(4))"),
           ((0, 0, 1, 1), "ab(3): depth 2 contact-type, ab(3) < k(7|4)"),
           ((1, 1, 1, 1), "ab(3): depth 2, g_-2 = 3|1"),
           ((2, 0, 0, 0), "ab(3): CONTACT, ab(3) < k(1|8)"),
           ((2, 0, 1, 1), "ab(3): depth 2, g_-2 = 2|2"),
           ((0, 1, 1, 2), "ab(3): depth 3"),
           ((3, 1, 1, 1), "ab(3): depth 3, g_-3 = 1|1")],
}


def model(which):
    fn = CACHE + "." + which
    if os.path.exists(fn):
        with open(fn, "rb") as f:
            return pickle.load(f)
    g = exc3.build_G3() if which == "G3" else exc3.build_F4()
    with open(fn, "wb") as f:
        pickle.dump(g, f)
    return g


if __name__ == "__main__":
    which = sys.argv[1]
    g = model(which)
    print("model: %s  N=%d  beta=%s" % (g.name, g.N, g.beta))
    todo = GRADINGS[which]
    if len(sys.argv) > 2:
        h = tuple(int(x) for x in sys.argv[2].split(","))
        todo = [(h, dict(todo).get(h, "ad hoc"))]
    mode = sys.argv[3] if len(sys.argv) > 3 else "full"
    for h, name in todo:
        tag = "".join(str(x) for x in h)
        run_grading(g, h, name, Vmode=mode,
                    dump=os.path.join(os.path.dirname(CACHE), "runs",
                                      "%s-%s.pkl" % (which, tag)))
