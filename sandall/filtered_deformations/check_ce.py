#!/usr/bin/env python3
"""Cross-check of the new numpy engine (ceh.py) against the verified one (coh.py)
on g = k^(1)(3; 1): H^2(g_-; g) and its g_0-invariants must agree block by block."""
from k3 import P, MONS, IDX, build_k3, degK, wpq
from coh import Complex, invariants
from ceh import CE, module_data
import run_k


class Adapter:
    """wraps a k3 LieAlg for ceh.CE."""

    def __init__(self, a):
        self.a = a
        self.n = a.n
        self.wt = [tuple(w) if isinstance(w, tuple) else (w,) for w in a.wt]
        self.deg = [w[0] for w in self.wt]
        self.names = a.names

    def row(self, i):
        return self.a.c[i]

    def brk(self, i, j):
        return self.a.c[i][j]


def main():
    k = build_k3()
    g = run_k.derived(k) if hasattr(run_k, "derived") else None
    # rebuild k^(1)(3;1) exactly as run_k does
    keep = [m for m in MONS if m != (2, 2, 2)]
    gk = run_k.monomial_subalgebra(k, keep)
    hidx = [i for i in range(gk.n) if gk.wt[i][0] < 0]
    g0idx = [i for i in range(gk.n) if gk.wt[i][0] == 0]
    print("dim g = %d, dim g_- = %d, dim g_0 = %d" % (gk.n, len(hidx), len(g0idx)))

    cx = Complex(gk, hidx)
    Hold = cx.cohomology(2)
    old = {w: v[0] for w, v in Hold.items()}
    invo, _, _ = invariants(cx, Hold, g0idx, 2)

    ad = Adapter(gk)
    ce = CE(ad, hidx)
    Hnew = ce.cohomology(2)
    new = {w: v[0] for w, v in Hnew.items()}

    print("coh.py :", sorted(old.items()))
    print("ceh.py :", sorted(new.items()))
    assert old == new, "the two engines disagree"

    info = module_data(ce, Hnew, g0idx, 2)
    tot_inv = sum(v["inv"] for v in info.values())
    print("invariants: coh.py %d, ceh.py %d" % (len(invo), tot_inv))
    assert len(invo) == tot_inv
    for d in sorted(info):
        print("  degree %3d: dim %d, invariants %d, submodule dims %s"
              % (d, info[d]["dim"], info[d]["inv"], info[d]["subdims"]))
    print("OK - the engines agree")


if __name__ == "__main__":
    main()
