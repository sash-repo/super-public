#!/usr/bin/env python3
"""Statement 3.5.2 of the paper.  The osp(3|2) case of [BLO] (their Sec. IV) re-done in THEIR conventions:

(a) are their explicit operators (4.1)-(4.2) invariant?
(b) what IS the dimension of the space of osp(3|2)-invariant bilinear
    operators F_tau (x) F_lam --> F_{tau+lam+k}, as a function of (tau, lam)?
    -- to be compared with their Theorem 4.1 (ii)-(v).

Usage:  python check_blo3.py [all|form|dim] [k1,k2,...]   (dim: default k = 1/2,1,3/2,2)
"""
import sys
from fractions import Fraction as Fr
from blo import BLO, Classifier, check_invariance, J_N3, label


def test_formulas(ks=(1, 2, 3, Fr(3, 2), Fr(5, 2)),
                  tau=Fr(3, 7), lam=Fr(-5, 11), tmax=3):
    alg = BLO(3)
    for k in ks:
        T = J_N3(alg, tau, lam, k)
        zero = all(not T(u, v) for u in
                   [{m: 1} for m in alg.monos(tmax)]
                   for v in [{m: 1} for m in alg.monos(tmax)])
        bad = check_invariance(alg, T, int(2 * k), tau, lam, tmax=tmax)
        print(f"  k={k} (order {2*k}): "
              f"{'IDENTICALLY ZERO, ' if zero else ''}"
              f"{'invariant' if not bad else str(len(bad)) + ' failures'}")
        sys.stdout.flush()


def dims(ks, grid, tmax=None):
    alg = BLO(3)
    for k in ks:
        ordr = int(2 * k)
        print(f"\n--- order {ordr} (k = {k}) ---")
        sys.stdout.flush()
        table = {}
        for i, tau in enumerate(grid):
            for lam in grid[i:]:
                ts, ns = Classifier(alg, tau, lam).solve(ordr, tmax=tmax)
                table[(tau, lam)] = len(ns)
        base = max(set(table.values()), key=list(table.values()).count)
        print(f"generic dim = {base}; exceptions:")
        for kk, v in sorted(table.items(), key=str):
            if v != base:
                print(f"    tau={kk[0]}, lam={kk[1]}: dim = {v}")
        sys.stdout.flush()
    return table


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('all', 'form'):
        print("[BLO]'s own osp(3|2) operators (4.1)-(4.2), generic weights:")
        test_formulas()
    if what in ('all', 'dim'):
        GRID = [Fr(x, 2) for x in range(-6, 3)] + [Fr(3, 7), Fr(-5, 11)]
        print("\nDimension of the space of osp(3|2)-invariant binary operators")
        print("(their normalization: F_tau (x) F_lam -> F_{tau+lam+k})")
        KS = [Fr(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 \
            else [Fr(1, 2), 1, Fr(3, 2), 2]
        dims(KS, GRID)
