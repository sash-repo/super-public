"""Run the fixed-multidegree computation for one multidegree mu.

    python3 run.py 2,2,1            one multidegree
    python3 run.py --degree 5       every multidegree of 5 (calibration)

For each mu it prints dim M_mu, dim I_mu, dim C_mu, the number of NEW identities
in that multidegree, and the prediction sum_lambda m_lambda K_{lambda,mu} made
by the multiplicities m_lambda already known in that degree (degrees 3, 4, 5).
Where the multiplicities are NOT known -- degrees 6 and up -- the computed
number is itself the news, and `reach` says which lambda it decides.
"""

import sys
import time

import gens as G
from mdeg import (Evaluator, count_monos, dim_C, dim_I, kostka, monos,
                  partitions, reach)

# multiplicities of [lambda] in the new multilinear identities I_n/C_n
KNOWN = {
    3: {(3,): 1, (2, 1): 1, (1, 1, 1): 1},
    4: {(4,): 1, (3, 1): 2, (2, 2): 1},
    5: {(5,): 1, (4, 1): 1, (3, 2): 1, (2, 2, 1): 1},
    6: {},                       # [BrP]: nothing new
    7: {},                       # [BrP]: nothing new
}


def predict(mu):
    n = sum(mu)
    if n not in KNOWN:
        return None
    smu = tuple(sorted(mu, reverse=True))
    return sum(m * kostka(lam, smu) for lam, m in KNOWN[n].items())


def one(mu, budget=None, verbose=True):
    mu = tuple(mu)
    n = sum(mu)
    t0 = time.time()
    dM = count_monos(mu)
    # the budget only bites when the target is NOT reached: it is what makes a
    # reported gap trustworthy, so let it grow with the space being searched.
    if budget is None:
        budget = max(400000, 50 * dM)
    print("mu = %-14s n = %d   dim M_mu = %d" % (str(mu), n, dM), flush=True)
    print("   sees: %s" % (", ".join(str(l) for l in reach(mu))), flush=True)
    trees = monos(mu)
    assert len(trees) == dM
    ev = Evaluator()
    dI, rk = dim_I(mu, trees, ev, verbose=verbose)
    print("   dim V_mu = %d,  dim I_mu = %d   [%.0fs]" % (rk, dI, time.time() - t0),
          flush=True)
    gg = {k: v for k, v in G.load().items() if k < n}
    dC, seen, exh = dim_C(mu, trees, gg, target=dI, budget=budget,
                          verbose=verbose)
    new = dI - dC
    pred = predict(mu)
    print("   dim C_mu = %d  (%d consequences used%s)"
          % (dC, seen, "" if exh else ", BUDGET NOT EXHAUSTED"), flush=True)
    print("   NEW in this multidegree: %d%s   [%.0fs]"
          % (new, "" if pred is None else "   (Kostka prediction %d)" % pred,
             time.time() - t0), flush=True)
    if pred is not None and new != pred:
        print("   *** MISMATCH with the prediction ***", flush=True)
    print(flush=True)
    return dict(mu=mu, dM=dM, dV=rk, dI=dI, dC=dC, new=new, pred=pred,
                seen=seen, secs=time.time() - t0)


if __name__ == "__main__":
    if sys.argv[1] == "--degree":
        n = int(sys.argv[2])
        for mu in partitions(n):
            one(mu, verbose=False)
    else:
        for a in sys.argv[1:]:
            one(tuple(int(x) for x in a.split(",")), verbose=True)
