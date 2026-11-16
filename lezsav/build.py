"""build.py -- the build of n4elim.py (q0 = A zeta, particular
solution and kernel of step 0, matrix M4 of the level-4 unknowns) for a
wider level-2 window, using the theta-truncated multiplication of fast.py.
Usage: build.py lo2 hi2 lo4 hi4"""
import os, sys, time, pickle
HERE = os.path.dirname(os.path.abspath(__file__))
import fast
fast.patch()
import n4elim as B
import n4noeth as N
from n4loop import dirs_abs
def main():
    lo2, hi2, lo4, hi4 = (int(x) for x in sys.argv[1:5])
    t0 = time.time()
    q0 = B.q0_of_loop(B.LOOPS["mono"](1))
    d2, _ = dirs_abs(1, lo2, hi2)
    d4, _ = dirs_abs(2, lo4, hi4)
    c2 = B.cols_at(q0, d2, 0)
    cols2 = [c2[i] for i in range(len(d2))]
    F = B.resid_full(q0)
    tgt0 = {k: -v for k, v in F.items() if B.step_of(k) == 0}
    x0 = N.sparse_solve(cols2, tgt0)
    ker0 = N.sparse_null(cols2)
    print("step 0: %d level-2 unknowns, kernel %d [%.0fs]" % (len(d2), len(ker0), time.time() - t0), flush=True)
    qs = B.qof(q0, d2, x0)
    c4 = B.cols_at(qs, d4, 1)
    M4 = [c4[i] for i in range(len(d4))]
    print("M4: %d columns [%.0fs]" % (len(d4), time.time() - t0), flush=True)
    tag = "w2_%d_%d-w4_%d_%d" % (lo2, hi2, lo4, hi4)
    pickle.dump(dict(q0=q0, d2=d2, x0=x0, ker0=ker0, M4=M4, lo2=lo2, hi2=hi2),
                open(os.path.join(HERE, "build-%s.pkl" % tag), "wb"))
    print("saved", tag, "[%.0fs]" % (time.time() - t0), flush=True)

if __name__ == "__main__":
    main()
