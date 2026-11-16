"""the full run: both families, six (n,m), the printed formulas and the repair"""
import sys, os
sys.path.insert(0, '.')
import runxu
from xuconf import Xu

runxu.log("backend:", runxu.backend())
runxu.log("Theorem 1: osp(2n+2|2m) on C[x_1..x_2n] (x) Lambda[th_1..th_2m]")
for n, m in ((1, 1), (2, 1), (1, 2), (2, 2)):
    X = runxu.report(n, m, False, 3)
    lo = min(2 * (n - m), 0) - 1
    for ell in (1, 2):
        if -ell <= lo:
            dim, sd, bad = runxu.check_findim(X, ell)
            runxu.log("  FIN   c=%d: dim A_[%d] = %d, sdim = (%d|%d), "
                      "non-invariant: %d" % (-ell, ell, dim, sd[0], sd[1], bad))
runxu.log("")
runxu.log("Theorem 2: osp(2n+3|2m), one more even indeterminate x_0")
runxu.log("--- (1.38) and (1.40) as printed ---")
os.environ["XU138_AS_PRINTED"] = "1"
os.environ["XU140_AS_PRINTED"] = "1"
for n, m in ((1, 1), (2, 1), (1, 2)):
    runxu.report(n, m, True, 3)
runxu.log("--- (1.38) and (1.40) repaired ---")
del os.environ["XU138_AS_PRINTED"]
del os.environ["XU140_AS_PRINTED"]
for n, m in ((1, 1), (2, 1), (1, 2)):
    runxu.report(n, m, True, 3)
runxu.log("=" * 72)
