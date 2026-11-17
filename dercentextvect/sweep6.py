"""dim out der of the simple relative of o_Pi(2n) for larger n (opi.py engine)."""
import sys, time
from opi import chevalley_span, der_dim, nullspace, echelon, reduce_by, same_span
from run_opi import quotient_by_centre

log = open("sweep6.log", "w")
def say(s):
    print(s); log.write(s + "\n"); log.flush()

for n in [int(a) for a in sys.argv[1:]]:
    t0 = time.time()
    g2 = chevalley_span(n)
    sc, m = quotient_by_centre(g2)
    rows = []
    for j in range(m):
        for k in range(m):
            r = 0
            for i in range(m):
                if (sc[i][j] >> k) & 1:
                    r |= 1 << i
            if r:
                rows.append(r)
    ctr = len(nullspace(rows, m))
    d = der_dim(sc, m)
    out = d - (m - ctr)
    say("n=%d: dim o^{(2)}_Pi(%d)=%d, center=%d, simple relative=%d, "
        "dim Der=%d, dim out=%d, 2n+1=%d  [%.1f s]"
        % (n, 2*n, g2.dim, len(g2.center()), m, d, out, 2*n+1, time.time()-t0))
log.close()
