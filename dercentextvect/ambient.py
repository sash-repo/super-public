"""How much of Der(simple relative of o_Pi(2n)) comes from the ambient o_Pi(2n)?

For X in o_Pi(2n), ad_X preserves o^{(2)}_Pi(2n) and its center, hence descends
to the simple relative g.  We compare Span(ad_X) with Der g.
"""
import sys, time
from opi import (E, bracket, coords, echelon, rank, reduce_by, nullspace,
                 der_dim, o_Pi, chevalley_span)
from run_opi import quotient_by_centre

log = open("ambient.log", "w")
def say(s):
    print(s); log.write(s + "\n"); log.flush()

for n in [int(a) for a in sys.argv[1:]]:
    t0 = time.time()
    N = 2 * n
    amb = o_Pi(n)
    g2 = chevalley_span(n)
    ctr = echelon(g2.center())                  # in g2-coordinates
    # basis of the simple relative: keep a complement of the center
    keep = []
    for k in range(g2.dim):
        v = 1 << k
        if reduce_by(echelon(ctr + keep), v):
            keep.append(v)
    idx = [v.bit_length() - 1 for v in keep]
    m = len(idx)

    def project(mask):
        v = mask
        for b in ctr:
            p = b.bit_length() - 1
            if (v >> p) & 1:
                v ^= b
        out = 0
        for a, k in enumerate(idx):
            if (v >> k) & 1:
                out |= 1 << a
        return out

    # o^{(2)} is an ideal in o_Pi: check it
    ideal = all(reduce_by(g2.basis, bracket(N, X, Y)) == 0
                for X in amb.basis for Y in g2.basis)
    # ad_X as a matrix on the simple relative
    mats = []
    for X in amb.basis:
        D = 0
        for b, k in enumerate(idx):
            im = bracket(N, X, g2.basis[k])
            c = coords(g2.basis, im)
            assert c is not None
            pc = project(c)
            for a in range(m):
                if (pc >> a) & 1:
                    D |= 1 << (a * m + b)
        mats.append(D)
    img = rank(mats)
    sc, mm = quotient_by_centre(g2)
    assert mm == m
    d = der_dim(sc, m)
    say("n=%d: dim o_Pi(%d)=%d (ideal ok: %s), simple relative=%d, "
        "dim ad(o_Pi)|_g=%d, dim Der g=%d, not from the ambient=%d  [%.1f s]"
        % (n, N, amb.dim, ideal, m, img, d, d - img, time.time() - t0))
log.close()
