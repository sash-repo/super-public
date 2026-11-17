"""Der H(2;N)^(2) against N_{Der O}(L) (Skryabin's normalizer), for p = 3, 5:
a check of Proposition 5.16 and of [BMP] = Burde--Moens--Paez-Guillan,
arXiv:2209.14822, Theorems 3.5, 3.6.

Usage: bmp.py [N ...], e.g. bmp.py "(1,2)" "(1,3)"; no argument runs all cases."""
import time
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.path.insert(0, HERE)
from dervect import Super, poisson, derived, normalizer, ad_dim, der_dim, Echelon

LOG = os.path.join(HERE, 'bmp.log')
def main():
    cases = [(3, (1, 1)), (3, (1, 2)), (3, (2, 1)), (3, (2, 2)), (3, (1, 3)),
             (5, (1, 2))]
    if len(sys.argv) > 1:
        cases = [c for c in cases if str(c[1]).replace(' ', '') in sys.argv[1:]]
    with open(LOG, 'w') as fh:
        for p, N in cases:
            t = time.time()
            S = Super(p, 2, 0, N)
            L1, _ = poisson(S, 'h', 1)          # span D_H(O) = H^(1)
            L = derived(S, L1)                  # H^(2)
            dL = Echelon([S.vec(X) for X in L], p).dim
            nz = normalizer(S, L)
            dN = Echelon([S.vec(X) for X in nz], p).dim
            dD, bl = der_dim(S, L, want_blocks=True)
            a, z = ad_dim(S, L)
            s = (f"p={p} N={N}: dim H^(2)={dL} centre={z} dim Der={dD} "
                 f"dim Out={dD-a} dim N_DerO={dN} gap={dD-dN} "
                 f"blocks={sorted(bl.items())} ({time.time()-t:.0f}s)")
            print(s, flush=True); fh.write(s + '\n'); fh.flush()
main()
