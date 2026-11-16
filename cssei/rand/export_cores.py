#!/usr/bin/env python3
"""Export the simple cores of the BGLLS rows (der.core) as ad tensors rand/tabs/<name>.npz."""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); os.chdir(os.path.dirname(HERE))
import numpy as np
from der import core
from absutil import Coords
os.makedirs(os.path.join(HERE, 'tabs'), exist_ok=True)
for name in sys.argv[1:]:
    t = time.time()
    sp, G, J = core(name)
    B = J.basis(); n = len(B); co = Coords(B)
    ad = np.zeros((n, n, n), np.uint8)
    deg = np.array([sp.kdeg(next(iter(v))) for v in B])
    for i in range(n):
        for j in range(n):
            c = co(sp.bracket(B[i], B[j]))
            ks = [k for k in range(n) if c >> k & 1] if isinstance(c, int) else list(c)
            for k in ks:
                ad[i, k, j] = 1
    np.savez_compressed(os.path.join(HERE, 'tabs', name.replace('~', 't') + '.npz'), ad=ad, deg=deg)
    print(name, n, round(time.time() - t, 1), flush=True)
