"""Calibration of gcnt.py / gcount.py against brute force (cmp/count.py) on graded CSS tables.
usage: calib.py TABLE.npz ...   (then gcnt.py TABLE.npz nil|tor must print the same numbers)"""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cmp'))
from count import invariants
for p in sys.argv[1:]:
    z = np.load(p); ad = z['ad']; n = ad.shape[0]
    tab = [[int(sum(int(ad[i][k, j]) << k for k in range(n))) for j in range(n)] for i in range(n)]
    print(p, 'brute nil,tor =', invariants(tab), flush=True)
