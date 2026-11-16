import sys, numpy as np
for p in sys.argv[1:]:
    z = np.load(p); ad = z['ad']; deg = z['deg']; d = len(deg); bad = 0
    for i in range(d):
        for j in range(d):
            for k in np.nonzero(ad[i][:, j] & 1)[0]:
                if deg[k] != deg[i] + deg[j]: bad += 1
    print(p, 'dim', d, 'bad', bad, 'degrees', dict(zip(*np.unique(deg, return_counts=True))) if len(set(deg.tolist()))<12 else len(set(deg.tolist())))
