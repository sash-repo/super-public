import witt2
m = witt2.make_module('adj')
bad = []
for w in range(-120, 121):
    for N in (20, 24):
        for q in (1, 2):
            h, _ = witt2.cohomology(q, N, w, m)
            if h: bad.append((q, w, N, h))
print("weights scanned: -120..120 at N = 20 and 24, q = 1,2")
print("nonzero found:", bad if bad else "NONE")
