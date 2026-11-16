import witt2
m = witt2.make_module('adj')
Ns = list(range(6, 39, 2))
ws = list(range(-8, 9))
print("dim H^2(Q_N) for M = witt (lambda = -1), rows = weight w, cols = N")
print("  w  " + "".join("%4d" % N for N in Ns))
for w in ws:
    row = [witt2.cohomology(2, N, w, m)[0] for N in Ns]
    print("%4d " % w + "".join("%4d" % h for h in row))
print()
print("dim of the spaces themselves, at w = 0:")
print("  N   dimC^1 dimC^2 dimC^3  rk d1  rk d2  dimZ^2  dimB^2")
for N in Ns:
    d1,s1,_ = witt2.differential(1, N, 0, m)
    d2,s2,t2 = witt2.differential(2, N, 0, m)
    d3,s3,t3 = witt2.differential(3, N, 0, m)
    r1, r2 = witt2._rank(d1), witt2._rank(d2)
    print("%4d %6d %6d %6d %6d %6d %7d %7d" % (N, len(s1), len(s2), len(s3), r1, r2, len(s2)-r2, r1))
