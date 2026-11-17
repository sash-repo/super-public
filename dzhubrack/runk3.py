import sys, time, random, kcomm
N = int(sys.argv[1]); deg = int(sys.argv[2]); seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
rng = random.Random(seed)
t0 = time.time()
Xs = [kcomm.contact_field(deg, rng) for _ in range(N)]
D = kcomm.aN_par(Xs, 3)
print('k3 N=%d deg=%d seed=%d' % (N, deg, seed), kcomm.classify(D, 3, True), 'ord', kcomm.order(D), '%.1fs' % (time.time() - t0), flush=True)
