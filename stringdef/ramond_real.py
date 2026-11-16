import sys, time
from real import *
n = int(sys.argv[1]); kind = sys.argv[2]; dmax = int(sys.argv[3]); dmin = int(sys.argv[4]) if len(sys.argv) > 4 else 0
Lb, br, Mb, act = extract(n, kind, dmax+1)
nM = len(Mb)
hol = CE(Lb, br, Mb, act)
Lb2, br2, act2 = double(Lb, br, act, nM)
rea = CE(Lb2, br2, Mb, act2)
tM, tact = trivial(Lb, br)
tri = CE(Lb, br, tM, tact)
for ce in (hol, rea, tri):
    for q in range(3):
        for d in range(dmin, dmax+1): ce.checkQ2(q, d)
print(f"{kind} n={n}: Q^2=0 checked; sdim g_-:", [(k,p) for k,p in Lb], flush=True)
for d in range(dmin-2, dmax+1):
    print(f" trivial H^1_{d}={tri.H(1,d)} H^2_{d}={tri.H(2,d)}", flush=True)
for d in range(-2, dmax+1):
    print(f" hol H^0_{d}={hol.H(0,d)} H^1_{d}={hol.H(1,d)}", flush=True)
for d in range(dmin, dmax+1):
    t = time.time()
    print(f" d={d}: hol H^2={hol.H(2,d)}  real(complexified, one conj half) H^2={rea.H(2,d)}  [{time.time()-t:.0f}s]", flush=True)
