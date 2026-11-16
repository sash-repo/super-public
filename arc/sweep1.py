from fractions import Fraction as F
import sl12, time, sys
sl12.set_numeric(True)
from sl12 import hom_dim, ansatz_terms
LOG=open("sweep1.log","w")
def say(s):
    print(s); LOG.write(s+"\n"); LOG.flush()
wa=(F(3,7),F(5,11)); wb=(F(-2,5),F(7,13))
say("generic numeric weights a=%s b=%s"%(wa,wb))
for N in range(-2,9):
    row=[]
    for q in range(-3,4):
        if (q-N)%2: continue
        if not ansatz_terms(N,q): continue
        t0=time.time()
        d=hom_dim(N,q,wa,wb,dmax=max(7,N+2),numeric=True)
        row.append("q=%+d:%d"%(q,d))
    say("N=%2d  %s"%(N," ".join(row)))
LOG.close()
