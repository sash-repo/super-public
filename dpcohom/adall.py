import itertools, ham4 as H
H.setup((1,1,1,1))
pairs=list(itertools.combinations(range(4),2))
B=lambda bits:{p:(bits>>t)&1 for t,p in enumerate(pairs)}
out={}
for bits in range(1,64):
    r=H.analyse(H.omega2(B(bits)),print,full=False,adr=True)
    b=B(bits); pf=(b[(0,1)]&b[(2,3)])^(b[(0,2)]&b[(1,3)])^(b[(0,3)]&b[(1,2)])
    out.setdefault((pf,str(r['adrank'])),[]).append(bits)
for k,v in out.items(): print(k,len(v),v)
