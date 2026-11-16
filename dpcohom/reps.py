import itertools, ham4 as H
H.setup((1,1,1,1))
pairs=list(itertools.combinations(range(4),2))
B=lambda bits:{p:(bits>>t)&1 for t,p in enumerate(pairs)}
import re
cls={}
for line in open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'ham4.log')):
    m=re.match(r"omega2 B=\[(.*)\]: (.*)",line)
    if m:
        ps=eval('['+m.group(1)+']'); bits=sum(1<<pairs.index(p) for p in ps)
        b=B(bits); pf=(b[(0,1)]&b[(2,3)])^(b[(0,2)]&b[(1,3)])^(b[(0,3)]&b[(1,2)])
        r=eval(m.group(2)); cls.setdefault((pf,r['h2'],r['outer'],r['H2']),[]).append(bits)
for k,v in cls.items(): print('Pf,h2,outer,H2',k,len(v))
for bits in (1,12,18):
    r=H.analyse(H.omega2(B(bits)),print,full=False,adr=True); print(bits,r)
