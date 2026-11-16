import ham4 as H
for N in [(1,1,1,1),(2,1,1,1),(1,1,2,1)]:
    H.setup(N); lam=[0,0,H.xvar(0),H.xvar(1)]
    for j in range(4):
        w=H.d1(H.scal1(H.expo(j),lam))
        print(N,'j=',j+1,H.analyse(w,print,full=True))
