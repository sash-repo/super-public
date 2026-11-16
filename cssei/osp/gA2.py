from galg import *
F = GFField(4)
for par in ([0, 1], [1, 1], [0, 0]):
    for lam in ([1, 0], [0, 1], [1, 1]):
        N, dims, el, central = generate([[0, 1], [1, 0]], par, lam, F)
        print('A=[[0,1],[1,0]] par', par, 'lam', lam, 'dim L', N, 'image sdim', sdim(dims, par), 'scalar in', central)
