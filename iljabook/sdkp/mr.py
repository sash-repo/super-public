import sys; sys.path.insert(0, '.')
from spdo import *
KMIN = -9
def lax(M, zero=()):
    t = {1: S.const(1)}
    U = {}
    for i in range(1, M + 1):
        if i in zero: continue
        U[i] = S.field(f'U{i}', i % 2)
        t[1 - i] = U[i]
    return Op(t, KMIN), U
def power(L, n):
    R = L
    for _ in range(n - 1): R = R * L
    return R
if __name__ == '__main__':
    L, U = lax(8, zero=(1,))
    for n in (2, 4):
        B = power(L, n).plus()
        print('B', n, {k: v for k, v in B.t.items()})
        C = scomm(B, L)
        print(' D^0 coeff of [B,L]:', C.t.get(0))
        print(' D^1 coeff:', C.t.get(1))
