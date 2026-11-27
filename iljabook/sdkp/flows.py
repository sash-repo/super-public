import sys; sys.path.insert(0, '.')
from mr import *
L, U = lax(7)
for n in (2, 4):
    B = power(L, n).plus()
    C = scomm(B, L)
    print('=== t_%d flow' % n)
    for i in range(1, 5):
        print(f' U{i}_t =', C.t.get(1 - i))
