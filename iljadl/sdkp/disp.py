"""Super mKP gauge of Manin-Radul: dL/dt_{2k} = [(L^{2k})_{>=1}, L], so t_2 = x.
Dispersionless limit: weights w(d_x)=1, w(D)=1/2, w(U_i body)=c_i with
c_odd=-1/2, c_even=0, theta-component c_i+1/2.  Keep minimal-weight terms."""
import sys; sys.path.insert(0, '.')
from mr import *
from fractions import Fraction as F
def ci(name):  # name like 'U31' -> i=3, comp=1
    i = int(name[1:-1]); comp = int(name[-1])
    c = F(-1, 2) if i % 2 else F(0)
    return c + F(comp, 2)
def wmono(m):
    ev, od = m
    w = F(0)
    for v, a in ev: w += a * (ci(v[0]) + v[1])
    for v in od: w += ci(v[0]) + v[1]
    return w
def leading(p, lhsw):
    if p is None or p.iszero(): return P(), None
    ws = {wmono(m) for m in p.t}
    mn = min(ws)
    return P({m: c for m, c in p.t.items() if wmono(m) == mn}), mn
def flows(M, n):
    L, U = lax(M)
    B = Op({k: f for k, f in power(L, n).t.items() if k >= 1}, KMIN)
    C = scomm(B, L)
    return C
if __name__ == '__main__':
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    for n in (2, 4, 6):
        C = flows(M, n)
        print(f'=== t_{n}')
        for i in range(1, M - n // 2):
            f = C.t.get(1 - i)
            for comp, part in ((0, f.a if f else None), (1, f.b if f else None)):
                name = f'U{i}{comp}'
                lw = ci(name) + 1
                lead, mn = leading(part, lw)
                flag = '' if mn is None or mn == lw else f'  !! min weight {mn} != {lw}'
                print(f' d{name} = {lead}{flag}')
