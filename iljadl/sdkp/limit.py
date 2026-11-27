"""The dispersionless limit of the flows t2, t4, t6 (orders, indices of the U_i involved,
superfield form of the leading parts) of of hierarchy (r1) [(L^n)_{>=1},L] and of MR [(L^n)_+,L], U1..U5."""
import sys
sys.path.insert(0, '.')
from mr import *
from disp import ci, wmono, leading
import re
M = 12
L, U = lax(M)
def Dder(p):
    """odd derivation D on component polynomials: U_i0 -> U_i1, U_i1 -> U_i0'"""
    r = P()
    for (ev, od), c in p.t.items():
        # expand monomial as product of factors in order: even part first, then odd vars
        facs = []
        for v, a in ev: facs += [v] * a
        facs += list(od)
        for j, v in enumerate(facs):
            name, k = v
            base, comp = name[:-1], int(name[-1])
            nv = (base + '1', k) if comp == 0 else (base + '0', k + 1)
            # sign: (-1)^{parity of factors before j}
            sgn = (-1) ** sum(PAR[w[0]] for w in facs[:j])
            prod = P.const(c * sgn)
            for jj, w in enumerate(facs):
                prod = prod * (P.var(*nv) if jj == j else P.var(*w))
            r = r + prod
    return r
def idx(p):
    return sorted({int(v[0][1:-1]) for m in p.t for v in ([w for w, a in m[0]] + list(m[1]))})
for hier, cut in (('r1', 1), ('MR', 0)):
    for n in (2, 4, 6):
        Ln = power(L, n)
        B = Op({k: f for k, f in Ln.t.items() if k >= cut}, KMIN)
        C = scomm(B, L)
        for i in range(1, 6):
            f = C.t.get(1 - i)
            out = []
            leads = {}
            for comp, part in ((0, f.a), (1, f.b)):
                name = f'U{i}{comp}'
                lw = ci(name) + 1
                lead, mn = leading(part, lw)
                leads[comp] = lead
                out.append(f'{name}: full idx {idx(part)} lead idx {idx(lead)} minw {mn} lhsw {lw} {"OK" if mn == lw else "!!"}')
            sf = (leads[1] - Dder(leads[0])).iszero()
            print(hier, f't{n}', ' ; '.join(out), 'superfield:', sf)
