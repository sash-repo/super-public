"""The super dispersionless KP system (r=1 gauge, t_2 = x, y = t_4, t = t_6):
component equations and their superfield form, fitted and verified."""
import sys, os; sys.path.insert(0, '.')
from disp import *
from fractions import Fraction as F
from itertools import combinations_with_replacement
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'lib'))
from exactlin import solve, nullspace

M = 7
lead4, lead6 = {}, {}
for n, store in ((4, lead4), (6, lead6)):
    C = flows(M, n)
    for i in range(1, 6):
        f = C.t.get(1 - i)
        store[i] = (leading(f.a, None)[0], leading(f.b, None)[0])

# superfield building blocks: D^k U_i, k = 0..3, i = 1..4 (and 5,6 never needed)
FIELDS = {i: S.field(f'U{i}', i % 2) for i in range(1, 8)}
def DU(i, k):
    f = FIELDS[i]
    for _ in range(k): f = f.D()
    return f
def w_block(i, k): return (F(-1, 2) if i % 2 else F(0)) + F(k, 2)
def p_block(i, k): return (i + k) % 2
BLOCKS = [(i, k) for i in range(1, 5) for k in range(0, 4)]

def monomials(weight, parity, maxdeg=3):
    out = []
    for d in range(1, maxdeg + 1):
        for combo in combinations_with_replacement(BLOCKS, d):
            if sum(w_block(*b) for b in combo) != weight: continue
            if sum(p_block(*b) for b in combo) % 2 != parity: continue
            # odd block repeated -> zero
            odd = [b for b in combo if p_block(*b)]
            if len(odd) != len(set(odd)): continue
            out.append(combo)
    return out

def sf(combo):
    r = None
    for b in combo:
        f = DU(*b)
        r = f if r is None else r * f
    return r

def fit(target_body, target_theta, weight, parity, name):
    mons = monomials(weight, parity)
    sfs = [sf(m) for m in mons]
    keys = sorted({k for s in sfs for k in list(s.a.t) + list(s.b.t)} |
                  set(target_body.t) | set(target_theta.t), key=str)
    rows = []; rhs = []
    for k in keys:
        rows.append([s.a.t.get(k, 0) for s in sfs]); rhs.append(target_body.t.get(k, 0))
        rows.append([s.b.t.get(k, 0) for s in sfs]); rhs.append(target_theta.t.get(k, 0))
    sol = solve(rows, rhs)
    assert sol is not None, name + ': no superfield form'
    terms = [(c, m) for c, m in zip(sol, mons) if c]
    return terms

def show(terms):
    def b(i, k):
        return ('D' if k % 2 else '') + f'U{i}' + "'" * (k // 2)
    return ' + '.join(f'{c}*' + '*'.join(b(*x) for x in m) for c, m in terms)

if __name__ == '__main__':
    out = {}
    # y-flows of U1, U2 (t_4); t-flows of U1, U2 with U5, U6 eliminated
    # via the y-flows of U3, U4:  t6(U_i) - (3/2) y(U_{i+2}) contains no U_{i+4}
    for i in (1, 2):
        wy = (F(-1, 2) if i % 2 else F(0)) + 1
        b, t = lead4[i]
        out[f'U{i}_y'] = fit(b, t, wy, i % 2, f'U{i}_y')
        b6, t6 = lead6[i]; b4, t4 = lead4[i + 2]
        b = b6 - b4 * F(3, 2); t = t6 - t4 * F(3, 2)
        names = {v[0] for m in list(b.t) + list(t.t) for v in [x[0] for x in m[0]] + list(m[1])}
        assert not any(n.startswith(('U5', 'U6', 'U7')) for n in names), names
        out[f'U{i}_t - 3/2 U{i+2}_y'] = fit(b, t, wy, i % 2, f'U{i}_t')
    for k, v in out.items(): print(k, '=', show(v))
