"""Super KdV families in components: generic superfields, the component system of a superfield
evolution equation, the symmetry test for evolutionary fields.  Dk_i = d_theta_i - theta_i d_x
(Dk_i^2 = -d_x); in sf.py, D_i = d_theta_i + theta_i d_x and Q_i = d_theta_i - theta_i d_x."""
from sf import *

def subsets(N):
    out = []
    for k in range(N + 1): out += list(combinations(range(1, N + 1), k))
    return out
def gen(N, pre, p):
    return SF.field(N, {I: pre + (''.join(map(str, I)) if I else '0') for I in subsets(N)}, p)
Dk = lambda X, i: X.Q(i)

def DkAll(X, N):
    for i in range(N, 0, -1): X = Dk(X, i)
    return X

def rhs_N3(U, a, b):
    F = DkAll(U, 3).scale(a)
    s = SF(3, {}, 0)
    for i in (1, 2, 3): s = s + Dk(U, i) * Dk(U, i)
    F = F + s.scale(b)
    r = DkAll(F, 3) + (U.dx() * F).scale(-2) - U * F.dx()
    for i in (1, 2, 3): r = r + Dk(U, i) * Dk(F, i)
    return r

def system(U, R):
    """dict component-name -> RHS (a P)"""
    out = {}
    for I, v in U.c.items():
        (ev, od), = v.t.keys()
        nm = (ev[0][0] if ev else od[0])[0]
        out[nm] = R.c.get(I, P())
    return out

def Dt_act(sysm):
    return lambda v: dxk(sysm[v[0]], v[1])

def is_symmetry(sysm, char, xpar):
    """char: dict comp-name -> P (values of the evolutionary field on the unknowns)"""
    act = lambda v: dxk(char[v[0]], v[1])
    for n in sysm:
        lhs = derive(sysm[n], act, xpar)
        rhs = derive(char[n], Dt_act(sysm), 0)
        if not (lhs - rhs).iszero(): return False, n
    return True, None

def evol_char(U, S, xpar):
    """component values of the evolutionary field X with X(U)=S (superfield)."""
    out = {}
    for I, v in U.c.items():
        (ev, od), = v.t.keys()
        nm = (ev[0][0] if ev else od[0])[0]
        w = S.c.get(I, P())
        out[nm] = w * (-1 if (xpar and len(I) % 2) else 1)
    return out
