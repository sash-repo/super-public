r"""svect^L_lambda(1|2): vector fields D = f_t d_t + f_1 d_1 + f_2 d_2 on (t|xi_1,xi_2), Laurent in t,
with div_lambda D := lambda f_t/t + d_t f_t + (-1)^{p(D)+1} sum_i d_i f_i = 0 (left derivatives).
Element: dict {(c, a, S): coef}, c in 't',1,2; a = power of t; S = sorted tuple of xi indices."""
import sympy as sp
lam = sp.Symbol('lam')

def gmul(S, T):
    """xi^S * xi^T -> (sign, U) or (0, None)."""
    if set(S) & set(T): return 0, None
    seq = list(S) + list(T); sign = 1
    for i in range(len(seq)):
        for j in range(len(seq) - 1 - i):
            if seq[j] > seq[j+1]: seq[j], seq[j+1] = seq[j+1], seq[j]; sign = -sign
    return sign, tuple(seq)

def fmul(F, G):
    out = {}
    for (a, S), x in F.items():
        for (b, T), y in G.items():
            s, U = gmul(S, T)
            if s: k = (a+b, U); out[k] = out.get(k, 0) + s*x*y
    return {k: v for k, v in out.items() if sp.simplify(v) != 0}

def dfun(c, F):
    out = {}
    for (a, S), x in F.items():
        if c == 't':
            if a: out[(a-1, S)] = out.get((a-1, S), 0) + a*x
        elif c in S:
            i = S.index(c); k = (a, S[:i] + S[i+1:])
            out[k] = out.get(k, 0) + (-1)**i * x
    return {k: v for k, v in out.items() if v != 0}

def comp(D, c):
    return {(a, S): x for (cc, a, S), x in D.items() if cc == c}

def apply(D, F):
    out = {}
    for c in ('t', 1, 2):
        for k, v in fmul(comp(D, c), dfun(c, F)).items(): out[k] = out.get(k, 0) + v
    return {k: v for k, v in out.items() if sp.simplify(v) != 0}

def parity(D):
    ps = {(len(S) + (0 if c == 't' else 1)) % 2 for (c, a, S) in D}
    assert len(ps) <= 1, D
    return ps.pop() if ps else 0

def br(D1, D2):
    p1, p2 = parity(D1), parity(D2); out = {}
    for c in ('t', 1, 2):
        x = {(0, ()) : 1} if c == 't' else None
        xc = {(1, ()): 1} if c == 't' else {(0, (c,)): 1}
        A = apply(D1, apply(D2, xc)); B = apply(D2, apply(D1, xc))
        for (a, S), v in A.items(): out[(c, a, S)] = out.get((c, a, S), 0) + v
        for (a, S), v in B.items(): out[(c, a, S)] = out.get((c, a, S), 0) - (-1)**(p1*p2)*v
    return {k: sp.expand(v) for k, v in out.items() if sp.simplify(v) != 0}

def div(D):
    p = parity(D); out = {}
    ft = comp(D, 't')
    for (a, S), x in ft.items():
        out[(a-1, S)] = out.get((a-1, S), 0) + (lam + a)*x
    for i in (1, 2):
        for k, v in dfun(i, comp(D, i)).items(): out[k] = out.get(k, 0) + (-1)**(p+1)*v
    return {k: sp.expand(v) for k, v in out.items() if sp.expand(v) != 0}

def add(*terms):
    out = {}
    for c, D in terms:
        for k, v in D.items(): out[k] = out.get(k, 0) + c*v
    return {k: sp.expand(v) for k, v in out.items() if sp.expand(v) != 0}

def mono(c, a, S, x=1): return {(c, a, tuple(S)): x}

# weights in tilde-h^* (basis eps, delta1, delta2): t^a xi^S d_c
def weight(D):
    ws = set()
    for (c, a, S) in D:
        e = a - (1 if c == 't' else 0)
        d1 = (1 if 1 in S else 0) - (1 if c == 1 else 0)
        d2 = (1 if 2 in S else 0) - (1 if c == 2 else 0)
        ws.add((e, d1, d2))
    assert len(ws) == 1, D
    return ws.pop()

# root vectors of the table (formula (43)), k in Z
def rv(name, k):
    L = lam
    return {
     'd1':  add((1, mono('t', k+1, (1,))), (L+k+1, mono(2, k, (1, 2)))),
     '-d1': mono(1, k, ()),
     'd2':  add((1, mono('t', k+1, (2,))), (-(L+k+1), mono(1, k, (1, 2)))),
     '-d2': mono(2, k, ()),
     'd2-d1': mono(1, k, (2,)),
     'd1-d2': mono(2, k, (1,)),
     '0a': add((1, mono(1, k, (1,))), (-1, mono(2, k, (2,)))),
     '0b': add((1, mono('t', k+1, ())), (L+k+1, mono(1, k, (1,)))),
     '0c': add((1, mono('t', k+1, ())), (L+k+1, mono(2, k, (2,)))),
    }[name]
