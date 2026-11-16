"""The cox.py test for osp(2|2n), n = 2, 3.

Weights of osp(2|2n): integer vectors (eps | delta_1..delta_n).  Roots: even
+-2delta_j, +-delta_i+-delta_j; odd +-eps+-delta_j.  Distinguished base
eps-delta_1 (odd), delta_1-delta_2, ..., delta_{n-1}-delta_n, 2delta_n.
osp(2|2n) is of type I: g_0 = gl(1)+sp(2n), g_{-1} spanned by the root vectors of
-eps+-delta_j.  A typical irreducible module is the Kac module, whose weight SET is
Gamma(L_0(lam)) + {sums of subsets of the roots -eps+-delta_j}; the eps-coordinate
of lam only translates that set (and is taken generic, so the module is typical),
so the modules are indexed by the dominant sp(2n)-weights lam'.  Plus the adjoint.
r_alpha acts on a weight set by g -> g + (p-q) alpha (uw.py, Subsec. 1.4 of the text).
C_g is realized faithfully as the Weyl group of the generalized Cartan matrix with
a_{st}a_{ts} = 1, 2, 4 for the edges 3, 4, infinity (s_i(e_j) = e_j - a_{ij} e_i).
"""
import sys, os, itertools
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cox import tits_gens, mul


def spweights(lam):
    """Weight set of the irreducible sp(2n)-module with dominant highest weight lam."""
    n, top = len(lam), lam[0]
    out = []
    for mu in itertools.product(range(-top, top + 1), repeat=n):
        if (sum(lam) - sum(mu)) % 2:
            continue
        a = sorted((abs(x) for x in mu), reverse=True)
        if all(sum(a[:t]) <= sum(lam[:t]) for t in range(1, n + 1)):
            out.append(mu)
    return out


def roots(n):
    N = n + 1
    e = lambda i: tuple(1 if k == i else 0 for k in range(N))
    add = lambda *vs: tuple(sum(c * v[k] for c, v in vs) for k in range(N))
    ev, od = [], []
    for j in range(1, N):
        for s in (1, -1):
            ev.append(add((2 * s, e(j))))
            for t in (1, -1):
                od.append(add((s, e(0)), (t, e(j))))
        for i in range(1, j):
            for s in (1, -1):
                for t in (1, -1):
                    ev.append(add((s, e(i)), (t, e(j))))
    return ev, od


def kac(lam):
    n = len(lam)
    negodd = [tuple([-1] + [s if k == j else 0 for k in range(n)]) for j in range(n) for s in (1, -1)]
    G = set()
    for mu in spweights(lam):
        for S in itertools.product((0, 1), repeat=len(negodd)):
            v = [0] + list(mu)
            for s, r in zip(S, negodd):
                if s:
                    v = [x + y for x, y in zip(v, r)]
            G.add(tuple(v))
    return frozenset(G)


def module_list(n, R):
    ev, od = roots(n)
    mods = [frozenset(ev + od + [tuple([0] * (n + 1))])]
    for lam in itertools.product(range(R, -1, -1), repeat=n):
        if all(lam[i] >= lam[i + 1] for i in range(n - 1)):
            mods.append(kac(lam))
    return mods


def refl(G, a):
    perm = {}
    for g in G:
        p = 0
        while tuple(x + (p + 1) * y for x, y in zip(g, a)) in G:
            p += 1
        q = 0
        while tuple(x - (q + 1) * y for x, y in zip(g, a)) in G:
            q += 1
        perm[g] = tuple(x + (p - q) * y for x, y in zip(g, a))
    assert sorted(perm.values()) == sorted(G)
    return perm


def gcm(n):
    # nodes: 0 = eps-delta_1 (odd), 1..n-1 = delta_i-delta_{i+1}, n = 2delta_n
    N = n + 1
    M = [[2 if i == j else 0 for j in range(N)] for i in range(N)]
    M[0][1] = M[1][0] = -2                      # infinity
    for i in range(1, n - 1):
        M[i][i + 1] = M[i + 1][i] = -1          # 3
    M[n - 1][n], M[n][n - 1] = -1, -2           # 4
    return M


def base(n):
    N = n + 1
    e = lambda i: [1 if k == i else 0 for k in range(N)]
    B = [tuple(x - y for x, y in zip(e(0), e(1)))]
    for i in range(1, n):
        B.append(tuple(x - y for x, y in zip(e(i), e(i + 1))))
    B.append(tuple(2 * x for x in e(n)))
    return B


def run(n, L, R):
    mods = module_list(n, R)
    B = base(n)
    srt = [sorted(G) for G in mods]
    idx = [{g: i for i, g in enumerate(s)} for s in srt]
    P = [[refl(G, a) for G in mods] for a in B]
    Pt = [tuple(tuple(idx[j][P[s][j][g]] for g in srt[j]) for j in range(len(mods))) for s in range(len(B))]
    act = lambda s, t: tuple(tuple(Pt[s][j][x] for x in t[j]) for j in range(len(mods)))
    T = tits_gens(gcm(n))
    N = len(B)
    I = tuple(tuple(1 if r == c else 0 for c in range(N)) for r in range(N))
    idt = tuple(tuple(range(len(s))) for s in srt)
    seen = {I: idt}; img = {idt: I}; front = [I]; clash = None
    for r in range(1, L + 1):
        nf = []
        for X in front:
            for s in range(N):
                Y = mul(T[s], X)
                w = act(s, seen[X])
                if Y in seen:
                    assert seen[Y] == w, 'a Coxeter relation fails in UW'
                    continue
                if w in img and img[w] != Y:
                    clash = (r, Y); break
                seen[Y] = w; img[w] = Y; nf.append(Y)
            if clash: break
        if clash: break
        front = nf
        print(f"  osp(2|{2*n}) radius {r}: ball {len(seen)}, images distinct", flush=True)
    return len(mods), len(seen), clash


if __name__ == '__main__':
    args = [(2, int(sys.argv[1]) if len(sys.argv) > 1 else 8, 3), (3, int(sys.argv[2]) if len(sys.argv) > 2 else 6, 2)]
    for n, L, R in args:
        k, b, c = run(n, L, R)
        line = f"osp(2|{2*n}): {k} modules, ball of radius {L} of C_g has {b} elements, " + \
               ("all images distinct" if c is None else f"CLASH at radius {c[0]}")
        print(line, flush=True)
