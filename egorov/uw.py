"""Serganova's universal super Weyl group UW for sl(m|n), m != n.

Weights of gl(m|n) are integer vectors (eps_1..eps_m | delta_1..delta_n); h^* of
sl(m|n) is the quotient by u = (1..1 | -1..-1).  The reflection r_alpha acts on
the weight SET Gamma of a finite-dimensional irreducible module by
    r_alpha(g) = g + (p - q) alpha,
p, q the lengths of the alpha-string through g inside Gamma (Subsec. 1.4 of the
text).  Modules used: the adjoint one (Gamma = R u {0}) and the typical
ones, Gamma = Gamma(L_0(lambda)) + {sums of subsets of the odd negative roots}.
"""
import itertools, sys
from math import gcd


def normalize(v, m, n):
    k = v[-1]                       # v + k*u has last coordinate 0
    return tuple(v[i] + k for i in range(m)) + tuple(v[m + j] - k for j in range(n))


def unit(i, N):
    return tuple(1 if k == i else 0 for k in range(N))


def roots(m, n):
    N = m + n
    ev, od = [], []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            a = tuple(unit(i, N)[k] - unit(j, N)[k] for k in range(N))
            ((i < m) == (j < m) and ev or od).append(a)
    return ev, od


def glweights(lam):
    """Weights of the irreducible gl(k)-module of highest weight lam (saturation)."""
    k, s = len(lam), sum(lam)
    lo, hi = min(lam), max(lam)
    out = []
    for mu in itertools.product(range(lo, hi + 1), repeat=k):
        if sum(mu) != s:
            continue
        a = sorted(mu, reverse=True)
        if all(sum(a[:t]) <= sum(lam[:t]) for t in range(1, k)):
            out.append(mu)
    return out


def typical(lam, m, n):
    # rho doubled: 2rho = sum even pos - sum odd pos, form (e,e)=1, (d,d)=-1
    two_rho = [0] * (m + n)
    for i in range(m):
        two_rho[i] = (m - 1 - 2 * i) - n
    for j in range(n):
        two_rho[m + j] = (n - 1 - 2 * j) + m
    # ((lam+rho), e_i - d_j) = (lam+rho)_i + (lam+rho)_{m+j}
    return all(2 * lam[i] + two_rho[i] + 2 * lam[m + j] + two_rho[m + j] != 0
               for i in range(m) for j in range(n))


def kac_weights(lam, m, n):
    neg_odd = [tuple(unit(m + j, m + n)[k] - unit(i, m + n)[k] for k in range(m + n))
               for i in range(m) for j in range(n)]
    base = [a + b for a in glweights(lam[:m]) for b in glweights(lam[m:])]
    G = set()
    for mu in base:
        for S in itertools.product((0, 1), repeat=len(neg_odd)):
            v = list(mu)
            for s, r in zip(S, neg_odd):
                if s:
                    v = [x + y for x, y in zip(v, r)]
            G.add(normalize(v, m, n))
    return frozenset(G)


def adjoint(m, n):
    ev, od = roots(m, n)
    return frozenset([normalize(a, m, n) for a in ev + od] + [tuple([0] * (m + n))])


def refl(G, a, m, n):
    a = normalize(a, m, n)
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


def order(perm):
    seen, o = set(), 1
    for x in perm:
        if x in seen:
            continue
        c, y = 0, x
        while y not in seen:
            seen.add(y); y = perm[y]; c += 1
        o = o * c // gcd(o, c)
    return o


def comp(p, q):                     # p after q
    return {x: p[q[x]] for x in q}


def module_list(m, n, R=3):
    mods = [adjoint(m, n)]
    for lam in itertools.product(range(-R, R + 1), repeat=m + n):
        if all(lam[i] >= lam[i + 1] for i in range(m - 1)) and \
           all(lam[m + j] >= lam[m + j + 1] for j in range(n - 1)) and \
           lam[-1] == 0 and typical(lam, m, n):
            mods.append(kac_weights(lam, m, n))
    return mods


def gen_closure(gens_tuple, limit=200000):
    """Closure of tuples of permutations (one per module) under composition."""
    key = lambda t: tuple(tuple(sorted(p.items())) for p in t)
    ident = tuple({x: x for x in p} for p in gens_tuple[0])
    seen = {key(ident): ident}
    frontier = [ident]
    while frontier:
        nf = []
        for t in frontier:
            for g in gens_tuple:
                s = tuple(comp(a, b) for a, b in zip(g, t))
                k = key(s)
                if k not in seen:
                    seen[k] = s; nf.append(s)
                    if len(seen) > limit:
                        return None
        frontier = nf
    return seen


if __name__ == "__main__":
    out = []
    say = lambda *a: (print(*a), out.append(" ".join(map(str, a))))
    for (m, n) in [(2, 1), (3, 1), (2, 3)]:
        N = m + n
        mods = module_list(m, n, R=3 if N <= 4 else 2)
        say(f"== sl({m}|{n}): {len(mods)} modules (adjoint + typical)")
        # distinguished base
        dist = [tuple(unit(i, N)[k] - unit(i + 1, N)[k] for k in range(N)) for i in range(N - 1)]
        odd_i = m - 1
        # Coxeter orders m_ij on the distinguished base, max over modules
        for i in range(N - 1):
            for j in range(i + 1, N - 1):
                orders = [order(comp(refl(G, dist[i], m, n), refl(G, dist[j], m, n))) for G in mods]
                say(f"  dist base: order(r{i+1} r{j+1}) max over modules = {max(orders)}"
                    f"  (largest 5: {sorted(orders)[-5:]})")
        # odd base(s): bases of sl(m|n) <-> orderings of eps/delta (shuffles)
        if (m, n) == (2, 1):
            e1, e2, d = unit(0, 3), unit(1, 3), unit(2, 3)
            sub = lambda x, y: tuple(a - b for a, b in zip(x, y))
            oddB = [sub(e1, d), sub(d, e2)]
            target = sub(e1, e2)
            small = [mods[0]] + [G for G in mods[1:] if len(G) <= 12][:2]
            gens = [tuple(refl(G, a, m, n) for G in small) for a in oddB]
            cl = gen_closure(gens)
            tgt = tuple(refl(G, target, m, n) for G in small)
            key = lambda t: tuple(tuple(sorted(p.items())) for p in t)
            say(f"  odd base {{e1-d, d-e2}}: group generated on {len(small)} modules has order"
                f" {len(cl)}; r_(e1-e2) in it: {key(tgt) in cl}")
            ad = mods[0]
            ra, rb = refl(ad, oddB[0], m, n), refl(ad, oddB[1], m, n)
            say(f"  on the adjoint: order(r_a r_b) = {order(comp(ra, rb))}")
