"""N-commutators of first-order differential operators (vector fields) in n variables.

a_N(X_1..X_N) = sum_s sign(s) X_s(1)...X_s(N), computed by the subset recursion
a_S = sum_{i in S} (-1)^{#{j in S, j<i}} X_i a_{S\\{i}}, level by level, mod a prime P.
A differential operator is a dict alpha -> poly, a poly a dict exponent -> int mod P.
"""
import random, sys, itertools, time
P = 1000003

def padd(a, b, s=1):
    for m, c in b.items():
        v = (a.get(m, 0) + s * c) % P
        if v: a[m] = v
        else: a.pop(m, None)

def pmul(a, b):
    r = {}
    for m1, c1 in a.items():
        for m2, c2 in b.items():
            m = tuple(x + y for x, y in zip(m1, m2))
            v = (r.get(m, 0) + c1 * c2) % P
            if v: r[m] = v
            else: r.pop(m, None)
    return r

def pdiff(a, i):
    r = {}
    for m, c in a.items():
        if m[i]:
            mm = list(m); mm[i] -= 1
            r[tuple(mm)] = (c * m[i]) % P
    return r

def compose(X, D, n):
    """X o D, X = list of n coefficient polys (first order, no constant term)."""
    R = {}
    for al, c in D.items():
        for i in range(n):
            if not X[i]: continue
            dc = pdiff(c, i)
            if dc:
                t = R.setdefault(al, {}); padd(t, pmul(X[i], dc))
            al2 = list(al); al2[i] += 1; al2 = tuple(al2)
            t = R.setdefault(al2, {}); padd(t, pmul(X[i], c))
    return {a: c for a, c in R.items() if c}

def aN(Xs, n):
    N = len(Xs)
    zero = tuple([0] * n)
    level = {0: {zero: {zero: 1}}}
    for k in range(1, N + 1):
        new = {}
        for S in range(1 << N):
            if bin(S).count('1') != k: continue
            acc = {}
            before = 0
            for i in range(N):
                if S >> i & 1:
                    T = S & ~(1 << i)
                    term = compose(Xs[i], level[T], n)
                    s = -1 if before % 2 else 1
                    for al, c in term.items():
                        t = acc.setdefault(al, {}); padd(t, c, s)
                    before += 1
            new[S] = {a: c for a, c in acc.items() if c}
        level = new
    return level[(1 << N) - 1]

def rpoly(n, deg, rng, nterms=None):
    mons = [m for m in itertools.product(range(deg + 1), repeat=n) if sum(m) <= deg]
    if nterms: mons = rng.sample(mons, min(nterms, len(mons)))
    return {m: rng.randrange(1, P) for m in mons}

def order(D):
    return max((sum(a) for a in D), default=-1)

def first_order_part(D, n):
    zero = tuple([0] * n)
    if any(sum(a) > 1 for a in D) or zero in D: return None
    X = []
    for i in range(n):
        e = [0] * n; e[i] = 1
        X.append(D.get(tuple(e), {}))
    return X

# --- the three algebras -------------------------------------------------------
def vect_field(n, deg, rng):
    return [rpoly(n, deg, rng) for _ in range(n)]

def ham_field(deg, rng):            # h(2) in (p,q): H_f = f_p d_q - f_q d_p, vars (q,p)
    f = rpoly(2, deg, rng)
    return [pdiff(f, 1), {m: (-c) % P for m, c in pdiff(f, 0).items()}]

def K(f):                          # vars (t,q,p), alpha = dt - p dq
    ft, fq, fp = pdiff(f, 0), pdiff(f, 1), pdiff(f, 2)
    pvar = {(0, 0, 1): 1}
    a = dict(f); padd(a, pmul(pvar, fp), -1)
    b = {m: (-c) % P for m, c in fp.items()}
    c = dict(fq); padd(c, pmul(pvar, ft))
    return [a, b, c]

def is_contact(X):
    pvar = {(0, 0, 1): 1}
    g = dict(X[0]); padd(g, pmul(pvar, X[1]), -1)
    Y = K(g)
    return all(not ({**Y[i]} != {**X[i]}) for i in range(3)), g

def contact_field(deg, rng):
    return K(rpoly(3, deg, rng))

def classify(D, n, contact=False):
    if not D: return 'zero'
    X = first_order_part(D, n)
    if X is None: return 'order%d' % order(D)
    if contact:
        ok, g = is_contact(X)
        return 'contact' if ok else 'vect-not-contact'
    return 'field'

if __name__ == '__main__':
    alg = sys.argv[1]; N = int(sys.argv[2]); deg = int(sys.argv[3]); trials = int(sys.argv[4])
    rng = random.Random(int(sys.argv[5]) if len(sys.argv) > 5 else 1)
    for tr in range(trials):
        t0 = time.time()
        if alg.startswith('vect'):
            n = int(alg[4:]); Xs = [vect_field(n, deg, rng) for _ in range(N)]; ctc = False
        elif alg == 'h2':
            n = 2; Xs = [ham_field(deg, rng) for _ in range(N)]; ctc = False
        elif alg == 'k3':
            n = 3; Xs = [contact_field(deg, rng) for _ in range(N)]; ctc = True
        D = aN(Xs, n)
        print(alg, 'N=%d deg=%d' % (N, deg), classify(D, n, ctc), 'ord', order(D), '%.1fs' % (time.time() - t0), flush=True)

# --- parallel version: one Pool per level (fork shares the previous level) -------
_G = {}
def _work(S):
    Xs, n, N, prev = _G['Xs'], _G['n'], _G['N'], _G['prev']
    acc = {}; before = 0
    for i in range(N):
        if S >> i & 1:
            term = compose(Xs[i], prev[S & ~(1 << i)], n)
            s = -1 if before % 2 else 1
            for al, c in term.items():
                t = acc.setdefault(al, {}); padd(t, c, s)
            before += 1
    return S, {a: c for a, c in acc.items() if c}

def aN_par(Xs, n, procs=10):
    import multiprocessing as mp
    N = len(Xs); zero = tuple([0] * n)
    prev = {0: {zero: {zero: 1}}}
    ctx = mp.get_context('fork')
    for k in range(1, N + 1):
        _G.update(Xs=Xs, n=n, N=N, prev=prev)
        subs = [S for S in range(1 << N) if bin(S).count('1') == k]
        if len(subs) < 40:
            prev = dict(_work(S) for S in subs)
        else:
            with ctx.Pool(procs) as pool:
                prev = dict(pool.map(_work, subs, chunksize=max(1, len(subs) // (4 * procs))))
    return prev[(1 << N) - 1]
