"""Identities of Volichenko algebras in a FIXED MULTIDEGREE.

The multilinear route stops at degree 5 here (dim M_5 = 1680) and at degree 7 in
Bremner--Peresi (dim M_7 = 665 280); degree 8 has dim M_8 = 8! Cat_7 = 17 297 280
and is out of reach for either.  Their own degree-5 identity is not multilinear:
it lives in three variables, in the multidegree (2,2,1), where the ambient has
dimension Cat_4 * 5!/(2!2!1!) = 14 * 30 = 420 instead of 1680.  That is the route
taken here.

WHAT A FIXED MULTIDEGREE SEES.  Write M_mu for the multidegree-mu component of
the free nonassociative algebra on k variables, |mu| = n, so that

    dim M_mu = Cat_{n-1} * n! / (mu_1! ... mu_k!).

Over a field of characteristic 0 a T-ideal is determined by its multilinear
parts, and W |--> W (x)_{S_n} V^{(x)n} carries the multilinear space W with
S_n-decomposition  sum_lambda m_lambda [lambda]  to the GL(V)-module
sum_lambda (S^lambda V)^{m_lambda}, whose mu-weight space has dimension
sum_lambda m_lambda K_{lambda,mu}, the K's being Kostka numbers.  Hence

    dim (I_mu / C_mu) = sum_lambda m_lambda K_{lambda, mu},                 (*)

with m_lambda the multiplicity of [lambda] in the NEW multilinear identities
I_n / C_n.  Since K_{lambda,mu} =/= 0 iff lambda >= mu in dominance, and since
lambda >= mu forces l(lambda) <= l(mu):

    the multidegrees in k variables see exactly the lambda with at most k rows,
    and the single balanced multidegree mu = (m^k) sees all of them at once
    (it is the minimum in dominance among the partitions of n with <= k parts).

So each computation is a detector of known reach, and the Kostka matrix being
unitriangular, running several mu recovers the individual m_lambda.

MODULAR SAFETY.  Ranks are taken over F_p, p = 2^19 - 1 > n, as in [BrP, Lemma 8].
A rank mod p is at most the rank over Q, so the computed dim I_mu is an upper
bound and the computed dim C_mu a lower bound for the true ones; therefore
    dim I_mu = dim C_mu mod p  ==>  the same over Q,
which is the direction in which "nothing new" is proved.  A computed gap is a
candidate, not a conclusion, and must be re-examined.
"""

import itertools
import math
import random
import sys
import time
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np

import volid

P = (1 << 19) - 1                   # 524287, a Mersenne prime; p > n always here


# ------------------------------------------------------- monomials of M_mu ---

def monos(mu):
    """All binary trees with leaves labelled 1..k, variable i used mu_i times.

    A tree is an int (a leaf = the index of the variable) or a pair (L, R).
    Their number is Cat_{n-1} * n!/prod(mu_i!).
    """
    mu = tuple(mu)
    cache = {}

    def rec(nu):
        if nu in cache:
            return cache[nu]
        n = sum(nu)
        if n == 1:
            out = [next(i + 1 for i, v in enumerate(nu) if v)]
            cache[nu] = out
            return out
        out = []
        for left in sub_multidegrees(nu):
            if not (0 < sum(left) < n):
                continue
            right = tuple(a - b for a, b in zip(nu, left))
            for tl in rec(left):
                for tr in rec(right):
                    out.append((tl, tr))
        cache[nu] = out
        return out

    return rec(mu)


def sub_multidegrees(nu):
    """Every nu' with 0 <= nu'_i <= nu_i."""
    return itertools.product(*(range(a + 1) for a in nu))


def count_monos(mu):
    n = sum(mu)
    c = catalan(n - 1) * math.factorial(n)
    for a in mu:
        c //= math.factorial(a)
    return c


def catalan(n):
    c = 1
    for i in range(n):
        c = c * 2 * (2 * i + 1) // (i + 2)
    return c


# ------------------------------------------------- evaluation, with memoing ---

class Evaluator:
    """tree -> its value in the free Lie superalgebra, memoized on subtrees."""

    def __init__(self):
        self.cache = {}

    def __call__(self, t):
        v = self.cache.get(t)
        if v is not None:
            return v
        if isinstance(t, int):
            v = volid.gen(t)
        else:
            v = volid.mul(self(t[0]), self(t[1]))
        self.cache[t] = v
        return v


# ------------------------------------------------ dense echelon over F_p ---

class Ech:
    """Row echelon basis of a subspace of F_p^dim, entries kept symmetric.

    Rows arrive in blocks; a block is reduced against the basis with one BLAS
    matrix product, then row-reduced among itself, then the new pivots are
    appended.  Symmetric representatives keep |entry| <= p/2, so a dot product
    of length `dim` stays inside the exact range of a float64 as long as
    dim * (p/2)^2 < 2^53 -- checked in __init__.
    """

    def __init__(self, dim, p=P, cap=None):
        assert dim * (p // 2) ** 2 < (1 << 53), "float64 would lose exactness"
        self.dim = dim
        self.p = p
        self.piv = []                       # pivot column of each basis row
        self.B = np.zeros((0, dim))
        self.cap = cap                      # stop growing at this rank

    def rank(self):
        return len(self.piv)

    def _sym(self, X):
        X %= self.p
        X[X > self.p // 2] -= self.p
        return X

    def add_block(self, X):
        """X: 2-d float64 array of rows.  Returns how many new pivots."""
        if X.size == 0:
            return 0
        X = self._sym(np.asarray(X, dtype=np.float64))
        if self.piv:
            X = self._sym(X - X[:, self.piv] @ self.B)
        before = len(self.piv)
        # row-reduce the block among itself
        r = 0
        m = X.shape[0]
        for c in range(self.dim):
            if r >= m:
                break
            col = X[r:, c]
            nz = np.nonzero(col)[0]
            if nz.size == 0:
                continue
            k = r + nz[0]
            if k != r:
                X[[r, k]] = X[[k, r]]
            inv = pow(int(X[r, c]) % self.p, self.p - 2, self.p)
            X[r] = self._sym(X[r] * inv)
            others = np.nonzero(X[:, c])[0]
            others = others[others != r]
            if others.size:
                X[others] = self._sym(X[others] - np.outer(X[others, c], X[r]))
            self.piv.append(c)
            r += 1
            if self.cap is not None and len(self.piv) >= self.cap:
                break
        if r:
            new = X[:r]
            if self.B.size:
                # keep the old basis clean of the new pivots
                cols = self.piv[before:]
                self.B = self._sym(self.B - self.B[:, cols] @ new)
                self.B = np.vstack([self.B, new])
            else:
                self.B = new
        return len(self.piv) - before


def rows_of(vals, keys=None):
    """Turn a list of dicts (word -> coeff) into a dense float64 block."""
    if keys is None:
        ks = set()
        for v in vals:
            ks.update(v)
        keys = {k: i for i, k in enumerate(sorted(ks))}
    X = np.zeros((len(vals), len(keys)))
    for i, v in enumerate(vals):
        for k, c in v.items():
            X[i, keys[k]] = float(c % P if isinstance(c, int) else
                                  (int(c.numerator) *
                                   pow(int(c.denominator), P - 2, P)) % P)
    return X, keys


# ------------------------------------------------ dim I_mu and the word space ---

def word_index(mu):
    """Index every word of the ambient of multidegree mu: length-n tuples of
    (variable, parity) with the variable content mu."""
    n = sum(mu)
    letters = []
    for i, a in enumerate(mu):
        letters += [i + 1] * a
    seen = {}
    for perm in set(itertools.permutations(letters)):
        for par in itertools.product((0, 1), repeat=n):
            seen[tuple(zip(perm, par))] = None
    return {w: i for i, w in enumerate(sorted(seen))}


def dim_I(mu, trees=None, ev=None, block=64, verbose=True):
    """dim I_mu = dim M_mu - rank(evaluation)."""
    trees = trees if trees is not None else monos(mu)
    ev = ev or Evaluator()
    keys = word_index(mu)
    e = Ech(len(keys))
    t0 = time.time()
    for s in range(0, len(trees), block):
        chunk = [ev(t) for t in trees[s:s + block]]
        X, _ = rows_of(chunk, keys)
        e.add_block(X)
        if verbose and (s // block) % 40 == 0:
            print("      eval %6d/%d   rank %5d   [%.0fs]"
                  % (s, len(trees), e.rank(), time.time() - t0), flush=True)
    return len(trees) - e.rank(), e.rank()


# ---------------------------------------------------------- the consequences ---

def relabel(t, sub):
    if isinstance(t, int):
        return sub[t]
    return (relabel(t[0], sub), relabel(t[1], sub))


def contexts(rho):
    """Trees of content rho with one extra leaf 0, the hole."""
    if sum(rho) == 0:
        return [0]
    ext = (1,) + tuple(rho)                       # variable 0 used once
    out = []
    for t in monos(ext):
        out.append(shift(t))
    return out


def shift(t):
    """monos() labels variables 1..k; here label 1 is the hole, shift the rest."""
    if isinstance(t, int):
        return 0 if t == 1 else t - 1
    return (shift(t[0]), shift(t[1]))


def ordered_splits(mu, k):
    """All (nu_1,...,nu_k), each nonzero, with sum nu_i <= mu componentwise."""
    if k == 0:
        yield ()
        return
    for nu in sub_multidegrees(mu):
        if sum(nu) == 0:
            continue
        rest = tuple(a - b for a, b in zip(mu, nu))
        for tail in ordered_splits(rest, k - 1):
            yield (nu,) + tail


def consequence_stream(gens, mu, rng=None, budget=None):
    """Stream of degree-mu consequences  c[ f(m_1,...,m_k) ]  as dicts."""
    mu = tuple(mu)
    n = sum(mu)
    cache = {}

    def mons(nu):
        if nu not in cache:
            cache[nu] = monos(nu)
        return cache[nu]

    plans = []
    for k, basis in sorted(gens.items()):
        if k > n:
            continue
        for nus in ordered_splits(mu, k):
            used = tuple(sum(x) for x in zip(*nus))
            rho = tuple(a - b for a, b in zip(mu, used))
            plans.append((k, nus, rho))
    if rng is not None:
        rng.shuffle(plans)
    made = 0
    for k, nus, rho in plans:
        basis = gens[k]
        ctxs = contexts(rho)
        pools = [mons(nu) for nu in nus]
        combos = list(itertools.product(*pools))
        if rng is not None:
            rng.shuffle(combos)
            rng.shuffle(ctxs)
        for choice in combos:
            sub = {i + 1: choice[i] for i in range(k)}
            for f in basis:
                # NB: distinct multilinear trees collapse onto one monomial
                # once the variables are identified -- accumulate, never assign.
                fs = {}
                for t, c in f.items():
                    key = relabel(t, sub)
                    fs[key] = fs.get(key, 0) + c
                fs = {t: c for t, c in fs.items() if c}
                if not fs:
                    continue
                for ctx in ctxs:
                    if ctx == 0:
                        yield fs
                    else:
                        g = {}
                        for t, c in fs.items():
                            key = plug(ctx, t)
                            g[key] = g.get(key, 0) + c
                        yield {t: c for t, c in g.items() if c}
                    made += 1
                    if budget and made >= budget:
                        return


def plug(ctx, t):
    if isinstance(ctx, int):
        return t if ctx == 0 else ctx
    return (plug(ctx[0], t), plug(ctx[1], t))


def dim_C(mu, trees, gens, target=None, block=64, budget=400000, seed=1,
          verbose=True):
    """dim C_mu, streaming the consequences in a randomized order.

    Stops as soon as the rank reaches `target` (= dim I_mu), which is the
    expected outcome; otherwise runs to the budget and reports the gap.
    """
    pos = {t: i for i, t in enumerate(trees)}
    e = Ech(len(trees), cap=target)
    rng = random.Random(seed)
    buf, seen, t0 = [], 0, time.time()
    hit = False                     # did we stop early, at the target?
    stream = consequence_stream(gens, mu, rng=rng, budget=budget)
    for g in stream:
        r = np.zeros(len(trees))
        ok = True
        for t, c in g.items():
            j = pos.get(t)
            if j is None:                      # cannot happen: same multidegree
                ok = False
                break
            r[j] = float(c)
        if not ok or not r.any():
            continue
        buf.append(r)
        seen += 1
        if len(buf) >= block:
            e.add_block(np.array(buf))
            buf = []
            if verbose and (seen // block) % 25 == 0:
                print("      cons %7d   rank %6d/%s   [%.0fs]"
                      % (seen, e.rank(), target, time.time() - t0), flush=True)
            if target is not None and e.rank() >= target:
                hit = True
                break
    if buf and not hit:
        e.add_block(np.array(buf))
        hit = target is not None and e.rank() >= target
    # `exhausted`: the whole supply of consequences was used, so the rank is
    # dim C_mu exactly and a gap is real (up to the modular caveat).
    exhausted = hit or (seen < budget)
    return e.rank(), seen, exhausted


# ------------------------------------------------------- Kostka & dominance ---

def partitions(n, mx=None):
    mx = mx or n
    if n == 0:
        yield ()
        return
    for k in range(min(n, mx), 0, -1):
        for rest in partitions(n - k, k):
            yield (k,) + rest


def kostka(lam, mu):
    """K_{lam,mu}: semistandard tableaux of shape lam and content mu."""
    cells = [(i, j) for i, r in enumerate(lam) for j in range(r)]

    def fill(k, tab, left):
        if k == len(cells):
            return 1
        i, j = cells[k]
        tot = 0
        for v in range(1, len(mu) + 1):
            if left[v - 1] == 0:
                continue
            if j > 0 and tab[(i, j - 1)] > v:
                continue
            if i > 0 and tab[(i - 1, j)] >= v:
                continue
            tab[(i, j)] = v
            left[v - 1] -= 1
            tot += fill(k + 1, tab, left)
            left[v - 1] += 1
            del tab[(i, j)]
        return tot

    return fill(0, {}, list(mu))


def dominates(lam, mu):
    """lam >= mu in the dominance order (both partitions of the same n)."""
    a = b = 0
    for i in range(max(len(lam), len(mu))):
        a += lam[i] if i < len(lam) else 0
        b += mu[i] if i < len(mu) else 0
        if a < b:
            return False
    return True


def reach(mu):
    """The partitions a multidegree mu can see."""
    n = sum(mu)
    smu = tuple(sorted(mu, reverse=True))
    return [lam for lam in partitions(n) if kostka(lam, smu)]


if __name__ == "__main__":
    mu = tuple(int(x) for x in sys.argv[1].split(","))
    print("mu =", mu, " dim M_mu =", count_monos(mu))
    t = monos(mu)
    assert len(t) == count_monos(mu)
    print("  reach:", reach(mu))
