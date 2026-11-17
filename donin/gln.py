"""Honest decomposition of S^k(gl(n)) and E^k(gl(n)) into irreducibles, by
characters -- so that Theorem 5.2 can be checked against it rather than assumed."""

from functools import lru_cache
from itertools import product


# --------------------------------------------------------------------------- #
#  Schur expansion
# --------------------------------------------------------------------------- #

def ssyt_count(lam, content):
    """Kostka number K_{lam,content}: semistandard Young tableaux of shape lam
    (a partition) with the given content."""
    lam = tuple(x for x in lam if x)
    if sum(lam) != sum(content):
        return 0
    return _ssyt(lam, tuple(content))


@lru_cache(maxsize=None)
def _ssyt(lam, content):
    if not lam:
        return 1 if all(c == 0 for c in content) else 0
    if not content:
        return 0
    # place the largest letter: it must occupy a horizontal strip at the outside
    m = len(content) - 1
    k = content[m]
    if k == 0:
        return _ssyt(lam, content[:m])
    total = 0
    for sub in _horizontal_strips(lam, k):
        total += _ssyt(sub, content[:m])
    return total


def _horizontal_strips(lam, k):
    """all partitions mu <= lam with lam/mu a horizontal strip of k cells."""
    out = []
    n = len(lam)

    def rec(i, left, acc):
        if i == n:
            if left == 0:
                out.append(tuple(x for x in acc if x))
            return
        lo = lam[i + 1] if i + 1 < n else 0          # mu_i >= lam_{i+1}
        hi = lam[i]
        prev = acc[-1] if acc else None
        for mui in range(lo, hi + 1):
            d = lam[i] - mui
            if d > left:
                continue
            if prev is not None and mui > prev:      # mu must be a partition
                continue
            rec(i + 1, left - d, acc + [mui])

    rec(0, k, [])
    return out


def schur_expand(char, n):
    """char: dict (weight as a tuple of n nonneg ints) -> multiplicity.
    Returns dict (partition of length n) -> multiplicity of the irreducible."""
    char = {w: c for w, c in char.items() if c}
    out = {}
    while char:
        lam = max((w for w in char if list(w) == sorted(w, reverse=True)),
                  default=None)
        if lam is None:
            raise ValueError("no dominant weight left but the character is nonzero: %r"
                             % char)
        m = char[lam]
        # subtract m * s_lam; every weight of s_lam is already in the support,
        # the character being a genuine (nonnegative) one
        for w in list(char.keys()):
            k = ssyt_count(lam, w)
            if k:
                char[w] = char.get(w, 0) - m * k
                if char[w] == 0:
                    del char[w]
        out[lam] = out.get(lam, 0) + m
    return out


# --------------------------------------------------------------------------- #
#  the characters of S^k and E^k of the adjoint module
# --------------------------------------------------------------------------- #

def adjoint_weights(n):
    """the n^2 weights of gl(n) as vectors in Z^n."""
    ws = []
    for i in range(n):
        for j in range(n):
            w = [0] * n
            w[i] += 1
            w[j] -= 1
            ws.append(tuple(w))
    return ws


def sym_ext_characters(n, kmax, exterior=False):
    """characters of S^k(gl(n)) (or E^k) for k = 0..kmax, weights shifted by
    (s,...,s) with s = kmax so that they are partitions."""
    ws = adjoint_weights(n)
    # dict k -> {weight (unshifted tuple) -> mult}
    cur = {0: {tuple([0] * n): 1}}
    for w in ws:
        nxt = {k: dict(v) for k, v in cur.items()}
        top = 1 if exterior else kmax
        for k in range(0, kmax + 1):
            for m in range(1, top + 1):
                if k + m > kmax:
                    break
                src = cur.get(k, {})
                dst = nxt.setdefault(k + m, {})
                for wt, c in src.items():
                    nw = tuple(a + m * b for a, b in zip(wt, w))
                    dst[nw] = dst.get(nw, 0) + c
        cur = {k: {a: b for a, b in v.items() if b} for k, v in nxt.items()}
    return cur


def decompose(n, kmax, exterior=False):
    """k -> {highest weight (a vector summing to 0) -> multiplicity}."""
    chars = sym_ext_characters(n, kmax, exterior)
    out = {}
    for k in range(kmax + 1):
        ch = chars.get(k, {})
        if not ch:
            out[k] = {}
            continue
        shift = -min(min(w) for w in ch)
        sh = {tuple(a + shift for a in w): c for w, c in ch.items()}
        dec = schur_expand(sh, n)
        out[k] = {tuple(a - shift for a in lam): m for lam, m in dec.items()}
    return out
