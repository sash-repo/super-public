#!/usr/bin/env python3
"""
Egorov, "How to superize gl(infty)": the classification of parity functions
p: Z -> Z/2 under the groups S_c, S_n (= S_l) and S_o.

The claim under test (Theorem 4.3.2 of the text, with Lemma 4.3.1, Example 4.3.3,
Proposition 4.3.4 and Remark 4.3.5; the shifts h, k of Theorem 4.3.2 are 0 here):

    let A  = p^{-1}(od),  A^c = p^{-1}(ev),  with increasing enumerations
    (a_n) and (c_n) anchored at 0 (index n >= 0 lists the part in [0,oo),
    index n < 0 the part in (-oo,-1] downwards), and likewise (a'_n), (c'_n)
    for p'.  Then

      p ~_{S_c} p'  <=>  sup_n |a_n - a'_n| < oo  and  sup_n |c_n - c'_n| < oo,
      p ~_{S_n} p'  <=>  |a_n - a'_n| = o(|a_n|)  and  |c_n - c'_n| = o(|c_n|),
      p ~_{S_o} p'  <=>  |a_n - a'_n| = |a_n|^{o(1)} and the same for c.

The two ingredients are
  (i)  sufficiency: the "sorted" bijection  a'_n |-> a_n,  c'_n |-> c_n
       is a bijection of Z with  p(sigma(i)) = p'(i);
  (ii) necessity: among all bijections between two subsets of Z the sorted one
       minimizes the maximal displacement, and does so locally.

Checks performed (all exact, integer arithmetic):

  1. sorted matching is a bijection of the window and transports p to p'
  2. bottleneck optimality of the sorted matching against an exhaustive /
     Hopcroft-Karp minimum-bottleneck matching, on random instances
  3. the local form of (ii): a witness of displacement >= a_n - a'_n sits at a
     position <= a'_n
  4. a pair of tight parity functions with one and the same one-point density
     spectrum which is NOT S_c-equivalent  (the density spectrum is therefore
     not a complete invariant)
  5. the pair of 4) IS S_n-equivalent, so the three groups really separate
  6. Egorov's own sigma_p (the normalization sigma_p(p_st) = p) against the
     sorted matching: they agree up to a bounded error for tight p
"""

import math
import random
import sys
from collections import deque

# ---------------------------------------------------------------- enumerations


def enumerate_set(bits, N, want):
    """Positions i in [-N, N] with bits[i] == want, split at 0 and indexed as in
    the text: index n >= 0 runs over [0, N] upwards, index n < 0 over [-N, -1]
    downwards.  Returns a dict n -> position."""
    out = {}
    n = 0
    for i in range(0, N + 1):
        if bits[i] == want:
            out[n] = i
            n += 1
    n = -1
    for i in range(-1, -N - 1, -1):
        if bits[i] == want:
            out[n] = i
            n -= 1
    return out


class Parity:
    """A parity function on [-N, N]; bits[i] in {0,1}, 1 = odd."""

    def __init__(self, bits, N):
        self.N = N
        self.bits = dict(bits)
        self.odd = enumerate_set(self.bits, N, 1)
        self.even = enumerate_set(self.bits, N, 0)

    def __getitem__(self, i):
        return self.bits[i]


def sorted_matching(p, q, margin):
    """sigma with p(sigma(i)) = q(i), built from the two sorted enumerations.
    Only indices whose both partners lie at distance <= margin from the window
    ends are used.  Returns (sigma, maxdisp_odd, maxdisp_even)."""
    sigma = {}
    d_odd = d_even = 0
    for src, dst, tag in ((q.odd, p.odd, "odd"), (q.even, p.even, "even")):
        for n, pos in src.items():
            if n not in dst:
                continue
            tgt = dst[n]
            if abs(pos) > p.N - margin or abs(tgt) > p.N - margin:
                continue
            sigma[pos] = tgt
            if tag == "odd":
                d_odd = max(d_odd, abs(tgt - pos))
            else:
                d_even = max(d_even, abs(tgt - pos))
    return sigma, d_odd, d_even


# ------------------------------------------------- minimum-bottleneck matching


def bottleneck(xs, ys):
    """Minimal over all bijections xs -> ys of the maximal |x - y|.
    xs, ys are equal-size lists of integers.  Hopcroft-Karp + binary search."""
    assert len(xs) == len(ys)
    if not xs:
        return 0
    cands = sorted({abs(x - y) for x in xs for y in ys})
    lo, hi = 0, len(cands) - 1
    if not feasible(xs, ys, cands[hi]):
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(xs, ys, cands[mid]):
            hi = mid
        else:
            lo = mid + 1
    return cands[lo]


def feasible(xs, ys, C):
    """Is there a perfect matching with |x - y| <= C?  Hopcroft-Karp."""
    n, m = len(xs), len(ys)
    adj = [[j for j in range(m) if abs(xs[i] - ys[j]) <= C] for i in range(n)]
    matchL = [-1] * n
    matchR = [-1] * m
    INF = float("inf")
    while True:
        dist = [INF] * n
        q = deque()
        for i in range(n):
            if matchL[i] == -1:
                dist[i] = 0
                q.append(i)
        found = False
        while q:
            i = q.popleft()
            for j in adj[i]:
                k = matchR[j]
                if k == -1:
                    found = True
                elif dist[k] == INF:
                    dist[k] = dist[i] + 1
                    q.append(k)
        if not found:
            break

        def dfs(i):
            for j in adj[i]:
                k = matchR[j]
                if k == -1 or (dist[k] == dist[i] + 1 and dfs(k)):
                    matchL[i] = j
                    matchR[j] = i
                    return True
            dist[i] = INF
            return False

        for i in range(n):
            if matchL[i] == -1:
                dfs(i)
    return all(v != -1 for v in matchL)


# ---------------------------------------------------------- parity constructors


def p_st(N):
    return Parity({i: i % 2 for i in range(-N, N + 1)}, N)


def p_from_odd_set(A, N):
    return Parity({i: (1 if i in A else 0) for i in range(-N, N + 1)}, N)


def p_blocks(N, block):
    """Density 1/2, but built of blocks of length block(x) of equal parity:
    odd on [s, s+L), even on [s+L, s+2L), L = block(s)."""
    bits = {}
    for side in (1, -1):
        s, val = 0, 1
        while abs(s) <= N:
            L = max(1, block(abs(s)))
            for t in range(L):
                i = s + side * t if side > 0 else s - t - 1
                if -N <= i <= N:
                    bits[i] = val
            s = s + side * L
            val ^= 1
    for i in range(-N, N + 1):
        bits.setdefault(i, 0)
    return Parity(bits, N)


def p_shift(N, shift):
    """p_st with a slowly drifting phase: odd iff (i + shift(i)) is odd."""
    return Parity({i: (i + shift(abs(i))) % 2 for i in range(-N, N + 1)}, N)


def density(p, a, b):
    n = sum(1 for i in range(a, b + 1) if p[i] == 1)
    return n / (b - a + 1)


def sigma_egorov(p, N):
    """Egorov's own sigma_p, the map with p_st(sigma_p(x)) = p(x)."""
    out = {}
    ev = 0
    od = 0
    for x in range(0, N + 1):
        if p[x] == 0:
            out[x] = 2 * ev
            ev += 1
        else:
            out[x] = 2 * od + 1
            od += 1
    ev = od = 0
    for x in range(-1, -N - 1, -1):
        if p[x] == 0:
            out[x] = -2 * (ev + 1)
            ev += 1
        else:
            out[x] = -2 * od - 1
            od += 1
    return out


# ---------------------------------------------------------------------- checks

FAIL = 0


def check(name, ok, extra=""):
    global FAIL
    if not ok:
        FAIL += 1
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' -- ' + extra) if extra else ''}")


def main():
    random.seed(20260828)
    N = 400
    margin = 60

    print("1) the sorted bijection transports p' to p  (random pairs)")
    worst = []
    for trial in range(40):
        d = random.uniform(0.2, 0.8)
        A = {i for i in range(-N, N + 1) if random.random() < d}
        B = {i for i in range(-N, N + 1) if random.random() < d}
        p = p_from_odd_set(A, N)
        q = p_from_odd_set(B, N)
        sig, do, de = sorted_matching(p, q, margin)
        inj = len(set(sig.values())) == len(sig)
        transports = all(p[v] == q[k] for k, v in sig.items())
        worst.append(max(do, de))
        if not (inj and transports):
            check(f"trial {trial}", False)
            break
    else:
        check("40 random pairs: injective and p(sigma(i)) = p'(i)", True,
              f"max displacement seen {max(worst)}")

    print("2) the sorted matching is bottleneck-optimal (random finite sets)")
    bad = 0
    for trial in range(200):
        k = random.randint(1, 9)
        span = random.randint(k, 40)
        xs = sorted(random.sample(range(span), k))
        ys = sorted(random.sample(range(span), k))
        srt = max(abs(x - y) for x, y in zip(xs, ys))
        opt = bottleneck(xs, ys)
        if srt != opt:
            bad += 1
            print(f"      xs={xs} ys={ys} sorted={srt} opt={opt}")
    check("200 random instances: sorted == minimum bottleneck", bad == 0)

    print("3) the local witness: some index at a position <= a'_n moves by >= a_n - a'_n")
    bad = 0
    for trial in range(200):
        k = random.randint(2, 9)
        span = random.randint(k, 40)
        xs = sorted(random.sample(range(span), k))   # A  (targets)
        ys = sorted(random.sample(range(span), k))   # A' (sources)
        perm = list(range(k))
        random.shuffle(perm)
        for n in range(k):
            gap = xs[n] - ys[n]
            if gap <= 0:
                continue
            # every bijection must move some source at a position <= ys[n]
            # by at least gap
            wit = max(abs(xs[perm[m]] - ys[m]) for m in range(k) if ys[m] <= ys[n])
            if wit < gap:
                bad += 1
    check("200 random bijections: the local bound holds", bad == 0)

    print("4) a tight pair with one and the same one-point density spectrum,")
    print("   not S_c-equivalent")
    # p  = p_st (alternating, tight, spectrum {1/2})
    # p' = p_st with a phase drifting like sqrt: tight, spectrum {1/2}
    p = p_st(N)
    q = p_shift(N, lambda x: int(math.isqrt(x)))
    runs = [max_run(Parity({i: q[i] for i in range(-M, M + 1)}, M), M)
            for M in (50, 100, 200, 400)]
    dens = [density(q, a, a + 200) for a in (0, 40, 90, 150)]
    check("p' is tight: the longest constant run does not grow with the window",
          len(set(runs)) == 1, f"longest run over [-M,M] for M = 50,100,200,400: {runs}")
    check("both have density 1/2 on every long window", all(abs(x - .5) < .06 for x in dens),
          f"densities {['%.3f' % x for x in dens]}")
    _, do, de = sorted_matching(p, q, margin)
    growth = []
    for M in (50, 100, 200, 340):
        pm = Parity({i: p[i] for i in range(-M, M + 1)}, M)
        qm = Parity({i: q[i] for i in range(-M, M + 1)}, M)
        _, d1, d2 = sorted_matching(pm, qm, 5)
        growth.append(max(d1, d2))
    check("the displacement grows with the window: no S_c-equivalence",
          growth == sorted(growth) and growth[-1] > growth[0],
          f"max |a_n - a'_n| over [-M,M] for M = 50,100,200,340: {growth}")
    # the growth is exactly of order sqrt: measure it far out on the half-line
    M2 = 20000
    A = [i for i in range(0, M2) if i % 2 == 1]
    B = [i for i in range(0, M2) if (i + math.isqrt(i)) % 2 == 1]
    rr = []
    for k in (100, 400, 1600, 6400, 9000):
        rr.append(round(max(abs(a - b) for a, b in zip(A[:k], B[:k])) / math.sqrt(A[k - 1]), 3))
    check("max_{n<=k} |a_n - a'_n| / sqrt(a_k) stays at 1: the parting is of order sqrt",
          all(abs(x - 1) < .05 for x in rr),
          f"ratios for k = 100,400,1600,6400,9000: {rr}")

    print("5) the same pair IS S_n-equivalent: |a_n - a'_n| = o(|a_n|)")
    ratios = []
    for M in (50, 100, 200, 340):
        pm = Parity({i: p[i] for i in range(-M, M + 1)}, M)
        qm = Parity({i: q[i] for i in range(-M, M + 1)}, M)
        sig, d1, d2 = sorted_matching(pm, qm, 5)
        r = max((abs(v - k) / abs(k)) for k, v in sig.items() if abs(k) > M // 2)
        ratios.append(round(r, 4))
    check("the ratio |sigma(i) - i| / |i| tends to 0",
          ratios == sorted(ratios, reverse=True) and ratios[-1] < ratios[0],
          f"max ratio on |i| > M/2 for M = 50,100,200,340: {ratios}")

    print("6) Egorov's sigma_p against the sorted matching, for a tight p")
    q2 = p_blocks(N, lambda x: 1 + (x % 2))    # tight, density 1/2
    eg = sigma_egorov(q2, N)
    # sigma_egorov carries q2 to p_st; the sorted matching from p_st to q2 is
    # its inverse up to bounded error
    pst = p_st(N)
    sig, do, de = sorted_matching(pst, q2, margin)
    diff = max(abs(eg[i] - sig[i]) for i in sig if abs(i) < N - margin)
    check("the two normalizations agree up to a bounded error", diff <= 4,
          f"max difference {diff}, sorted-matching displacement {max(do, de)}")

    print("7) S_m: two one-point spectra that DIFFER, yet the functions are S_m-equivalent")
    # p  : alternating, density 1/2 ; p' : odd exactly on 3Z, density 1/3
    alt = p_st(N)
    third = p_from_odd_set({i for i in range(-N, N + 1) if i % 3 == 0}, N)
    d1 = [round(density(alt, a, a + 300), 3) for a in (0, 50)]
    d2 = [round(density(third, a, a + 300), 3) for a in (0, 50)]
    check("the two densities differ", abs(d1[0] - 0.5) < .02 and abs(d2[0] - 1 / 3) < .02,
          f"densities {d1} and {d2}")
    sig, do, de = sorted_matching(alt, third, margin)
    transports = all(alt[v] == third[k] for k, v in sig.items())
    inj = len(set(sig.values())) == len(sig)
    rat = [abs(v) / abs(k) for k, v in sig.items() if abs(k) > N // 2 and v != 0]
    check("sigma is an injection transporting p' to p", transports and inj)
    check("|sigma(i)| / |i| is bounded above and away from 0: sigma is in S_m",
          0 < min(rat) and max(rat) < 10,
          f"ratio on |i| > {N // 2} runs in [{min(rat):.3f}, {max(rat):.3f}]")
    check("sigma is NOT in S_c: the displacement is unbounded",
          max(do, de) > 40, f"max displacement {max(do, de)} on a window of half-width {N}")

    print()
    print(f"failures: {FAIL}")
    return 1 if FAIL else 0


def max_run(p, N):
    best = run = 1
    for i in range(-N + 1, N + 1):
        if p[i] == p[i - 1]:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


if __name__ == "__main__":
    sys.exit(main())
