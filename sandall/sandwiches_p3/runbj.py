#!/usr/bin/env python3
r"""
runbj.py --- which of the two W-gradings of the Bouarroudj superalgebra
Bj(3; \un|3) is the one given by n_L(s(L))?

The tie-breaker needs no knowledge of the whole sandwich variety S.  Write
L_0^A, L_0^B for the two maximal subalgebras of codimension 3|3.  Both are
maximal and they are distinct, so L_0^A + L_0^B = L.  Hence:

 *  s := s(L) is a non-zero nilpotent subalgebra (Zelmanov-Kostrikin), so it is
    not an ideal of the simple L, so n_L(s) is a proper subalgebra;
 *  if s were stable under both L_0^A and L_0^B it would be stable under L,
    i.e. an ideal --- excluded.  So it is stable under at most one of them;
 *  therefore: take any sandwich s_0, and close it under ad of L_0^X and under
    brackets.  If that closure is not nilpotent (in practice: it is the whole
    of L), then s is not L_0^X-stable, so n_L(s) does not contain L_0^X, so
    n_L(s) != L_0^X.

Exactly one of A, B can survive; and whichever survives is n_L(s), because
n_L(s) is then a proper subalgebra containing the maximal L_0^X.

Output: runs/bj.log
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bj import (P, N, MONS, IDX, PAR, mvec, kbracket, parity, rref, inside,   # noqa
                reduce_by, std_component, deg_std, deg_A, deg_B, V_BASIS,
                prolong)

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "bj.log")
NAMES = ("t", "p", "q", "xi", "eta")


def show(v):
    out = []
    for k, c in enumerate(v):
        if c % P:
            m = MONS[k]
            s = "".join(n + ("" if x == 1 else "^(%d)" % x)
                        for n, x in zip(NAMES, m) if x) or "1"
            pre = "" if c == 1 else ("-" if c == P - 1 else str(c))
            out.append(pre + s)
    return " + ".join(out) if out else "0"


def submodule(gens, acting):
    rows, piv = rref(gens)
    while True:
        new = []
        for h in acting:
            for r in rows:
                v = kbracket(h, r, pf=parity(h))
                if v.any() and not inside(rows, piv, v):
                    new.append(v)
        if not new:
            return rows, piv
        rows, piv = rref(np.concatenate([rows, np.array(new)]))


def lie_closure(gens, extra=()):
    """the smallest subspace containing gens, stable under ad of `extra` and
    closed under its own bracket."""
    rows, piv = rref([g for g in gens if np.asarray(g).any()])
    while True:
        new = []
        for a in list(rows) + list(extra):
            for b in rows:
                v = kbracket(a, b, pf=parity(a))
                if v.any() and not inside(rows, piv, v):
                    new.append(v)
        if not new:
            return rows, piv
        rows, piv = rref(np.concatenate([rows, np.array(new)]))


def lower_central(rows, piv):
    """the lower central series of the subalgebra span(rows); returns the list
    of dimensions."""
    cur, cpiv = rows, piv
    dims = [len(cur)]
    for _ in range(40):
        new = []
        for a in rows:
            for b in cur:
                v = kbracket(a, b, pf=parity(a))
                if v.any():
                    new.append(v)
        R, pv = rref(new) if new else (np.zeros((0, N), np.int64), [])
        dims.append(len(R))
        if len(R) == 0 or len(R) == dims[-2]:
            break
        cur, cpiv = R, pv
    return dims


def normaliser(alg_rows, alg_piv, sub_rows, sub_piv):
    """{x in span(alg_rows) | [x, sub] subset sub}."""
    eqs = []
    for x_i, x in enumerate(alg_rows):
        pass
    # linear system in the coordinates of x w.r.t. alg_rows
    k = len(alg_rows)
    comp = [c for c in range(N) if c not in sub_piv]
    rows = []
    for s in sub_rows:
        for c in comp:
            r = []
            for x in alg_rows:
                v = reduce_by(sub_rows, sub_piv, kbracket(x, s, pf=parity(x)))
                r.append(int(v[c]) % P)
            rows.append(r)
    M = np.array(rows, dtype=np.int64) % P if rows else np.zeros((0, k), np.int64)
    R, piv = rref(M) if M.size else (np.zeros((0, k), np.int64), [])
    free = [c for c in range(k) if c not in piv]
    out = []
    for f in free:
        s = np.zeros(k, dtype=np.int64)
        s[f] = 1
        for t, c in enumerate(piv):
            s[c] = (-R[t, f]) % P
        v = np.zeros(N, dtype=np.int64)
        for co, a in zip(s, alg_rows):
            if co:
                v = (v + co * a) % P
        out.append(v)
    return rref(out) if out else (np.zeros((0, N), np.int64), [])


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    log = open(LOG, "w", buffering=1)

    def say(m):
        print(m)
        log.write(m + "\n")

    k0 = std_component(0)
    W_eta, _ = submodule([mvec(0, 0, 2, 0, 1)], k0)       # the eta-family
    W_xi, _ = submodule([mvec(0, 0, 2, 1, 0)], k0)
    say("k(3; 1|2): dim %d;  W_eta dim %d, W_xi dim %d" % (N, len(W_eta), len(W_xi)))
    for r in W_eta:
        say("    W_eta: %s" % show(r))

    g = prolong(list(V_BASIS) + list(W_eta))
    rows = [v for i in sorted(g) for v in g[i]]
    R, piv = rref(rows)
    ev = sum(1 for r in R if parity(r) == 0)
    say("")
    say("Bj(3; 1|2) = prolong of k_{<=0} + V + W_eta:  sdim = (%d|%d), dim %d"
        % (ev, len(R) - ev, len(R)))
    say("    standard grading: " + ", ".join(
        "g_%d: %d" % (i, len(g[i])) for i in sorted(g) if g[i]))

    # closure and simplicity
    closed = all(inside(R, piv, kbracket(a, b, pf=parity(a))) for a in R for b in R)
    say("    closed under the bracket: %s" % closed)

    def ideal_of(v):
        r, p_ = rref([v])
        while True:
            new = []
            for a in R:
                for b in r:
                    w = kbracket(a, b, pf=parity(a))
                    if w.any() and not inside(r, p_, w):
                        new.append(w)
            if not new:
                return r
            r, p_ = rref(np.concatenate([r, np.array(new)]))
    simple = all(len(ideal_of(v)) == len(R) for v in R)
    say("    simple: %s" % simple)

    # ------------------------------------------------ the two W-gradings
    out = {}
    for tag, dg in (("A  Bj(3;un|3)  [deg xi = 0]", deg_A),
                    ("B  ~Bj(3;un|3) [deg eta = 0]", deg_B)):
        comp = {}
        homog = True
        for r in R:
            ds = {dg(MONS[k]) for k, c in enumerate(r) if c % P}
            if len(ds) != 1:
                homog = False
            comp.setdefault(min(ds), []).append(r)
        say("")
        say("grading %s : basis homogeneous %s" % (tag, homog))
        for i in sorted(comp):
            b = comp[i]
            e_ = sum(1 for v in b if parity(v) == 0)
            say("    g_%-3d sdim (%d|%d)" % (i, e_, len(b) - e_))
        neg = [v for i in comp for v in comp[i] if i < 0]
        nonneg = [v for i in comp for v in comp[i] if i >= 0]
        e_ = sum(1 for v in neg if parity(v) == 0)
        say("    codim of L_0 = (%d|%d)" % (e_, len(neg) - e_))
        out[tag[0]] = rref(nonneg)

    # ------------------------------------------------ the sandwiches
    say("")
    say("sandwiches: (ad_s)^2 = 0 on L")

    def ad2_zero(v):
        pv = parity(v)
        for b in R:
            if kbracket(v, kbracket(v, b, pf=pv), pf=pv).any():
                return False
        return True

    # a homogeneous scan: one representative per line of each (deg_std, deg_A,
    # deg_B, parity, weight) block, plus every basis vector of the algebra
    sand = []
    blocks = {}
    for r in R:
        ks = [k for k, c in enumerate(r) if c % P]
        key = (deg_std(MONS[ks[0]]), deg_A(MONS[ks[0]]), deg_B(MONS[ks[0]]),
               parity(r), tuple(MONS[ks[0]][i] for i in (1, 2)))
        blocks.setdefault(key, []).append(r)
    for key, bs in blocks.items():
        d = len(bs)
        for lead in range(d):
            for tail in range(P ** (d - lead - 1)):
                co = [0] * lead + [1]
                t = tail
                for _ in range(d - lead - 1):
                    co.append(t % P)
                    t //= P
                v = np.zeros(N, dtype=np.int64)
                for c, b in zip(co, bs):
                    v = (v + c * b) % P
                if v.any() and ad2_zero(v):
                    sand.append(v)
    Rs, _ = rref(sand) if sand else (np.zeros((0, N), np.int64), [])
    e_ = sum(1 for r in Rs if parity(r) == 0)
    say("    %d homogeneous sandwiches found; their span has sdim (%d|%d)"
        % (len(sand), e_, len(Rs) - e_))
    for r in Rs:
        say("        %s   (deg_A %d, deg_B %d)"
            % (show(r), deg_A(MONS[[k for k, c in enumerate(r) if c][0]]),
               deg_B(MONS[[k for k, c in enumerate(r) if c][0]])))

    # ------------------------------------------------ the tie-breaker
    say("")
    say("the tie-breaker: close one sandwich under ad L_0^X and the bracket")
    for tag in ("A", "B"):
        L0, L0p = out[tag]
        cl, clp = lie_closure(sand, extra=list(L0))
        e_ = sum(1 for r in cl if parity(r) == 0)
        lcs = lower_central(cl, clp)
        say("    X = %s : the closure has sdim (%d|%d) = dim %d of %d; "
            "lower central series %s"
            % (tag, e_, len(cl) - e_, len(cl), len(R), lcs))
        say("        nilpotent: %s" % (lcs[-1] == 0))

    # ------------------------------------------------ the normalisers
    say("")
    sgen, sgenp = lie_closure(sand)
    e_ = sum(1 for r in sgen if parity(r) == 0)
    say("the subalgebra generated by the homogeneous sandwiches: sdim (%d|%d)"
        % (e_, len(sgen) - e_))
    say("    lower central series %s" % lower_central(sgen, sgenp))
    nn, _ = normaliser(R, piv, sgen, sgenp)
    e_ = sum(1 for r in nn if parity(r) == 0)
    say("    its normaliser in L: sdim (%d|%d), codim (%d|%d)"
        % (e_, len(nn) - e_,
           sum(1 for r in R if parity(r) == 0) - e_,
           (len(R) - sum(1 for r in R if parity(r) == 0)) - (len(nn) - e_)))
    for tag in ("A", "B"):
        L0, L0p = out[tag]
        same = len(L0) == len(nn) and all(inside(L0, L0p, r) for r in nn) \
            and all(inside(*rref(nn), r) for r in L0)
        inc = all(inside(*rref(nn), r) for r in L0)
        say("    equals L_0^%s: %s   (L_0^%s inside it: %s)" % (tag, same, tag, inc))

    log.close()


if __name__ == "__main__":
    main()
