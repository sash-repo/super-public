#!/usr/bin/env python3
r"""
runbj6.py --- the Bouarroudj superalgebra Bj(3; \One|3) against the repaired
definition of a sandwich.

Comment 2) of Subsec. remBjs of the text exhibits six odd elements of
Bj(3; \One|2),

        xi, eta, p xi, q xi, p eta, q eta,

each with (ad_s)^2 = 0, generating a 13-dimensional subalgebra that coincides
with its own derived algebra --- whence the Correction (impC) is not sufficient
for the Zelmanov--Kostrikin theorem.

Here we ask whether those six are sandwiches at all under the repaired
definition, i.e., whether they satisfy

    (8)  [[g, s], [g, s]] = 0,   equivalently (7)  ad_s ad_x ad_s = 0,

the condition [ZK] imposes at p = 2 and which, by the identity

    (1 + (-1)^{p(s)}) ad_s ad_x ad_s = 0,

does NOT follow from (ad_s)^2 = 0 for an odd s in any characteristic.

Checked here:
  0) the identity (I) ad_s ad_x ad_s (y) = [[s,x],[s,y]] on Bj itself;
  1) the six elements, one by one;
  2) every odd line supported in a single homogeneous component of each of the
     three gradings (the same sweep as runbj5.py), and every F_3-combination
     of two of them that again has (ad_s)^2 = 0;
  3) the even sandwiches: they satisfy (8) automatically (p = 3), so
     s(Bj) under the repaired definition is what Theorem ThBj item 2) says.

Output: runs/bj6.log
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bj import (P, N, MONS, parity, rref, inside, deg_std, deg_A, deg_B,      # noqa
                kbracket, mvec)
from runbj import show, lie_closure, lower_central
from runbj3 import build, points

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "bj6.log")

SIX = [("xi", mvec(0, 0, 0, 1, 0)),
       ("eta", mvec(0, 0, 0, 0, 1)),
       ("p xi", mvec(0, 1, 0, 1, 0)),
       ("q xi", mvec(0, 0, 1, 1, 0)),
       ("p eta", mvec(0, 1, 0, 0, 1)),
       ("q eta", mvec(0, 0, 1, 0, 1))]


def ad2_zero(v, R):
    pv = parity(v)
    return not any(kbracket(v, kbracket(v, b, pf=pv), pf=pv).any() for b in R)


def image_basis(v, R):
    """a basis of [g, v], split by parity: (even rows, odd rows)."""
    pv = parity(v)
    ev, od = [], []
    for b in R:
        w = kbracket(v, b, pf=pv)
        if w.any():
            (ev if parity(w) == 0 else od).append(w)
    out = []
    for rows, par in ((ev, 0), (od, 1)):
        if rows:
            Rr, _ = rref(rows)
            out += [(r, par) for r in Rr if r.any()]
    return out


def cond8(v, R):
    """[[g, v], [g, v]] = 0 ?"""
    B = image_basis(v, R)
    for (a, pa), (b, _) in itertools.combinations_with_replacement(B, 2):
        if kbracket(a, b, pf=pa).any():
            return False
    return True


def cond7(v, R):
    """ad_v ad_x ad_v = 0 for every x in the basis R ?"""
    pv = parity(v)
    for x in R:
        px = parity(x)
        for b in R:
            w = kbracket(v, b, pf=pv)
            if not w.any():
                continue
            u = kbracket(x, w, pf=px)
            if not u.any():
                continue
            if kbracket(v, u, pf=pv).any():
                return False
    return True


def identity_I(v, R):
    """ad_v ad_x ad_v (y) == [[v,x],[v,y]] for all x, y in R."""
    pv = parity(v)
    for x in R:
        px, vx = parity(x), kbracket(v, x, pf=pv)
        for y in R:
            w = kbracket(v, y, pf=pv)
            lhs = kbracket(v, kbracket(x, w, pf=px), pf=pv) if w.any() else np.zeros(N, np.int64)
            rhs = kbracket(vx, w, pf=parity(vx)) if (vx.any() and w.any()) else np.zeros(N, np.int64)
            if ((lhs - rhs) % P).any():
                return False
    return True


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    log = open(LOG, "w", buffering=1)

    def say(m):
        print(m)
        log.write(m + "\n")

    R, piv = build()
    say("Bj(3; \\One|2): dim %d" % len(R))
    ev = [r for r in R if parity(r) == 0]
    say("sdim (%d|%d)" % (len(ev), len(R) - len(ev)))

    # ------------------------------------------------------------ 0) identity
    say("")
    say("0) the identity (I) ad_s ad_x ad_s(y) = [[s,x],[s,y]] on Bj,")
    say("   for every s with (ad_s)^2 = 0 among the six and among 20 random")
    say("   basis elements of each parity:")
    tested = 0
    for _, v in SIX:
        assert ad2_zero(v, R)
        assert identity_I(v, R)
        tested += 1
    for b in R:
        if ad2_zero(b, R):
            assert identity_I(b, R)
            tested += 1
    say("   OK on all %d of them" % tested)

    # --------------------------------------------------------------- 1) the six
    say("")
    say("1) the six odd elements of eq. (sixs):")
    for name, v in SIX:
        a2, c8, c7 = ad2_zero(v, R), cond8(v, R), cond7(v, R)
        say("   %-6s : (ad_s)^2 = 0 : %-3s   (8) : %-3s   (7) : %-3s"
            % (name, "yes" if a2 else "no", "yes" if c8 else "NO",
               "yes" if c7 else "NO"))
        assert a2
        assert c7 == c8
    say("   so not one of the six is a sandwich under the repaired definition.")

    # ---------------------------------------------------------- 2) the sweep
    say("")
    say("2) the sweep over the homogeneous odd lines of the three gradings:")
    cand = []
    for dg in (deg_std, deg_A, deg_B):
        comp = {}
        for r in R:
            k0 = [k for k, c in enumerate(r) if c % P][0]
            comp.setdefault((dg(MONS[k0]), parity(r)), []).append(r)
        for (d, par), bs in comp.items():
            if par != 1:
                continue
            for co in points(len(bs)):
                v = np.zeros(N, dtype=np.int64)
                for c, b in zip(co, bs):
                    v = (v + c * b) % P
                if v.any() and ad2_zero(v, R):
                    cand.append(v)
    seen, uniq = set(), []
    for v in cand:
        k = tuple(int(c) for c in v)
        if k not in seen:
            seen.add(k)
            uniq.append(v)
    say("   distinct odd s with (ad_s)^2 = 0 found: %d" % len(uniq))
    strong = [v for v in uniq if cond8(v, R)]
    say("   of them satisfying (8): %d" % len(strong))

    S, spiv = rref(strong) if strong else (np.zeros((0, N), np.int64), [])
    say("   they span a subspace of dim %d; every line of that span is" % len(S))
    inspan_bad = 0
    for co in points(len(S)) if len(S) else []:
        v = np.zeros(N, dtype=np.int64)
        for c, b in zip(co, S):
            v = (v + c * b) % P
        if not v.any():
            continue
        if not (ad2_zero(v, R) and cond8(v, R)):
            inspan_bad += 1
    say("   again a sandwich: %s (%d of the %d lines fail)"
        % ("yes" if not inspan_bad else "NO", inspan_bad,
           (P ** len(S) - 1) // (P - 1) if len(S) else 0))

    # a random sample of the combinations of two cone elements
    say("")
    say("   a random sample of F_3-combinations of two of the 323:")
    rng = np.random.default_rng(20260916)
    pairs, incone, strong2, outside = 0, 0, 0, 0
    for _ in range(4000):
        i, j = rng.integers(0, len(uniq), 2)
        if i == j:
            continue
        c = int(rng.integers(1, P))
        v = (uniq[i] + c * uniq[j]) % P
        pairs += 1
        if not v.any() or not ad2_zero(v, R):
            continue
        incone += 1
        if cond8(v, R):
            strong2 += 1
            if len(S) and not inside(S, spiv, v):
                outside += 1
    say("   combinations tried: %d; of them with (ad_s)^2 = 0: %d;"
        " of those satisfying (8): %d, of which outside the span above: %d"
        % (pairs, incone, strong2, outside))

    # ------------------------------------------------------- 3) the even ones
    say("")
    say("3) the even sandwiches (p = 3, so (8) is automatic for them):")
    evens = []
    for dg in (deg_std, deg_A, deg_B):
        comp = {}
        for r in R:
            k0 = [k for k, c in enumerate(r) if c % P][0]
            comp.setdefault((dg(MONS[k0]), parity(r)), []).append(r)
        for (d, par), bs in comp.items():
            if par != 0:
                continue
            for co in points(len(bs)):
                v = np.zeros(N, dtype=np.int64)
                for c, b in zip(co, bs):
                    v = (v + c * b) % P
                if v.any() and ad2_zero(v, R):
                    evens.append(v)
    bad = [v for v in evens if not cond8(v, R)]
    E, epiv = rref(evens) if evens else (np.zeros((0, N), np.int64), [])
    say("   even s with (ad_s)^2 = 0 found: %d; all satisfy (8): %s;"
        " they span dim %d" % (len(evens), "yes" if not bad else "NO", len(E)))
    rows, pv = lie_closure(list(E) + list(strong))
    lcs = lower_central(rows, pv)
    nev = sum(1 for r in rows if parity(r) == 0)
    say("   s(Bj) under the repaired definition: sdim (%d|%d), dim %d,"
        " lower central series %s%s"
        % (nev, len(rows) - nev, len(rows), lcs,
           "  -- NILPOTENT" if lcs[-1] == 0 else "  -- NOT nilpotent"))

    say("")
    say("done.")
    log.close()


if __name__ == "__main__":
    main()
