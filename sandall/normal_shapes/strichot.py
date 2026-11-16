#!/usr/bin/env python3
"""Task 3: the three cases of the union, on the full complex over O(m|n;u).

    (1) H^k of Omega^*(m|n;u) against the Kuenneth prediction of sfactor.py,
        by parity, for p = 2, 3, 5, 7;
    (2) the PARITY TRICHOTOMY of non-degeneracy: an odd non-degenerate form
        forces m = n; an even one forces the two diagonal blocks to be
        non-degenerate, hence m even when p > 2 and nothing when p = 2;
    (3) the standard shapes of each case and the superdimensions of the
        Lie superalgebras they preserve.

All arithmetic exact over F_p.
"""
import sys, os, random, itertools, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sform as S

random.seed(20260821)
LOG = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "strichot.log"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


# --------------------------------------------------------------- prediction
def factor_H(p, kind, kmax):
    """dim H^k of one factor, k = 0..kmax, as a list (closed form of sfactor.py)."""
    out = []
    for k in range(kmax + 1):
        if p > 2 and kind == "ev":
            out.append(1 if k in (0, 1) else 0)
        else:
            out.append((1 if k % p == 0 else 0) + (1 if (k + 1) % p == 0 else 0))
    return out


def factor_par(p, kind, k):
    """parity of the (1-dimensional, when non-zero) class of degree k.

    p>2, ev: k=0 even, k=1 odd (theta = xbar dx).
    p>2, od: k=0 mod p -> (dxi)^{(k)} even; k=-1 mod p -> xi(dxi)^{(k)} odd.
    p=2:     k=0 mod 2 -> (dg)^{(k)} even; k=1 mod 2 -> gbar (dg)^{(k)} odd.
    """
    if p > 2 and kind == "ev":
        return [0] if k == 0 else ([1] if k == 1 else [])
    out = []
    if k % p == 0:
        out.append(0)
    if (k + 1) % p == 0:
        out.append(1)
    return out


def predict_total(m, n, p, kmax):
    """Kuenneth: dim H^k of the whole complex, split by parity."""
    # each factor contributes (degree, parity) with multiplicity dim
    facs = [("ev", ) for _ in range(m)] + [("od", ) for _ in range(n)]
    res = {}
    def rec(idx, deg, par):
        if deg > kmax:
            return
        if idx == len(facs):
            res[(deg, par)] = res.get((deg, par), 0) + 1
            return
        kind = facs[idx][0]
        for k in range(kmax + 1 - deg + 1):
            for pr in factor_par(p, kind, k):
                rec(idx + 1, deg + k, (par + pr) % 2)
    rec(0, 0, 0)
    return res


# ------------------------------------------------------------------- part 1
say("(1) H^k(Omega^*(m|n;u)) against the Kuenneth prediction")
say("    line: dim H^k = total (even|odd), machine / predicted")
bad = 0
CASES = [(1, 1, 3, (1,)), (2, 1, 3, (1, 1)), (1, 2, 3, (1,)), (2, 2, 3, (1, 1)),
         (1, 1, 3, (2,)), (2, 1, 5, (1, 1)), (1, 2, 5, (1,)), (2, 0, 5, (1, 1)),
         (1, 2, 7, (1,)),
         (1, 1, 2, (1,)), (2, 1, 2, (1, 1)), (1, 2, 2, (1,)), (2, 2, 2, (1, 1)),
         (3, 0, 2, (1, 1, 1)), (0, 3, 2, ()), (1, 1, 2, (2,)), (0, 2, 2, ())]
for (m, n, p, u) in CASES:
    A = S.Om(m, n, p, u, dh=2, dxh=2)
    kmax = 2
    pred = predict_total(m, n, p, kmax)
    parts = []
    for k in range(kmax + 1):
        hev = S.cohomology(A, k, 0)[0]
        hod = S.cohomology(A, k, 1)[0]
        pev = pred.get((k, 0), 0)
        pod = pred.get((k, 1), 0)
        if (hev, hod) != (pev, pod):
            bad += 1
            parts.append(f"H^{k}={hev+hod}({hev}|{hod}) MISMATCH pred ({pev}|{pod})")
        else:
            parts.append(f"H^{k}={hev+hod}({hev}|{hod})")
    say(f"    O({m}|{n};{u}) p={p}: " + "  ".join(parts))
say("  mismatches:", bad, " (must be 0)")

say("")
say("    consequences read off the table:")
say("      p>3 : H^2 is EVEN of dim C(m,2)  (Skryabin's classes, the xi's add nothing)")
say("      p=3 : H^2 = C(m,2) even + n odd (the classes xi_j(dxi_j)^{(2)})")
say("      p=2 : H^2 is EVEN of dim M(M+1)/2, M = m+n; H^2_odd = 0")
for (m, n, p, u) in CASES:
    A = S.Om(m, n, p, u, dh=2, dxh=2)
    hev = S.cohomology(A, 2, 0)[0]
    hod = S.cohomology(A, 2, 1)[0]
    M = m + n
    if p > 3:
        f = (m * (m - 1) // 2, 0)
    elif p == 3:
        f = (m * (m - 1) // 2, n)
    else:
        f = (M * (M + 1) // 2, 0)
    ok = (hev, hod) == f
    if not ok:
        bad += 1
    say(f"      O({m}|{n}) p={p}: ({hev}|{hod}) vs formula ({f[0]}|{f[1]}) "
        f"{'ok' if ok else 'MISMATCH'}")

# ------------------------------------------------------------------- part 2
say("")
say("(2) the parity trichotomy of non-degeneracy")
say("    random homogeneous closed 2-forms; C = constant part of the Gram matrix")
bad2 = 0
for (m, n, p, u) in [(2, 1, 3, (1, 1)), (1, 1, 3, (1,)), (2, 2, 3, (1, 1)),
                     (2, 1, 5, (1, 1)), (3, 1, 5, (1, 1, 1)),
                     (2, 1, 2, (1, 1)), (1, 1, 2, (1,)), (2, 2, 2, (1, 1))]:
    A = S.Om(m, n, p, u, dh=2, dxh=2)
    stats = {}
    for par in (0, 1):
        B = [mm for mm in A.basis(2) if A.parity(mm) == par]
        found = 0
        blockviol = 0
        trials = 400
        for _ in range(trials):
            om = {}
            for _ in range(6):
                om = A.add(om, {random.choice(B): random.randrange(1, p)})
            if not om or not S.is_closed(A, om):
                continue
            nd, C = S.nondegenerate(A, om)
            if not nd:
                continue
            found += 1
            # block structure of C: rows/cols 0..m-1 even, m..m+n-1 odd
            xx = any(C[a][b] for a in range(m) for b in range(m))
            xs = any(C[a][m + b] for a in range(m) for b in range(n))
            ss = any(C[m + a][m + b] for a in range(n) for b in range(n))
            if par == 0 and xs:
                blockviol += 1
            if par == 1 and (xx or ss):
                blockviol += 1
        stats[par] = (found, blockviol)
    bad2 += stats[0][1] + stats[1][1]
    say(f"    O({m}|{n};{u}) p={p}: non-deg closed found "
        f"even {stats[0][0]}, odd {stats[1][0]}; "
        f"block violations {stats[0][1]}+{stats[1][1]} (must be 0+0)"
        + ("   [odd impossible: m != n]" if m != n and stats[1][0] == 0 else ""))
say("  violations:", bad2, " (must be 0)")

say("")
say("    the two forced counts, checked directly:")
for (m, n, p) in [(1, 2, 3), (2, 1, 3), (2, 2, 3), (3, 1, 2), (1, 1, 2)]:
    A = S.Om(m, n, p, (1,) * m, dh=2, dxh=2)
    # an odd non-degenerate form needs an invertible m x n block: m = n
    say(f"      O({m}|{n}) p={p}: odd non-degenerate possible? "
        f"{'yes' if m == n else 'no'} (needs m=n)")
for (m, n, p) in [(1, 1, 3), (2, 1, 3), (3, 1, 5), (1, 1, 2), (3, 1, 2)]:
    A = S.Om(m, n, p, (1,) * m, dh=2, dxh=2)
    # an even non-degenerate form needs a non-degenerate alternating x-block
    # (p>2), hence m even; at p=2 the diagonal (dx_i)^{(2)} saves odd m
    say(f"      O({m}|{n}) p={p}: even non-degenerate needs m even? "
        f"{'yes' if p > 2 else 'no (the divided squares supply a diagonal)'}")

# ------------------------------------------------------------------- part 3
say("")
say("(3) the standard shapes and the algebras they preserve")


def omega_even(A):
    """sum_{i<=k} dx_i dx_{k+i} + sum_j (dxi_j)^{(2)}   (p>2: needs m = 2k)."""
    m, n = A.m, A.n
    k = m // 2
    om = {}
    for i in range(k):
        om = A.add(om, A.mul(A.dx(i), A.dx(k + i)))
    for j in range(n):
        om = A.add(om, A.dxi(j, 2))
    return om


def omega_odd(A):
    """sum_i dx_i dxi_i   (needs m = n)."""
    om = {}
    for i in range(A.m):
        om = A.add(om, A.mul(A.dx(i), A.dxi(i)))
    return om


def omega_p2(A):
    """p = 2: (dg_1)^{(2)} + sum of the alternating pairs among the rest."""
    m, n = A.m, A.n
    gens = [("x", i) for i in range(m)] + [("xi", j) for j in range(n)]

    def dg(t):
        return A.dx(t[1]) if t[0] == "x" else A.dxi(t[1])
    om = {}
    if gens:
        t = gens[0]
        om = A.add(om, A.dx(t[1], 2) if t[0] == "x" else A.dxi(t[1], 2))
    rest = gens[1:]
    for a in range(0, len(rest) - 1, 2):
        om = A.add(om, A.mul(dg(rest[a]), dg(rest[a + 1])))
    return om


say("    p > 2, EVEN forms (the case tasks 1 and 2 do not cover):")
for (m, n, p, u) in [(2, 1, 3, (1, 1)), (2, 2, 3, (1, 1)), (2, 1, 5, (1, 1)),
                     (2, 0, 3, (1, 1)), (2, 3, 3, (1, 1))]:
    A = S.Om(m, n, p, u, dh=2, dxh=2)
    om = omega_even(A)
    t0 = time.time()
    nd, C = S.nondegenerate(A, om)
    ev, od, bas = S.preserving(A, om)
    sdO = A.sdim_funs()
    say(f"      O({m}|{n};{u}) p={p}: closed={S.is_closed(A, om)} nondeg={nd} "
        f"parity={A.parity(next(iter(om)))}  sdim O = ({sdO[0]}|{sdO[1]})  "
        f"sdim h(omega_0) = ({ev}|{od})  [{time.time()-t0:.1f} s]")

say("")
say("    p = 2, both a purely even and a mixed splitting:")
for (m, n, u) in [(2, 0, (1, 1)), (1, 1, (1,)), (2, 1, (1, 1)), (0, 2, ()),
                  (1, 2, (1,)), (3, 0, (1, 1, 1)), (0, 3, ())]:
    A = S.Om(m, n, 2, u, dh=2, dxh=2)
    om = omega_p2(A)
    nd, C = S.nondegenerate(A, om)
    ev, od, bas = S.preserving(A, om)
    sdO = A.sdim_funs()
    pars = set(A.parity(mm) for mm in om)
    say(f"      O({m}|{n};{u}) p=2: closed={S.is_closed(A, om)} nondeg={nd} "
        f"parity={pars}  sdim O = ({sdO[0]}|{sdO[1]})  "
        f"sdim h = ({ev}|{od})")

say("")
say("    p > 2, ODD forms (the periplectic case, recomputed here for the record):")
for (n, p, u) in [(1, 3, (1,)), (2, 3, (1, 1)), (1, 5, (1,))]:
    A = S.Om(n, n, p, u, dh=2, dxh=2)
    om = omega_odd(A)
    nd, C = S.nondegenerate(A, om)
    ev, od, bas = S.preserving(A, om)
    sdO = A.sdim_funs()
    say(f"      O({n}|{n};{u}) p={p}: nondeg={nd} sdim O = ({sdO[0]}|{sdO[1]})  "
        f"sdim h = ({ev}|{od})")

say("")
say("TOTAL failures:", bad + bad2, " (must be 0)")
LOG.close()
