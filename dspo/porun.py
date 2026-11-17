#!/usr/bin/env python3
"""
Driver on top of po.py.

  1. reproduces the tables of Subsecs. 3.1.3, 3.1.4 of the text (m=5,6)
     with the even|odd splits of Ker and Im;
  2. the same for m=7: table (24);
  3. checks Theorem 3.2 (graded duality inside g_x) with the reflection
     centre (m-4)/2, and the other reading [m/2]-2 that Remark 3.2.2 rules out;
  4. prints derived and lower central series of g_x and the perfect part
     when g_x is not solvable (Remarks 3.1.6, 3.2.4);
  5. scans non-monomial homologic x built out of commuting cliques of monomials
     (the table after Conjecture 3.2.3; Remark 3.1.2 for m=5).

A monomial x is determined, up to the automorphisms of po(0|m) permuting the
pairs (xi_j, eta_j) and swapping xi_j <-> eta_j, by its TYPE (a, b, t):
  a = number of complete pairs xi_j eta_j dividing x,
  b = number of unpaired xi/eta factors,
  t = 1 if theta | x.
Only one representative per type is computed.

Usage: python3 porun.py [m ...]
"""
import sys
from fractions import Fraction
from itertools import combinations
from po import (Po, Quot, rank, homologic, monomial_reps, independent_subset,
                null_space)


def mono_type(alg, msk):
    a = b = 0
    for j in range(alg.k):
        has_xi = bool(msk & (1 << j))
        has_eta = bool(msk & (1 << (alg.k + j)))
        if has_xi and has_eta:
            a += 1
        elif has_xi or has_eta:
            b += 1
    t = 1 if (alg.odd_m and (msk & (1 << (2 * alg.k)))) else 0
    return (a, b, t)


def odd_monomials(alg):
    out = []
    for d in range(1, alg.m + 1, 2):
        out.extend(monomial_reps(alg, d))
    return out


def type_reps(alg):
    """one representative monomial per type, plus the multiplicity of the type."""
    reps, mult = {}, {}
    for msk in odd_monomials(alg):
        ty = mono_type(alg, msk)
        mult[ty] = mult.get(ty, 0) + 1
        reps.setdefault(ty, msk)
    return reps, mult


def sdim_split(alg, vecs):
    ev = od = 0
    for v in vecs:
        supp = [i for i, c in enumerate(v) if c != 0]
        ps = {alg.parity(i) for i in supp}
        if len(ps) != 1:
            return None
        if ps.pop() == 0:
            ev += 1
        else:
            od += 1
    return ev, od


def graded_dims(q):
    out = {}
    for p, d in zip(q.par, q.deg):
        e, o = out.get(d, (0, 0))
        out[d] = (e + (p == 0), o + (p == 1))
    return out


def perfect_part(q):
    """terminal member of the derived series, with its sdim."""
    seq, term = q.derived_series()
    ev = od = 0
    for v in term:
        # parity of a coordinate vector w.r.t. the (homogeneous) reps
        ps = {q.par[i] for i, c in enumerate(v) if c != 0}
        if len(ps) == 1:
            if ps.pop() == 0:
                ev += 1
            else:
                od += 1
        else:
            return seq, term, None
    return seq, term, (ev, od)


def report(m, struct=True):
    alg = Po(m)
    err = alg.check_identities()
    print(f"=== po(0|{m}):  dim = {alg.dim} = {alg.dim//2}|{alg.dim//2};  "
          f"bracket identities: {'verified' if err is None else err}")
    reps, mult = type_reps(alg)
    data = {}
    for ty, msk in sorted(reps.items()):
        x = {msk: 1}
        if not homologic(alg, x):
            print(f"    type {ty}: {alg.name(msk)} is NOT homologic "
                  f"({mult[ty]} monomials)")
            continue
        q = Quot(alg, x)
        data[ty] = (msk, q)
    print(f"{'deg':>4} {'type(a,b,t)':>12} {'rank':>6} {'dim Ker':>10} "
          f"{'dim Im':>9} {'sdim g_x':>10} {'#mono':>6}   representative")
    for ty in sorted(data, key=lambda t: (2*t[0]+t[1]+t[2], data[t][1].rank)):
        msk, q = data[ty]
        ker = null_space(q.M, alg.dim)
        ke = sdim_split(alg, ker)
        ie = sdim_split(alg, q.im)
        ev, od = q.sdim()
        d = 2 * ty[0] + ty[1] + ty[2]
        print(f"{d:>4} {str(ty):>12} {q.rank:>6} {str(ke[0])+'|'+str(ke[1]):>10} "
              f"{str(ie[0])+'|'+str(ie[1]):>9} {str(ev)+'|'+str(od):>10} "
              f"{mult[ty]:>6}   {alg.name(msk)}")
    if not struct:
        return alg, data

    print("\n  -- graded duality inside g_x (Theorem 3.2):")
    print(f"     (A) n_i <-> n_{{{m-4}-i}}  [centre (m-4)/2, the one the NIS forces]")
    print(f"     (B) n_i <-> n_{{{2*(m//2)-4}-i}}  [centre [m/2]-2 = {m//2-2}, cf. Remark 3.2.2]")
    okA = okB = True
    for ty in sorted(data):
        msk, q = data[ty]
        if q.dim == 0:
            continue
        g = graded_dims(q)
        pi = m % 2
        def dual_ok(cs):
            for i, (e, o) in g.items():
                e2, o2 = g.get(cs - i, (0, 0))
                if (e, o) != ((o2, e2) if pi else (e2, o2)):
                    return False
            return True
        a, b = dual_ok(m - 4), dual_ok(2 * (m // 2) - 4)
        okA &= a; okB &= b
        gd = " ".join(f"n_{i}={g[i][0]}|{g[i][1]}" for i in sorted(g))
        print(f"     x={alg.name(msk):<26} rank {q.rank:>3}  {gd}")
        print(f"        (A) {'holds' if a else 'FAILS'}   (B) {'holds' if b else 'FAILS'}")
    print(f"     ==> (A) {'holds for every homogeneous x above' if okA else 'FAILS'};"
          f"  (B) {'holds' if okB else 'FAILS'}")

    print("\n  -- solvability (Remarks 3.1.6, 3.2.4):")
    live = {ty: v for ty, v in data.items() if v[1].dim > 0}
    rmax = max(v[1].rank for v in live.values()) if live else None
    for ty in sorted(live, key=lambda t: live[t][1].rank):
        msk, q = live[ty]
        seq, term, sd = perfect_part(q)
        lcs = q.lower_central()
        solv = seq[-1] == 0
        tag = "   <== max rank with g_x != 0 (monomials)" if q.rank == rmax else ""
        extra = "" if solv else f"  perfect part sdim {sd[0]}|{sd[1]}" if sd else ""
        print(f"     x={alg.name(msk):<26} rank {q.rank:>3} "
              f"sdim g_x={q.sdim()[0]}|{q.sdim()[1]:<3} derived {seq} LCS {lcs} "
              f"{'solvable' if solv else 'NOT solvable'}{extra}{tag}")
    return alg, data


def clique_scan(m, tries=3000, seed=11, deep=True):
    import random
    rnd = random.Random(seed)
    alg = Po(m)
    mons = [x for x in odd_monomials(alg) if homologic(alg, {x: 1})]
    n = len(mons)
    comm = [[not alg.bracket({mons[i]: 1}, {mons[j]: 1}) for j in range(n)]
            for i in range(n)]
    seen = {}
    for _ in range(tries):
        order = list(range(n))
        rnd.shuffle(order)
        cl = []
        for i in order:
            if all(comm[i][j] for j in cl):
                cl.append(i)
            if len(cl) >= 7:
                break
        x = {}
        for i in cl:
            x[mons[i]] = rnd.choice([1, -1, 2, 3])
        if not x or not homologic(alg, x):
            continue
        r = rank(alg.ad_matrix(x))
        if r not in seen:
            seen[r] = (alg.dim - 2 * r, x)
    print(f"\n  -- non-monomial homologic x in po(0|{m}) (random elements of "
          f"commuting cliques of monomials):")
    live = [r for r, (dg, _) in seen.items() if dg > 0]
    for r in sorted(seen):
        dg, x = seen[r]
        nm = " + ".join(f"{c}*{alg.name(k)}" for k, c in x.items())
        mark = ""
        if dg > 0 and r == max(live):
            q = Quot(alg, x)
            seq, term, sd = perfect_part(q)
            mark = (f"   derived {seq} -> "
                    f"{'solvable' if seq[-1]==0 else 'NOT solvable'}"
                    f"  <== max rank with g_x != 0")
        print(f"     rank {r:>3}  dim g_x = {dg:>3}   x = {nm}{mark}")
    print(f"     max rank with g_x != 0 among these: {max(live) if live else None}")


if __name__ == "__main__":
    ms = [int(a) for a in sys.argv[1:]] or [5, 6]
    for m in ms:
        report(m)
        clique_scan(m)
        print()
