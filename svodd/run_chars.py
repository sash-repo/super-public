#!/usr/bin/env python3
"""Driver: is the number printed in each row of Table 1.5 the superdimension of
the centralizer of *some* characteristic of that superalgebra, and of the one
drawn?

For every simple system (= every shuffle of the eps's and the delta's) and every
set of marks we compute sdim g_0 exactly, and record

  (a) whether the printed pair occurs at all,
  (b) the diagrams that produce it, in the notation
        e = white (even) node,  o = crossed (isotropic odd) node,
        b = black (odd non-isotropic) node;  UPPER CASE = marked with 1.

Run:  python3 run_chars.py
"""
import itertools
import sys

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from chars import roots_sl, simple_sl, roots_osp, simple_osp, expand

LOG = None


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    if LOG is not None:
        LOG.write(s + "\n")
        LOG.flush()


def node_types_sl(seq, par):
    return ["o" if par[seq[i]] ^ par[seq[i + 1]] else "e" for i in range(len(seq) - 1)]


def node_types_osp(seq, M, N, P):
    par = [0] * M + [1] * N
    t = ["o" if par[seq[i]] ^ par[seq[i + 1]] else "e" for i in range(len(seq) - 1)]
    last = seq[-1]
    if P % 2:
        t.append("e" if last < M else "b")      # eps  -> short even ;  delta -> black
    else:
        if last >= M:
            t.append("e")                       # 2delta, long even
        else:
            t.append("o" if par[seq[-2]] ^ par[last] else "e")   # the fork
    return t


def sweep(roots, dim, simple_list, rank, types_list):
    """Return {sdim : set of shape strings}."""
    out = {}
    for simple, types in zip(simple_list, types_list):
        exp = expand(roots, simple, dim)
        if exp is None:
            continue
        r = len(simple)
        for bits in range(1 << r):
            marks = [(bits >> k) & 1 for k in range(r)]
            ev = od = 0
            for (c, p) in exp:
                if any(marks[k] and c[k] for k in range(r)):
                    continue
                if p:
                    od += 1
                else:
                    ev += 1
            key = (rank + ev, od)
            sh = "".join(t.upper() if marks[k] else t for k, t in enumerate(types))
            out.setdefault(key, set()).add(sh)
    return out


def sl_sweep(m, n, rank=None):
    roots, dim, par = roots_sl(m, n)
    rank = m + n - 1 if rank is None else rank
    simples, typs = [], []
    for bits in sorted(set(itertools.permutations([0] * m + [1] * n))):
        e_it, d_it = iter(range(m)), iter(range(m, m + n))
        seq = [next(d_it) if b else next(e_it) for b in bits]
        simples.append(simple_sl(seq))
        typs.append(node_types_sl(seq, par))
    return sweep(roots, dim, simples, rank, typs)


def osp_sweep(P, twoN):
    M, N = P // 2, twoN // 2
    roots, dim, par = roots_osp(M, twoN, P)
    simples, typs = [], []
    pats = sorted(set(itertools.permutations([0] * M + [1] * N))) or [()]
    for bits in pats:
        e_it, d_it = iter(range(M)), iter(range(M, M + N))
        seq = [next(d_it) if b else next(e_it) for b in bits]
        simples.append(simple_osp(seq, M, N, P))
        typs.append(node_types_osp(seq, M, N, P))
    return sweep(roots, dim, simples, M + N, typs)


def report(label, table, printed, drawn=None):
    hit = table.get(printed)
    tag = "REALIZED" if hit else "NOT REALIZED"
    say("  %-24s printed %-12s  %s%s" %
        (label, "%d|%d" % printed, tag,
         "  (%d diagram%s)" % (len(hit), "" if len(hit) == 1 else "s") if hit else ""))
    if hit:
        for s in sorted(hit)[:4]:
            say("        %s" % s)
    if drawn is not None:
        ok = hit is not None and drawn in hit
        say("        drawn: %-28s %s" % (drawn, "MATCHES" if ok else "does NOT match"))
        if not ok:
            for k, v in table.items():
                if drawn in v:
                    say("        drawn diagram in fact gives %d|%d" % k)
    return hit


if __name__ == "__main__":
    say("=" * 78)
    say("ROWS 1-2   sl(m|n), m>n")
    for m in range(3, 8):
        for n in range(1, m):
            t = sl_sweep(m, n)
            if n % 2 == 0:
                report("sl(%d|%d) row 1" % (m, n), t, ((m - n) ** 2 + 2 * n - 1, 0),
                       drawn="O" * n + "e" * (m - n - 1) + "O" * n)
            else:
                report("sl(%d|%d) row 2" % (m, n), t,
                       ((m - n - 1) ** 2 + 2 * n, 2 * (m - n - 1)),
                       drawn="O" * n + "e" * (m - n - 1) + "o" + "O" * n)
                say("        as corrected (m-n-2 white nodes): %s" %
                    ("MATCHES" if ("O" * n + "e" * (m - n - 2) + "o" + "O" * n)
                     in t.get(((m - n - 1) ** 2 + 2 * n, 2 * (m - n - 1)), set())
                     else "no"))

    say("=" * 78)
    say("ROWS 3-4   psl(n|n)")
    for n in range(2, 6):
        t = sl_sweep(n, n, rank=2 * n - 2)
        report("psl(%d|%d) row 3" % (n, n), t, (2 * n, 0),
               drawn="O" * (n - 1) + "e" + "O" * (n - 1))
        report("psl(%d|%d) row 4" % (n, n), t, (2 * n - 2, 2),
               drawn="O" * (n - 2) + "e" + "o" + "O" * (n - 1))

    say("=" * 78)
    say("ROW 5   osp(2|2n), n>2      printed 2n^2-7n+1 | 4n-8")
    for n in range(3, 7):
        report("osp(2|%d)" % (2 * n), osp_sweep(2, 2 * n), (2 * n * n - 7 * n + 1, 4 * n - 8))

    say("=" * 78)
    say("ROW 6   osp(2m|2n), 2<=m<n-1   printed 2(n-m)^2+n+m")
    for m in range(2, 5):
        for n in range(m + 2, m + 5):
            report("osp(%d|%d)" % (2 * m, 2 * n), osp_sweep(2 * m, 2 * n),
                   (2 * (n - m) ** 2 + n + m, 0))

    say("=" * 78)
    say("ROW 7   osp(2m|2m+2)          printed 2m+3")
    for m in range(1, 5):
        report("osp(%d|%d)" % (2 * m, 2 * m + 2), osp_sweep(2 * m, 2 * m + 2), (2 * m + 3, 0))

    say("=" * 78)
    say("ROW 8   `osp(2n|2m)'          printed 2n     [tested as osp(2m|2m)]")
    for m in range(2, 6):
        report("osp(%d|%d)" % (2 * m, 2 * m), osp_sweep(2 * m, 2 * m), (2 * m, 0))

    say("=" * 78)
    say("ROW 9   osp(2n+2|2n)          printed 2n+1")
    for n in range(1, 5):
        report("osp(%d|%d)" % (2 * n + 2, 2 * n), osp_sweep(2 * n + 2, 2 * n), (2 * n + 1, 0))

    say("=" * 78)
    say("ROW 10  osp(2m|2n), m>n+1     printed 2(m-n)^2+3n-m")
    for n in range(1, 4):
        for m in range(n + 2, n + 5):
            report("osp(%d|%d)" % (2 * m, 2 * n), osp_sweep(2 * m, 2 * n),
                   (2 * (m - n) ** 2 + 3 * n - m, 0))

    say("=" * 78)
    say("ROWS 11-12  osp(1|2n)         printed 1 ; 2n^2-n | 2(n-1)")
    report("osp(1|2)", osp_sweep(1, 2), (1, 0))
    for n in range(2, 6):
        report("osp(1|%d)" % (2 * n), osp_sweep(1, 2 * n), (2 * n * n - n, 2 * (n - 1)))

    say("=" * 78)
    say("ROW 13  osp(2m+1|2n), 1<m<n-1  printed 2(n-m)^2+3m-n")
    for m in range(2, 4):
        for n in range(m + 2, m + 5):
            report("osp(%d|%d)" % (2 * m + 1, 2 * n), osp_sweep(2 * m + 1, 2 * n),
                   (2 * (n - m) ** 2 + 3 * m - n, 0))

    say("=" * 78)
    say("ROW 14  osp(2m+1|2m+2)        printed 2m+1")
    for m in range(1, 5):
        report("osp(%d|%d)" % (2 * m + 1, 2 * m + 2), osp_sweep(2 * m + 1, 2 * m + 2),
               (2 * m + 1, 0))

    say("=" * 78)
    say("ROW 15  osp(2m+1|2m)          printed 2m")
    for m in range(1, 5):
        report("osp(%d|%d)" % (2 * m + 1, 2 * m), osp_sweep(2 * m + 1, 2 * m), (2 * m, 0))

    say("=" * 78)
    say("ROW 16  osp(2m+1|2n), m>=n+1  printed 2(m-n)^2+m+n")
    for n in range(1, 4):
        for m in range(n + 1, n + 4):
            report("osp(%d|%d)" % (2 * m + 1, 2 * n), osp_sweep(2 * m + 1, 2 * n),
                   (2 * (m - n) ** 2 + m + n, 0))
