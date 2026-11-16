#!/usr/bin/env python3
"""What the characteristic *as drawn* in each row of Table 1.5 of "Seminar on
Supersymmetries", vol. 2, Ch. 1, gives.

run_chars.py answers the weaker question -- is the printed number sdim g_0 of
SOME characteristic of that superalgebra.  This driver asks the stronger
question of every row: take
the diagram exactly as printed (node types and marks read off the page), and
compute sdim g_0 of that characteristic alone.

The shape of a diagram is written as a string:
    e = white (even) node,  o = crossed (isotropic odd) node,
    b = black (odd non-isotropic) node;  UPPER CASE = carrying the mark 1.
A row whose drawn string has a length other than the rank cannot be realized at
all, and that is reported as such.

Exact arithmetic; the sweep is chars.py's, over every simple system.
"""
import sys

from run_chars import sl_sweep, osp_sweep

LOG = None


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    if LOG is not None:
        LOG.write(s + "\n")
        LOG.flush()


def where(table, shape):
    """Every sdim under which this shape string occurs."""
    return sorted(k for k, v in table.items() if shape in v)


def check(label, table, printed, shape, rank):
    hits = where(table, shape)
    if len(shape) != rank:
        say("  %-22s printed %-14s drawn %-22s NOT REALIZABLE "
            "(%d nodes, rank %d)" % (label, "%d|%d" % printed, shape,
                                     len(shape), rank))
        return "unrealizable"
    if not hits:
        say("  %-22s printed %-14s drawn %-22s no simple system has this shape"
            % (label, "%d|%d" % printed, shape))
        return "noshape"
    tag = ("MATCHES" if hits == [printed] else
           ("matches (and %d other value(s))" % (len(hits) - 1)
            if printed in hits else "drawn gives " +
            ", ".join("%d|%d" % h for h in hits)))
    say("  %-22s printed %-14s drawn %-22s %s" %
        (label, "%d|%d" % printed, shape, tag))
    return "ok" if printed in hits else "wrong"


if __name__ == "__main__":
    verdict = {}

    say("=" * 78)
    say("ROW 1)  sl(m|n), m>n, n even     drawn: n crossed marked, m-n-1 white, "
        "n crossed marked")
    for m in range(3, 7):
        for n in range(2, m, 2):
            t = sl_sweep(m, n)
            verdict[(1, m, n)] = check(
                "sl(%d|%d)" % (m, n), t, ((m - n) ** 2 + 2 * n - 1, 0),
                "O" * n + "e" * (m - n - 1) + "O" * n, m + n - 1)

    say("=" * 78)
    say("ROW 2)  sl(m|n), m>n, n odd")
    for m in range(3, 7):
        for n in range(1, m, 2):
            t = sl_sweep(m, n)
            verdict[(2, m, n)] = check(
                "sl(%d|%d)" % (m, n), t,
                ((m - n - 1) ** 2 + 2 * n, 2 * (m - n - 1)),
                "O" * n + "e" * (m - n - 1) + "o" + "O" * n, m + n - 1)
            check("  ... one white less", t,
                  ((m - n - 1) ** 2 + 2 * n, 2 * (m - n - 1)),
                  "O" * n + "e" * (m - n - 2) + "o" + "O" * n, m + n - 1)

    say("=" * 78)
    say("ROWS 3) and 4)  psl(n|n)")
    for n in range(2, 6):
        t = sl_sweep(n, n, rank=2 * n - 2)
        verdict[(3, n)] = check("psl(%d|%d)" % (n, n), t, (2 * n, 0),
                                "O" * (n - 1) + "e" + "O" * (n - 1), 2 * n - 1)
        verdict[(4, n)] = check("psl(%d|%d)" % (n, n), t, (2 * n - 2, 2),
                                "O" * (n - 2) + "e" + "o" + "O" * (n - 1),
                                2 * n - 1)
        check("  ... white marked", t, (2 * n - 2, 2),
              "O" * (n - 2) + "E" + "o" + "O" * (n - 1), 2 * n - 1)

    say("=" * 78)
    say("ROW 5)  osp(2|2n), n>2    drawn: white, crossed marked, crossed, "
        "n-3 white, white")
    for n in range(3, 7):
        t = osp_sweep(2, 2 * n)
        verdict[(5, n)] = check("osp(2|%d)" % (2 * n), t,
                                (2 * n * n - 7 * n + 1, 4 * n - 8),
                                "eOo" + "e" * (n - 2), n + 1)

    say("=" * 78)
    say("ROW 6)  osp(2m|2n), 2<=m<n-1")
    for m in range(2, 4):
        for n in range(m + 2, m + 4):
            t = osp_sweep(2 * m, 2 * n)
            sh = "O" * (2 * m - 2) + "e" + "O" + "o" + "e" * (n - m - 1)
            verdict[(6, m, n)] = check(
                "osp(%d|%d)" % (2 * m, 2 * n), t,
                (2 * (n - m) ** 2 + n + m, 0), sh, m + n)
            say("        the printed number is the TOTAL of the drawn %s "
                "(odd part 4(n-m-1) = %d)"
                % (", ".join("%d|%d" % k for k, v in t.items() if sh in v),
                   4 * (n - m - 1)))

    say("=" * 78)
    say("ROW 7)  osp(2m|2m+2)")
    for m in range(2, 5):
        t = osp_sweep(2 * m, 2 * m + 2)
        verdict[(7, m)] = check("osp(%d|%d)" % (2 * m, 2 * m + 2), t,
                                (2 * m + 3, 0),
                                "O" * (2 * m - 2) + "e" + "OO", 2 * m + 1)

    say("=" * 78)
    say("ROW 8)  the row labeled osp(2n|2m), read as osp(2m|2m)")
    for m in range(2, 6):
        t = osp_sweep(2 * m, 2 * m)
        verdict[(8, m)] = check("osp(%d|%d)" % (2 * m, 2 * m), t, (2 * m, 0),
                                "O" * (2 * m), 2 * m)

    say("=" * 78)
    say("ROW 9)  osp(2n+2|2n)")
    for n in range(1, 5):
        t = osp_sweep(2 * n + 2, 2 * n)
        verdict[(9, n)] = check("osp(%d|%d)" % (2 * n + 2, 2 * n), t,
                                (2 * n + 1, 0), "O" * (2 * n + 1), 2 * n + 1)

    say("=" * 78)
    say("ROW 10)  osp(2m|2n), m>n+1")
    for n in range(1, 4):
        for m in range(n + 2, n + 5):
            t = osp_sweep(2 * m, 2 * n)
            verdict[(10, m, n)] = check(
                "osp(%d|%d)" % (2 * m, 2 * n), t,
                (2 * (m - n) ** 2 + 3 * n - m, 0),
                "O" * (2 * n) + "e" * (m - n - 2) + "ee", m + n)

    say("=" * 78)
    say("ROW 11)  osp(1|2), drawn: one black node, unmarked")
    t = osp_sweep(1, 2)
    verdict[(11,)] = check("osp(1|2)", t, (1, 0), "b", 1)
    check("  ... node marked", t, (1, 0), "B", 1)

    say("=" * 78)
    say("ROW 12)  osp(1|2n), n>1")
    for n in range(2, 6):
        t = osp_sweep(1, 2 * n)
        verdict[(12, n)] = check("osp(1|%d)" % (2 * n), t,
                                 (2 * n * n - n, 2 * (n - 1)),
                                 "E" + "e" * (n - 2) + "b", n)
        say("        the printed even entry %d is the TOTAL of the drawn %s"
            % (2 * n * n - n,
               ", ".join("%d|%d" % k for k, v in t.items()
                         if ("E" + "e" * (n - 2) + "b") in v)))

    say("=" * 78)
    say("ROW 13)  osp(2m+1|2n), 1<m<n-1")
    for m in range(2, 4):
        for n in range(m + 2, m + 4):
            t = osp_sweep(2 * m + 1, 2 * n)
            verdict[(13, m, n)] = check(
                "osp(%d|%d)" % (2 * m + 1, 2 * n), t,
                (2 * (n - m) ** 2 + 3 * m - n, 0),
                "O" * (2 * m + 1) + "e" * (n - m - 2) + "b", m + n)
            shapes = set(s for v in t.values() for s in v)
            say("        the drawn diagram claims %d crossed nodes; the most "
                "any simple system has is %d"
                % (2 * m + 1,
                   max(sum(1 for c in s if c in "oO") for s in shapes)))

    say("=" * 78)
    say("ROW 14)  osp(2m+1|2m+2)")
    for m in range(1, 5):
        t = osp_sweep(2 * m + 1, 2 * m + 2)
        verdict[(14, m)] = check("osp(%d|%d)" % (2 * m + 1, 2 * m + 2), t,
                                 (2 * m + 1, 0), "O" * (2 * m) + "B", 2 * m + 1)

    say("=" * 78)
    say("ROW 15)  osp(2m+1|2m)")
    for m in range(1, 5):
        t = osp_sweep(2 * m + 1, 2 * m)
        verdict[(15, m)] = check("osp(%d|%d)" % (2 * m + 1, 2 * m), t,
                                 (2 * m, 0), "O" * (2 * m - 1) + "B", 2 * m)

    say("=" * 78)
    say("ROW 16)  osp(2m+1|2n), m>=n+1")
    for n in range(1, 4):
        for m in range(n + 1, n + 4):
            t = osp_sweep(2 * m + 1, 2 * n)
            verdict[(16, m, n)] = check(
                "osp(%d|%d)" % (2 * m + 1, 2 * n), t,
                (2 * (m - n) ** 2 + m + n, 0),
                "O" * (2 * n) + "e" * (m - n - 1) + "e", m + n)

    say("=" * 78)
    say("SUMMARY by row")
    rows = {}
    for k, v in verdict.items():
        rows.setdefault(k[0], set()).add(v)
    for r in sorted(rows):
        say("  row %2d): %s" % (r, ", ".join(sorted(rows[r]))))
