#!/usr/bin/env python3
"""Every row of the corrected table of regular odd elements, verified.

The characteristic *as printed* in Table 1.5 of Chapter 1 of
"Seminar on Supersymmetries", vol. 2, is right in ten of the nineteen rows.  This
driver checks the table that replaces it: for each row it takes the corrected
diagram and the corrected entry and asks

    (a) is the diagram realized by a simple system of that superalgebra at all,
    (b) is the tabulated pair sdim g_0 of that diagram, and of nothing else.

Shape strings:
    e = white (even) node,  o = crossed (isotropic odd) node,
    b = black (odd non-isotropic) node;  UPPER CASE = carrying the mark 1.

Two rows have no printed diagram to correct, the printed one being unrealizable:
  row 2), sl(m|n) with n odd -- the picture claims 2n+1 crossed nodes and m+n
    of them, against 2n and m+n-1; the entry is right, and the characteristic
    that has it is the one whose unmarked part is the diagram of sl(1|m-n-1);
  row 13), osp(2m+1|2n) with m<n-1 -- the picture claims 2m+1 crossed nodes
    against 2m; the row is the third member of the family of rows 10) and 16)
    (all the crossed nodes marked, the unmarked tail the smaller factor), and
    that characteristic is the one tabulated here.  Its value is NOT the printed
    number, which belongs to no characteristic of that shape.

Exact arithmetic throughout; the sweep is chars.py's, over every simple system.

Run:  python3 corrected.py
"""
import sys

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from run_chars import sl_sweep, osp_sweep

LOG = None


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    if LOG is not None:
        LOG.write(s + "\n")
        LOG.flush()


def check(label, table, entry, shape, rank):
    """Is `entry` sdim g_0 of `shape`, and is `shape` realizable at all?"""
    if len(shape) != rank:
        say("  %-22s table %-16s %-24s NODE COUNT %d != rank %d"
            % (label, "%d|%d" % entry, shape, len(shape), rank))
        return "badcount"
    vals = sorted(k for k, v in table.items() if shape in v)
    if not vals:
        say("  %-22s table %-16s %-24s NOT REALIZABLE"
            % (label, "%d|%d" % entry, shape))
        return "noshape"
    if vals == [entry]:
        say("  %-22s table %-16s %-24s OK" % (label, "%d|%d" % entry, shape))
        return "ok"
    say("  %-22s table %-16s %-24s MISMATCH: the diagram gives %s"
        % (label, "%d|%d" % entry, shape,
           ", ".join("%d|%d" % v for v in vals)))
    return "wrong"


if __name__ == "__main__":
    verdict = {}

    say("=" * 78)
    say("ROW 1)  sl(m|n), m>n, n even:  O^n e^{m-n-1} O^n")
    for m in range(3, 8):
        for n in range(2, min(m, 11 - m), 2):
            verdict[(1, m, n)] = check(
                "sl(%d|%d)" % (m, n), sl_sweep(m, n),
                ((m - n) ** 2 + 2 * n - 1, 0),
                "O" * n + "e" * (m - n - 1) + "O" * n, m + n - 1)

    say("=" * 78)
    say("ROW 2)  sl(m|n), m>n, n odd:  O^n o e^{m-n-2} O^{n-1} E")
    for m in range(3, 8):
        for n in range(1, min(m, 11 - m), 2):
            # the unmarked block is the diagram of sl(1|m-n-1): it is empty
            # when m = n+1, and then every node is marked.
            shape = ("O" * n + ("o" + "e" * (m - n - 2) if m > n + 1 else "")
                     + "O" * (n - 1) + "E")
            verdict[(2, m, n)] = check(
                "sl(%d|%d)" % (m, n), sl_sweep(m, n),
                ((m - n - 1) ** 2 + 2 * n, 2 * (m - n - 1)), shape, m + n - 1)

    say("=" * 78)
    say("ROW 3)  psl(n|n), n even:  O^{n-1} e O^{n-1}")
    for n in range(2, 6, 2):
        verdict[(3, n)] = check(
            "psl(%d|%d)" % (n, n), sl_sweep(n, n, rank=2 * n - 2), (2 * n, 0),
            "O" * (n - 1) + "e" + "O" * (n - 1), 2 * n - 1)

    say("=" * 78)
    say("ROW 4)  psl(n|n), n odd:  O^{n-2} E o O^{n-1}")
    for n in range(3, 6, 2):
        verdict[(4, n)] = check(
            "psl(%d|%d)" % (n, n), sl_sweep(n, n, rank=2 * n - 2),
            (2 * n - 2, 2), "O" * (n - 2) + "E" + "o" + "O" * (n - 1),
            2 * n - 1)

    say("=" * 78)
    say("ROW 5)  osp(2|2n), n>2:  e O o e^{n-2}")
    for n in range(3, 8):
        verdict[(5, n)] = check(
            "osp(2|%d)" % (2 * n), osp_sweep(2, 2 * n),
            (2 * n * n - 7 * n + 11, 4 * n - 8), "eOo" + "e" * (n - 2), n + 1)

    say("=" * 78)
    say("ROW 6)  osp(2m|2n), 2<=m<n-1:  O^{2m-2} e O o e^{n-m-1}")
    for m in range(2, 4):
        for n in range(m + 2, m + 4):
            verdict[(6, m, n)] = check(
                "osp(%d|%d)" % (2 * m, 2 * n), osp_sweep(2 * m, 2 * n),
                (2 * (n - m) ** 2 - 4 * (n - m) + 4 + n + m, 4 * (n - m - 1)),
                "O" * (2 * m - 2) + "e" + "O" + "o" + "e" * (n - m - 1), m + n)

    say("=" * 78)
    say("ROW 7)  osp(2m|2m+2):  O^{2m-2} e O O")
    for m in range(2, 5):
        verdict[(7, m)] = check(
            "osp(%d|%d)" % (2 * m, 2 * m + 2), osp_sweep(2 * m, 2 * m + 2),
            (2 * m + 3, 0), "O" * (2 * m - 2) + "e" + "OO", 2 * m + 1)

    say("=" * 78)
    say("ROW 8)  osp(2m|2m):  O^{2m}")
    for m in range(2, 6):
        verdict[(8, m)] = check(
            "osp(%d|%d)" % (2 * m, 2 * m), osp_sweep(2 * m, 2 * m),
            (2 * m, 0), "O" * (2 * m), 2 * m)

    say("=" * 78)
    say("ROW 9)  osp(2n+2|2n):  O^{2n+1}")
    for n in range(1, 5):
        verdict[(9, n)] = check(
            "osp(%d|%d)" % (2 * n + 2, 2 * n), osp_sweep(2 * n + 2, 2 * n),
            (2 * n + 1, 0), "O" * (2 * n + 1), 2 * n + 1)

    say("=" * 78)
    say("ROW 10)  osp(2m|2n), m>n+1:  O^{2n} e^{m-n}")
    for n in range(1, 4):
        for m in range(n + 2, n + 5):
            verdict[(10, m, n)] = check(
                "osp(%d|%d)" % (2 * m, 2 * n), osp_sweep(2 * m, 2 * n),
                (2 * (m - n) ** 2 + 3 * n - m, 0),
                "O" * (2 * n) + "e" * (m - n), m + n)

    say("=" * 78)
    say("ROW 11)  osp(1|2):  B")
    verdict[(11,)] = check("osp(1|2)", osp_sweep(1, 2), (1, 0), "B", 1)

    say("=" * 78)
    say("ROW 12)  osp(1|2n), n>1:  E e^{n-2} b")
    for n in range(2, 7):
        verdict[(12, n)] = check(
            "osp(1|%d)" % (2 * n), osp_sweep(1, 2 * n),
            (2 * n * n - 3 * n + 2, 2 * n - 2), "E" + "e" * (n - 2) + "b", n)

    say("=" * 78)
    say("ROW 13)  osp(2m+1|2n), 1<m<n-1:  O^{2m} e^{n-m-1} b")
    for m in range(2, 4):
        for n in range(m + 2, m + 4):
            verdict[(13, m, n)] = check(
                "osp(%d|%d)" % (2 * m + 1, 2 * n), osp_sweep(2 * m + 1, 2 * n),
                (2 * (n - m) ** 2 + m + n, 2 * (n - m)),
                "O" * (2 * m) + "e" * (n - m - 1) + "b", m + n)

    say("=" * 78)
    say("ROW 14)  osp(2m+1|2m+2):  O^{2m} B")
    for m in range(1, 5):
        verdict[(14, m)] = check(
            "osp(%d|%d)" % (2 * m + 1, 2 * m + 2),
            osp_sweep(2 * m + 1, 2 * m + 2), (2 * m + 1, 0),
            "O" * (2 * m) + "B", 2 * m + 1)

    say("=" * 78)
    say("ROW 15)  osp(2m+1|2m):  O^{2m-1} B")
    for m in range(1, 5):
        verdict[(15, m)] = check(
            "osp(%d|%d)" % (2 * m + 1, 2 * m), osp_sweep(2 * m + 1, 2 * m),
            (2 * m, 0), "O" * (2 * m - 1) + "B", 2 * m)

    say("=" * 78)
    say("ROW 16)  osp(2m+1|2n), m>=n+1:  O^{2n} e^{m-n}")
    for n in range(1, 4):
        for m in range(n + 1, n + 4):
            verdict[(16, m, n)] = check(
                "osp(%d|%d)" % (2 * m + 1, 2 * n), osp_sweep(2 * m + 1, 2 * n),
                (2 * (m - n) ** 2 + m + n, 0),
                "O" * (2 * n) + "e" * (m - n), m + n)

    say("=" * 78)
    say("SUMMARY by row  (rows 17)--19) are exc.py's)")
    rows = {}
    for k, v in verdict.items():
        rows.setdefault(k[0], set()).add(v)
    bad = 0
    for r in sorted(rows):
        say("  row %2d): %s" % (r, ", ".join(sorted(rows[r]))))
        if rows[r] != {"ok"}:
            bad += 1
    say("  %d row(s) not OK" % bad)
