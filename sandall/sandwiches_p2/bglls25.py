#!/usr/bin/env python3
"""Table 25.5 of [BGLLS1] (SIGMA 16:089 (2020), arXiv:1510.07255): the 15
exceptional vectorial Lie superalgebras over K (p=2) and their desuperizations.

The table gives, for each g, the F(g_i) for i=-1,-2 as K-modules, and the name
of F(g) -- whose first argument is the number of indeterminates F(g) is
realized on, i.e. the codimension of F(g)_0.

This script recomputes, from the module descriptions alone,
  * dim F(g_-1), dim F(g_-2), dim F(g_-3);
  * the growth vector of the preserved distribution;
and checks it twice:
  (a) the last entry of the growth vector must equal the number of
      indeterminates in the name of F(g)  --  an independent datum of the table;
  (b) the eleven rows of depth >= 2 must reproduce the growth vectors already
      tabulated in table (grVexF) of the text (taken from [BGLLS2, Sec. 25.5]).

Run:  python3 bglls25.py
"""

# dim O(m;1) = 2^m for p = 2.
def O(m):
    return 2 ** m


# The 15 rows.  "m2"/"m1"/"m3" are dim F(g_{-2}) / F(g_{-1}) / F(g_{-3}),
# each given as an arithmetic expression mirroring the module in the table.
# "ind" is the number of indeterminates in the name of F(g).
# "old" is the name used in table (grVexF) of the text (None = row absent there).
ROWS = [
    # g                F(g_-2)                   F(g_-1)                    F(g_-3)  F(g)          ind old
    ("vle(4;N|3)",     ("---", 0),               ("O(3;1)/K1", O(3) - 1),   None,    "vle(7)",      7, None),
    ("vle(3;N|6)",     ("id_sl(3) x K[*]", 3),   ("id*_sl(3) x id_sl(2) x K[*]", 3 * 2), None,
                                                                                     "vle(9)",      9, "Fvle(4|3;K)"),
    ("kas(1;N|6)",     ("K[*]", 1),              ("id", 6),                 None,    "kas(7)",      7, "Fkas"),
    ("kas(5;N|5)",     ("O(1;1)", O(1)),         ("id_sl(2) x id_gl(2) x O(1;1)", 2 * 2 * O(1)), None,
                                                                                     "kas(10)",    10, "Fkas(;1xi)"),
    ("kas(4;N|4)",     ("---", 0),               ("O(3;1)", O(3)),          None,    "kas(8)",      8, None),
    ("kas(4;N|3)",     ("---", 0),               ("Vol_0(3;1)", O(3) - 1),  None,    "kas~(7)",     7, None),
    ("mb(4;N|5)",      ("K[*]", 1),              ("O(3;1)", O(3)),          None,    "mb(9)",       9, "Fmb(4|5)"),
    ("mb(3;N|8)",      ("id_sl(3) x K[*]", 3),   ("id*_sl(3) x id_sl(2) x K[*]", 3 * 2),
                                                 ("K x id_sl(2) x K[*]", 1 * 2 * 1), "mb_3(11)",   11, "Fmb(4|5;K)"),
    ("mb(5;N|6)",      ("O(2;1)/K1", O(2) - 1),  ("id_sl(2) x O(2;1)", 2 * O(2)), None,
                                                                                     "mb_2(11)",   11, "Fmb(4|5;1)"),
    ("kle(5;N|10)",    ("id", 5),                ("Lambda^2(id*)", 5 * 4 // 2), None,
                                                                                     "kle(15)",    15, "Fkle(9|6;K)"),
    ("kle(11;N|9)",    ("id_sl(4)", 4),          ("id_sl(2) x O(3;1)", 2 * O(3)), None,
                                                                                     "kle_2(20)",  20, "Fkle(9|6;2)"),
    ("kle(9;N|11)",    ("id*_sl(3) x O(1;1)", 3 * O(1)),
                                                 ("id_sl(2) x (id_sl(3) x O(1;1))", 2 * 3 * O(1)),
                                                 ("id_sl(2) x K[*]", 2 * 1),         "kle_3(20)",  20, "Fkle(9|6;CK)"),
    ("vle(5;N|4)",     ("K[*]", 1),              ("id x O(2;1)", 2 * O(2)), None,    "vle~(9)",     9, "Fvle(4|3;1)"),
    ("kle(9;N|6)",     ("K[*]", 1),              ("T_0^0", 14),             None,    "kle~(15)",   15, "Fkle(9|6)"),
    ("vas(4;N|4)",     ("---", 0),               ("id_F(as)", 8),           None,    "vas(8)",      8, None),
]

# table (grVexF) of the text, verbatim: growth vector, and whether marked "C".
SAND2_21 = {
    "Fvle(4|3;1)":  ((8, 9),      True),
    "Fvle(4|3;K)":  ((6, 9),      False),
    "Fkle(9|6)":    ((14, 15),    True),
    "Fkle(9|6;2)":  ((16, 20),    False),
    "Fkle(9|6;K)":  ((10, 15),    False),
    "Fkle(9|6;CK)": ((12, 18, 20), False),
    "Fkas":         ((6, 7),      True),
    "Fkas(;1xi)":   ((8, 10),     False),
    "Fmb(4|5)":     ((8, 9),      True),
    "Fmb(4|5;1)":   ((8, 11),     False),
    "Fmb(4|5;K)":   ((6, 9, 11),  False),
}


def growth(m1, m2, m3):
    """Growth vector of the distribution: partial sums of dim g_-1, g_-2, ..."""
    v = [m1]
    if m2:
        v.append(v[-1] + m2)
    if m3:
        v.append(v[-1] + m3)
    return tuple(v)


def main():
    bad = 0
    print("%-14s %5s %5s %5s  %-14s %-6s %s"
          % ("g", "d-1", "d-2", "d-3", "growth", "depth", "F(g)"))
    print("-" * 74)
    seen = set()
    for g, (s2, m2), (s1, m1), t3, fg, ind, old in ROWS:
        m3 = t3[1] if t3 else 0
        v = growth(m1, m2, m3)
        depth = len(v)
        contact = (m2 == 1) if depth >= 2 else False
        mark = "C" if contact else ""
        print("%-14s %5d %5d %5d  %-14s %-6d %s%s"
              % (g, m1, m2, m3, str(v), depth, fg, " " + mark if mark else ""))

        # check (a): the growth vector ends at the number of indeterminates
        if v[-1] != ind:
            print("   !! growth ends at %d but F(g) = %s has %d indeterminates"
                  % (v[-1], fg, ind))
            bad += 1

        # check (b): the depth >= 2 rows must match table (grVexF) of the text
        if old is not None:
            seen.add(old)
            want, wantC = SAND2_21[old]
            if v != want:
                print("   !! table (grVexF) row %s says %s, we get %s" % (old, want, v))
                bad += 1
            if contact != wantC:
                print("   !! table (grVexF) row %s C-mark %s, we get %s"
                      % (old, wantC, contact))
                bad += 1

    print("-" * 74)
    # every row of Subsec. 2.1 must have been accounted for
    for old in SAND2_21:
        if old not in seen:
            print("!! row %s of table (grVexF) has no counterpart in Table 25.5" % old)
            bad += 1

    # Grouping by base algebra: where is the smallest codimension attained,
    # and is the row attaining it contact?  This is the question Subsec. 2.1
    # asks; with only the eleven rows of depth >= 2 it gets a different answer.
    print()
    for label, rows in (("all fifteen rows", ROWS),
                        ("only the eleven of depth >= 2",
                         [r for r in ROWS if r[1][1] != 0])):
        print("smallest codimension per base algebra, %s:" % label)
        base = {}
        for g, (s2, m2), (s1, m1), t3, fg, ind, old in rows:
            b = g.split("(")[0]
            v = growth(m1, m2, t3[1] if t3 else 0)
            base.setdefault(b, []).append((ind, fg, (m2 == 1), len(v)))
        for b in sorted(base):
            lo = min(e[0] for e in base[b])
            at = [e for e in base[b] if e[0] == lo]
            print("   %-4s min codim %2d attained %d time(s): %s"
                  % (b, lo, len(at),
                     ", ".join("%s%s [depth %d]" % (e[1], " C" if e[2] else "", e[3])
                               for e in at)))
        print()

    depth1 = [r[0] for r in ROWS if r[1][1] == 0]
    print("rows of Table 25.5 : %d" % len(ROWS))
    print("of depth 1 (F(g_-2) is '---', so no distribution) : %d  %s"
          % (len(depth1), depth1))
    print("of depth >= 2, i.e. tabulated in table (grVexF) : %d"
          % (len(ROWS) - len(depth1)))
    print("contact (dim F(g_-2) = 1) : %s"
          % [r[0] for r in ROWS if r[1][1] == 1])
    print()
    print("MISMATCHES: %d" % bad)
    return bad


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
