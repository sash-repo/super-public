#!/usr/bin/env python3
"""Which tabulated column of "From Supergravity to Ballbearings" is an enlargement \\widetilde R(N)?

The Table identifies each model by its parity string Par alone.  The marking of
the simple roots -- which is what actually fixes g_- and g_0 -- is recovered by
the rule stated in the text:

    cut Par into consecutive blocks of size <= 2, a change of parity forcing a cut.

Each block of size 2 contributes an sl(2) factor to the Levi of g_0, a block of
size 1 contributes nothing.  The one invariant the Table does print is the number
of entries of a weight tuple, which must equal the number of sl(2) factors; that
is the calibration below.

Then \\widetilde R(N) := SL(N|4)/P' with Lie(P') of Levi Q x SL_L(2) x SL_R(2),
Q a parabolic of SL(N) -- i.e. an enlargement of \\widetilde M(N), fibered over it
with fiber SL(N)/Q.  For that, the four "spacetime" positions must form ONE
contiguous run (so that they carry sl(4), cut 2|2 into sl_L(2)+sl_R(2)) and the N
positions of the other parity must form one contiguous run as well (so that they
carry sl(N) and the extra marks inside it are a parabolic Q).  That is: Par must
be 0^4 1^N or 1^N 0^4, up to the global flip 0<->1 under which the Table is
invariant.
"""

# Par -> the arity of the weight tuples printed in that column of the Table.
TABLE = [
    ("0000",         2),
    ("00100",        2),
    ("00001",        2),
    ("001100",       3),
    ("100001",       2),
    ("000011",       3),
    ("00111100",     4),
    ("00001111",     4),
    ("111100001111", 6),
]

# the two options displayed in eq. (5) for N=8
EQ5 = ["001111111100", "000011110000"]


def blocks(par):
    """Cut Par into runs of equal parity, then each run into pieces of size <= 2."""
    runs, cur = [], par[0]
    start = 0
    for i in range(1, len(par) + 1):
        if i == len(par) or par[i] != cur:
            runs.append((cur, i - start))
            if i < len(par):
                cur, start = par[i], i
    out = []
    for sym, n in runs:
        while n > 0:
            out.append((sym, min(n, 2)))
            n -= min(n, 2)
    return out


def n_sl2(par):
    return sum(1 for _, n in blocks(par) if n == 2)


def standard_format(par):
    """Is Par = 0^4 1^N or 1^N 0^4 (up to the global flip)?"""
    runs, cur, start = [], par[0], 0
    for i in range(1, len(par) + 1):
        if i == len(par) or par[i] != cur:
            runs.append((cur, i - start))
            if i < len(par):
                cur, start = par[i], i
    if len(runs) == 1:
        # no second parity at all: this is Minkowski space, N = 0
        return 0 if runs[0][1] == 4 else None
    if len(runs) != 2:
        return None
    # the block of size 4 is the spacetime sl(4); the other is sl(N)
    sizes = [n for _, n in runs]
    if 4 not in sizes:
        return None
    N = sizes[1] if sizes[0] == 4 else sizes[0]
    return N


def q_marked(par, N):
    """Inside the sl(N) run, is any internal node marked (i.e. is Q != SL(N))?"""
    # the sl(N) run is cut into pieces of 2; Q = SL(N) exactly when N <= 2
    return N > 2


def main():
    print("Par            sl(2) factors  weight arity  ok   standard  N   R(N)?")
    ok_all = True
    for par, arity in TABLE:
        k = n_sl2(par)
        ok = (k == arity)
        ok_all &= ok
        N = standard_format(par)
        if N is None:
            verdict = "no (spacetime block split by the other parity)"
            std = "-"
        else:
            std = "yes"
            if N == 0:
                verdict = "M(0) = Minkowski space itself"
            elif not q_marked(par, N):
                verdict = "= M(%d) itself (Q = SL(%d), nothing added)" % (N, N)
            else:
                verdict = "YES, a proper enlargement, N=%d" % N
        print("%-14s %-14d %-13d %-4s %-9s %-3s %s"
              % (par, k, arity, "ok" if ok else "MISMATCH", std,
                 "-" if N is None else N, verdict))
    print()
    print("calibration against the Table's weight arities:",
          "all agree" if ok_all else "MISMATCH")
    print()
    for par in EQ5:
        N = standard_format(par)
        print("eq. (5) option %s: sl(2) factors %d, standard format %s"
              % (par, n_sl2(par), "yes (N=%d)" % N if N else "no"))
    print()
    # the one proper enlargement, in Hartwell-Howe's (N,p,q) terms
    par = "00001111"
    N = standard_format(par)
    # the sl(N) run is cut 2|2, so Q stabilizes a 2-plane in C^N
    p, Nmq = 2, 2
    q = N - Nmq
    print("%s: N=%d, SL(N)/Q = flag of type (p, N-q) = (%d, %d), so p=%d, q=%d, p+q=%d <= N=%d : %s"
          % (par, N, p, Nmq, p, q, p + q, N, "ok" if p + q <= N else "FAILS"))


if __name__ == "__main__":
    main()
