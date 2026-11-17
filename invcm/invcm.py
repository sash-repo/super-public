#!/usr/bin/env python3
"""Exact check of every entry of the tables of Subsections 9.1 and 9.2.

For each of the 370 entries (name, A, A^{-1}, det A) we verify over Q:

  * det A equals the value printed in the last column;
  * A times the printed inverse is the identity matrix;
  * det A < 0;
  * the mark !!! sits exactly on the entries whose inverse has no zero element;
  * the inverses of the marked entries have ALL elements negative;
  * the zeros of every inverse occur on the main diagonal only
    (the last two are the phenomena described in Subsection 1.5 and explained by
    Theorem 3.3.1).

Usage:  python invcm.py
"""
from tables import load, det, mul

entries = load()
print("read %d table entries (%d from Subsection 9.1, %d from Subsection 9.2)"
      % (len(entries), sum(e["subsection"] == "9.1" for e in entries),
         sum(e["subsection"] == "9.2" for e in entries)))

problems = []
allneg_marked, zeros_offdiag = [], []
nmark = 0
for e in entries:
    name, A, B, d, marked = e["name"], e["A"], e["inverse"], e["det"], e["marked"]
    nmark += marked
    n = len(A)
    dA = det(A)
    if dA != d:
        problems.append(("DET", name, "printed %s, true %s" % (d, dA)))
    P = mul(A, B)
    if any(P[i][j] != (1 if i == j else 0) for i in range(n) for j in range(n)):
        problems.append(("INV", name, "A*B != 1"))
    if dA >= 0:
        problems.append(("SIGN", name, "det = %s is not negative" % dA))
    haszero = any(B[i][j] == 0 for i in range(n) for j in range(n))
    if marked and haszero:
        problems.append(("MARK", name, "marked !!! but the inverse has a 0"))
    if (not marked) and (not haszero):
        problems.append(("MARK", name, "not marked but the inverse has no 0"))
    if marked and any(B[i][j] >= 0 for i in range(n) for j in range(n)):
        allneg_marked.append(name)
    off = [(i, j) for i in range(n) for j in range(n) if i != j and B[i][j] == 0]
    if off:
        zeros_offdiag.append(name)

print("marked !!!: %d" % nmark)
print("all elements of a marked inverse are negative: %s"
      % ("OK" if not allneg_marked else "FAILS for " + ", ".join(allneg_marked)))
print("0s of the inverses only on the main diagonal: %s"
      % ("OK" if not zeros_offdiag else "FAILS for " + ", ".join(zeros_offdiag[:20])))
print("all determinants negative: %s"
      % ("OK" if not [p for p in problems if p[0] == "SIGN"] else "see below"))
print("\n%d problems" % len(problems))
for kind, name, msg in problems:
    print("  %-6s %-14s %s" % (kind, name, msg))
