#!/usr/bin/env python3
"""
mktable.py -- generate the LaTeX of the new Table columns from out/*.json, in
the notation of the paper's own Table: the Riemannian answer is listed, what
survives conformally is boxed, a split multiplicity is written with a boxed
exponent, and a degree all of whose classes are conformally invariant has its
degree label boxed instead.

    ../../tools/venv/bin/python mktable.py 00110011 0011111100 ...  > tab.tex
"""

import json
import os
import sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
WIDTH = 82          # wrap a cell at about this many characters


def load(par):
    with open(os.path.join(OUT, par + ".json")) as f:
        return json.load(f)


def wt(w):
    return "(" + (",".join(map(str, w)) if len(w) == 2
                  else "".join(map(str, w))) + ")"


def entry(w, par, r, c):
    """One weight: r copies Riemannian, c of them conformal."""
    body = wt(w) + ("_1" if par else "")
    if c == 0:
        return f"${body}" + (f"^{{{r}}}$" if r > 1 else "$")
    if c >= r:
        return "\\fbox{$" + body + (f"^{{{r}}}" if r > 1 else "") + "$}"
    return f"${body}^{{\\fbox{{$\\scriptstyle {c}$}}+{r - c}}}$"


def cell(rec, d):
    riem = rec["kinds"]["riem"]
    full = rec["kinds"]["full"]
    R = {(tuple(w), p): m for w, p, m in riem["H"].get(str(d), [])}
    C = {(tuple(w), p): m for w, p, m in full["H"].get(str(d), [])}
    if not R:
        if not riem["c2dim"].get(str(d)):
            return "not defined", False, []
        return "---", False, [k for k in C]
    allbox = all(C.get(k, 0) >= m for k, m in R.items()) and set(C) == set(R)
    items, extra = [], [k for k in C if k not in R]
    for (w, p), m in sorted(R.items(), key=lambda t: (-sum(t[0][0]), t[0][1],
                                                      t[0][0])):
        c = C.get((w, p), 0)
        items.append(entry(w, p, m, 0 if allbox else c))
    return ", ".join(items), allbox, extra


def wrap(s):
    out, line = [], ""
    for piece in s.split(", "):
        if line and len(line) + len(piece) + 2 > WIDTH:
            out.append(line + ",")
            line = " " + piece
        else:
            line = (line + ", " + piece) if line else piece
    out.append(line)
    return out


def table(par):
    rec = load(par)
    degs = sorted({int(d) for k in rec["kinds"]
                   for d in rec["kinds"][k]["c2dim"]}
                  | {int(d) for k in rec["kinds"]
                     for d in rec["kinds"][k]["H"]})
    # collapse the empty tail into one ">= k" row
    body, tail = [], None
    for d in degs:
        txt, allbox, extra = cell(rec, d)
        if extra:
            print(f"%% NB: {par} degree {d} has conformal-only classes {extra}",
                  file=sys.stderr)
        if txt == "---" and tail is None and d > 2:
            tail = d
        if tail is not None and txt not in ("---", "not defined"):
            tail = None
        body.append((d, txt, allbox))
    lines = ["\\noindent", "\\begin{tabular}{|c|c|}", "\\hline",
             "$\\deg(SF) \\backslash Par$&$(%s)$\\\\" % par, "\\hline"]
    for (d, txt, allbox) in body:
        if tail is not None and d > tail:
            continue
        lab = ("\\fbox{$%d$}" % d) if allbox else "$%d$" % d
        if tail is not None and d == tail:
            lab = "$\\geqslant %d$" % d
        ws = wrap(txt)
        lines.append(f"{lab}&{ws[0]}\\\\")
        for w in ws[1:]:
            lines.append(f"&{w}\\\\")
        lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


if __name__ == "__main__":
    for par in sys.argv[1:]:
        print(table(par))
        print("\n\\vspace{4pt}\n")
