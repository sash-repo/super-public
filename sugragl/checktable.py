#!/usr/bin/env python3
r"""
checktable.py -- read the Table out of the LaTeX source of the text and compare every cell with the
recomputation in out/*.json.  Nothing is transcribed: the expected content of a
cell comes from the JSON, the printed content is parsed from the LaTeX.

    python3 checktable.py <the LaTeX source of the text>

The conventions decoded are the paper's own, stated above the Table:
  * a weight inside a single \fbox is conformally invariant ("on shell");
  * a boxed DEGREE label says every entry of that row is;
  * a boxed EXPONENT, as (0011)^{\fbox{2}+1}, splits a multiplicity into the
    conformal part and the rest;
  * a plain exponent is a multiplicity;
  * a DOUBLE box is the "off shell" mark and says nothing about conformality,
    so the parser looks through it.
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


# ----------------------------------------------------------------- the LaTeX

def tabulars(tex):
    """The four tabulars of the Table, as lists of physical rows."""
    start = tex.index(r"\subsection{Table}")
    end = tex.index(r"\subsection{More models")
    body = tex[start:end]
    return re.findall(r"\\begin\{tabular\}(.*?)\\end\{tabular\}", body, re.S)


def rows(block):
    """[(label, [cell, ...]), ...] -- continuation lines merged into the row."""
    out = []
    for chunk in block.split(r"\hline"):
        cells, lab = None, None
        for line in chunk.strip().split("\\\\"):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("&")]
            if cells is None:
                lab, cells = parts[0], parts[1:]
            else:
                if parts[0]:               # the label may sit on any line of a row
                    if lab:
                        sys.exit("two labels in one row: " + line)
                    lab = parts[0]
                for i, p in enumerate(parts[1:]):
                    cells[i] = (cells[i] + " " + p).strip()
        if cells is not None:
            out.append((lab, cells))
    return out


def degree(label):
    m = re.search(r"(-?\d+)", label)
    if not m:
        return None
    if r"\geqslant" in label:
        return None
    return int(m.group(1)), label.startswith(r"\fbox")


def brace(s, i):
    """i points at '{'; return the index just past its match."""
    d = 0
    while i < len(s):
        if s[i] == "{":
            d += 1
        elif s[i] == "}":
            d -= 1
            if d == 0:
                return i + 1
        i += 1
    sys.exit("unbalanced braces in " + s)


def parse_cell(text, rowboxed):
    """-> None (not defined) | {} (dash) | {(weight, parity): (mult, conf)}."""
    if "not defined" in text:
        return None
    if not re.search(r"\(\s*\d", text):
        return {}
    res, i, boxes = {}, 0, []          # boxes: brace-depths at which an fbox is open
    depth = 0
    while i < len(text):
        if text.startswith(r"\fbox{", i):
            depth += 1
            boxes.append(depth)
            i += len(r"\fbox{")
            continue
        if text[i] == "{":
            depth += 1
            i += 1
            continue
        if text[i] == "}":
            if boxes and boxes[-1] == depth:
                boxes.pop()
            depth -= 1
            i += 1
            continue
        m = re.match(r"\((\s*\d[\d,\s]*)\)", text[i:])
        if not m:
            i += 1
            continue
        inside = m.group(1)
        w = tuple(int(x) for x in inside.split(",")) if "," in inside \
            else tuple(int(c) for c in inside.strip())
        nbox = len(boxes)
        i += m.end()
        par = 0
        if text.startswith("_1", i):
            par, i = 1, i + 2
        mult, conf = 1, None
        if text.startswith("^", i):
            j = i + 1
            if text[j] == "{":
                k = brace(text, j)
                exp = text[j + 1:k - 1]
                i = k
            else:
                exp, i = text[j], j + 1
            mb = re.match(r"\\fbox\{\$\\scriptstyle (\d+)\$\}\+(\d+)$", exp)
            if mb:
                conf = int(mb.group(1))
                mult = conf + int(mb.group(2))
            else:
                mult = int(exp)
        if conf is None:
            conf = mult if (rowboxed or nbox == 1) else 0
        if (w, par) in res:
            sys.exit("weight twice in one cell: %s" % (w,))
        res[(w, par)] = (mult, conf)
    return res


# ------------------------------------------------------------------ the JSON

def expected(par):
    with open(os.path.join(OUT, par + ".json")) as f:
        rec = json.load(f)
    riem, full = rec["kinds"]["riem"], rec["kinds"]["full"]
    degs = sorted({int(d) for k in (riem, full) for d in k["H"]}
                  | {int(d) for k in (riem, full) for d in k["c2dim"]})
    exp = {}
    for d in degs:
        R = {(tuple(w), p): m for w, p, m in riem["H"].get(str(d), [])}
        C = {(tuple(w), p): m for w, p, m in full["H"].get(str(d), [])}
        for k in C:
            if k not in R:
                sys.exit("conformal-only class in %s degree %d: %s" % (par, d, k))
        if R:
            exp[d] = {k: (m, C.get(k, 0)) for k, m in R.items()}
        elif riem["c2dim"].get(str(d)):
            exp[d] = {}                       # a dash: cochains, no cohomology
        else:
            exp[d] = None                     # not defined
    return exp


# --------------------------------------------------------------------- main

def main(path):
    tex = open(path, encoding="utf-8").read()
    printed, bad, cells = {}, 0, 0
    for block in tabulars(tex):
        rs = [r for r in rows(block) if r[1]]     # drop the column spec
        head = rs[0][1]
        pars = [re.fullmatch(r"\$\((\w+)\)\$", h).group(1) for h in head]
        for lab, cs in rs[1:]:
            dg = degree(lab)
            if dg is None:
                continue
            d, boxed = dg
            for par, c in zip(pars, cs):
                printed.setdefault(par, {})[d] = parse_cell(c, boxed)

    for par, got in printed.items():
        exp = expected(par)
        for d, g in sorted(got.items()):
            cells += 1
            e = exp.get(d, "MISSING")
            if e == "MISSING":
                # a degree the print shows but the computation does not reach
                if g:
                    print(f"  [{par}] degree {d}: printed {g}, not computed")
                    bad += 1
                continue
            if g is None or e is None:
                if (g is None) != (e is None):
                    print(f"  [{par}] degree {d}: 'not defined' vs {e if g is None else g}")
                    bad += 1
                continue
            if g != e:
                print(f"  [{par}] degree {d}:")
                for k in sorted(set(g) | set(e), key=lambda t: (t[1], t[0])):
                    if g.get(k) != e.get(k):
                        print(f"      {k}: table {g.get(k)}  computed {e.get(k)}")
                bad += 1
        for d, e in exp.items():
            if e and d not in got:
                print(f"  [{par}] degree {d} computed but absent from the Table: {e}")
                bad += 1
    print(f"{cells} cells checked in {len(printed)} columns, {bad} disagreements")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else os.path.join(HERE, "paper.tex")))
