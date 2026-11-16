#!/usr/bin/env python3
"""Turn the dumps of run3.py into the LaTeX tables of Statements 7.1-7.5, 8.1-8.8.

   python3 tables3.py runs/G3-*.pkl > tables-G3.tex
"""
import pickle, sys, re
from fractions import Fraction as F
import run3, ident3
from sweep3 import Ech, sdim_of, derived, centre_dim
from post3 import SIMPLE, name_of, is_simple

TEX = {
    "sl": r"\fsl", "osp": r"\fosp", "sp": r"\fsp", "o": r"\fo", "g": r"\fg",
    "psl": r"\fpsl", "spe": r"\fspe", "ag": r"\fag", "ab": r"\fab",
}


def tex_name(n):
    n = n.replace("osp_a(4|2)=D(2,1;a)", r"\fosp_a(4|2)")
    n = n.replace("sl(2|1)=osp(2|2)", r"\fsl(2|1)")
    n = n.replace("sp(4)=o(5)", r"\fsp(4)")
    n = n.replace("sl(4)=o(6)", r"\fsl(4)")
    n = n.replace("ag(2)=G(3)", r"\fag(2)").replace("ab(3)=F(4)", r"\fab(3)")
    for k in ("psl", "spe", "osp", "sl", "sp", "o", "g"):
        rep = TEX[k] + "("            # a lambda: re.sub would eat the backslash
        n = re.sub(r"(?<![\\a-z])%s\(" % k, lambda m, r=rep: r, n)
    return "$" + n + "$"


def analyse(g, ub):
    E = Ech(g.N, ub)
    mins = ident3.minimal_ideals(g, ub)
    tot = Ech(g.N, [v for I in mins for v in I.basis()])
    names, simple = [], True
    for I in mins:
        isd = sdim_of(g, I.basis())
        s = is_simple(g, I)
        simple = simple and s
        names.append(name_of(isd, s))
    ss = tot.dim() == E.dim() and simple
    return ss, " + ".join(sorted(names)), mins


def load(p):
    with open(p, "rb") as f:
        d = pickle.load(f)
    which = "G3" if "G3" in p else "F4"
    return which, d


def main(paths):
    summary = []
    for p in paths:
        which, d = load(p)
        g = run3.model(which)
        rows, nonss = [], {}
        for tags, basis, sd, pieces, verdict in d["answers"]:
            ub = [[F(x) for x in v] for v in basis]
            ss, nm, mins = analyse(g, ub)
            cases = sorted({t.split()[0] for t in tags})
            if ss:
                extra = ""
                for I in ([Ech(g.N, ub)] if tuple(sd) == (9, 8) else
                          [I for I in mins if sdim_of(g, I.basis()) == (9, 8)]):
                    r = ident3.d21a_parameter(g, I.basis())
                    if r:
                        # a is defined only up to the S_3-action; print the
                        # largest positive member of the orbit as the canonical
                        # representative
                        pos = [x for x in r[1] if x > 0]
                        extra = r" with $a=%s$" % (max(pos) if pos else r[0])
                rows.append((sd, pieces, (nm, extra), cases, verdict,
                             len(tags)))
            else:
                key = (tuple(sd), verdict)
                nonss[key] = nonss.get(key, 0) + 1
        # a row whose ideals could not be identified (no simple superalgebra of
        # that superdimension) is a conjugate copy of an identified row with the
        # same sdim whose torus was big enough to split it: drop it
        clean = {tuple(r[0]) for r in rows if "(?)" not in r[2][0]}
        rows = [r for r in rows
                if "(?)" not in r[2][0] or tuple(r[0]) not in clean]
        # dedupe identical rows
        seen, out = set(), []
        for r in rows:
            k = (r[0], r[1], r[2][0], r[2][1], r[4])
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
        lbl = p.split("/")[-1].split(".")[0]
        print("%% ---------------- %s : %s" % (p, d["name"]))
        print(r"\ssbegin{Statement}")
        print(r"\label{S%s}" % lbl)
        print(r"For the grading $h=(%s)$ of %s $(%s)$ the semi-simple maximal"
              % (", ".join(d["h"]), "$\\fag(2)$" if which == "G3"
                 else "$\\fab(3)$", d["name"].split(":")[1].strip()
                 .replace("_", "_").replace("|", "{\\mid}")))
        print(r"graded subalgebras are")
        print(r"\begin{equation}\label{%s}" % lbl)
        print(r"\footnotesize\begin{tabular}{llll}")
        print(r"$\sdim\fu$&the components of $\fu$&$\fu$&case, "
              r"certificate\\ \hline")
        for sd, pieces, nm, cases, verdict, cnt in sorted(
                out, key=lambda r: (-r[0][0] - r[0][1], r[0])):
            print(r"$%d|%d$&$%s$&%s&%s, %s\\"
                  % (sd[0], sd[1], tex_pieces(pieces),
                     tex_name(nm[0]) + nm[1],
                     ", ".join(c.replace("o", "$^o$") if c[0].isdigit()
                               else c.replace("_", r"\_") for c in cases),
                     "max" if verdict == "MAX" else "max(b)"))
        print(r"\end{tabular}\end{equation}")
        tot = sum(nonss.values())
        sds = sorted({k[0] for k in nonss})
        print(r"Besides them the sweep returns %d maximal graded subalgebras "
              r"that are not semi-simple, of superdimensions %s; each of them "
              r"is either $\fg_{\leq0}$, or $\fg_{\geq0}$, or the "
              r"stabiliser-with-prolongation of a subspace $V\subset\fg_{-1}$ "
              r"$($case $2^o$ or $3^o)$."
              % (tot, ", ".join("$%d|%d$" % s for s in sds)))
        print(r"\end{Statement}")
        print()
        summary.append((p, d, out, nonss))
    print(r"%% ================= SUMMARY")
    for p, d, out, nonss in summary:
        print("%% %-14s %-46s %s" % (p.split("/")[-1], d["name"],
              "; ".join("%d|%d %s%s [%s]" % (r[0][0], r[0][1], r[2][0],
                                             r[2][1], r[4])
                        for r in out)))
    # the union table: one row per isomorphism type
    types = {}
    for p, d, out, nonss in summary:
        for r in out:
            key = (tuple(r[0]), r[2][0] + r[2][1])
            types.setdefault(key, {})[p] = r[4]
    print()
    print(r"\begin{equation}\label{union}")
    print(r"\begin{tabular}{ll%s}" % ("c" * len(summary)))
    print(r"$\sdim\fu$&$\fu$&%s\\ \hline"
          % "&".join(p.split("/")[-1].split(".")[0].split("-")[1]
                     for p, _, _, _ in summary))
    for key in sorted(types, key=lambda k: (-k[0][0] - k[0][1], k[0])):
        sd, nm = key
        cells = []
        for p, _, _, _ in summary:
            v = types[key].get(p)
            cells.append("" if v is None else
                         ("$+$" if v == "MAX" else "$(b)$"))
        print(r"$%d|%d$&%s&%s\\" % (sd[0], sd[1],
              tex_name(nm.split(" with ")[0]) +
              (" with $a=%s$" % nm.split("$a=")[1].rstrip("$")
               if " with " in nm else ""), "&".join(cells)))
    print(r"\end{tabular}\end{equation}")
    return summary


def tex_pieces(pieces):
    out = []
    for tok in pieces.split():
        k, sd = tok.split(":")
        out.append(r"{\scriptstyle %s{:}}%s" % (k, sd))
    return ",".join(out)


if __name__ == "__main__":
    main(sys.argv[1:])
