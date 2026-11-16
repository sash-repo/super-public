#!/usr/bin/env python3
"""Post-processing: identify every maximal graded subalgebra dumped by run3.py.

   python3 post3.py runs/G3-100.pkl [...]
"""
import pickle, sys
from fractions import Fraction as F
import run3, ident3
from sweep3 import Ech, sdim_of, derived, centre_dim

# simple Lie (super)algebras by superdimension (small ones; ambiguities noted)
SIMPLE = {
    (3, 0): "sl(2)", (8, 0): "sl(3)", (10, 0): "sp(4)=o(5)", (14, 0): "g(2)",
    (15, 0): "sl(4)=o(6)", (21, 0): "o(7)", (24, 0): "sp(6)",
    (3, 2): "osp(1|2)", (4, 4): "sl(2|1)=osp(2|2)", (6, 6): "osp(3|2)",
    (9, 8): "osp_a(4|2)=D(2,1;a)", (10, 4): "osp(1|4)", (11, 8): "osp(2|4)",
    (13, 10): "osp(5|2)", (13, 12): "osp(3|4)", (18, 12): "osp(6|2)",
    (9, 6): "sl(3|1)", (12, 12): "sl(3|2)", (16, 8): "sl(4|1)",
    (6, 8): "psl(2|2)", (7, 8): "sl(2|2)", (8, 9): "spe(3)",
    (17, 14): "ag(2)=G(3)", (24, 16): "ab(3)=F(4)",
}


def name_of(sd, simple):
    n = SIMPLE.get(tuple(sd))
    if n is None:
        return "sdim %d|%d (?)" % tuple(sd)
    return n if simple else "%s?? (not simple)" % n


def main(paths):
    for p in paths:
        with open(p, "rb") as f:
            d = pickle.load(f)
        which = "G3" if "G3" in p else "F4"
        g = run3.model(which)
        print("\n" + "#" * 100)
        print("# %s   %s   h = (%s)" % (p, d["name"], ", ".join(d["h"])))
        print("#" * 100)
        for tags, basis, sd, pieces, verdict in d["answers"]:
            ub = [[F(x) for x in v] for v in basis]
            E = Ech(g.N, ub)
            ds = derived(g, E)
            perfect = len(ds) >= 2 and ds[0] == ds[1]
            mins = ident3.minimal_ideals(g, ub)
            tot = Ech(g.N, [v for I in mins for v in I.basis()])
            names = []
            for I in mins:
                isd = sdim_of(g, I.basis())
                names.append(name_of(isd, is_simple(g, I)))
            semisimple = tot.dim() == E.dim() and all("??" not in n
                                                      for n in names)
            print("\n * %d|%d   %s   %s   %s"
                  % (sd[0], sd[1], pieces, verdict,
                     (" + ".join(names)) if semisimple
                     else "NOT semisimple (ideals %s sum to %d of %d)"
                     % ("+".join("%d|%d" % sdim_of(g, I.basis())
                                 for I in mins), tot.dim(), E.dim())))
            print("      from: %s%s" % (", ".join(tags[:3]),
                                        "  [%d V's]" % len(tags)))
            print("      derived %s  centre %d  perfect %s"
                  % (ds, centre_dim(g, E), perfect))
            if tuple(sd) == (9, 8) or any(sdim_of(g, I.basis()) == (9, 8)
                                          for I in mins):
                for I in ([E] if tuple(sd) == (9, 8) else
                          [I for I in mins if sdim_of(g, I.basis()) == (9, 8)]):
                    r = ident3.d21a_parameter(g, I.basis())
                    if r:
                        print("      ==> D(2,1;a) with a = %s  (S_3-orbit %s)"
                              % (r[0], ", ".join(str(x) for x in r[1])))


def is_simple(g, I):
    b = ident3.weight_gens(g, I.basis())
    for x in b:
        if ident3.ideal_generated(g, I.basis(), x).dim() != I.dim():
            return False
    return True


if __name__ == "__main__":
    main(sys.argv[1:])
