#!/usr/bin/env python3
"""Full report for the 15- and 16-dimensional simple Lie algebras over F_2 of
Cushing--Stagg--Stewart: invariants, sandwiches, the smallest codimension of a
maximal subalgebra, and the two Weisfeiler gradings."""
import ast, glob, os, sys
from css import LieF2, rref, in_span
from thin import algebra_from_thin_table, load_table
from minmax import smallest_codim
from wgrad import wgrading


def build(path):
    name = os.path.basename(path)[:-4]
    data = load_table(path)
    if isinstance(data[0][0], list):                 # list of matrices
        dim = len(data[0])
        if len(data) == dim:
            return LieF2.from_adjoint_basis(data, name)
        return LieF2.from_generators(data, name, dim)
    return algebra_from_thin_table(data, name)       # thin table


def report(path):
    L = build(path)
    name = L.name
    ok, msg = L.check_jacobi()
    print("=" * 72, flush=True)
    print("%s  dim %d   axioms: %s" % (name, L.n, msg), flush=True)
    if not ok:
        return
    print("   simple %s | dim[L,L] %d | dim Der %d | Num2Nilps %d (CSS: %d)"
          % (L.is_simple(), len(L.derived()), L.dim_der(),
             L.num_2nilps(), L.num_2nilps() + 1), flush=True)
    S = rref(L.sandwiches())
    print("   sandwiches over F_2 span dim %d; the whole span consists of"
          " sandwiches: %s" % (len(S), L.flat_family(S)), flush=True)
    sg = L.subalgebra_generated(S)
    nn = L.normaliser(sg)
    print("   dim s(L) = %d ; n_L(s): dim %d, codim %d, maximal %s"
          % (len(sg), len(nn), L.n - len(nn),
             L.is_maximal_subalgebra(nn)[0]), flush=True)
    neg, pos = wgrading(L, nn)
    print("   W-grading of n_L(s):  (g_-1, g_-2, ...) = %s ; (g_0, g_1, ...) = %s"
          % (neg, pos), flush=True)
    c, k, M = smallest_codim(L)
    print("   smallest codimension of a proper subalgebra: %s (%d of them)"
          % (c, k), flush=True)
    if M is not None:
        neg2, pos2 = wgrading(L, M)
        print("   W-grading of one of them: (g_-1, ...) = %s ; (g_0, g_1, ...) = %s"
              % (neg2, pos2), flush=True)
        print("   >>> n_L(s) IS of the smallest codimension: %s"
              % (L.n - len(nn) == c), flush=True)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        try:
            report(p)
        except Exception as e:
            print("FAILED on", p, repr(e), flush=True)
