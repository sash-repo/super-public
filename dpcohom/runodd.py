#!/usr/bin/env python3
"""runodd.py -- the checks for the case p > 2, and the Hypothesis of [BGL, Sec. 4].

Everything is exact: the complexes are built over Z and reduced mod p.

  A) U(g) for p > 2 with the signs restored: the axioms, and PBW through the
     associativity of the rewriting  x_i x_j = (-1)^{p_i p_j} x_j x_i + [x_i,x_j],
     f f = s(f) = (1/2)[f,f].
  B) The two lattices in one rational complex: the super anti-symmetric cochains
     e_X (X an argument tuple) and the divided power cochains f_X = c_X e_X with
     c_X = prod_j a_j(X)!, a_j the multiplicity of the odd index j.  Checked:
     d^2 = 0, and the matrix of d in the f-basis is INTEGRAL -- so the divided
     power complex exists over Z, hence mod every p.
  C) The comparison map Phi = diag(c_X) is the isomorphism the Hypothesis asks
     for; it is invertible mod p exactly in the degrees in which every odd
     multiplicity is < p, in particular in all degrees < p.
  D) Hence the two cohomologies agree in degrees <= p - 2; and they DIFFER in
     degree p - 1 already, which the anchor below exhibits.  The Hypothesis of
     [BGL] ("when p = 0 or n < p") is therefore right about the cochains and
     off by one about the cohomology: H^{p-1} is computed from a differential
     that lands in degree p, where Phi is not invertible.
  E) The anchor: sdim (1|1) with [f,f] = 2e and e central has U(g) = K[f], a
     polynomial algebra, so Ext^n_{U(g)}(K;M) = K, K, 0, 0, ... -- the p > 2
     analog of the (1|1) example of the text for p = 2.  The divided power theory gives exactly
     that for every p; the anti-symmetric one does not.

Usage:  python runodd.py      (writes runodd.log)
"""

import sys

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import backend                                       # noqa: E402

from podd import SLAp, SLAz, ModZ, ACZ, EnvP                       # noqa: E402

LOG = []


def out(s=""):
    print(s)
    LOG.append(s)
    sys.stdout.flush()


def samples():
    return [
        ("sdim (0|1), abelian            (U = Lambda(f))", SLAz(0, 1, {})),
        ("sdim (0|2), abelian            (U = Lambda(f_1,f_2))", SLAz(0, 2, {})),
        ("sdim (1|1), [e,f] = f, [f,f] = 0", SLAz(1, 1, {(0, 1): (0, 1)})),
        ("sdim (1|1), [f,f] = 2e, e central  (U = K[f])",
         SLAz(1, 1, {(1, 1): (2, 0)})),
        ("sdim (1|2), [f_1,f_2] = e, e central",
         SLAz(1, 2, {(1, 2): (1, 0, 0)})),
    ]


def first_diff(a, b):
    for n in range(min(len(a), len(b))):
        if a[n] != b[n]:
            return n
    return None


def main():
    out("runodd.py -- p > 2: the two lattices and the Hypothesis of [BGL, Sec. 4]")
    out("17 August 2026;  exact over Z and mod p;  linear algebra backend: %s"
        % backend())
    out()

    # ---------------------------------------------------- A) U(g) and PBW
    out("=" * 78)
    out("A)  U(g) for p > 2: the axioms and PBW")
    out("=" * 78)
    for p in (3, 5):
        for (nm, gz) in samples():
            g = gz.mod(p)
            ax = g.check_axioms()
            env = EnvP(g)
            ass = env.check_associativity(3)
            md = g.adjoint().check_module()
            out("   p = %d  %-44s axioms: %-4s  U(g) associative on all words "
                "of length <= 4: %-4s  adjoint restricted: %s"
                % (p, nm, "OK" if not ax else "BAD",
                   "OK" if ass == 0 else "BAD (%d)" % ass,
                   "OK" if md == 0 else "BAD"))
    out()

    # ------------------------------------ B), C), D) the two lattices
    out("=" * 78)
    out("B), C), D)  the two lattices, the map Phi = diag(prod a_j!), and the")
    out("            comparison of the two cohomologies")
    out("=" * 78)
    maxdeg = 6
    for (nm, gz) in samples():
        out("-" * 78)
        out(nm)
        for (mn, M) in [("K", ModZ.trivial(gz)), ("g", ModZ.adjoint(gz))]:
            ac = ACZ(gz, M)
            nonint = 0
            dims = []
            for n in range(maxdeg + 1):
                rows, ns, nd, bad = ac.dmat_dp(n)
                nonint += bad
                dims.append(ns)
            d2z = ac.dsquared(maxdeg - 1)
            d2dp = ac.dsquared(maxdeg - 1, divided=True)
            out("   M = %-2s dim C^n = %s" % (mn, dims))
            out("      d^2 = 0 over Z: anti-symmetric %s, divided power %s;  "
                "non-integral entries of the divided power d: %d"
                % ("OK" if d2z == 0 else "BAD", "OK" if d2dp == 0 else "BAD",
                   nonint))
            for p in (3, 5, 7):
                ah = ac.cohom(maxdeg, p, divided=False)
                dp = ac.cohom(maxdeg, p, divided=True)
                phi = [ac.phi_rank(n, p) for n in range(maxdeg + 1)]
                inv = [n for n in range(maxdeg + 1) if phi[n][0] == phi[n][1]]
                fd = first_diff(ah, dp)
                out("      p = %d  AH^n = %-24s dpH^n = %-24s Phi bijective in "
                    "degrees %s; first difference: %s"
                    % (p, ah, dp, inv,
                       "none up to %d" % (maxdeg - 1) if fd is None
                       else "degree %d" % fd))
    out()

    # ------------------------------------------------------- E) the anchor
    out("=" * 78)
    out("E)  the anchor: U(g) = K[f], so Ext^n_{U(g)}(K;M) = M, M, 0, 0, ...")
    out("=" * 78)
    gz = SLAz(1, 1, {(1, 1): (2, 0)})
    for (mn, M) in [("K", ModZ.trivial(gz)), ("g", ModZ.adjoint(gz))]:
        ac = ACZ(gz, M)
        for p in (3, 5, 7, 11):
            dp = ac.cohom(7, p, divided=True)
            ah = ac.cohom(7, p, divided=False)
            # as a K[f]-module, K is trivial and g is K[f]/(f^2): in both cases
            # Ext^0 = Ext^1 = K and Ext^n = 0 for n >= 2
            exp = [1, 1] + [0] * 5
            out("   M = %-2s p = %-3d dpH^n = %-22s (Ext = %s): %s;   AH^n = %s"
                % (mn, p, dp, exp, "AGREE" if dp == exp else "MISMATCH", ah))

    with open("runodd.log", "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
