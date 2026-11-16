"""Self-tests of poi.py, then the quadratic Casimir of po(0|2n)."""
import sympy
from poi import Po, Verma, check_algebra, parity

LOG = open("run-poi.log", "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


def casimir_variants(g):
    """Four candidate quadratic Casimirs; only the central one is kept."""
    basis = g.basis
    G = sympy.Matrix([[g.form(a, b) for b in basis] for a in basis])
    Gi, GiT = G.inv(), G.T.inv()
    out = {}
    for name, M in (("Ginv", Gi), ("GinvT", GiT)):
        for rule in ("plain", "parity"):
            pairs = []
            for i in range(len(basis)):
                for j in range(len(basis)):
                    c = M[i, j]
                    if c:
                        s = c * ((-1) ** parity(basis[i]) if rule == "parity" else 1)
                        pairs.append((basis[i], basis[j], s))
            out[f"{name}/{rule}"] = pairs
    return out


def casimir_on(V, word, pairs, memo=None):
    if memo is None:
        memo = {}
    tot = {}
    for a, b, c in pairs:
        for w, cc in V.act_word([a, b], {word: sympy.Integer(1)}, memo).items():
            tot[w] = tot.get(w, 0) + c * cc
    return {k: sympy.expand(v) for k, v in tot.items() if sympy.expand(v) != 0}


for n in (2, 3):
    say(f"\n{'='*66}\n=== po(0|{2*n}) ===")
    g = Po(n)
    if n == 2:
        say("structural checks:")
        say("  all pass:", check_algebra(g))
    V = Verma(g)
    say(f"dim g = {g.dim},  dim n_- = {len(V.neg)},  dim h = {len(V.zero)}")
    say("weights of n_-:", sorted({V.env.wt[i] for i in range(len(V.neg))}))

    # test words: a few genuine PBW basis vectors of low weight
    tests = []
    for nu in ([(0, 1)], [(1, 0)], [(1, 1)], [(2, 0)])[: 4 if n == 2 else 2]:
        for t in V.env.basis_of_weight(nu[0] if isinstance(nu, list) else nu):
            tests.append(t)
    tests = [()] + tests[:6]
    say(f"centrality tested on {len(tests)} vectors of M(lambda)")

    for name, pairs in casimir_variants(g).items():
        memo = {}
        hc = casimir_on(V, (), pairs, memo).get((), 0)
        central = True
        for w in tests:
            out = casimir_on(V, w, pairs, memo)
            if len(out) != 1 or sympy.expand(out.get(w, 0) - hc) != 0:
                central = False
                break
        flag = "CENTRAL" if central else "not central"
        say(f"  {name:12s} HC(C) = {sympy.factor(hc)}   [{flag}]")

LOG.close()
