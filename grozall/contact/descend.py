"""Does an invariant operator over k(2n+1) descend to one over k(2n-1)?

This is the missing half of Problem "Higher dimensions, without a bound".  The half
already verified (restrict.py) is that the generating functions free of p_n, q_n form a
graded subalgebra of k(2n+1) isomorphic to k(2n-1), on which the frame restricts.  The
question here is whether the k_1-CONDITION restricts with it -- i.e., whether a solution
vector, read in the scheme basis of Proposition "The complete contractions span" (which
does not depend on n), is again a solution one dimension down.  If it is, then

    K(n) subset K(n-1)   in scheme coordinates,

and the census at n = 2 settles every n >= 2 at once, exactly as Sect. 14 of Grozman's
detailed proof settles every dimension from dimensions 1 and 2.

The two maps.  Let N = {p_n = q_n = 0}, a contact submanifold of M, invariant under the
subalgebra, and let

    pi : M -> M',  (t,p,q) |-> (t, p_{<n}, q_{<n}),

which is k(2n-1)-projectable (the only term of K_f that is not is (1/2) f_t (p_n d_{p_n}
+ q_n d_{q_n}), and that one is pi-vertical).  Then

    pi^*  : T'(k,lam) -> T(k,lam)      a field of M' read on M, with no dependence
                                       on p_n, q_n and no fiber index 2n-1, 2n;
    iota^*: T(k,lam) -> T'(k,lam)      restrict to N and drop the fiber indices 2n-1, 2n.

iota^* IS equivariant: e_a for a <= 2n-2 is tangent to N, K_f is tangent to N and
restricts to K'_f there, and C_a^b for a <= 2n-2 < b is a multiple of p_n or q_n.
Whether pi^* is equivariant is exactly what is in question: C_{2n-1}^b for b <= 2n-2 is
a multiple of q_n, which is not zero on M, and the composite iota^* B (pi^* . , pi^* .)
can differentiate that q_n away.  So the question is decided here by computation, not by
inspection: everything below is exact rational arithmetic.

Jobs:
  rho     the restriction matrix rho on the scheme basis, for one configuration
  equiv   is pi^* equivariant?  (the failure term, printed)
  descent K(n) vs K(n-1) in scheme coordinates, over a grid of weights
  scan    descent over several configurations at once
"""

import os, sys
import time
from fractions import Fraction as Fr

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import nullspace, rank, solve  # noqa: E402

from contactn import Contact, Module, fiber_monomials  # noqa: E402
from genn import (Problem, schemes, scheme_name, expand, eval_terms,  # noqa: E402
                  target_weight, rand_field)

LOG = None


def say(s):
    print(s, flush=True)
    if LOG:
        LOG.write(s + "\n")
        LOG.flush()


# ------------------------------------------------------------------ the maps
class Descent:
    """The pair (pi^*, iota^*) between k(2n-1) and k(2n+1), n >= 2."""

    def __init__(self, n):
        assert n >= 2
        self.n = n
        self.big = Contact(n)
        self.small = Contact(n - 1)

    # exponent tuples: 0 = t, 1..n = p, n+1..2n = q
    def up_exp(self, e):
        """Small (2n-1 slots) -> big (2n+1 slots), inserting p_n = q_n = 0."""
        m = self.n - 1
        return (e[0],) + e[1:m + 1] + (0,) + e[m + 1:2 * m + 1] + (0,)

    def down_exp(self, e):
        """Big -> small; None if the monomial contains p_n or q_n."""
        n = self.n
        if e[n] or e[2 * n]:
            return None
        return (e[0],) + e[1:n] + e[n + 1:2 * n]

    def up_fib(self, mu):
        """Fiber monomial of length 2n-2 -> length 2n."""
        return tuple(mu) + (0, 0)

    def down_fib(self, mu):
        if mu[-1] or mu[-2]:
            return None
        return tuple(mu[:-2])

    def pull(self, F):
        """pi^*: a field of the small manifold, read on the big one."""
        return {(self.up_exp(e), self.up_fib(mu)): v for (e, mu), v in F.items()}

    def restrict(self, F):
        """iota^*: restrict to N and to the small fiber."""
        out = {}
        for (e, mu), v in F.items():
            ee, mm = self.down_exp(e), self.down_fib(mu)
            if ee is None or mm is None:
                continue
            key = (ee, mm)
            nv = out.get(key, Fr(0)) + v
            if nv:
                out[key] = nv
            elif key in out:
                del out[key]
        return out


# --------------------------------------------------- is pi^* equivariant?
def job_equiv(n, k=2, lam=Fr(1, 3), deg=2):
    """L_{K_f} pi^* F  -  pi^* L_{K'_f} F  for f in the small algebra."""
    D = Descent(n)
    Mb, Ms = Module(D.big, k, lam), Module(D.small, k, lam)
    say("=== pi^* equivariance, k(%d) -> k(%d), T(%d,%s) ==="
        % (2 * n - 1, 2 * n + 1, k, lam))
    bad = []
    fs = []
    for i in (0, 1, 2):
        fs += D.small.k_piece(i)
    Fs = [rand_field(D.small, k, deg, 40 + i) for i in range(3)]
    for f in fs:
        fbig = {D.up_exp(e): v for e, v in f.items()}
        for F in Fs:
            lhs = Mb.act(fbig, D.pull(F))
            rhs = D.pull(Ms.act(f, F))
            d = dict(lhs)
            for key, v in rhs.items():
                d[key] = d.get(key, Fr(0)) - v
            d = {a: b for a, b in d.items() if b}
            if d:
                bad.append((f, d))
    if not bad:
        say("  pi^* IS equivariant on every test  (%d generators, %d fields)"
            % (len(fs), len(Fs)))
        return True
    say("  pi^* is NOT equivariant: %d of %d (generator, field) pairs fail"
        % (len(bad), len(fs) * len(Fs)))
    f, d = bad[0]
    say("  the first failure, f = %s:" % _pname(D.small, f))
    for key, v in sorted(d.items())[:6]:
        say("      %s * %s * u^%s" % (v, _mname(D.big, key[0]), list(key[1])))
    # WHERE it sits: every failing term should carry p_n or q_n
    off = 0
    for f, d in bad:
        for (e, mu) in d:
            if not (e[n] or e[2 * n]):
                off += 1
    say("  terms of the failure free of p_%d and q_%d: %d  (so the failure vanishes ON N"
        " iff this is 0)" % (n, n, off))
    return False


def _mname(alg, e):
    n = alg.n
    parts = []
    if e[0]:
        parts.append("t^%d" % e[0] if e[0] > 1 else "t")
    for i in range(1, n + 1):
        if e[i]:
            parts.append("p%d^%d" % (i, e[i]) if e[i] > 1 else "p%d" % i)
    for i in range(1, n + 1):
        if e[n + i]:
            parts.append("q%d^%d" % (i, e[n + i]) if e[n + i] > 1 else "q%d" % i)
    return "*".join(parts) or "1"


def _pname(alg, f):
    return "+".join("%s%s" % ("" if v == 1 else str(v) + "*", _mname(alg, e))
                    for e, v in sorted(f.items()))


# ------------------------------------------------- the restriction matrix rho
def _fieldset(alg, k, deg, seeds):
    return [rand_field(alg, k, deg, s) for s in seeds]


def _vector(op_vals, keys):
    return [op_vals.get(kk, Fr(0)) for kk in keys]


class Rho:
    """The matrix of iota^* s_n (pi^* . , pi^* .) in the scheme basis at n-1."""

    def __init__(self, n, k1, k2, k3, W, deg=None, nfields=3):
        self.D = Descent(n)
        self.n, self.k1, self.k2, self.k3, self.W = n, k1, k2, k3, W
        self.schemes = schemes(k1, k2, k3, W)
        big, small = self.D.big, self.D.small
        self.exp_big = [expand(big, s) for s in self.schemes]
        self.exp_small = [expand(small, s) for s in self.schemes]
        keep = [i for i in range(len(self.schemes))
                if self.exp_big[i] or self.exp_small[i]]
        self.schemes = [self.schemes[i] for i in keep]
        self.exp_big = [self.exp_big[i] for i in keep]
        self.exp_small = [self.exp_small[i] for i in keep]
        if deg is None:
            deg = W + 1
        self.Fs = _fieldset(small, k1, deg, [11 + i for i in range(nfields)])
        self.Gs = _fieldset(small, k2, deg, [71 + i for i in range(nfields)])

    def _values(self, which):
        """which = 'big' (restricted) or 'small'; one column per scheme."""
        cols = []
        keys = set()
        for j in range(len(self.schemes)):
            d = {}
            for a, F in enumerate(self.Fs):
                for b, G in enumerate(self.Gs):
                    if which == "big":
                        terms = self.exp_big[j]
                        if not terms:
                            continue
                        v = self.D.restrict(
                            eval_terms(self.D.big, terms, self.D.pull(F),
                                       self.D.pull(G)))
                    else:
                        terms = self.exp_small[j]
                        if not terms:
                            continue
                        v = eval_terms(self.D.small, terms, F, G)
                    for kk, vv in v.items():
                        if vv:
                            d[(a, b) + kk] = vv
            cols.append(d)
            keys |= set(d)
        return cols, sorted(keys)

    def matrix(self):
        """rho with  iota^* s_n (pi^*,pi^*) = sum_s' rho[s',s] s'_{n-1}.

        Returns (rho, ok) where ok says the expansion exists and is unique.
        """
        colsB, kB = self._values("big")
        colsS, kS = self._values("small")
        keys = sorted(set(kB) | set(kS))
        A = [[c.get(kk, Fr(0)) for c in colsS] for kk in keys]   # small schemes
        m = len(self.schemes)
        rho = [[Fr(0)] * m for _ in range(m)]
        unique = rank(A) == m if A else (m == 0)
        for j in range(m):
            b = [colsB[j].get(kk, Fr(0)) for kk in keys]
            x = solve(A, b)
            if x is None:
                return None, False, unique
            for i in range(m):
                rho[i][j] = x[i]
        return rho, True, unique


def _ecount(s):
    """The number of frame derivatives e in a scheme monomial."""
    per = {"Pi": 2, "ThF": 1, "ThG": 1, "DvF": 1, "DvG": 1, "EeF": 1, "EeG": 1}
    return sum(per.get(nm, 0) * m for nm, m in s.items())


def job_rho(n, k1, k2, k3, W):
    say("=== rho for k(%d) -> k(%d), (k1,k2,k3)=(%d,%d,%d), W=%d ==="
        % (2 * n + 1, 2 * n - 1, k1, k2, k3, W))
    Rh = Rho(n, k1, k2, k3, W)
    names = [scheme_name(s) for s in Rh.schemes]
    say("  %d schemes: %s" % (len(names), ", ".join(names)))
    rho, ok, unique = Rh.matrix()
    if not ok:
        say("  the restriction of some scheme is NOT in the span of the schemes"
            " one dimension down")
        return None
    say("  the small schemes are independent: %s" % unique)
    ident = True
    tri = True
    for j, s in enumerate(Rh.schemes):
        row = []
        for i, s2 in enumerate(Rh.schemes):
            if rho[i][j]:
                row.append("%s*%s" % (rho[i][j], names[i]))
                if i != j and _ecount(Rh.schemes[i]) >= _ecount(s):
                    tri = False
                if i != j:
                    ident = False
                elif rho[i][j] != 1:
                    ident = False
        say("    %-22s |-> %s" % (names[j], " + ".join(row) or "0"))
    say("  rho = identity: %s;  unitriangular in the number of frame derivatives: %s"
        % (ident, tri))
    return rho


# ------------------------------------------------------------- the descent
def kernel(n, k1, l1, k2, l2, k3, W, vpieces=(0, 1, 2)):
    """The solution space in SCHEME coordinates, and the operator dimension."""
    P = Problem(n, k1, k2, k3, W)
    if not P.schemes:
        return [], 0, []
    l3, ns = P.solve(l1, l2, vpieces=vpieces)
    return ns, P.op_dim(ns, deg=2), P.schemes


def _align(ns, sch_from, sch_to):
    """Re-index a solution vector from one scheme list to another (same labels)."""
    idx = {scheme_name(s): i for i, s in enumerate(sch_to)}
    out = []
    for v in ns:
        w = [Fr(0)] * len(sch_to)
        okv = True
        for c, s in zip(v, sch_from):
            nm = scheme_name(s)
            if nm in idx:
                w[idx[nm]] += c
            elif c:
                okv = False
        if okv:
            out.append(w)
    return out


def _in_span(vecs, target):
    """Is target in the span of vecs?"""
    if not any(target):
        return True
    if not vecs:
        return False
    A = [[v[i] for v in vecs] for i in range(len(target))]
    return solve(A, list(target)) is not None


def job_descent(n, k1, k2, k3, W, weights=None):
    say("=== descent k(%d) -> k(%d), (k1,k2,k3)=(%d,%d,%d), W=%d ==="
        % (2 * n + 1, 2 * n - 1, k1, k2, k3, W))
    if weights is None:
        weights = [(Fr(0), Fr(0)), (Fr(1, 3), Fr(-1, 3)), (Fr(1), Fr(-1)),
                   (Fr(2, 3), Fr(-1, 5)), (Fr(-1), Fr(-1)), (Fr(1, 2), Fr(3, 7))]
    Rh = Rho(n, k1, k2, k3, W)
    rho, ok, unique = Rh.matrix()
    if not ok:
        say("  rho does not exist -- the restriction leaves the span")
        return
    names = [scheme_name(s) for s in Rh.schemes]
    ident = all(rho[i][j] == (1 if i == j else 0)
                for i in range(len(names)) for j in range(len(names)))
    say("  rho is the identity: %s" % ident)
    allok = True
    for l1, l2 in weights:
        t0 = time.time()
        nsB, dB, schB = kernel(n, k1, l1, k2, l2, k3, W)
        nsS, dS, schS = kernel(n - 1, k1, l1, k2, l2, k3, W)
        vB = _align(nsB, schB, Rh.schemes)
        vS = _align(nsS, schS, Rh.schemes)
        good = True
        for v in vB:
            w = [sum(rho[i][j] * v[j] for j in range(len(v)))
                 for i in range(len(v))]
            if not _in_span(vS, w):
                good = False
        allok = allok and good
        say("  (%s,%s): dim K(%d)=%d (op %d), dim K(%d)=%d (op %d)  "
            "rho K(n) in K(n-1): %s   [%.0f s]"
            % (l1, l2, n, len(vB), dB, n - 1, len(vS), dS, good, time.time() - t0))
    say("  DESCENT HOLDS on every weight tested: %s" % allok)
    return allok


def job_iota(n, k=2, lam=Fr(1, 3), deg=3):
    """iota^* is claimed equivariant; this checks it, as the control on job_equiv."""
    D = Descent(n)
    Mb, Ms = Module(D.big, k, lam), Module(D.small, k, lam)
    say("=== iota^* equivariance, k(%d) -> k(%d), T(%d,%s) ==="
        % (2 * n + 1, 2 * n - 1, k, lam))
    fs = []
    for i in (0, 1, 2):
        fs += D.small.k_piece(i)
    Fs = [rand_field(D.big, k, deg, 900 + i) for i in range(3)]
    bad = 0
    for f in fs:
        fbig = {D.up_exp(e): v for e, v in f.items()}
        for F in Fs:
            d = D.restrict(Mb.act(fbig, F))
            for key, v in Ms.act(f, D.restrict(F)).items():
                d[key] = d.get(key, Fr(0)) - v
            if any(d.values()):
                bad += 1
    say("  %d of %d (generator, field) pairs fail" % (bad, len(fs) * len(Fs)))
    return bad == 0


def job_where(n, k1, k2, k3, W, l1=Fr(0), l2=Fr(0)):
    """WHICH generators of the small algebra the descended operator fails."""
    say("=== where the descent fails, k(%d) -> k(%d), (%d,%d,%d), W=%d, (%s,%s) ==="
        % (2 * n + 1, 2 * n - 1, k1, k2, k3, W, l1, l2))
    D = Descent(n)
    sch = schemes(k1, k2, k3, W)
    expB = [expand(D.big, s) for s in sch]
    l3 = target_weight(k1, l1, k2, l2, k3, W)
    nsB, dB, schB = kernel(n, k1, l1, k2, l2, k3, W)
    if not nsB:
        say("  K(%d) = 0" % n)
        return
    vec = _align(nsB, schB, sch)[0]
    say("  the operator: %s"
        % " + ".join("%s*%s" % (c, scheme_name(s))
                     for c, s in zip(vec, sch) if c))
    Ms1, Ms2 = Module(D.small, k1, l1), Module(D.small, k2, l2)
    Ms3 = Module(D.small, k3, l3)
    Fs = [rand_field(D.small, k1, W + 2, 200 + i) for i in range(2)]
    Gs = [rand_field(D.small, k2, W + 2, 300 + i) for i in range(2)]

    def Bp(F, G):
        acc = {}
        for c, terms in zip(vec, expB):
            if not c or not terms:
                continue
            for kk, vv in eval_terms(D.big, terms, D.pull(F), D.pull(G)).items():
                acc[kk] = acc.get(kk, Fr(0)) + c * vv
        return D.restrict({kk: vv for kk, vv in acc.items() if vv})

    for i in (-2, -1, 0, 1, 2):
        gens = D.small.k_piece(i)
        bad = []
        for f in gens:
            for F in Fs:
                for G in Gs:
                    d = Ms3.act(f, Bp(F, G))
                    for kk, vv in Bp(Ms1.act(f, F), G).items():
                        d[kk] = d.get(kk, Fr(0)) - vv
                    for kk, vv in Bp(F, Ms2.act(f, G)).items():
                        d[kk] = d.get(kk, Fr(0)) - vv
                    if any(d.values()):
                        bad.append(_pname(D.small, f))
                        break
                else:
                    continue
                break
        say("  k_%-2d (%2d generators): %d fail   %s"
            % (i, len(gens), len(bad), ", ".join(sorted(set(bad))[:6])))


def job_direct(n, k1, k2, k3, W, weights=None):
    """Is  B' := iota^* B (pi^* . , pi^* .)  invariant under k(2n-1), for B in K(n)?

    This is the descent itself, tested on the OPERATOR and not on its coefficient
    vector, so that the relations among the schemes one dimension down -- which are
    what makes the vector reading ambiguous for n-1 = 1 -- cannot affect the answer.
    """
    say("=== direct descent k(%d) -> k(%d), (k1,k2,k3)=(%d,%d,%d), W=%d ==="
        % (2 * n + 1, 2 * n - 1, k1, k2, k3, W))
    if weights is None:
        weights = [(Fr(0), Fr(0)), (Fr(1, 3), Fr(-1, 3)), (Fr(1), Fr(-1)),
                   (Fr(2, 3), Fr(-1, 5))]
    D = Descent(n)
    sch = schemes(k1, k2, k3, W)
    expB = [expand(D.big, s) for s in sch]
    for l1, l2 in weights:
        l3 = target_weight(k1, l1, k2, l2, k3, W)
        nsB, dB, schB = kernel(n, k1, l1, k2, l2, k3, W)
        if not nsB:
            say("  (%s,%s): K(%d) = 0, nothing to descend" % (l1, l2, n))
            continue
        vB = _align(nsB, schB, sch)
        Ms1 = Module(D.small, k1, l1)
        Ms2 = Module(D.small, k2, l2)
        Ms3 = Module(D.small, k3, l3)
        gens = []
        for i in (0, 1, 2):
            gens += D.small.k_piece(i)
        Fs = [rand_field(D.small, k1, W + 2, 200 + i) for i in range(2)]
        Gs = [rand_field(D.small, k2, W + 2, 300 + i) for i in range(2)]

        def Bp(vec, F, G):
            acc = {}
            for c, terms in zip(vec, expB):
                if not c or not terms:
                    continue
                for kk, vv in eval_terms(D.big, terms, D.pull(F),
                                         D.pull(G)).items():
                    acc[kk] = acc.get(kk, Fr(0)) + c * vv
            return D.restrict({kk: vv for kk, vv in acc.items() if vv})

        nz, bad = 0, 0
        for vec in vB:
            zero = True
            for F in Fs:
                for G in Gs:
                    if Bp(vec, F, G):
                        zero = False
            if zero:
                continue
            nz += 1
            for f in gens:
                for F in Fs:
                    for G in Gs:
                        d = Ms3.act(f, Bp(vec, F, G))
                        for kk, vv in Bp(vec, Ms1.act(f, F), G).items():
                            d[kk] = d.get(kk, Fr(0)) - vv
                        for kk, vv in Bp(vec, F, Ms2.act(f, G)).items():
                            d[kk] = d.get(kk, Fr(0)) - vv
                        if any(d.values()):
                            bad += 1
                            break
                    else:
                        continue
                    break
        say("  (%s,%s): dim K(%d)=%d, of them %d descend to a nonzero operator, "
            "%d of those FAIL invariance" % (l1, l2, n, len(vB), nz, bad))


CONFIGS = [
    (0, 0, 0, 2),      # the densities, the contact-bracket family
    (1, 1, 0, 0),      # the pairing
    (1, 1, 0, 2),      # the first configuration with Dv and Th together
    (2, 2, 0, 0),      # the three contractions
    (2, 2, 0, 2),      # the configuration of kGz
    (2, 0, 0, 2),      # divergence-type
    (1, 1, 2, 2),
]


def job_scan(n):
    for (k1, k2, k3, W) in CONFIGS:
        try:
            job_descent(n, k1, k2, k3, W)
        except Exception as exc:                                  # noqa: BLE001
            say("  *** %s: %s" % ((k1, k2, k3, W), exc))
        say("")


def main():
    global LOG
    job = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    LOG = open("descend-%s-%d.log" % (job, n), "w")
    rest = [int(x) for x in sys.argv[3:]]
    if job == "equiv":
        job_equiv(n, *(rest or []))
    elif job == "rho":
        job_rho(n, *rest)
    elif job == "descent":
        job_descent(n, *rest)
    elif job == "direct":
        job_direct(n, *rest)
    elif job == "iota":
        job_iota(n, *rest)
    elif job == "where":
        job_where(n, *rest)
    elif job == "scan":
        job_scan(n)
    else:
        raise SystemExit("unknown job")


if __name__ == "__main__":
    main()
