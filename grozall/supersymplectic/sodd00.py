"""The three invariant operators of order 0 over a periplectic form, written out.

Statement 11.2 3) of the text counts them -- the space is 3-dimensional for
k = 2 and k = 3, against 1 over an even form -- and names none.  Here they are
named and the naming is checked.

An operator of order 0 differentiates nothing, so it contracts two of the four
fiber slots of the pair (A, A') and leaves two free.  There are three ways to do
that, and each is an operator one can write down:

    T1(A,A') = (ptr A)  A'_{ij}
    T2(A,A') = (ptr A') A_{ij}
    T3(A,A') = sym_{ij} sum_{k,l} omega^{kl} A_{ik} A'_{jl}

with ptr the periplectic trace (26) of the text.  Over an EVEN form ptr vanishes
identically and only T3 is left -- which is the dimension 1 the even case has.

Usage:  python3 sodd00.py [k]
"""

import os
import pickle
import random
import sys
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import rank                                  # noqa: E402

from ssuper import SSetup                                  # noqa: E402
from sgen import enumerate_schemes, evaluate               # noqa: E402
from srun import op_parity, operator                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def ptr(S, T):
    """The periplectic trace sum omega^{kl} d_{u^k} d_{u^l} T, an odd function."""
    A = S.A
    return A.summ([A.scale(S.du(S.du(T, k), l), w) for k, l, w in S.Omnz])


def T1(S, P, Q):
    return S.A.mul(ptr(S, P), Q)


def T2(S, P, Q):
    return S.A.mul(ptr(S, Q), P)


def T3(S, P, Q):
    """sum omega^{kl} (d_{u^k} P)(d_{u^l} Q).

    The operator d_{u^l} has to be moved past the first factor, which costs
    a Koszul sign.  On metrics, which are even, the sign is invisible -- and
    that is exactly the trap Remark rem:trap2 records: the invariance test
    feeds the operator the fields L_X A, which over a periplectic form are odd
    for an even Hamiltonian, and there the sign decides.
    """
    A = S.A
    out = []
    for k, l, w in S.Omnz:
        PP = S.du(P, k)
        if not PP:
            continue
        QQ = S.du(Q, l)
        if not QQ:
            continue
        if S.upar[l] and A.parity(PP):
            w = -w
        out.append(A.scale(A.mul(PP, QQ), w))
    return A.summ(out)


NAMED = [("T1 = (ptr A) A'", T1), ("T2 = (ptr A') A", T2),
         ("T3 = om^{kl} A_ik A'_jl", T3)]


def defect_named(S, fun, f, P, Q, pT):
    A = S.A
    pX = (A.parity(f) + (1 if S.form == "odd" else 0)) % 2
    e = -1 if (pT * pX) % 2 else 1
    lhs = S.lie(f, fun(S, P, Q))
    rhs = A.add(fun(S, S.lie(f, P), Q), fun(S, P, S.lie(f, Q)))
    return A.add(lhs, A.scale(rhs, -e))


def coords(S, vals, keys):
    out = []
    for v, ks in zip(vals, keys):
        for k in ks:
            out.append(v.get(k, Fraction(0)))
    return out


def traceless_fields(S, rng, deg, want=4):
    """Even metric fields with ptr = 0, built from the fiber monomials the
    trace kills -- not by subtracting a trace, which needs a division."""
    A = S.A
    keep = [fib for (i, j), fib in S.fiber_monomials() if not ptr(S, fib)]
    mons = [p for d in range(deg + 1) for p in S.monomials(d)]
    out = []
    for _ in range(want):
        terms = []
        for fib in keep:
            pf = A.parity(fib)
            pool = [p for p in mons if A.parity(p) == pf]
            if not pool:
                continue
            for p in rng.sample(pool, min(3, len(pool))):
                c = rng.randint(-4, 4)
                if c:
                    terms.append(A.scale(A.mul(p, fib), c))
        T = A.summ(terms)
        if T and not ptr(S, T):
            out.append(T)
    return out


def run(k, form):
    m, n = (k, k) if form == "odd" else (k, 0)
    S = SSetup(m, n, form=form)
    schemes = enumerate_schemes(0)
    pT = op_parity(S, 0)
    tag = f"periplectic k = {k}" if form == "odd" else f"even, m = {k}, n = 0"
    print(f"\n=== {tag};  order 0 is "
          f"{'ODD' if pT else 'even'};  {len(schemes)} schemes")

    rng = random.Random(555 + 7 * k + (1 if form == "odd" else 0))
    hams = [p for d in (1, 2, 3) for p in S.monomials(d)]
    probes = [(S.random_field(3, rng, nterms=3), S.random_field(3, rng, nterms=3))
              for _ in range(3)]

    # --- 1) each named operator: invariant?  non-zero?
    live = []
    for (name, fun) in NAMED:
        bad = 0
        nonzero = False
        for (P, Q) in probes:
            if fun(S, P, Q):
                nonzero = True
            for f in hams:
                if defect_named(S, fun, f, P, Q, pT):
                    bad += 1
        print(f"  {name:26s} invariant: {'YES' if not bad else f'NO ({bad})'}"
              f"   non-zero: {'YES' if nonzero else 'NO '}"
              f"   ({len(hams)*len(probes)} checks)")
        if nonzero:
            live.append((name, fun))

    # --- 2) are they independent, and do they span the invariant space?
    keys = []
    for (P, Q) in probes:
        s = set()
        for (name, fun) in NAMED:
            s |= set(fun(S, P, Q))
        keys.append(sorted(s))
    rowsN = [coords(S, [fun(S, P, Q) for (P, Q) in probes], keys)
             for (name, fun) in NAMED]
    rN = rank(rowsN)
    print(f"  the three named operators span a space of dimension {rN}")

    path = os.path.join(HERE, f"sgen_{form}_m{m}_n{n}_N0.pkl")
    if not os.path.exists(path):
        print(f"  (no census file {os.path.basename(path)}; span not compared)")
        return
    with open(path, "rb") as fh:
        D = pickle.load(fh)
    kernel = [[Fraction(c) for c in v] for v in D["kernel"]]
    rowsK = []
    for v in kernel:
        vals = [operator(S, v, schemes, P, Q) for (P, Q) in probes]
        s = [set(x) for x in vals]
        rowsK.append(vals)
    allkeys = []
    for i in range(len(probes)):
        s = set(keys[i])
        for vals in rowsK:
            s |= set(vals[i])
        allkeys.append(sorted(s))
    RK = [coords(S, vals, allkeys) for vals in rowsK]
    RN = [coords(S, [fun(S, P, Q) for (P, Q) in probes], allkeys)
          for (name, fun) in NAMED]
    rk = rank(RK)
    rboth = rank(RK + RN)
    print(f"  the census space has dimension {rk} "
          f"(recorded: {D['dim']}); with the named ones adjoined: {rboth}")
    print(f"  => the named operators {'SPAN it' if rboth == rk and rN == rk else 'do NOT span it'}")

    # --- 3) does everything factor through the trace, as it does in order 2?
    #        Build fields the trace kills and see whether T3 survives them.
    tl = traceless_fields(S, rng, 4)
    print(f"  traceless fields built: {len(tl)}"
          + ("  -- NOTHING TO TEST" if len(tl) < 2 else ""))
    if len(tl) >= 2:
        seen = False
        for a in range(len(tl)):
            for b in range(len(tl)):
                assert not ptr(S, tl[a]) and not ptr(S, tl[b])
                if T3(S, tl[a], tl[b]):
                    seen = True
        print(f"  on ptr-free metrics: T1 = T2 = 0 by construction, "
              f"T3 {'SURVIVES' if seen else 'vanishes too'}")


def main():
    ks = [int(x) for x in sys.argv[1:]] or [2, 3]
    for k in ks:
        run(k, "odd")
    for k in ks:
        run(k, "even")


if __name__ == "__main__":
    main()
