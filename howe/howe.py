#!/usr/bin/env python3
"""Exact engine for the flat model of the symplectic and periplectic Howe pairs.

The flat model of Section 2: the superalgebra of differential
forms on a symplectic supermanifold of superdimension (2n|l),

    Omega = K[q,p] (x) Lambda(qhat,phat) (x) Lambda(theta) (x) K[thetahat],

with p(yhat) = p(y)+1, so qhat,phat are odd and thetahat is even.  Every
operator below is a polynomial-coefficient differential operator in the
generators; the arithmetic is exact (Fraction), hbar is a rational parameter,
and operator identities are certified by evaluating both sides on every basis
monomial of bounded degree.

What is checked (the command line of this file runs (1)-(5) on one model, the
second part of (5) for l = 0,...,3; run_howe.py runs all of them):

  (1) [X^+,X^-] = H, i.e. which of the two sign conventions for X^- Bernstein's
      Theorem uses, and that H is (form degree - n);
  (2) the osp(1|2) relations for the explicit D^+, D^- of eq. (10) --
      with the exact hbar-dependence of every structure constant;
  (3) that D^- = [X^-,D^+], as Subsec. 2.5 defines it;
  (4) that the po(2n)-action rho_hbar(f) = L_{H_f} + hbar*(E(f)/2 - f) on the
      twisted forms commutes with all five operators -- the commuting pair of
      Theorem 2.5.1 -- and that the constant f=1 acts by -hbar;
  (5) at hbar = 0 the same five operators span the contraction: d^2 = 0,
      delta^2 = 0, [d,delta] = ?  -- the (a=0) equation
      of [LKae], checked for l = 0,1,2,3;
  (6) the contact reading: with alpha = dt + (hbar/2) sum (p dq - q dp) on the
      contactization (alpha = dt + sum p dq in run_howe.py), L_{K_f} alpha = (d f/d t) alpha, so that alpha^ commutes
      with the strict contactomorphisms po(2n) = {f | df/dt = 0} and no more;
  (7) the centralizer of the po(2n)-action inside a bounded space of operators,
      for hbar != 0 and for hbar = 0 -- the quantitative form of "only
      a portion of g is used".

Usage:  python3 howe.py [n] [l] [degree]      
"""
import itertools
import sys
from fractions import Fraction as F

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]


# ---------------------------------------------------------------- generators

class Model:
    """Names, parities and indices of the generators of Omega."""

    def __init__(self, n, l, contact=False):
        self.n, self.l, self.contact = n, l, contact
        self.ev, self.od = [], []          # names of even / odd generators
        self.q, self.p, self.th = [], [], []
        self.qh, self.ph, self.thh = [], [], []
        for i in range(n):
            self.q.append(self._add("q%d" % i, 0))
            self.p.append(self._add("p%d" % i, 0))
        for j in range(l):
            self.th.append(self._add("th%d" % j, 1))
        if contact:
            self.t = self._add("t", 0)
        for i in range(n):
            self.qh.append(self._add("qh%d" % i, 1))
            self.ph.append(self._add("ph%d" % i, 1))
        for j in range(l):
            self.thh.append(self._add("thh%d" % j, 0))
        if contact:
            self.th_t = self._add("th_t", 1)      # dt

    def _add(self, name, par):
        """Register a generator; return the pair (parity, index)."""
        if par == 0:
            self.ev.append(name)
            return (0, len(self.ev) - 1)
        self.od.append(name)
        return (1, len(self.od) - 1)

    def one(self):
        return ((0,) * len(self.ev), ())

    def basis(self, deg):
        """All monomials of total degree <= deg."""
        out = []
        ne, no = len(self.ev), len(self.od)
        for k in range(no + 1):
            for odd in itertools.combinations(range(no), k):
                if k > deg:
                    continue
                for e in _compositions(ne, deg - k):
                    out.append((e, odd))
        return out


def _compositions(nvar, total):
    """All exponent tuples of length nvar with sum <= total."""
    if nvar == 0:
        yield ()
        return
    for first in range(total + 1):
        for rest in _compositions(nvar - 1, total - first):
            yield (first,) + rest


# ------------------------------------------------------------------- vectors
# A vector is a dict {monomial: Fraction}; a monomial is (exponents, odd_tuple).

def vadd(v, w, c=1):
    for m, a in w.items():
        s = v.get(m, F(0)) + c * a
        if s:
            v[m] = s
        elif m in v:
            del v[m]
    return v


def apply_factor(fac, vec):
    """Apply one elementary factor to a vector."""
    kind = fac[0]
    out = {}
    if kind == "poly":                     # multiply by the monomial fac[1]
        pe = fac[1]
        for (e, o), c in vec.items():
            out[(tuple(a + b for a, b in zip(e, pe)), o)] = out.get(
                (tuple(a + b for a, b in zip(e, pe)), o), F(0)) + c
        return {m: c for m, c in out.items() if c}
    par, idx = fac[1]
    if kind == "mul":
        if par == 0:
            for (e, o), c in vec.items():
                ee = list(e)
                ee[idx] += 1
                m = (tuple(ee), o)
                out[m] = out.get(m, F(0)) + c
        else:
            for (e, o), c in vec.items():
                if idx in o:
                    continue
                pos = sum(1 for j in o if j < idx)
                oo = tuple(sorted(o + (idx,)))
                m = (e, oo)
                out[m] = out.get(m, F(0)) + (c if pos % 2 == 0 else -c)
    elif kind == "der":
        if par == 0:
            for (e, o), c in vec.items():
                if e[idx] == 0:
                    continue
                ee = list(e)
                k = ee[idx]
                ee[idx] -= 1
                m = (tuple(ee), o)
                out[m] = out.get(m, F(0)) + c * k
        else:
            for (e, o), c in vec.items():
                if idx not in o:
                    continue
                pos = o.index(idx)
                oo = tuple(j for j in o if j != idx)
                m = (e, oo)
                out[m] = out.get(m, F(0)) + (c if pos % 2 == 0 else -c)
    else:
        raise ValueError(kind)
    return {m: c for m, c in out.items() if c}


class Op:
    """A differential operator: a list of (coefficient, factors) terms.

    Factors are applied right to left, as in the composition they denote."""

    def __init__(self, terms=(), parity=0, name="?"):
        self.terms = list(terms)
        self.parity = parity
        self.name = name

    def __call__(self, vec):
        out = {}
        for c, facs in self.terms:
            v = dict(vec)
            for fac in reversed(facs):
                v = apply_factor(fac, v)
                if not v:
                    break
            vadd(out, v, c)
        return out

    def __add__(self, other):
        return Op(self.terms + other.terms, self.parity,
                  "%s+%s" % (self.name, other.name))

    def scaled(self, c):
        return Op([(c * a, f) for a, f in self.terms], self.parity,
                  "%s*%s" % (c, self.name))


def collect(terms):
    """Sum the coefficients of identical factor tuples; drop the zeros.

    Without this the repeated brackets of the closure computations grow
    factorially in the number of terms."""
    acc = {}
    for c, f in terms:
        acc[f] = acc.get(f, F(0)) + c
    return [(c, f) for f, c in acc.items() if c]


def compose(A, B):
    return Op(collect([(a * b, fa + fb) for a, fa in A.terms for b, fb in B.terms]),
              (A.parity + B.parity) % 2, "%s.%s" % (A.name, B.name))


def bracket(A, B):
    sign = -1 if (A.parity and B.parity) else 1
    return Op(collect(compose(A, B).terms
                      + [(-sign * c, f) for c, f in compose(B, A).terms]),
              (A.parity + B.parity) % 2, "[%s,%s]" % (A.name, B.name))


def is_zero(A, mod, deg):
    return all(not A({m: F(1)}) for m in mod.basis(deg))


def equal(A, B, mod, deg):
    for m in mod.basis(deg):
        v = A({m: F(1)})
        w = B({m: F(1)})
        if v != w:
            return False, m, v, w
    return True, None, None, None


# ---------------------------------------------------------------- operators

def ops(mod, hbar):
    """The operators of Subsecs. 2.1.1 and 2.6 in the flat model."""
    n, l = mod.n, mod.l
    half = F(hbar, 2) if isinstance(hbar, int) else hbar / 2

    # d = sum yhat d/dy over all coordinates
    d_terms = []
    for y, yh in list(zip(mod.q, mod.qh)) + list(zip(mod.p, mod.ph)) + \
            list(zip(mod.th, mod.thh)):
        d_terms.append((F(1), (("mul", yh), ("der", y))))
    d = Op(d_terms, 1, "d")

    # X^+ = multiplication by omega = sum qhat_i phat_i + (1/2) sum thetahat_j^2
    xp_terms = [(F(1), (("mul", mod.qh[i]), ("mul", mod.ph[i]))) for i in range(n)]
    xp_terms += [(F(1, 2), (("mul", mod.thh[j]), ("mul", mod.thh[j]))) for j in range(l)]
    Xp = Op(xp_terms, 0, "X+")

    # X^- : the convolution with the dual bivector; the sign convention is
    # determined below by [X+,X-] = H.
    xm_terms = [(F(1), (("der", mod.ph[i]), ("der", mod.qh[i]))) for i in range(n)]
    xm_terms += [(F(1, 2), (("der", mod.thh[j]), ("der", mod.thh[j]))) for j in range(l)]
    Xm = Op(xm_terms, 0, "X-")

    # H = [X+,X-]; the expected form is (form degree - n)
    H = bracket(Xp, Xm)
    formdeg = Op([(F(1), (("mul", yh), ("der", yh)))
                  for yh in mod.qh + mod.ph + mod.thh], 0, "deg")
    Hexp = Op(formdeg.terms + [(F(-n), ())], 0, "deg-n")

    # D^+ = d - (hbar/2)( sum (qhat_i p_i - phat_i q_i) + sum thetahat_j theta_j )
    dp_terms = list(d.terms)
    for i in range(n):
        dp_terms.append((-half, (("mul", mod.qh[i]), ("mul", mod.p[i]))))
        dp_terms.append((half, (("mul", mod.ph[i]), ("mul", mod.q[i]))))
    for j in range(l):
        dp_terms.append((-half, (("mul", mod.thh[j]), ("mul", mod.th[j]))))
    Dp = Op(dp_terms, 1, "D+")

    # D^- = sum (d/dphat d/dq - d/dqhat d/dp) + sum d/dthetahat d/dtheta
    #        - (hbar/2)( sum (p d/dphat + q d/dqhat) + sum theta d/dthetahat )
    dm_terms = []
    for i in range(n):
        dm_terms.append((F(1), (("der", mod.ph[i]), ("der", mod.q[i]))))
        dm_terms.append((F(-1), (("der", mod.qh[i]), ("der", mod.p[i]))))
    for j in range(l):
        dm_terms.append((F(1), (("der", mod.thh[j]), ("der", mod.th[j]))))
    for i in range(n):
        dm_terms.append((-half, (("mul", mod.p[i]), ("der", mod.ph[i]))))
        dm_terms.append((-half, (("mul", mod.q[i]), ("der", mod.qh[i]))))
    for j in range(l):
        dm_terms.append((-half, (("mul", mod.th[j]), ("der", mod.thh[j]))))
    Dm = Op(dm_terms, 1, "D-")

    delta = Op([t for t in dm_terms if t[0] in (F(1), F(-1))], 1, "delta")
    return dict(d=d, delta=delta, Xp=Xp, Xm=Xm, H=H, Hexp=Hexp, Dp=Dp, Dm=Dm)


# ------------------------------------------------------- the po(2n)-action

def poly_ham(mod, f):
    """Components of the Hamiltonian field H_f = sum (f_q d/dp - f_p d/dq).

    f is a dict {exponent tuple over the even generators: coefficient}."""
    comp = {}                                   # generator -> {monomial: coeff}
    for i in range(mod.n):
        qi, pi = mod.q[i], mod.p[i]
        comp.setdefault(pi, {})
        comp.setdefault(qi, {})
        for e, c in f.items():
            k = e[qi[1]]
            if k:
                ee = list(e)
                ee[qi[1]] -= 1
                vadd(comp[pi], {tuple(ee): c * k})
            k = e[pi[1]]
            if k:
                ee = list(e)
                ee[pi[1]] -= 1
                vadd(comp[qi], {tuple(ee): -c * k})
    return comp


def rho(mod, f, hbar):
    """The prequantum action of f in po(2n) on the twisted forms:
       rho(f) = L_{H_f} + hbar*(E(f)/2 - f), as in eq. (13)."""
    comp = poly_ham(mod, f)
    terms = []
    for gen, coefs in comp.items():
        for e, c in coefs.items():
            terms.append((c, (("poly", e), ("der", gen))))
            # the form part: sum_j yhat_j d(f_i)/dy_j d/dyhat_i
            for gen2 in [g for g in (mod.q + mod.p) if g[0] == 0]:
                k = e[gen2[1]]
                if not k:
                    continue
                ee = list(e)
                ee[gen2[1]] -= 1
                hat2 = _hat(mod, gen2)
                hat1 = _hat(mod, gen)
                terms.append((c * k, (("mul", hat2), ("poly", tuple(ee)),
                                      ("der", hat1))))
    for e, c in f.items():
        deg = sum(e)
        terms.append((c * hbar * (F(deg, 2) - 1), (("poly", e),)))
    return Op(terms, 0, "rho(f)")


def _hat(mod, gen):
    for i in range(mod.n):
        if gen == mod.q[i]:
            return mod.qh[i]
        if gen == mod.p[i]:
            return mod.ph[i]
    for j in range(mod.l):
        if gen == mod.th[j]:
            return mod.thh[j]
    raise KeyError(gen)


def gens_po(mod, maxdeg):
    """All monomials in q,p of degree <= maxdeg, as elements of po(2n)."""
    out = []
    ne = len(mod.ev)
    for e in _compositions(ne, maxdeg):
        if any(e[g[1]] for g in mod.th):        # even part only
            continue
        out.append({e: F(1)})
    return out


# ---------------------------------------------------------------- the checks

def check_sl2(mod, deg, verbose=True):
    o = ops(mod, 0)
    ok, m, v, w = equal(o["H"], o["Hexp"], mod, deg)
    if verbose:
        print("  [X+,X-] = (form degree) - n :", "OK" if ok else "FAILS %s" % (m,))
    return ok


def check_osp(mod, hbar, deg):
    """The osp(1|2)-relations, with the exact hbar-dependence."""
    o = ops(mod, hbar)
    Xp, Xm, H, Dp, Dm = o["Xp"], o["Xm"], o["H"], o["Dp"], o["Dm"]
    res = {}
    res["[D+,D+] =  2h X+"] = equal(bracket(Dp, Dp), Xp.scaled(2 * hbar), mod, deg)[0]
    res["[D-,D-] = -2h X-"] = equal(bracket(Dm, Dm), Xm.scaled(-2 * hbar), mod, deg)[0]
    res["[D+,D-] =   -h H "] = equal(bracket(Dp, Dm), H.scaled(-hbar), mod, deg)[0]
    res["[H,D+] =     D+  "] = equal(bracket(H, Dp), Dp, mod, deg)[0]
    res["[H,D-] =    -D-  "] = equal(bracket(H, Dm), Dm.scaled(-1), mod, deg)[0]
    res["[X+,D-] =    D+  "] = equal(bracket(Xp, Dm), Dp, mod, deg)[0]
    res["[X-,D+] =    D-  "] = equal(bracket(Xm, Dp), Dm, mod, deg)[0]
    res["[H,X+] =    2X+  "] = equal(bracket(H, Xp), Xp.scaled(2), mod, deg)[0]
    res["[H,X-] =   -2X-  "] = equal(bracket(H, Xm), Xm.scaled(-2), mod, deg)[0]
    return res


def check_commuting(mod, hbar, deg, fdeg=4):
    """Theorem 2.5.1 in the flat model: [rho(f), A] = 0."""
    o = ops(mod, hbar)
    bad = []
    for f in gens_po(mod, fdeg):
        R = rho(mod, f, hbar)
        for nm in ("Xp", "Xm", "H", "Dp", "Dm"):
            if not is_zero(bracket(R, o[nm]), mod, deg):
                bad.append((f, nm))
    return bad


def check_homomorphism(mod, hbar, deg, fdeg=3):
    """rho is a homomorphism po(2n) -> End(Omega_hbar): [rho(f),rho(g)] =
    rho({f,g}), and the constant 1 acts by -hbar."""
    bad = []
    gs = gens_po(mod, fdeg)
    for f in gs:
        for g in gs:
            pb = poisson(mod, f, g)
            lhs = bracket(rho(mod, f, hbar), rho(mod, g, hbar))
            rhs = rho(mod, pb, hbar)
            if not is_zero(Op(lhs.terms + [(-c, t) for c, t in rhs.terms]), mod, deg):
                bad.append((f, g))
    one = mod.one()[0]
    R1 = rho(mod, {one: F(1)}, hbar)
    scal = Op([(F(-hbar), ())])
    central = is_zero(Op(R1.terms + [(-c, t) for c, t in scal.terms]), mod, deg)
    return bad, central


def poisson(mod, f, g):
    """{f,g} = sum (df/dq dg/dp - df/dp dg/dq)."""
    out = {}
    for i in range(mod.n):
        qi, pi = mod.q[i][1], mod.p[i][1]
        for e, c in f.items():
            for e2, c2 in g.items():
                if e[qi] and e2[pi]:
                    ee = list(e)
                    ee2 = list(e2)
                    k1, k2 = ee[qi], ee2[pi]
                    ee[qi] -= 1
                    ee2[pi] -= 1
                    vadd(out, {tuple(a + b for a, b in zip(ee, ee2)): c * c2 * k1 * k2})
                if e[pi] and e2[qi]:
                    ee = list(e)
                    ee2 = list(e2)
                    k1, k2 = ee[pi], ee2[qi]
                    ee[pi] -= 1
                    ee2[qi] -= 1
                    vadd(out, {tuple(a + b for a, b in zip(ee, ee2)): -c * c2 * k1 * k2})
    return out


def check_contraction(mod, deg):
    """At hbar = 0: the algebra spanned by X^+, X^-, H, d, delta."""
    o = ops(mod, 0)
    d, delta, Xp, Xm, H = o["d"], o["delta"], o["Xp"], o["Xm"], o["H"]
    out = {}
    out["[d,d]"] = is_zero(bracket(d, d), mod, deg)
    out["[delta,delta]"] = is_zero(bracket(delta, delta), mod, deg)
    dd = bracket(d, delta)
    out["[d,delta]"] = is_zero(dd, mod, deg)
    if not out["[d,delta]"]:
        # is it central in the algebra spanned?
        out["[d,delta] central"] = all(
            is_zero(bracket(dd, A), mod, deg) for A in (d, delta, Xp, Xm, H))
    out["[X+,d]"] = is_zero(bracket(Xp, d), mod, deg)
    out["[X-,delta]"] = is_zero(bracket(Xm, delta), mod, deg)
    out["[X-,d] = delta"] = equal(bracket(Xm, d), delta, mod, deg)[0]
    out["[X+,delta] = d"] = equal(bracket(Xp, delta), d, mod, deg)[0]
    return out


# ------------------------------------------------------------------- driver

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    l = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    deg = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    mod = Model(n, l)
    print("model: sdim (%d|%d), monomials of degree <= %d: %d"
          % (2 * n, l, deg, len(mod.basis(deg))))

    print("\n(1) the sl(2)-triple")
    check_sl2(mod, deg)

    print("\n(2) the osp(1|2)-relations of eq. (5), for the explicit D+, D-")
    for hbar in (F(1), F(2), F(-3, 5)):
        print("  hbar = %s" % hbar)
        for k, v in check_osp(mod, hbar, deg).items():
            print("    %s : %s" % (k, "OK" if v else "FAILS"))

    print("\n(3) D- = [X-,D+] ?")
    for hbar in (F(1), F(2)):
        o = ops(mod, hbar)
        print("  hbar = %s : %s" % (hbar, equal(bracket(o["Xm"], o["Dp"]),
                                                o["Dm"], mod, deg)[0]))

    print("\n(4) the commuting pair: [rho_hbar(f), A] = 0")
    for hbar in (F(0), F(1), F(2)):
        bad = check_commuting(mod, hbar, deg, fdeg=4)
        print("  hbar = %s : %s" % (hbar, "OK (all f of degree <= 4)" if not bad
                                    else "FAILS for %d pairs, e.g. %s" % (len(bad), bad[:3])))

    print("\n(5) rho is a homomorphism, and 1 acts by -hbar")
    for hbar in (F(0), F(1)):
        bad, central = check_homomorphism(mod, hbar, deg, fdeg=3)
        print("  hbar = %s : homomorphism %s ; rho(1) = -hbar*id %s"
              % (hbar, "OK" if not bad else "FAILS %s" % bad[:2], central))

    print("\n(6) hbar = 0: the contraction")
    for ll in range(0, 4):
        m2 = Model(n, ll)
        print("  l = %d : %s" % (ll, check_contraction(m2, min(deg, 3))))


if __name__ == "__main__":
    main()


# ------------------------------------------- operators as vectors: spans etc.

def opvec(A, mod, deg):
    """Coordinates of an operator: its values on all monomials of degree<=deg."""
    out = {}
    for m in mod.basis(deg):
        for k, c in A({m: F(1)}).items():
            out[(m, k)] = c
    return out


def reduce_row(basis, row):
    """Sparse elimination; basis maps pivot -> normalized row."""
    row = dict(row)
    while row:
        piv = min(row)
        if piv not in basis:
            c = row[piv]
            return piv, {k: v / c for k, v in row.items()}
        b = basis[piv]
        c = row[piv]
        for k, v in b.items():
            nv = row.get(k, F(0)) - c * v
            if nv:
                row[k] = nv
            elif k in row:
                del row[k]
    return None, None


def span_dim(vecs):
    basis = {}
    for v in vecs:
        piv, r = reduce_row(basis, v)
        if piv is not None:
            basis[piv] = r
    return len(basis), basis


def in_span(basis, v):
    piv, _ = reduce_row(basis, v)
    return piv is None


def algebra_generated(gens, mod, deg, cap=40):
    """The Lie superalgebra generated by the operators gens (as operators on the
    monomials of degree <= deg).  Returns (list of ops, dimension, (even|odd))."""
    elts, basis = [], {}
    frontier = list(gens)
    while frontier and len(elts) < cap:
        A = frontier.pop(0)
        v = opvec(A, mod, deg)
        if not v:
            continue
        piv, r = reduce_row(basis, v)
        if piv is None:
            continue
        basis[piv] = r
        elts.append(A)
        for B in list(elts):
            frontier.append(bracket(A, B))
    ev = sum(1 for A in elts if A.parity == 0)
    return elts, len(elts), (ev, len(elts) - ev)


# ----------------------------------------------------------- the centralizer

def ansatz(mod, cmax, omax):
    """All operators (monomial)(hats)(even derivatives)(odd derivatives) with
    coefficient degree <= cmax and even derivative order <= omax."""
    out = []
    ne, no = len(mod.ev), len(mod.od)
    coefs = [e for e in _compositions(ne, cmax)]
    ders = [e for e in _compositions(ne, omax)]
    odd = [s for k in range(no + 1) for s in itertools.combinations(range(no), k)]
    for e in coefs:
        for S in odd:
            for dd in ders:
                for T in odd:
                    facs = [("poly", e)]
                    facs += [("mul", (1, i)) for i in S]
                    for i, k in enumerate(dd):
                        facs += [("der", (0, i))] * k
                    facs += [("der", (1, i)) for i in T]
                    out.append(Op([(F(1), tuple(facs))], (len(S) + len(T)) % 2,
                                  "A%d" % len(out)))
    return out


def centralizer(mod, hbar, cmax=2, omax=2, fdeg=3, deg=4):
    """The centralizer of the po(2n)-action inside the ansatz."""
    from exactlin import nullspace
    A = ansatz(mod, cmax, omax)
    cols = []
    keys = {}
    for op in A:
        v = {}
        for f in gens_po(mod, fdeg):
            R = rho(mod, f, hbar)
            for (m, k), c in opvec(bracket(R, op), mod, deg).items():
                v[(tuple(sorted(f.items())), m, k)] = c
        for k in v:
            keys.setdefault(k, len(keys))
        cols.append(v)
    rows = [[F(0)] * len(cols) for _ in range(len(keys))]
    for j, v in enumerate(cols):
        for k, c in v.items():
            rows[keys[k]][j] = c
    ns = nullspace(rows) if rows else []
    sols = []
    for vec in ns:
        terms = []
        for j, c in enumerate(vec):
            if c:
                terms += [(c * a, f) for a, f in A[j].terms]
        sols.append(Op(terms, 0, "c"))
    return sols, len(A)


def products(gens, mod, deg):
    """Vectors of 1, the generators, and all their pairwise products."""
    vs = []
    idop = Op([(F(1), ())], 0, "1")
    lst = [idop] + list(gens)
    for A in lst:
        vs.append(opvec(A, mod, deg))
    for A in gens:
        for B in gens:
            vs.append(opvec(compose(A, B), mod, deg))
    return [v for v in vs if v]


# ------------------------------------------------------------- contact model

def contact_alpha(mod):
    """alpha = dt + sum p_i dq_i  as a vector (a 1-form)."""
    v = {}
    e0 = (0,) * len(mod.ev)
    v[(e0, (mod.th_t[1],))] = F(1)
    for i in range(mod.n):
        e = list(e0)
        e[mod.p[i][1]] += 1
        v[(tuple(e), (mod.qh[i][1],))] = F(1)
    return v


def lie_field(mod, comps):
    """L_D for the vector field D = sum comps[g]*d/dg, all coordinates even."""
    terms = []
    for gen, coefs in comps.items():
        for e, c in coefs.items():
            terms.append((c, (("poly", e), ("der", gen))))
            for gen2 in [g for g in (mod.q + mod.p + [mod.t]) if True]:
                k = e[gen2[1]]
                if not k:
                    continue
                ee = list(e)
                ee[gen2[1]] -= 1
                terms.append((c * k, (("mul", _hat_c(mod, gen2)),
                                      ("poly", tuple(ee)),
                                      ("der", _hat_c(mod, gen)))))
    return Op(terms, 0, "L_D")


def _hat_c(mod, gen):
    if gen == mod.t:
        return mod.th_t
    return _hat(mod, gen)


def contact_fields(mod, maxdeg, deg=3):
    """Solve L_K alpha = u alpha for K with polynomial coefficients of degree
    <= maxdeg and u of degree <= maxdeg.  Returns the list of solutions
    (components, u) and the dictionary f = alpha(K)."""
    from exactlin import nullspace
    gens = mod.q + mod.p + [mod.t]
    mons = [e for e in _compositions(len(mod.ev), maxdeg)]
    unknowns = [(g, e) for g in gens for e in mons] + [("u", e) for e in mons]
    alpha = contact_alpha(mod)
    cols = []
    keys = {}
    for un in unknowns:
        if un[0] == "u":
            v = {k: -c for k, c in apply_factor(("poly", un[1]), alpha).items()}
        else:
            g, e = un
            L = lie_field(mod, {g: {e: F(1)}})
            v = L(alpha)
        for k in v:
            keys.setdefault(k, len(keys))
        cols.append(v)
    rows = [[F(0)] * len(cols) for _ in range(len(keys))]
    for j, v in enumerate(cols):
        for k, c in v.items():
            rows[keys[k]][j] = c
    return nullspace(rows), unknowns


# --------------------------------------------- the periplectic (odd) model

class PeriModel:
    """The flat periplectic model: x_1..x_n even, xi_1..xi_n odd, with
    p(yhat) = p(y)+1, so that xhat is odd and xihat is even, and

        omegatilde = sum xhat_i xihat_i     (odd, omegatilde^2 = 0).

    One more odd generator, hb, is the central charge hbar; it is never
    differentiated, so C[hb] = Lambda(1) is the ground ring, and hb^2 = 0."""

    def __init__(self, n, odd_hbar=True):
        self.n = n
        self.ev, self.od = [], []
        self.x = [self._add("x%d" % i, 0) for i in range(n)]
        self.xi = [self._add("xi%d" % i, 1) for i in range(n)]
        self.xh = [self._add("xh%d" % i, 1) for i in range(n)]
        self.xih = [self._add("xih%d" % i, 0) for i in range(n)]
        self.hb = self._add("hb", 1 if odd_hbar else 0)
        self.l = 0

    _add = Model._add
    one = Model.one
    basis = Model.basis


def peri_ops(pm, with_hbar=True):
    """d, X^+ = omegatilde^, X^- , and D^+ = d + a^ with da = hbar*omegatilde."""
    n = pm.n
    d = Op([(F(1), (("mul", pm.xh[i]), ("der", pm.x[i]))) for i in range(n)] +
           [(F(1), (("mul", pm.xih[i]), ("der", pm.xi[i]))) for i in range(n)],
           1, "d")
    Xp = Op([(F(1), (("mul", pm.xh[i]), ("mul", pm.xih[i]))) for i in range(n)],
            1, "X+")
    Xm = Op([(F(1), (("der", pm.xih[i]), ("der", pm.xh[i]))) for i in range(n)],
            1, "X-")
    # a = (1/2) hb sum (xi_i dx_i + x_i dxi_i), an odd 1-form: the odd contact
    # form alpha of eq. (43), multiplied by the odd central charge hb.
    half = F(1, 2) if with_hbar else F(0)
    a_terms = []
    for i in range(n):
        a_terms.append((half, (("mul", pm.hb), ("mul", pm.xi[i]), ("mul", pm.xh[i]))))
        a_terms.append((half, (("mul", pm.hb), ("mul", pm.x[i]), ("mul", pm.xih[i]))))
    Dp = Op(d.terms + a_terms, 1, "D+")
    Dm = bracket(Xm, Dp)
    Dm.name = "D-"
    H = bracket(Xp, Xm)
    H.name = "H"
    return dict(d=d, Xp=Xp, Xm=Xm, H=H, Dp=Dp, Dm=Dm,
                hbXp=compose(Op([(F(1), (("mul", pm.hb),))], 1, "hb"), Xp))


def mult_hb(pm, A):
    return compose(Op([(F(1), (("mul", pm.hb),))], 1, "hb"), A)
