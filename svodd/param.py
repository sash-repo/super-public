#!/usr/bin/env python3
"""
param.py -- the PARAMETRIC certificate: a node at which the classes form a
family is closed by classifying the fibre over the GENERIC member.

THE OBSTRUCTION.  A step of the enumeration may leave a family of classes

        r(mu) = r + mu_1 d_1 + ... + mu_t d_t ,          mu in K^t,

and the classification must then continue underneath EVERY member.  The
stabiliser doing the continuing,

        S(mu) = Ker Phi_mu ,     Phi_mu(M) = Abar_M r(mu) = B_0(M) + sum mu_i B_i(M),

need not be the same subalgebra for different mu: affine.uniform_family proves
it is in two cases (a zero action; a family of scalar multiples) and declines
otherwise, since a constant S(mu) forces S(mu) = S* = Ker B_0 cap ... cap Ker B_t,
so a node with dim S(mu) constant and larger than dim S* has a stabiliser that
MOVES while keeping its dimension.  Corank 1|5 has such a node.

THE CERTIFICATE.  Do the computation over the field K(mu) = Q(mu_1,...,mu_t) --
at the generic point of the family.  This is a change of the ground ring and
not of the logic: every step the machinery takes is exact there, and every
branch it takes is a statement about the generic member.  The result specialises:
the elimination inverts finitely many polynomials p_1, ..., p_N in mu (ratfun
records each one as it is inverted), and at every member with p_i(a) != 0 for
all i the same computation runs verbatim over K.  Hence

    THEOREM.  The list of classes of the fibre over r(a), with their orbit
    dimensions, is the specialisation at mu = a of the generic list, for every
    a outside the hypersurface p_1 ... p_N = 0.

For a one-parameter family that exceptional set is the root set of one
univariate polynomial: finitely many members, each of which is then run
separately over K -- over Q at a rational root, and over the number field
Q[mu]/(p) at an irrational one (`special_members` splits them, `rerun_at` runs
a rational one).

WHAT IS NOT CLAIMED.  The generic answer says nothing about a special member
until that member is run; a member at which the answer differs is a separate
line of the census, not an anomaly.  This is the same discipline as everywhere
in this programme: an uncertified branch is not a provisional result.
"""

from fractions import Fraction as F

import ratfun

# every parametric node that has been opened, with what is needed to re-run it
# at a special member; see special_members / rerun_at
NODES = []

_POOL = []          # the generators of K(mu) not yet handed out
_NPARAM = 0


def begin(nparams=6):
    """
    Make K(mu_1..mu_nparams) the ground field of the whole run and reset the
    trace.  One field for the run, so that nested family nodes -- a family
    inside the fibre of a family -- live in the same ring and may be combined.
    """
    global _POOL, _NPARAM
    _NPARAM = nparams
    _POOL = list(ratfun.setup(nparams))
    NODES.clear()
    return _NPARAM


def active():
    return bool(_POOL) or _NPARAM > 0


LAST_PARAMS = []        # the indices of the parameters handed out last


def fresh(t):
    """t parameters not used by any other node, or None if the pool is out"""
    global _POOL, LAST_PARAMS
    if len(_POOL) < t:
        return None
    LAST_PARAMS = list(range(_NPARAM - len(_POOL), _NPARAM - len(_POOL) + t))
    out, _POOL = _POOL[:t], _POOL[t:]
    return out


def generic_member(r, dirs):
    """
    r + mu_1 d_1 + ... + mu_t d_t over K(mu), with fresh parameters.
    Returns (rgen, mus) or None if the pool is exhausted.
    """
    mus = fresh(len(dirs))
    if mus is None:
        return None
    out = [ratfun.Rat.of(x) for x in r]
    for c, d in zip(mus, dirs):
        for i in range(len(out)):
            if d[i]:
                out[i] = out[i] + c * d[i]
    return out, mus


def note_node(kind, data):
    """open a parametric node; the trace from here on belongs to it"""
    NODES.append(dict(kind=kind, trace_start=len(ratfun.INVERTED), **data))
    return len(NODES) - 1


def close_node(idx, generic):
    """record what the node produced over the generic member, and what it inverted"""
    if idx is None or idx >= len(NODES):
        return
    nd = NODES[idx]
    nd["generic"] = generic
    nd["inverted"] = ratfun.INVERTED[nd["trace_start"]:]


def node_exceptional(idx, param_index=0):
    """the irreducible factors inverted while the node `idx` was being solved"""
    out, seen = [], []
    for p in NODES[idx].get("inverted", []):
        for g, _ in p.factor():
            if g.is_constant():
                continue
            k = g.key()
            if k not in seen:
                seen.append(k)
                out.append(g)
    return out


# ------------------------------------------------------ the exceptional set

def exceptional():
    """
    The irreducible factors of everything the run inverted, deduplicated and
    with the constants dropped: the polynomials whose vanishing is the only
    thing that can invalidate the generic answer.
    """
    out, seen = [], []
    for p in ratfun.trace_polys():
        for g, _ in p.factor():
            if g.is_constant():
                continue
            k = g.key()
            if k in seen:
                continue
            seen.append(k)
            out.append(g)
    return out


def special_members(polys, param=0):
    """
    For a ONE-parameter family: the special members, as
    (rational roots, irrational factors).  A rational root is a member that can
    be re-run over Q; an irrational factor names the members that need the
    number field Q[mu]/(p).  Factors not involving the parameter `param` at all
    are irrelevant to it and are dropped.
    """
    rats, others = [], []
    for g in polys:
        if g.degree_in(param) == 0:
            continue
        d = _uni_coeffs(g, param)
        if d is None:
            others.append(g)
            continue
        if len(d) == 2:                      # a mu + b
            rats.append(-d[0] / d[1])
        elif len(d) == 1:
            continue
        else:
            others.append(g)
    return sorted(set(rats)), others


def _uni_coeffs(g, param):
    """the coefficients of g as a polynomial in mu_param, or None if it is not"""
    try:
        terms = g.p.to_dict() if hasattr(g, "p") else g.d
    except Exception:
        return None
    out = {}
    for mon, c in terms.items():
        for i, e in enumerate(mon):
            if i != param and e:
                return None
        v = F(int(c.numer()), int(c.denom())) if hasattr(c, "numer") else F(c)
        out[int(mon[param])] = out.get(int(mon[param]), F(0)) + v
    n = max(out)
    return [out.get(i, F(0)) for i in range(n + 1)]


# ------------------------------------------- the ranks the computation uses

# Every rank the elimination decides is recorded here, in the order in which it
# is decided.  A rank is INTRINSIC -- it does not depend on which minor the
# pivoting happened to invert -- and the specialisation theorem needs no more
# than that the ranks do not drop.  Hence the refinement of \S rank: a member at
# which the recorded ranks are the generic ones is not exceptional, whatever the
# elimination inverted while it ran there.  See sym.py.
RANKS = []
RECORD = False          # the primitives record only while this is on


def record_rank(r, n=None):
    RANKS.append(r if n is None else (r, n))


def ranks_reset():
    RANKS.clear()


def report():
    """a human-readable summary of the parametric nodes and their exceptions"""
    polys = exceptional()
    lines = [f"parametric nodes: {len(NODES)}",
             f"inverted (irreducible factors): {len(polys)}"]
    for g in polys[:40]:
        lines.append(f"    {g}")
    return "\n".join(lines)
