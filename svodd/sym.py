#!/usr/bin/env python3
"""
sym.py -- why the exceptional members of IV_1(3,2) cluster at mu = -1.

The question: the ten parametric nodes of IV_1(3,2) pair up as (1,2), (4,5), (6,7),
(8,9), and four of the five genuine jumps sit at mu = -1.  Is that the shadow of
a symmetry of the normal form permuting the odd indeterminates?

Two independent computations here, and they answer it:

PART A -- the symmetry group of the germ.  All monomial substitutions
    x |-> alpha x,   xi_i |-> t_i xi_{pi(i)},   alpha, t_i in K^*, pi in S_q
that fix the germ f, and their action on the layer Nt_f and on the class list.
If the pairing were a symmetry permuting the odd indeterminates, it would be
here.

PART B -- the ranks.  Every step the elimination takes is decided by the RANK of
a matrix over K(mu); the pivot it inverts is a choice, the rank is not.  So the
member a is harmless as soon as no recorded rank drops at a -- whatever the
elimination happened to invert while running there.  The script re-runs each
node at each of its rational special members with the ranks recorded (see
param.RECORD) and compares the rank sequence with the one at an ordinary member.

Usage:
    python3 sym.py A          # the symmetry group
    python3 sym.py B          # the rank criterion on IV_1(3,2)
    python3 sym.py B III_1 4 3
"""

import sys
import random
from fractions import Fraction as F
from itertools import permutations

import sv
import stab
import orb
import t3
import param
from probe import TYPES


# --------------------------------------------------------------- the type

def load(tname="IV_1", n=3, m=2, D=None):
    p, q, terms = TYPES[tname](n, m)
    if D is None:
        D = 2 * n + 4
    res = stab.analyse(tname, p, q, terms, D, verbose=False)
    return res, p, q, terms


# ============================================================== PART A

def monomial_syms(p, q, terms):
    """
    All (alpha, pi, t) with  f(alpha x, t_i xi_{pi(i)}) = f,  as a list of
    (alpha, pi, t).  The search is exact: a monomial substitution sends a
    monomial of f to a monomial, so it must permute the monomials of f, and the
    coefficients then have to match.  Over Q, so alpha^n = 1 means alpha = +-1.
    """
    f = stab.build(terms)
    mons = sorted(f, key=stab.mkey)
    out = []
    for pi in permutations(range(q)):
        # the induced map on the monomials of f must be a bijection of them
        img = {}
        ok = True
        for (a, S) in mons:
            T = frozenset(pi[i] for i in S)
            if (a, T) not in f:
                ok = False
                break
            img[(a, S)] = (a, T)
        if not ok or len(set(img.values())) != len(mons):
            continue
        # now solve for alpha, t.  Multiplicatively, over Q^*, we only need the
        # signs and the rational scalars: solve the linear system in the
        # exponents by brute force over the sign group, which is what a
        # rational solution can differ from a positive one by.
        for sol in _solve_scalars(f, pi, p, q):
            out.append(sol)
    return out


def _sign_of_sort(S, pi):
    """the sign of reordering xi_{pi(i)}, i in S (increasingly), into order"""
    seq = [pi[i] for i in sorted(S)]
    sgn, seq = 1, list(seq)
    for i in range(len(seq)):
        for j in range(len(seq) - 1 - i):
            if seq[j] > seq[j + 1]:
                seq[j], seq[j + 1] = seq[j + 1], seq[j]
                sgn = -sgn
    return sgn


def _solve_scalars(f, pi, p, q, vals=(F(1), F(-1))):
    """
    the (alpha, t) making the substitution fix f.

    Searching the SIGNS only is exhaustive here, and not an approximation.  The
    normal forms of the Serganova-Vaintrob table have all their coefficients equal to 1, so every equation
    reads  alpha^{|a|} eps Prod_{i in S} t_i = 1  with eps = +-1; the solutions
    for a fixed pi are a coset of the solution group of pi = id, which is the
    torus T of the germ -- and T is CONNECTED, so it changes nothing about which
    component of the symmetry group we have found.  Moreover the equations are
    multiplicative with right-hand sides +-1, so with any solution t their signs
    sign(t_i) are a solution as well: a solvable system is solvable in +-1.
    Over Q the only roots of unity are +-1, whence alpha in {1,-1} too.
    """
    mons = sorted(f, key=stab.mkey)
    out = []
    for alpha in (F(1), F(-1)):
        # t_i enters each equation as a monomial; solve by search over a small
        # grid, which is exhaustive for the germs at hand (see the docstring)
        def rec(i, t):
            if i == q:
                for (a, S) in mons:
                    T = frozenset(pi[j] for j in S)
                    c = alpha ** sum(a) * _sign_of_sort(S, pi)
                    for j in S:
                        c *= t[j]
                    if c * f[(a, S)] != f[(a, T)]:
                        return
                out.append((alpha, pi, tuple(t)))
                return
            # xi_i not occurring in f at all is free; pin it to 1
            if not any(i in S for (a, S) in mons):
                rec(i + 1, t + [F(1)])
                return
            for v in vals:
                rec(i + 1, t + [v])
        rec(0, [])
    return out


def layer_matrix(res, alpha, pi, t):
    """the matrix of the substitution on the layer Nt_f, in its monomial basis"""
    R, mI, basis, index = res["R"], res["mI"], res["basis"], res["index"]
    n = len(basis)
    M = [[F(0)] * n for _ in range(n)]
    for j, (a, S) in enumerate(basis):
        c = alpha ** sum(a) * _sign_of_sort(S, pi)
        for i in S:
            c *= t[i]
        T = frozenset(pi[i] for i in S)
        v = mI.reduce({(a, T): c})          # the class of the image
        for mon, co in (v or {}).items():
            if mon in index:
                M[index[mon]][j] = co
    return M


def part_A(tname="IV_1", n=3, m=2):
    res, p, q, terms = load(tname, n, m)
    f = stab.build(terms)
    print(f"germ {tname}({n},{m}):  f = "
          + " + ".join(sv.show_mon(mo, "xy", "xi") for mo in sorted(f, key=stab.mkey)))
    syms = monomial_syms(p, q, terms)
    nontriv = [s for s in syms if s[1] != tuple(range(q)) or s[0] != F(1)
               or any(x != F(1) for x in s[2])]
    print(f"\nmonomial symmetries fixing f: {len(syms)}")
    perms = sorted(set(s[1] for s in syms))
    print(f"    permutations of the odd indeterminates occurring: {len(perms)}")
    for pi in perms:
        cyc = " ".join(f"xi_{i+1}->xi_{pi[i]+1}" for i in range(q) if pi[i] != i)
        print(f"      {pi}   {cyc if cyc else '(identity on the indices)'}")
    if not nontriv:
        print("    -- only the identity substitution")
        return

    # what they do to the classes
    L, classes, cert, notes = t3.classify(res)
    vecs = [c["v"] for c in classes]
    import run_b2
    spec = run_b2.specialised(vecs)
    if spec is not None:
        vecs = spec
    print(f"\nthe class list: {len(classes)} classes, certified={cert}")
    for (alpha, pi, t) in syms:
        if pi == tuple(range(q)) and alpha == F(1) and all(x == F(1) for x in t):
            continue
        M = layer_matrix(res, alpha, pi, t)
        moved, cache = 0, {}
        for i, v in enumerate(vecs):
            w = [sum(M[r][c] * v[c] for c in range(len(v)))
                 for r in range(len(v))]
            same, _ = orb.same_orbit(L.mats, w, v, degs=L.degs, cache=cache)
            if same is not True:
                moved += 1
        print(f"    alpha={alpha}, pi={pi}, t={t}:  classes not fixed: "
              f"{moved} of {len(vecs)}")


# ============================================================== PART B

def rank_signature():
    """the sequence of ranks recorded since the last reset"""
    return tuple(param.RANKS)


def probe_node(nd, member, keep=None):
    """
    Re-run the node's fibre at the member `member` (a list of values for the
    node's parameters), with the ranks recorded.  Returns (nclasses, ranks).
    """
    n0 = len(param.NODES)
    param.ranks_reset()
    param.RECORD = True
    try:
        rf, cf = nd["run"]([F(x) for x in member])
    finally:
        param.RECORD = False
        del param.NODES[n0:]                 # a nested node opened by the probe
    # the ANSWER of the node: how many classes, and how many of them are
    # families of what dimension -- a family that appears where there was a
    # single class is a jump even though the count does not move (node 0)
    ans = None if rf is None else (len(rf), tuple(sorted(len(fd)
                                                        for y, ff, fd in rf)))
    sig = rank_signature()
    # THE CRITERION: the dimensions of the pure-translation spaces T_0 that the
    # elimination met, each against the dimension of the fibre it sat in.  A
    # node whose generic fibre is one class BECAUSE T_0 fills it is one class at
    # every member where T_0 still fills it -- that is the whole of branch (A1),
    # so this is a proof and not a sample.  dim s, by contrast, may grow without
    # anything happening (mu = 0 in node 1), and the inverted polynomials are
    # pivots, hence choices.
    dims = tuple(tuple(v) for tag, *v in sig if tag == "dim_T0")
    return ans, sig, dims


def part_B(tname="IV_1", n=3, m=2, seed=11):
    res, p, q, terms = load(tname, n, m)
    L, classes, cert, notes = t3.classify(res)
    nodes = list(param.NODES)
    print(f"--- {tname}({n},{m}):  {len(classes)} classes, certified={cert}, "
          f"{len(nodes)} parametric node(s)")
    allex = param.exceptional()
    print(f"    the run inverted {len(allex)} irreducible factor(s)")

    rnd = random.Random(seed)
    tot = dict(jump=0, quiet=0, rk_jump=0, rk_quiet=0, agree=0, disagree=0)
    rows = []
    for i, nd in enumerate(nodes):
        pars = nd["params"]
        if len(pars) != 1:
            print(f"  node {i}: {len(pars)} parameters -- skipped "
                  f"(its exceptional set is a curve)")
            continue
        ex = param.node_exceptional(i)
        roots, others = param.special_members(ex, param=pars[0])
        # an ordinary member: a rational point at which nothing the node
        # inverted vanishes
        while True:
            a0 = F(rnd.randint(5, 60))
            if all(g.eval([a0 if j == pars[0] else F(rnd.randint(5, 60))
                           for j in range(param._NPARAM)]) != 0 for g in ex):
                break
        n0, r0, d0 = probe_node(nd, [a0])
        print(f"  node {i}: generic answer {n0} at mu={a0}, T_0 = {d0}, "
              f"{len(r0)} rank decisions; special members {roots}"
              + (f" (+{len(others)} curve(s))" if others else ""))
        for a in roots:
            na, ra, da = probe_node(nd, [a])
            jump = (na != n0)
            rkdrop = (ra != r0)
            sjump = (da != d0)
            rows.append((i, a, n0, na, jump, rkdrop, d0, da))
            tot["jump" if jump else "quiet"] += 1
            tot["rk_jump" if rkdrop else "rk_quiet"] += 1
            tot["agree" if jump == sjump else "disagree"] += 1
            print(f"      mu = {a}:  answer {na} "
                  f"{'JUMP' if jump else 'same'} | all ranks "
                  f"{'differ' if rkdrop else 'equal'} | T_0 {da} "
                  f"{'COLLAPSES' if sjump else 'unchanged'}"
                  f"{'' if jump == sjump else '   <-- CRITERION FAILS'}")
    print(f"\n  members tested: {tot['jump'] + tot['quiet']}   "
          f"jumps: {tot['jump']}   rank drops: {tot['rk_jump']}   "
          f"criterion agrees: {tot['agree']}, fails: {tot['disagree']}")
    return rows, tot


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "B"
    tn = sys.argv[2] if len(sys.argv) > 2 else "IV_1"
    nn = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    mm = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    if what.upper() == "A":
        part_A(tn, nn, mm)
    else:
        part_B(tn, nn, mm)
