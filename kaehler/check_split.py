r"""Item 3) of Theorem 1.9.4a: is  0 -> hbar*g -> g -> h -> 0  a split extension,
for  g := <D^+,D^-,X^+,X^->  with  D^- := [X^-, D^+]  and  h := ba(2) + sl(1|1)?

The ideal I := hbar*g is abelian and, as an h-module, is Pi(h).  A splitting is a
parity-preserving linear section sigma(u_i) = s_i + sum_k beta_ik e_k that is a
homomorphism.  Since I is abelian, the condition is affine-linear in beta:

  [s_i,s_j] - sum_m c^m_ij s_m + sum_k beta_jk [s_i,e_k] + sum_k beta_ik [e_k,s_j]
                               - sum_m c^m_ij sum_k beta_mk e_k = 0.

We solve it exactly over Q; solvable <=> the extension splits.  The machinery is
validated on  h (x) C[hbar]/(hbar^2), which splits by construction.

Run (from this folder):  ../venv/bin/python check_split.py
"""
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from fractions import Fraction
from flatmodel import Model, Echelon, closure, d_operator

half = Fraction(1, 2)


def solve_affine(rows, rhs):
    """Is  A x = b  solvable?  rows: list of lists (A), rhs: list (b)."""
    m, n = len(rows), (len(rows[0]) if rows else 0)
    aug = [list(rows[i]) + [rhs[i]] for i in range(m)]
    r0 = 0
    for col in range(n):
        piv = next((r for r in range(r0, m) if aug[r][col] != 0), None)
        if piv is None:
            continue
        aug[r0], aug[piv] = aug[piv], aug[r0]
        pv = aug[r0][col]
        aug[r0] = [v / pv for v in aug[r0]]
        for r in range(m):
            if r != r0 and aug[r][col] != 0:
                f = aug[r][col]
                aug[r] = [a - f * b for a, b in zip(aug[r], aug[r0])]
        r0 += 1
    for r in range(r0, m):
        if all(v == 0 for v in aug[r][:n]) and aug[r][n] != 0:
            return False, r0
    return True, r0


def analyse(P, gens, label):
    basis, sd = closure(P, gens, limit=300)

    E = {0: Echelon(), 1: Echelon()}
    gb = [e for e in basis if E[P.parity(e)].add(e)]
    EI = {0: Echelon(), 1: Echelon()}
    ideal = [e for e in gb if all(m and m[0][0] == 'h' for m in e)
             and EI[P.parity(e)].add(e)]
    EL = {0: Echelon(), 1: Echelon()}
    for e in ideal:
        EL[P.parity(e)].add(e)
    lifts = [e for e in gb if EL[P.parity(e)].add(e)]
    print('%s sdim g = %s, sdim hbar*g = (%d|%d), quotient = (%d|%d)'
          % (label, sd, len(EI[0]), len(EI[1]),
             sum(1 for e in lifts if P.parity(e) == 0),
             sum(1 for e in lifts if P.parity(e) == 1)))

    def coords_mod_I(e):
        """write e = sum_m a_m lifts_m + (element of I); return the a_m"""
        par = P.parity(e)
        vecs = [(k, v) for k, v in enumerate(lifts) if P.parity(v) == par]
        cand = [v for _, v in vecs] + [v for v in ideal if P.parity(v) == par]
        mons = sorted({m for v in cand + [e] for m in v})
        A = [[v.get(m, Fraction(0)) for v in cand] for m in mons]
        rhs = [e.get(m, Fraction(0)) for m in mons]
        m_, n_ = len(A), len(cand)
        aug = [A[i] + [rhs[i]] for i in range(m_)]
        r0, piv_col = 0, {}
        for col in range(n_):
            piv = next((r for r in range(r0, m_) if aug[r][col] != 0), None)
            if piv is None:
                continue
            aug[r0], aug[piv] = aug[piv], aug[r0]
            pv = aug[r0][col]
            aug[r0] = [v / pv for v in aug[r0]]
            for r in range(m_):
                if r != r0 and aug[r][col] != 0:
                    f = aug[r][col]
                    aug[r] = [a - f * bb for a, bb in zip(aug[r], aug[r0])]
            piv_col[col] = r0
            r0 += 1
        sol = [Fraction(0)] * n_
        for col, r in piv_col.items():
            sol[col] = aug[r][n_]
        return {k: sol[idx] for idx, (k, _) in enumerate(vecs) if sol[idx]}

    L = len(lifts)
    cstr = {(i, j): coords_mod_I(P.br(lifts[i], lifts[j]))
            for i in range(L) for j in range(L)}

    unk = [(i, k) for i in range(L) for k in range(len(ideal))
           if P.parity(lifts[i]) == P.parity(ideal[k])]
    ui = {u: idx for idx, u in enumerate(unk)}

    rows, rhs = [], []
    for i in range(L):
        for j in range(L):
            const = P.add(P.br(lifts[i], lifts[j]),
                          *[P.smul(-a, lifts[m]) for m, a in cstr[(i, j)].items()])
            terms = {}

            def acc(idx, elt):
                terms[idx] = P.add(terms.get(idx, {}), elt)

            for k in range(len(ideal)):
                if (j, k) in ui:
                    acc(ui[(j, k)], P.br(lifts[i], ideal[k]))
                if (i, k) in ui:
                    acc(ui[(i, k)], P.br(ideal[k], lifts[j]))
            for m, a in cstr[(i, j)].items():
                for k in range(len(ideal)):
                    if (m, k) in ui:
                        acc(ui[(m, k)], P.smul(-a, ideal[k]))
            mons = sorted({mm for e in list(terms.values()) + [const] for mm in e})
            for mm in mons:
                rows.append([terms.get(idx, {}).get(mm, Fraction(0))
                             for idx in range(len(unk))])
                rhs.append(-const.get(mm, Fraction(0)))

    ok, rank = solve_affine(rows, rhs)
    print('   unknowns %d, equations %d, rank %d  ->  splitting exists: %s'
          % (len(unk), len(rows), rank, ok))
    return ok


for n in (1, 2, 3):
    P = Model([0] * n + [1] * n)
    q, pi = list(range(n)), list(range(n, 2 * n))
    c, b, x, t = (lambda a: P.gen('c', a)), (lambda a: P.gen('b', a)), \
                 (lambda a: P.gen('x', a)), (lambda a: P.gen('t', a))
    hb = P.gen('h', 0)
    d = d_operator(P)
    Xp = P.add(*[P.mul(c(q[i]), c(pi[i])) for i in range(n)])
    Xm = P.add(*[P.mul(b(pi[i]), b(q[i])) for i in range(n)])
    Dp = P.add(d, P.smul(-half, P.mul(hb, P.add(
        *[P.add(P.mul(c(q[i]), x(pi[i])),
                P.smul(-1, P.mul(c(pi[i]), x(q[i])))) for i in range(n)]))))
    Dm = P.br(Xm, Dp)

    analyse(P, [Dp, Dm, Xp, Xm], '=== C^{%d|%d}, g := <D^+,D^-,X^+,X^->: ' % (n, n))
    # control: the split model h (x) C[hbar]/(hbar^2)
    analyse(P, [Xp, Xm, d, P.mul(hb, Xp), P.mul(hb, Xm), P.mul(hb, d)],
            '    control  h (x) C[hbar]/(hbar^2):    ')
    print()
