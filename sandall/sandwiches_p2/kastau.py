#!/usr/bin/env python3
r"""
kastau.py --- the parity theorem for the mirror automorphisms of
\fkas(7;\widetilde\un) = \textbf{F}(\fkas(1;\un|6)), and its two consequences for
table (grVexF) of the text.

For S \subseteq {1,2,3} let \tau_S be the contact transformation

    \tau_S:  \xi_i \leftrightarrow \eta_i  for i \in S,   t \mapsto t + \Phi_S,
             \Phi_S = \sum_{i \in S} \xi_i\eta_i,

extended to \cO(7;\widetilde\un) as a homomorphism of divided power algebras.  Each
\tau_S preserves the contact form \alpha = dt + \sum_i \xi_i d\eta_i exactly, because
\eta_i d\xi_i = d(\xi_i\eta_i) + \xi_i d\eta_i, so \tau_S \in \Aut\fk(7;\widetilde\un);
and \tau_S\tau_{S'} = \tau_{S \triangle S'}, so S \mapsto \tau_S is an embedding of
(\Zee/2)^3.

CLAIM.  \tau_S(\fkas) = \fkas if and only if |S| is even.

Consequences, since \tau_S turns the degree vector (d; a; b) into the one with a_i and
b_i interchanged for i \in S:
  * |S| = 2 gives an isomorphism of the graded algebras behind rows 2 and 3 of table (20)
    ("1\xi" and "1\eta"), and behind rows 4 and 5;
  * |S| = 3 -- the full mirror, which would take R(3\xi) to R(3\eta) -- is NOT available,
    which is Remark 2.11.3(1).

Run:  python3 kastau.py         (KAS_NT=3 by default)
"""

import itertools
import os

import kas2 as K

NVAR, T, XI, ETA = K.NVAR, K.T, K.XI, K.ETA
MONS, IDX, DIMO = K.MONS, K.IDX, K.DIMO
NT = K.NT

def var(i):
    e = [0] * NVAR
    e[i] = 1
    return 1 << IDX[tuple(e)]

ONE = 1 << IDX[tuple([0] * NVAR)]
TT = var(T)
X = [var(i) for i in XI]
E = [var(i) for i in ETA]

# ---------------------------------------------------------------- tau_S

def tau_table(S):
    r"""the linear map \tau_S on \cO(7;\widetilde\un), as a table over the monomials"""
    PhiS = 0
    for i in S:
        PhiS ^= K.mul(X[i], E[i])
    # images of the six height-one indeterminates
    img = {}
    for i in range(3):
        img[XI[i]] = E[i] if i in S else X[i]
        img[ETA[i]] = X[i] if i in S else E[i]
    # \tau(t^{(k)}) = (t+\Phi_S)^{(k)} = sum_j t^{(k-j)} e_j(monomials of \Phi_S)
    mons_phi = [1 << b for b in K.bits(PhiS)]
    tpow = {}
    for k in range(2 ** NT):
        r = 0
        for j in range(0, len(mons_phi) + 1):
            if j > k:
                break
            e = [0] * NVAR
            e[T] = k - j
            base = 1 << IDX[tuple(e)]
            gj = 0
            if j == 0:
                gj = ONE
            else:
                for comb in itertools.combinations(mons_phi, j):
                    p = comb[0]
                    for c in comb[1:]:
                        p = K.mul(p, c)
                        if p == 0:
                            break
                    gj ^= p
            if gj:
                r ^= K.mul(base, gj)
        tpow[k] = r
    tab = []
    for m in MONS:
        v = tpow[m[T]]
        for i in XI + ETA:
            if m[i]:
                v = K.mul(v, img[i])
        tab.append(v)
    return tab

def apply_tab(tab, f):
    r = 0
    for k in K.bits(f):
        r ^= tab[k]
    return r

def check_automorphism(tab):
    """(multiplicative, preserves the bracket, commutes with the squaring)"""
    mult = all(apply_tab(tab, K.mul(1 << i, 1 << j)) == K.mul(tab[i], tab[j])
               for i in range(DIMO) for j in range(DIMO))
    br = all(apply_tab(tab, K.kb(1 << i, 1 << j)) == K.kb(tab[i], tab[j])
             for i in range(DIMO) for j in range(DIMO))
    return mult, br

# ---------------------------------------------------------------- main

def main():
    print('kastau.py --- the parity theorem for tau_S,  N_t = %d' % NT)
    comp = K.build_kas()
    kas = K.echelon([v for j in sorted(comp) for v in comp[j]])
    U = comp[1]
    pvkas = K.pivots(kas)
    pvU = K.pivots(U)
    print('dim O(7;N) = %d, dim kas = %d, dim U = %d\n' % (DIMO, len(kas), len(U)))

    print('  S            |S|  automorphism of k(7)?   tau_S(U)=U   tau_S(kas)=kas')
    results = {}
    for r in range(4):
        for S in itertools.combinations(range(3), r):
            tab = tau_table(set(S))
            mult, br = check_automorphism(tab)
            keepU = all(K.in_span(apply_tab(tab, u), pvU) for u in U)
            keepkas = all(K.in_span(apply_tab(tab, v), pvkas) for v in kas)
            results[S] = keepkas
            print('  %-12s %d    %s              %-6s       %-6s'
                  % ('{%s}' % ','.join(str(i + 1) for i in S) if S else '{}',
                     len(S),
                     'yes' if (mult and br) else 'NO (mult %s, bracket %s)' % (mult, br),
                     'yes' if keepU else 'no',
                     'yes' if keepkas else 'no'))

    even = all(v for S, v in results.items() if len(S) % 2 == 0)
    odd = all(not v for S, v in results.items() if len(S) % 2 == 1)
    print('\n  CLAIM  tau_S(kas) = kas  <=>  |S| even :  %s'
          % ('CONFIRMED' if (even and odd) else 'FAILS'))

    # ------------------------------------------------ the consequences for table (20)
    print('\nConsequences for table (20).')
    rows = {
        'row 1  Fkas          ': (2, (1, 1, 1), (1, 1, 1)),
        'row 2  Fkas(;1xi)    ': (2, (0, 1, 1), (2, 1, 1)),
        'row 3  Fkas(;1eta)   ': (2, (2, 1, 1), (0, 1, 1)),
        'row 4  R(3xi)        ': (1, (0, 0, 0), (1, 1, 1)),
        'row 5  R(3xi)-variant': (1, (0, 1, 1), (1, 0, 0)),
        'R(3eta)              ': (1, (1, 1, 1), (0, 0, 0)),
    }
    pieces = {}
    for name, (d, a, b) in rows.items():
        dv = (d,) + a + b
        gr = K.graded_pieces(kas, dv)
        ok, info = K.w_test(gr) if gr is not None else (False, 'not graded')
        pieces[name] = (dv, gr, ok, info)
        print('  %-22s d=%s  a=%s b=%s   %s' % (name, d, a, b, 'W-grading' if ok else info))

    def transported(S, dv):
        """the degree vector obtained from dv by the mirror in the indices of S"""
        d, a, b = dv[0], list(dv[1:4]), list(dv[4:7])
        for i in S:
            a[i], b[i] = b[i], a[i]
        return (d,) + tuple(a) + tuple(b)

    print('\n  isomorphisms of graded algebras exhibited by the even tau_S:')
    for S in [(0, 2), (1, 2), (0, 1)]:
        tab = tau_table(set(S))
        for name, (dv, gr, ok, info) in pieces.items():
            if gr is None or not ok:
                continue
            tgt = transported(S, dv)
            gr2 = K.graded_pieces(kas, tgt)
            if gr2 is None:
                continue
            same = all(all(K.in_span(apply_tab(tab, v), K.pivots(gr2.get(c, [])))
                           for v in gr[c]) for c in sorted(gr))
            if same and tgt != dv:
                print('    tau_{%s}:  %s  -->  degree vector %s   (graded iso: %s)'
                      % (','.join(str(i + 1) for i in S), name.strip(), tgt,
                         'YES' if same else 'no'))


if __name__ == '__main__':
    main()
