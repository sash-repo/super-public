#!/usr/bin/env python3
r"""fvlewt2.py --- the weight-homogeneous sandwiches of F vle over F_{2^e}.

In a weight space V = span(v_1..v_k) the sandwich conditions on
s = sum c_i v_i are quadratic forms in c with coefficients in L (bit vectors):
  (ad s)^2 x = sum_i c_i^2 [v_i,[v_i,x]] + sum_{i<j} c_i c_j ([v_i,[v_j,x]]+[v_j,[v_i,x]])
  [[x,s],[y,s]] = sum_i c_i^2 [[x,v_i],[y,v_i]]
                 + sum_{i<j} c_i c_j ([[x,v_i],[y,v_j]] + [[x,v_j],[y,v_i]]).
The points over GF(2^e) are enumerated; the K-span of the variety is spanned
by its points over a large enough finite field, and we report the span of the
points over GF(2), GF(4), GF(16) for every weight space (it stabilizes).
Then: the subalgebra generated, and its normalizer."""
import os as _os_runs; _os_runs.makedirs('runs', exist_ok=True)  # logs go to ./runs
import itertools
import fvle
from fvle import irw, bracket
from fvlesand import closure
from fvlesand import L4
from fvle import graded_pieces, echelon, in_span
import ir

POLY = {1: 0b11, 2: 0b111, 4: 0b10011}


def gmul(a, b, e):
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if a >> e & 1:
            a ^= POLY[e]
    return r


def main():
    basis = fvle.build()
    n = len(basis)
    spaces = {(): basis}
    for d in L4:
        new = {}
        for w, vs in spaces.items():
            p = graded_pieces(vs, [irw.terms(v) for v in vs], d)
            for k, b in p.items():
                new[w + (k,)] = b
        spaces = new
    ads = {}
    gens_total = []
    report = {}
    for w, vs in spaces.items():
        k = len(vs)
        # coefficient vectors of the quadratic forms: one long bit vector per
        # monomial c_i c_j, concatenating all the conditions
        adv = [[bracket(v, x) for x in basis] for v in vs]
        def cond(i, j):
            out = []
            for xi, x in enumerate(basis):
                if i == j:
                    out.append(bracket(vs[i], adv[i][xi]))
                else:
                    out.append(bracket(vs[i], adv[j][xi]) ^ bracket(vs[j], adv[i][xi]))
            for a in range(n):
                for b in range(a):
                    if i == j:
                        out.append(bracket(adv[i][a], adv[i][b]))
                    else:
                        out.append(bracket(adv[i][a], adv[j][b]) ^ bracket(adv[j][a], adv[i][b]))
            key = 0
            for r in out:
                key = (key << 896) | r
            return key
        C = {(i, j): cond(i, j) for i in range(k) for j in range(i, k)}
        C = {ij: c for ij, c in C.items() if c}
        span_by_e = {}
        for e in (1, 2, 4):
            pts = []
            q = 1 << e
            for cs in itertools.product(range(q), repeat=k):
                if not any(cs):
                    continue
                # evaluate: sum over monomials of (c_i c_j) * C_ij, coefficient in GF(q)
                # zero iff for each bit-plane the XOR vanishes
                planes = [0] * e
                for (i, j), c in C.items():
                    m = gmul(cs[i], cs[j], e) if e > 1 else cs[i] & cs[j]
                    for t in range(e):
                        if m >> t & 1:
                            planes[t] ^= c
                if not any(planes):
                    pts.append(cs)
            # dimension of the GF(q)-span of the points = rank over GF(q);
            # it equals the F_2-dim of the F_2-span of all GF(2)-traces of
            # multiples -- simpler: Galois-stable, so take the F_2-span of the
            # bit-planes of {lambda * p}
            vecs = []
            for cs in pts:
                for lam in range(1, q):
                    for t in range(e):
                        v = 0
                        for i in range(k):
                            ci = gmul(lam, cs[i], e) if e > 1 else cs[i]
                            if ci >> t & 1:
                                v ^= vs[i]
                        vecs.append(v)
            span_by_e[e] = echelon(vecs)
        report[w] = (k, [len(span_by_e[e]) for e in (1, 2, 4)])
        gens_total.extend(v for _, v in span_by_e[4])
    bad = {w: r for w, r in report.items() if len(set(r[1])) > 1}
    fvle.say('weight spaces where the span still grows from F_2 to F_16:', bad)
    hom = sum(r[1][2] for r in report.values())
    fvle.say('dim of the span of the homogeneous sandwiches:', hom,
             '; spaces entirely of sandwiches:', sum(1 for r in report.values() if r[1][2] == r[0]),
             'of', len(report))
    S = closure(gens_total)
    fvle.say('the subalgebra they generate: dim', len(S))
    Sv = [v for _, v in S]
    rec, ker = [], []
    for j, x in enumerate(basis):
        key = 0
        for s in Sv:
            key = (key << 896) | ir.reduce_vec(bracket(x, s), S)
        c = 1 << j
        for p, b, cc in rec:
            if key >> p & 1:
                key ^= b; c ^= cc
        if key:
            rec.append((key.bit_length() - 1, key, c)); rec.sort(key=lambda t: -t[0])
        else:
            ker.append(c)
    fvle.say('its normalizer: dim', len(ker), ', codim', n - len(ker))
    import pickle
    pickle.dump({'S': Sv, 'N': [sum(basis[i] for i in range(n) if c >> i & 1) for c in ker]},
                open('runs/fvlesand.pkl', 'wb'))


if __name__ == '__main__':
    main()
