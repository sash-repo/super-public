#!/usr/bin/env python3
r"""kas(7;1) = F(kas(1;N|6)) in its standard grading, BGLLS (17.4): the partial
prolong of k(7)_{<=0} and g_1 = Y + (V + P).  The generating functions are those
of kas2.py (contact bracket (17.1)); the non-positive part is realized on O(7;1)
by blattner.py and U is lifted to vector fields through its brackets with g_{-1}.

usage: python3 kas7.py [KAS_NT]
"""
import os
import sys
os.environ['KAS_NT'] = sys.argv[1] if len(sys.argv) > 1 else '1'
import kas2
from f2vf import Space, find_weights, kernel_bits, BitEchelon
from prolong import prolong, log, Graded
from absutil import abstract, Coords
from blattner import Realization
from run_row import analyse, save


def fpar(f):
    ps = {sum(kas2.MONS[k][i] for i in kas2.XI + kas2.ETA) % 2 for k in kas2.bits(f)}
    assert len(ps) == 1
    return ps.pop()


def main():
    comp = kas2.build_kas()
    log('kas in generating functions, N_t=%d: dims %s, total %d'
        % (kas2.NT, {j: len(v) for j, v in comp.items() if v}, sum(len(v) for v in comp.values())))
    basis, degs, pars = [], [], []
    for j in (-2, -1, 0):
        for v in comp[j]:
            # split by parity
            parts = {}
            for k in kas2.bits(v):
                p = sum(kas2.MONS[k][i] for i in kas2.XI + kas2.ETA) % 2
                parts[p] = parts.get(p, 0) | (1 << k)
            assert len(parts) == 1
            basis.append(v)
            degs.append(j)
            pars.append(next(iter(parts)))
    A = abstract(basis, degs, pars, kas2.kb, as_int=True)
    Rz = Realization(A)
    D = Rz.realize()
    fl = [D[i] for i in range(A.n)]
    W = find_weights(Rz.sp, fl)
    sp = Space([1] * Rz.m, Rz.sp.degs, Rz.sp.par, W)
    log('realized on O(%d;1), p-map nonzero on %d, weight rank %d'
        % (Rz.m, sum(1 for b in Rz.pmap if Rz.pmap[b]), len(W[0])))
    neg = [D[i] for i in range(A.n) if A.deg[i] < 0]
    g0 = [D[i] for i in range(A.n) if A.deg[i] == 0]
    G1full = prolong(sp, neg, g0, maxdeg=1, verbose=False)
    log('complete prolong degree 1 (k(7) realized): %d' % G1full.dim(1))
    # lift U: brackets with the g_{-1} basis, in g_0 coordinates
    co = Coords(basis, as_int=True)
    im1 = [i for i in range(A.n) if A.deg[i] == -1]
    i0 = [i for i in range(A.n) if A.deg[i] == 0]
    Dco = Coords([D[i] for i in i0])               # g_0 fields -> index into i0
    cand = G1full.basis(1)

    def sig_field(F):
        out = []
        for e in im1:
            c = Dco(sp.bracket(F, D[e]))
            out.append(c)
        return out

    def sig_func(u):
        out = []
        for e in im1:
            cc = co(kas2.kb(u, basis[e]))           # over the whole abstract basis
            # translate to i0-local indices
            loc = 0
            for t, i in enumerate(i0):
                if (cc >> i) & 1:
                    loc |= 1 << t
            assert cc & ~sum(1 << i for i in i0) == 0
            out.append(loc)
        return out
    nb = len(i0)

    def flat(sig):
        v = 0
        for t, c in enumerate(sig):
            v |= c << (t * nb)
        return v
    candsig = [flat(sig_field(F)) for F in cand]
    lifted = []
    for u in comp[1]:
        target = flat(sig_func(u))
        # solve sum y_c candsig[c] = target
        from blattner import solve_f2
        y = solve_f2(candsig, target, len(cand))
        assert y is not None, 'U element does not lift'
        F = set()
        for c in range(len(cand)):
            if (y >> c) & 1:
                F ^= cand[c]
        lifted.append(F)
    G = prolong(sp, neg, g0, g1=lifted)
    out = analyse(sp, G, 'kas(7;1) standard (17.4), N_t=%d' % kas2.NT)
    save('kas7_NT%d' % kas2.NT, {'row': 'kas7', 'table': 'kas(1;N|6) -> kas(7;N)', 'A': out})


if __name__ == '__main__':
    main()
