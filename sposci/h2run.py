#!/usr/bin/env python3
"""h2run.py -- the Mumford-type exponents as ratios of classes in H^2(g),
for the algebras where a Witt probe is blind (m^L(1)) and as a cross-check
(k(1|1), k(1|2), vect(1|1)).  Output: h2run.log."""
import sys
sys.path.insert(0, ".")
import mumford as mu
from mumford import *

OUT = []


def out(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


def gb_from(fields, ell0, W):
    byE = {}
    for D in fields:
        if vzero(D):
            continue
        Dl = br(ell0, D)
        fD = flat(D)
        piv = next(iter(fD))
        lam = flat(Dl).get(piv, 0) / fD[piv]
        check(vzero(vadd(Dl, vsmul(-lam, D))), "not ell0-eigen")
        if abs(lam) <= W:
            byE.setdefault(lam, []).append(D)
    basis = [((lam, j), D) for lam, Ds in byE.items() for j, D in enumerate(Ds)]

    def coord(D):
        if vzero(D):
            return {}
        fD = flat(D)
        piv = next(iter(fD))
        lam = flat(br(ell0, D)).get(piv, 0) / fD[piv]
        vecs = byE.get(lam, [])
        if not vecs:
            return {}          # outside the window
        cols = sorted(set().union(*[flat(V).keys() for V in vecs]) | set(fD), key=str)
        x = solve([[flat(V).get(c, 0) for V in vecs] for c in cols],
                  [fD.get(c, 0) for c in cols])
        check(x is not None, "outside g")
        return {(lam, j): x[j] for j in range(len(vecs)) if x[j] != 0}
    return basis, coord


def run(name, n, fields_of, ell0, mods, wmax=Q(2), W=Q(3)):
    fields = fields_of(int(4 * W) + 6)
    gb, coord = gb_from(fields, ell0, W)
    r = h2_ratio(gb, coord, ell0, mods, wmax, name)
    out("%s: %s" % (name, r))
    return r


Wm = 9
# ---- k(1|1), k(1|2)
for n in (1, 2):
    ell = contact_ell(n)
    l0 = ell(0)
    fo = lambda N, n=n: [K(mono(k, S), n) for S in subsets(n) for k in range(-N, N + 1)]
    gK = gb_from(fo(40), l0, Q(Wm))
    adj = (Module("adj", gK[0], lambda D, X: br(D, X), None, flat), gK[1])
    pars = {k: vpar(D) for k, D in gK[0]}
    adj[0].par = lambda k, pars=pars: pars[k]
    ber = density_module(n, ell, n % 2, contact_mult(n), Q(2 - n) / 2, Wm)
    fun = density_module(n, ell, 0, contact_mult(n), Q(0), Wm)
    run("k(1|%d)" % n, n, fo, l0, {"Ber": ber, "adj": adj, "functions": fun})

# ---- vect(1|1)
ell = vect_ell(1)
l0 = ell(0)


def fv(N):
    out_ = []
    for S in subsets(1):
        for k in range(-N, N + 1):
            for var in ("x", 1):
                D = vf(1)
                D[var] = mono(k, S)
                out_.append(D)
    return out_


gV = gb_from(fv(40), l0, Q(Wm))
adjV = Module("adj", gV[0], lambda D, X: br(D, X), None, flat)
parsV = {k: vpar(D) for k, D in gV[0]}
adjV.par = lambda k: parsV[k]
run("vect(1|1)", 1, fv, l0,
    {"Ber": density_module(1, ell, 1, Div, Q(1), Wm), "adj": (adjV, gV[1]),
     "Vol^2": density_module(1, ell, 0, Div, Q(2), Wm)})

# ---- m^L(1)
l0 = m_ell(0)


def fm(N):
    return [Mf(mono(k, S)) for S in subsets(2) for k in range(-N, N + 1)]


gM = gb_from(fm(40), l0, Q(Wm))
adjM = Module("adj", gM[0], lambda D, X: br(D, X), None, flat)
parsM = {k: vpar(D) for k, D in gM[0]}
adjM.par = lambda k: parsM[k]
rm = run("m^L(1)", 2, fm, l0,
         {"Ber": density_module(2, m_ell, 0, Div, Q(1), Wm), "adj": (adjM, gM[1]),
          "functions": density_module(2, m_ell, 0, Div, Q(0), Wm),
          "Vol^2": density_module(2, m_ell, 0, Div, Q(2), Wm)})
# ---- the cocycle identity for psi_V on m^L(1), odd and even arguments alike
gs, cs = gb_from(fm(40), l0, Q(2))
fl = [X for _, X in gs]
for lab, (M, co) in {"Ber": density_module(2, m_ell, 0, Div, Q(1), Wm),
                     "adj": (adjM, gM[1])}.items():
    E = energies(M, l0)
    wt = lambda D: flat(br(l0, D)).get(next(iter(flat(D))), 0) / next(iter(flat(D).values()))
    nt = 0
    for X in fl:
        for Y in fl:
            for Z in fl:
                if wt(X) + wt(Y) + wt(Z) != 0 or abs(wt(X) + wt(Y)) > 2:
                    continue
                if vzero(br(X, Y)) and vzero(br(Y, Z)) and vzero(br(X, Z)):
                    continue
                pX, pY, pZ = vpar(X), vpar(Y), vpar(Z)
                lhs = (psi_pair(M, co, E, br(X, Y), Z) - psi_pair(M, co, E, X, br(Y, Z))
                       + (-1) ** (pX * pY) * psi_pair(M, co, E, Y, br(X, Z)))
                check(lhs == 0, "cocycle identity %s" % lab)
                nt += 1
    out("m^L(1), %s: cocycle identity on %d triples" % (lab, nt))
out("checks: %d of %d" % (mu.CHECKS[1], mu.CHECKS[0]))
open("h2run.log", "w").write("\n".join(OUT) + "\n")
