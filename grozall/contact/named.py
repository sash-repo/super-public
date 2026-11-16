"""Named contraction schemes, so that an operator found as a coefficient vector can be
written invariantly instead of in components.

A bi-field is a polynomial in (t,p,q) and in TWO sets of fiber variables, u_F and u_G.
On it act (w = weighted order, and (d_F,d_G) = how many fiber indices are used up):

    Om     = w^{ab} d_{u_F^a} d_{u_G^b}          w=0, (1,1)   the fiber pairing
    Pi     = w^{ab} e_a^{(F)} e_b^{(G)}          w=2, (0,0)   the Poisson bracket along D
    ThF    = w^{ab} e_a^{(F)} d_{u_G^b}          w=1, (0,1)
    ThG    = w^{ab} d_{u_F^a} e_b^{(G)}          w=1, (1,0)
    dvF    = w^{ab} e_a^{(F)} d_{u_F^b}          w=1, (1,0)   the divergence of F along D
    dvG    = w^{ab} e_a^{(G)} d_{u_G^b}          w=1, (0,1)
    RF, RG = the Reeb derivative of one factor   w=2, (0,0)

with w^{12} = -1, w^{21} = +1 -- the value read off the contact bracket, not chosen.
"""

import os, sys
from fractions import Fraction as Fr
from itertools import product

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import solve as lsolve, rank  # noqa: E402

from contact import pmul, padd, pscal, e1, e2, reeb  # noqa: E402

# a bi-field: {(a,b,c,iF,jF,iG,jG): coef}


def bi_from(F, G, kF, kG):
    """F, G given as {(a,b,c,i)} with u_1^i u_2^(k-i)."""
    out = {}
    for kf, vf in F.items():
        for kg, vg in G.items():
            key = (kf[0] + kg[0], kf[1] + kg[1], kf[2] + kg[2],
                   kf[3], kF - kf[3], kg[3], kG - kg[3])
            out[key] = out.get(key, Fr(0)) + vf * vg
    return {k: v for k, v in out.items() if v}


def _deriv(B, which, op):
    """Apply a (t,p,q)-derivation to one factor -- impossible on a product, so the
    bi-field carries the two factors' variables apart: the derivation acts on the
    monomial through the factor it belongs to.  We therefore keep the factors apart by
    working with F and G separately; see apply_word."""
    raise NotImplementedError


def apply_word(word, F, G, kF, kG):
    """Apply a word to the PAIR (F,G), each a {(a,b,c,i)} dict, returning a scalar
    polynomial when all fiber indices are used up.

    The two factors are kept apart until the very end: a state is a list of
    (coef, Fpart, Gpart, uF-degree, uG-degree) with Fpart, Gpart plain polynomials
    indexed by the remaining fiber powers.
    """
    # state: dict (iF,jF,iG,jG) -> list of (polyF, polyG) is awkward; instead keep
    # a dict {(iF,jF,iG,jG): list of (coef, polyF, polyG)} and multiply at the end.
    state = {}
    for kf, vf in F.items():
        for kg, vg in G.items():
            key = (kf[3], kF - kf[3], kg[3], kG - kg[3])
            state.setdefault(key, []).append(
                (vf * vg, {kf[:3]: Fr(1)}, {kg[:3]: Fr(1)}))
    for letter in word:
        state = _step(letter, state)
        if not state:
            return {}
    out = {}
    for key, lst in state.items():
        if key != (0, 0, 0, 0):
            continue
        for c, pf, pg in lst:
            for k, v in pmul(pf, pg).items():
                out[k] = out.get(k, Fr(0)) + c * v
    return {k: v for k, v in out.items() if v}


W_OF = {"Om": 0, "Pi": 2, "ThF": 1, "ThG": 1, "dvF": 1, "dvG": 1, "RF": 2, "RG": 2}
U_OF = {"Om": (1, 1), "Pi": (0, 0), "ThF": (0, 1), "ThG": (1, 0),
        "dvF": (1, 0), "dvG": (0, 1), "RF": (0, 0), "RG": (0, 0)}
WPI = {(1, 2): Fr(-1), (2, 1): Fr(1)}


def _du(key, side, a):
    """d/d u_side^a on the fiber multidegree key=(iF,jF,iG,jG); returns (coef, key)."""
    iF, jF, iG, jG = key
    if side == "F":
        if a == 1:
            return (iF, (iF - 1, jF, iG, jG)) if iF else (0, None)
        return (jF, (iF, jF - 1, iG, jG)) if jF else (0, None)
    if a == 1:
        return (iG, (iF, jF, iG - 1, jG)) if iG else (0, None)
    return (jG, (iF, jF, iG, jG - 1)) if jG else (0, None)


EA = {1: e1, 2: e2}


def _step(letter, state):
    out = {}

    def add(key, c, pf, pg):
        if not c or not pf or not pg:
            return
        out.setdefault(key, []).append((c, pf, pg))

    for key, lst in state.items():
        for c, pf, pg in lst:
            if letter == "Om":
                for (a, b), w in WPI.items():
                    ca, k1 = _du(key, "F", a)
                    if not ca:
                        continue
                    cb, k2 = _du(k1, "G", b)
                    if not cb:
                        continue
                    add(k2, c * w * ca * cb, pf, pg)
            elif letter == "Pi":
                for (a, b), w in WPI.items():
                    add(key, c * w, EA[a](pf), EA[b](pg))
            elif letter == "ThF":
                for (a, b), w in WPI.items():
                    cb, k2 = _du(key, "G", b)
                    if not cb:
                        continue
                    add(k2, c * w * cb, EA[a](pf), pg)
            elif letter == "ThG":
                for (a, b), w in WPI.items():
                    ca, k1 = _du(key, "F", a)
                    if not ca:
                        continue
                    add(k1, c * w * ca, pf, EA[b](pg))
            elif letter == "dvF":
                for (a, b), w in WPI.items():
                    cb, k1 = _du(key, "F", b)
                    if not cb:
                        continue
                    add(k1, c * w * cb, EA[a](pf), pg)
            elif letter == "dvG":
                for (a, b), w in WPI.items():
                    cb, k1 = _du(key, "G", b)
                    if not cb:
                        continue
                    add(k1, c * w * cb, pf, EA[a](pg))
            elif letter == "RF":
                add(key, c, reeb(pf), pg)
            elif letter == "RG":
                add(key, c, pf, reeb(pg))
            else:
                raise ValueError(letter)
    return {k: v for k, v in out.items() if v}


def words(kF, kG, k3, W, maxlen=5):
    """All words of the right weighted order that use up the fiber indices."""
    letters = list(W_OF)
    out = []
    for n in range(1, maxlen + 1):
        for w in product(letters, repeat=n):
            if sum(W_OF[x] for x in w) != W:
                continue
            dF = sum(U_OF[x][0] for x in w)
            dG = sum(U_OF[x][1] for x in w)
            if dF != kF or dG != kG or k3 != 0:
                continue
            out.append(w)
    return out
