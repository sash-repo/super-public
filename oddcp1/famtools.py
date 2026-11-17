"""From a class in H^1(T_q) to the honest family over Spec Lambda(tau_1..tau_n).

A class is a dict {(gen, exponent): coefficient} in the coordinates of
bashkin.py, or a coefficient vector against quot.Quot(...).basis.  For q odd
the derivation z is odd, so the automorphism exp(tau z) of the retract is even,
as it must be; tau^2 = 0 makes exp(tau z) = 1 + tau z on the coordinates, which
is what ns.SM.from_cocycle composes with the split transition functions.
"""

from fractions import Fraction
import ns
from ns import SM, g_from, g_mul, g_add, g_zero, lp


def cocycle_to_family(ks, coc, npar=1, par_index=None, name=""):
    """coc = {(('u',I),e): c} u {(('w',J,t),e): c}; the family exp(tau.z).split."""
    m = len(ks)
    p = par_index or (m + 1)
    tau = g_from((p,), lp((0, 1)))
    zx = g_zero()
    zxi = {}
    for (g, e), c in coc.items():
        c = Fraction(c)
        if not c:
            continue
        if g[0] == "u":
            zx = g_add(zx, g_mul(tau, g_from(g[1], lp((e, c)))))
        else:
            _, J, t = g
            zxi[t] = g_add(zxi.get(t, g_zero()),
                           g_mul(tau, g_from(J, lp((e, c)))))
    return SM.from_cocycle(ks, zx, zxi, name=name, npar=npar)


def vec_to_coc(Q, v):
    """A coefficient vector against Q.basis as a cocycle dict."""
    out = {}
    for c, (g, e) in zip(v, Q.basis):
        if c:
            out[(g, e)] = out.get((g, e), Fraction(0)) + Fraction(c)
    return out


def show(Q, v):
    """A class in readable form."""
    bits = []
    for c, (g, e) in zip(v, Q.basis):
        if not c:
            continue
        s = (f"x^({e}) xi_{''.join(map(str, g[1]))} d/dx" if g[0] == "u"
             else f"x^({e}) xi_{''.join(map(str, g[1]))} d/dxi_{g[2]}")
        bits.append((f"{c}*" if c != 1 else "") + s)
    return " + ".join(bits) or "0"
