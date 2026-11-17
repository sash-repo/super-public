"""Two single cases of Section 7 checked by hand-made polynomials in S(gl(n)):

  * Proposition 7.4: for gl(4), f = (1,1,-1,-1), Donin's vector (3) for k = 3 is
    compared with w * tr, where w = e14 e23 - e13 e24 and (3) for k = 2 gives 2w;
  * Statement 7.2 and Remark 7.4.2: for gl(4), k = 6, f = (3,1,-2,-2) (the Example
    of Section 5), the span of the vectors (3) has dimension 2 = formula (5.7),
    the multiplicity of f in S^6(gl(4)) is 2, and in Kostant's harmonics it is 1.
"""

import contextlib
import io
from fractions import Fraction

from prim import delta, mul, add, smul, is_primitive, trace, e, donin_vector

with contextlib.redirect_stdout(io.StringIO()):      # run4 prints its own report on import
    from run4 import HS_dim
    from run3 import essential_by_kostant, essential_by_theorem
    from gln import decompose

line = "-" * 70
n = 4

print(line)
print("Proposition 7.4: gl(4), f = (1,1,-1,-1), k = 3")
print(line)
w = add(mul(e(1, 4), e(2, 3)), smul(-1, mul(e(1, 3), e(2, 4))))
v = add(mul(delta(2, 1, 4, n), e(2, 3)), mul(delta(2, 2, 3, n), e(1, 4)),
        smul(-1, mul(delta(2, 1, 3, n), e(2, 4))), smul(-1, mul(delta(2, 2, 4, n), e(1, 3))))
half_w_tr = {m: Fraction(c, 2) for m, c in mul(w, trace(n)).items()}
print(f"w primitive: {is_primitive(w, n)}")
print(f"number of monomials of v: {len(v)}")
print(f"v == w * tr: {v == mul(w, trace(n))}")
print(f"v == (1/2) w * tr: {v == half_w_tr}")
w3 = donin_vector((1, 1), (1, 2), (4, 3), n)          # the vector (3) itself for k = 2
print(f"vector (3) for k = 2 equals 2w: {w3 == smul(2, w)}")
print(f"v is the vector (3) for kappa = (2,1): {donin_vector((2, 1), (1, 2), (4, 3), n) == v}")
print(f"v != 0: {bool(v)}")
print(f"dim HS_f^3 = {HS_dim((1, 1, -1, -1), 3, n)}")
print(f"dim PS_f^3 (multiplicity of f in S^3(gl(4))) = {decompose(n, 3)[3].get((1, 1, -1, -1), 0)}")

print(line)
print("Example of Section 5: gl(4), k = 6, f = (3,1,-2,-2)")
print(line)
f, k = (3, 1, -2, -2), 6
dec = decompose(n, k)
ess = essential_by_kostant(dec, n, k)
print(f"dim HS_f^6 (span of the vectors (3))  = {HS_dim(f, k, n)}")
print(f"formula (5.7)                         = {essential_by_theorem(f, k, n)}")
print(f"multiplicity of f in S^6(gl(4))       = {dec[k].get(f, 0)}")
print(f"multiplicity of f in the harmonics    = {ess[f][k]}")
