"""Print the eleven blocks of formula (9), superized, as LaTeX.

The order in which the derivatives of one factor are applied matters -- two odd
derivatives super-commute with a sign -- so the display is generated from the
BLOCKS table of sblocks.py itself rather than transcribed by hand.  Convention:
the operators are written in the usual composition order, the one applied FIRST
standing innermost, i.e. rightmost.

Usage:  python3 sblockprint.py
"""

import os
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from sblocks import BLOCKS                                 # noqa: E402

LETTERS = [("k", "l"), ("p", "q")]
NAME = {"P": r"\mathsf p", "G1a": r"\mathsf g_1'", "G1b": r"\mathsf g_1''",
        "G2a": r"\mathsf g_2'", "G2b": r"\mathsf g_2''",
        "G3a": r"\mathsf g_3'", "G3b": r"\mathsf g_3''",
        "G4": r"\mathsf g_4", "G5a": r"\mathsf g_5'", "G5b": r"\mathsf g_5''",
        "M": r"\mathsf m"}


def render(kinds, FA, FB):
    """The block as (omega factors, A-part, B-part), in LaTeX."""
    om = []
    apA, apB = [], []
    for n, (k1, k2) in enumerate(kinds):
        i1, i2 = LETTERS[n]
        om.append(rf"\omega^{{{i1}{i2}}}")
        for kind, ix in ((k1, i1), (k2, i2)):
            op = rf"\partial_{{z^{ix}}}" if kind[1] == "z" else rf"\partial_{{u^{ix}}}"
            (apA if kind[0] == "A" else apB).append(op)
    A = "".join(reversed(apA))
    B = "".join(reversed(apB))
    pre_A = pre_B = ""
    if FA:
        A = r"\partial_{z^{i}}" + A
        pre_A = "u^{i}"
    if FB:
        B = r"\partial_{z^{j}}" + B
        pre_B = "u^{j}"
    return "".join(om), f"{pre_A}({A}\\widetilde A)", f"{pre_B}({B}\\widetilde{{A'}})"


def frac(c):
    if c.denominator == 1:
        return f"{c.numerator}"
    return rf"\frac{{{abs(c.numerator)}}}{{{c.denominator}}}" if c > 0 \
        else rf"-\frac{{{abs(c.numerator)}}}{{{c.denominator}}}"


def main():
    # the constants of the encoding, as sblocks.py measured them
    KAPPA = {"P": Fraction(-1, 4), "G1a": Fraction(1, 2), "G1b": Fraction(1, 2),
             "G2a": Fraction(-1, 2), "G2b": Fraction(1, 2),
             "G3a": Fraction(1, 4), "G3b": Fraction(1, 4), "G4": Fraction(1, 4),
             "G5a": Fraction(1, 4), "G5b": Fraction(-1, 4), "M": Fraction(1, 4)}

    print("% --- the eleven blocks")
    lines = []
    coeffs = {}
    for (name, fun, kinds, FA, FB, c) in BLOCKS:
        om, A, B = render(kinds, FA, FB)
        lines.append(rf"{NAME[name]}&:={om}\,{A}\,{B},\\")
        coeffs[name] = c * KAPPA[name]
    print("\n".join(lines))

    print("\n% --- the coefficient of each block in Gz, before normalization")
    for (name, fun, kinds, FA, FB, c) in BLOCKS:
        print(f"%   {name:4s}  (9) gives {str(c):>3s},  kappa = {str(KAPPA[name]):>5s}"
              f",  product = {coeffs[name]}")

    scale = Fraction(1) / coeffs["P"]
    print(f"\n% --- normalized so that the first coefficient is 1 (scale {scale})")
    terms = []
    for (name, fun, kinds, FA, FB, c) in BLOCKS:
        v = coeffs[name] * scale
        sign = "+" if v > 0 else "-"
        av = abs(v)
        body = NAME[name] if av == 1 else rf"{frac(av)}\,{NAME[name]}"
        terms.append((sign, body))
    out = terms[0][1] if terms[0][0] == "+" else "-" + terms[0][1]
    for sign, body in terms[1:]:
        out += f"{sign}{body}"
    print(rf"\Gz(2m|n)={out}.")


if __name__ == "__main__":
    main()
