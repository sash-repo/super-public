#!/usr/bin/env python3
"""Reads the two scans runs-degen-N3.txt (order <= 3) and runs-degen-N4.txt
(order <= 4) and tests, mechanically, the four things the eye is tempted to
assert about them:

 1) at order <= 4 the deviating pairs are exactly {0,-1,-2,-3}^2, i.e. both
    weights in Zee_{<=0} --- the guess printed after the table (132);
 2) at order <= 3 they are exactly {0,-1,-2}^2, so the seven new pairs of the
    order-4 scan are precisely those with a weight -3;
 3) on the shifts s = 0..6 the two scans agree row by row, so the order does
    not create dimensions inside a window already scanned --- the deviation is
    governed by the shift alone;
 4) the first failing shift is 1 - 2 min(mu1, mu2), one level above the top
    level of the finite-dimensional submodule of the worse factor.

    ../venv/bin/python checkdegen.py > runs-checkdegen.txt
"""
import re
import sys

PAT = re.compile(r'\s*(-?\d+(?:/\d+)?)\s+(-?\d+(?:/\d+)?)\s+(\[[^\]]*\])\s+<-- differs')


def rows(fn):
    out = {}
    for line in open(fn):
        m = PAT.match(line)
        if m:
            out[(m.group(1), m.group(2))] = eval(m.group(3))
    return out


def main():
    n3, n4 = rows('runs-degen-N3.txt'), rows('runs-degen-N4.txt')
    zle = ['0', '-1', '-2', '-3']
    checks = [
        ("order <= 4: deviating set = {0,-1,-2,-3}^2",
         set(n4) == {(a, b) for a in zle for b in zle}),
        ("order <= 3: deviating set = {0,-1,-2}^2",
         set(n3) == {(a, b) for a in zle[:3] for b in zle[:3]}),
        ("the two scans agree on the shifts 0..6",
         all(n4[k][:7] == n3[k] for k in n3)),
        ("first failing shift = 1 - 2 min(mu1, mu2)",
         all(min(i for i, d in enumerate(v) if d != 1)
             == 1 - 2 * min(int(k[0]), int(k[1])) for k, v in n4.items())),
    ]
    print(f"# deviating pairs: {len(n3)} of 169 at order <= 3, "
          f"{len(n4)} of 169 at order <= 4")
    bad = 0
    for name, ok in checks:
        print(f"{'ok  ' if ok else 'FAIL'}  {name}")
        bad += not ok
    print(f"\n# {len(checks) - bad} checks, {bad} failures")
    print("# note: the grid is mu in (1/2)Zee with |mu| <= 3, so the guess is "
          "tested only there;\n#       a pair with min(mu) = -4 first deviates "
          "at the shift 9 and would need order <= 5.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
