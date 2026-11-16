"""Statement 4.3.4, item 2): the dimensions of the components of g(C), C as in formula
(28) of the paper, by height with respect to its simple roots.

Usage:  python gC.py [H]     (H defaults to 26)
Prints one line per height: height, dim of the even part, dim of the odd part, total.
"""
import sys
from gA import build

C = [[2, -1, 0], [-1, 0, -2], [0, -2, 0]]
PAR = [0, 1, 1]

if __name__ == "__main__":
    H = int(sys.argv[1]) if len(sys.argv) > 1 else 26
    d, l = build(C, PAR, H)
    for h in sorted(l):
        ev = sum(d[b] for b in l[h] if sum(b[t] * PAR[t] for t in range(3)) % 2 == 0)
        tot = sum(d[b] for b in l[h])
        print(h, ev, tot - ev, tot, flush=True)
