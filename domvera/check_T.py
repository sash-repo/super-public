"""Proposition 5.1 of the text: the closed form of the dominance conditions in the base of the ab(3) matrix,
checked against the engine over the whole box (dom_T.json)."""
import json
from itertools import product
S = set(map(tuple, json.load(open('dom_T.json'))))
def formula(a, b, c, d):
    if a < 0 or c < 0 or d < 0 or (a - c) % 2: return False
    k = (c - 3*a)//2 - b
    e1 = int(b != 0); e2 = int(3*a + b + 3*e1 != 0)
    return k >= e1 + e2
n = bad = 0
for l in product(range(0, 9), range(-10, 16), range(0, 7), range(0, 7)):
    n += 1
    if (l in S) != formula(*l): bad += 1
print("weights", n, "dominant", len(S), "mismatches", bad)
