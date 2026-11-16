"""Compare the census (explicit2.py output) with the closed form of Proposition 6.3 (dims per pair).
Usage: python3 closed2.py p N explicit-p-N.txt"""
import sys, re, itertools
from collections import Counter
p, N, log = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
P = p**N; q = P//p
got = Counter()
for line in open(log):
    m = re.match(r"F_\((\d+), (\d+)\) -> F_\((\d+), (\d+)\) (even|odd)", line)
    if m: got[(int(m[1]), int(m[2]), int(m[3]), int(m[4]), m[5])] += 1
pred = Counter()
for a, mu in itertools.product(range(p), repeat=2):
    pred[(a, mu, a, mu, 'even')] += q
    for s in range(1, P+1):
        if (a + s) % p == 0 and (s % p or mu == 0):
            pred[(a, mu, s % p, mu, 'even')] += 1
for a in range(p):
    pred[(a, (-a) % p, (a+1) % p, (-a-1) % p, 'odd')] += q
    pred[(a, a % p, (a+1) % p, (a+1) % p, 'odd')] += q
# the census omits the identity alone (dim-1 even endomorphism)
for k in list(pred):
    if k[0:2] == k[2:4] and k[4] == 'even' and pred[k] == 1: del pred[k]
print(f"p={p} N={N}: {'AGREE' if pred == got else 'DISAGREE'}  ({sum(got.values())} maps in {len(got)} pairs)")
if pred != got:
    for k in set(pred) | set(got):
        if pred[k] != got[k]: print(k, pred[k], got[k])
