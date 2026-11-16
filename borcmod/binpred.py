"""Closed form for osp(1|2)-invariant bilinear maps F_a x F_b -> F_c (N finite), from the defect formula
 m D1^i D2^j  |->  (c-a-b-i-j) theta m D1^i D2^j + al_i(a) m D1^{i-1}D2^j + (-1)^i al_j(b) m D1^i D2^{j-1},
 al_{2m}=m, al_{2m+1}=x+m.  Block k: chain c_{i,k-i}, i in [max(0,k-2P+1), min(k,2P-1)],
 equations al_{i+1}(a) c_{i+1,j} + (-1)^i al_{j+1}(b) c_{i,j+1} = 0 for i+j=k-1.
Usage: python3 bin1.py p N > bin1-p-N.txt; python3 binpred.py p N bin1-p-N.txt   (Theorem 7.4, Subsec. 7.5)"""
def al(n, x, p): return (n//2) % p if n % 2 == 0 else (x + n//2) % p

def block(k, a, b, p, P, vecs=False):
    lo, hi = max(0, k-2*P+1), min(k, 2*P-1)
    if lo > hi: return 0
    L = hi - lo + 1
    links = []; killed = set()
    for r in range(L-1):
        i = lo + r; j = k - 1 - i            # equation (i,j): variables x_{r+1}=c_{i+1,j}, x_r=c_{i,j+1}
        A, B = al(i+1, a, p), al(j+1, b, p)
        if A and B: links.append(r)
        elif A and not B: killed.add(r+1)
        elif B and not A: killed.add(r)
    if lo > 0 and al(lo, a, p): killed.add(0)                 # equation (lo-1, 2P-1): al_lo(a) c_{lo,2P-1} = 0
    if hi < k and al(k-hi, b, p): killed.add(L-1)              # equation (2P-1, k-2P): al_{k-2P+1}(b) c_{2P-1,k-2P+1} = 0
    comps = []; cur = [0]
    for r in range(L-1):
        if r in links: cur.append(r+1)
        else: comps.append(cur); cur = [r+1]
    comps.append(cur)
    good = [cmp for cmp in comps if not (set(cmp) & killed)]
    return [[lo+x for x in cmp] for cmp in good] if vecs else len(good)

def predicted(a, b, c, par, p, P):
    return sum(block(k, a, b, p, P) for k in range(0, 4*P-1) if k % 2 == par and (k - (c-a-b)) % p == 0)

if __name__ == "__main__":
    import sys, re
    p, N, log = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    P = p**N
    got = {}
    for line in open(log):
        m = re.search(r'a=(\d+) b=(\d+) c=(\d+) (even|odd): dim (\d+)', line)
        if m: got[(int(m[1]), int(m[2]), int(m[3]), m[4] == 'odd')] = int(m[5])
    bad = 0; n = 0
    for a in range(p):
        for b in range(p):
            for c in range(p):
                for par in (0, 1):
                    n += 1
                    if predicted(a, b, c, par, p, P) != got.get((a, b, c, bool(par)), 0):
                        bad += 1; print("MISMATCH", a, b, c, par, predicted(a, b, c, par, p, P), got.get((a, b, c, bool(par)), 0))
    print(f"p={p} N={N}: {n} cases, {bad} mismatches")
