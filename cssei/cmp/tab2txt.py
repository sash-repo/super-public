#!/usr/bin/env python3
"""Convert a structure table to count.c input; also read the CSS library format
(mats[i] = ad(e_i); mats[i][j][k] = coefficient of e_k in [e_i,e_j]); check Jacobi."""
import json, sys, ast, re
def css(path):
    txt = open(path).read()
    mats = ast.literal_eval(re.sub(r'\s+', ' ', txt).strip().rstrip(';'))
    n = len(mats)
    return [[sum((mats[i][j][k] & 1) << k for k in range(n)) for j in range(n)] for i in range(n)]
def brv(tab, a, b):
    n = len(tab); out = 0
    for i in range(n):
        if a >> i & 1:
            for j in range(n):
                if b >> j & 1: out ^= tab[i][j]
    return out
def jacobi_ok(tab):
    n = len(tab)
    for i in range(n):
        if tab[i][i]: return False
        for j in range(n):
            if tab[i][j] != tab[j][i]: return False
            for k in range(n):
                s = brv(tab, 1 << i, tab[j][k]) ^ brv(tab, 1 << j, tab[k][i]) ^ brv(tab, 1 << k, tab[i][j])
                if s: return False
    return True
def load(path):
    return json.load(open(path))['tab'] if path.endswith('.json') else css(path)
if __name__ == '__main__':
    tab = load(sys.argv[1])
    assert jacobi_ok(tab), 'Jacobi fails'
    n = len(tab)
    print(n); print('\n'.join(' '.join(map(str, r)) for r in tab))
