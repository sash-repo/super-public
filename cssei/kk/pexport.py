import sys, json, itertools, kk, pnu
n = int(sys.argv[1]); u = tuple(int(c) for c in sys.argv[2].split(','))
B = pnu.build(n, u)
B.sort(key=lambda f: sum(next(iter(f))))
tab = kk.table(B)
json.dump({'tab': tab, 'deg': [sum(next(iter(f))) - 2 for f in B]}, open('P%d_%s.json' % (n, ''.join(map(str, u))), 'w'))
