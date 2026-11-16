import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sand import Alg, Span
from walk import nil2
from wgrade import wgrading
for p in sys.argv[1:]:
    d = json.load(open(p)); A = Alg(d['tab'])
    sw = [x for x in nil2(d['tab']) if A.is_sandwich2(x)]
    S = A.generated(sw); N = A.normaliser(S)
    neg, pos = wgrading(A, N)
    print(os.path.basename(p), 'dim s', len(S), 'codim n', A.n - len(N), 'g_-..:', neg, '| g_0..:', pos, flush=True)
