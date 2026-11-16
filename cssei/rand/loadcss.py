"""Load the CSS library files: an adjoint table (d matrices d x d) or a set of
generating matrices of a matrix Lie algebra (GAP files); returns an ad tensor."""
import numpy as np, ast, re, glob, os
from lie import MatAmb, closure, table_of
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'css-lib', 'repo', 'SimpleLieAlgebraBases')
def load(path):
    txt = open(path).read()
    txt = re.sub(r'^[^\[]*', '', txt.strip())
    mats = ast.literal_eval(re.sub(r'\s+', ' ', txt).strip().rstrip(';'))
    M = np.array(mats, np.int64) & 1
    k, n, _ = M.shape
    if k == n:      # try as adjoint table: mats[i][j][k] = coeff of e_k in [e_i,e_j]
        ad = M.transpose(0, 2, 1).astype(np.uint8)   # ad[i][k][j]
        if all(not ad[i, :, i].any() for i in range(n)):
            return ad
    amb = MatAmb(n, 'gl')
    E = closure(amb, [m.ravel().astype(np.uint8) for m in M], 10 ** 6)
    return table_of(amb, E)
def library():
    for p in sorted(glob.glob(ROOT + '/dim*/*.txt')):
        yield os.path.basename(p)[:-4], p
