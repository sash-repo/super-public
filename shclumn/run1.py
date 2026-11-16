import time
from models import build_vle, build_kas, build_vas
from excalg import prolong
from rigid1 import rigid_degree0

for name, builder in (('vle(4|3)', build_vle), ('kas(1|6)', build_kas), ('vas(4|4)', build_vas)):
    t=time.time()
    Sp, gm, g0, top = builder()
    comp = prolong(Sp, gm, g0, 1)
    ev=lambda L: (sum(1 for D in L if Sp.vf_par(D)==0), sum(1 for D in L if Sp.vf_par(D)==1))
    print('%s: sdim g_-1 = %s, g_0 = %s, g_1 = %s' % (name, ev(comp[-1]), ev(comp[0]), ev(comp[1])))
    rigid_degree0(Sp, comp, name)
    print('   %.1fs' % (time.time()-t))
