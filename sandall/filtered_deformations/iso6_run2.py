import sys
from iso6_tori import split_tori, LABEL
for nm in sys.argv[1:]:
    print("=" * 72); print(LABEL[nm]); r = split_tori(nm); print("   types:", sorted(r)); sys.stdout.flush()
