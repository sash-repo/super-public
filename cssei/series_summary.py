#!/usr/bin/env python3
"""Compact summary of results/series/*.json (Table 25.2 at N~ = 1)."""
import json, glob, os
def one(label, v):
    sup = v.get('superizations', {})
    s = '; '.join('%s: J %s, J^s %s' % (k, tuple(x['J']), tuple(x['J+odd squares'])) if isinstance(x, dict) else '%s: %s' % (k, x)
                  for k, x in sup.items())
    print('%-34s G %-5d J %-5d simple %-5s 2-cl %-5d centr %s | %s' % (
        label, v['dim'], v['J'][0], v['J simple']['simple'], v['2-closure of J'][0],
        v['centralizer of J in 2-closure'], s))
for f in sorted(glob.glob('results/series/*.json')):
    d = json.load(open(f))
    base = os.path.basename(f)[:-5]
    if 'dim' in d:
        one(base, d)
    else:
        for k, v in d.items():
            one(base + ' ' + k, v)
