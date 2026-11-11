"""Rough explicit-integration check of the kart jbeam: omega*dt per node at 2 kHz (keep < ~0.9)."""
import collections
import json
import math
import sys

jb = json.load(open(sys.argv[1]))
for pn, p in jb.items():
    if "nodes" not in p:
        continue
    m = {r[0]: r[4]["nodeWeight"] for r in p["nodes"][1:] if isinstance(r, list)}
    k = collections.Counter()
    c = collections.Counter()
    for sec in ("beams", "hydros"):
        spring = 0
        for r in p.get(sec, [])[1:]:
            if isinstance(r, dict) and isinstance(r.get("beamSpring"), (int, float)):
                spring = r["beamSpring"]
            elif isinstance(r, list):
                ks = r[2]["beamSpring"] if len(r) > 2 and isinstance(r[2], dict) and "beamSpring" in r[2] else spring
                k[r[0]] += ks
                k[r[1]] += ks
                cs = r[2].get("beamDamp", 0) if len(r) > 2 and isinstance(r[2], dict) else 0
                c[r[0]] += cs
                c[r[1]] += cs
    worst = sorted(((math.sqrt(k[n] / m[n]) * 0.0005, n, m[n]) for n in k), reverse=True)[:5]
    damp = max((c[n] * 0.0005 / m[n], n) for n in c) if c else (0, "-")
    print(pn, "min weight %.2f" % min(m.values()), ["%s %.1fkg %.2f" % (n, w, x) for x, n, w in worst],
          "max c*dt/m %.2f (%s)" % damp)
