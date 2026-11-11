"""Print the node tree, materials and textures of a kn5 file."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import kn5  # noqa: E402

k = kn5.read(sys.argv[1])
print("version", k.version, "textures", len(k.textures), "materials", len(k.materials))
for t, b in k.textures:
    print("  tex", t, len(b), b[:4])
for m in k.materials:
    print("  mat", m.name, m.shader, "blend", m.blend, "at", m.alpha_tested,
          [(p[0], round(p[1], 3)) for p in m.props], [(x[0], x[1], x[2]) for x in m.maps])
tot = 0
for n, d, w in kn5.walk(k.root):
    if n.cls == 1:
        t = w[3, :3]
        print("  " * d + f"[T] {n.name}  world=({t[0]:.3f},{t[1]:.3f},{t[2]:.3f})")
    else:
        p = n.verts[:, :3] @ w[:3, :3] + w[3, :3]
        tot += len(n.indices) // 3
        print("  " * d + f"[M] {n.name} v={len(n.verts)} t={len(n.indices)//3} "
              f"mat={k.materials[n.material_id].name} min={np.round(p.min(0),3)} max={np.round(p.max(0),3)}"
              + ("" if n.visible else " HIDDEN") + ("" if n.cast_shadows else " noshadow"))
print("tris", tot)
