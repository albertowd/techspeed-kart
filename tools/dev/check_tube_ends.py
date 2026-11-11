"""Find tube ends that touch nothing (a tube stopping short in the air). Builds a variant recording every
open ml.tube's end points, following the tube through transformed() / mirrored_x() / merge into the part it
ends up in, then through the part's node and the fore-aft stretch (_stretch), and measures each end against
everything else in the kart.
    python tools/dev/check_tube_ends.py <variant> [margin_mm]

The pedal pivot shafts show up by design: their ends stick out past the pedal loops."""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import meshlib as ml
import kart_model as km
import bpy  # noqa: F401  (loads mathutils)
from mathutils import Vector
from mathutils.bvhtree import BVHTree

variant = sys.argv[1]
margin = float(sys.argv[2]) / 1000 if len(sys.argv) > 2 else 0.008
recs = []
_tube, _transformed, _mirrored = ml.tube, ml.Mesh.transformed, ml.Mesh.mirrored_x


def tube(points, radius, segs=12, bend=0.0, caps=True, closed=False):
    m = _tube(points, radius, segs, bend, caps, closed)
    if not closed:
        p = np.asarray(points, float)
        m._ends = [(p[0], p[-1], radius)]
    return m


def transformed(self, M):
    new = _transformed(self, M)
    if getattr(self, "_ends", None):
        M = np.asarray(M, float)
        new._ends = [((M @ np.r_[a, 1])[:3], (M @ np.r_[b, 1])[:3], r) for a, b, r in self._ends]
    return new


def mirrored(self):
    new = _mirrored(self)
    if getattr(self, "_ends", None):
        f = np.array([-1.0, 1, 1])
        new._ends = [(a * f, b * f, r) for a, b, r in self._ends]
    return new


_merge = ml.merge


def merge(meshes):
    new = _merge(meshes)
    new._ends = [e for m in meshes for e in (getattr(m, "_ends", None) or [])]
    return new


ml.tube, ml.Mesh.transformed, ml.Mesh.mirrored_x, ml.merge = tube, transformed, mirrored, merge
km.ml = ml
shift = {}
_stretch = km._stretch


def stretch(sc):
    for part in sc.parts:                       # how each part's ends move: as _stretch moves its verts
        v = part.mesh.verts
        if part.node != "BODY":
            shift[id(part)] = None
        elif part.name.startswith(km._STRETCH_PARTS):
            shift[id(part)] = "warp"
        else:
            c = (v[:, 1].min() + v[:, 1].max()) / 2
            shift[id(part)] = float(km.warp_y(c)) - c
    _stretch(sc)


km._stretch = stretch
sc = km.build(variant)
seen = set()
for part in sc.parts:
    for a, b, r in getattr(part.mesh, "_ends", None) or []:
        key = (part.name, tuple(np.round(a, 5)), tuple(np.round(b, 5)))
        if key in seen:
            continue
        seen.add(key)
        s = shift.get(id(part))
        ends = []
        for e in (a, b):
            e = np.array(e, float)
            if s == "warp":
                e[1] = float(km.warp_y(e[1]))
            elif isinstance(s, float):
                e[1] += s
            ends.append(e)
        M = sc.world(part.node)
        ends = [(M @ np.r_[e, 1])[:3] if part.node != "BODY" else e for e in ends]
        recs.append(dict(part=part.name, ends=ends, r=r, pid=id(part)))


def world_verts(part):
    v = np.asarray(part.mesh.verts, float)[:, :3]
    if part.node == "BODY":
        return v
    M = sc.world(part.node)
    return (np.c_[v, np.ones(len(v))] @ M.T)[:, :3]


W = {id(p): (world_verts(p), p.mesh.faces, p.name) for p in sc.parts}


def bvh(ids):
    vs, fs = [], []
    for i in ids:
        v, f, _ = W[i]
        b = len(vs)
        vs += [tuple(x) for x in v]
        fs += [tuple(b + k for k in face) for face in f]
    return BVHTree.FromPolygons(vs, fs)


flagged = []
trees = {}
for t in recs:
    if t["pid"] not in trees:
        trees[t["pid"]] = bvh([i for i in W if i != t["pid"]])
    for k, e in enumerate(t["ends"]):
        hit = trees[t["pid"]].find_nearest(Vector(e))
        d = hit[3] if hit[0] is not None else 9
        if d > t["r"] + margin:
            flagged.append((t["part"], k, np.round(e, 3).tolist(), round(d * 1000), round(t["r"] * 1000, 1)))
for f in sorted(flagged):
    print("%-28s end %d at %-26s nearest %4d mm (r %s mm)" % f)
print("%d free tube ends of %d tubes (ends checked against the other parts; tubes merged in one part may touch only each other)"
      % (len(flagged), len(recs)))
