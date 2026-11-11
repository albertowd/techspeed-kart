"""Reader/writer for Assetto Corsa .kn5 model files (version 5).

Layout (little endian):
  magic "sc6969", int32 version (if version > 5 an extra int32 follows)
  textures:  int32 n, { int32 active, str name, int32 size, bytes }
  materials: int32 n, { str name, str shader, u8 blend, u8 alphaTested,
                        int32 depthMode, int32 nProps { str, f32, f32*2, f32*3, f32*4 },
                        int32 nMaps { str slotName, int32 slot, str texture } }
  node tree: { int32 class, str name, int32 nChildren, u8 active, payload, children... }
    class 1 (transform): 16 x f32 matrix (row-vector convention, translation in 12..14)
    class 2 (mesh): u8 castShadows, u8 visible, u8 transparent,
                    int32 nVerts { pos3, normal3, uv2, tangent3 },
                    int32 nIdx { u16 }, int32 materialId, int32 layer,
                    f32 lodIn, f32 lodOut, f32*3 sphereCenter, f32 sphereRadius, u8 renderable
"""
import struct
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Material:
    name: str
    shader: str = "ksPerPixel"
    blend: int = 0
    alpha_tested: bool = False
    depth_mode: int = 0
    props: list = field(default_factory=list)  # (name, a, (b2), (c3), (d4))
    maps: list = field(default_factory=list)   # (slotName, slot, textureName)


@dataclass
class Node:
    name: str
    cls: int = 1
    active: bool = True
    matrix: np.ndarray = None           # 4x4, row-vector convention
    children: list = field(default_factory=list)
    # mesh payload
    cast_shadows: bool = True
    visible: bool = True
    transparent: bool = False
    verts: np.ndarray = None            # (n, 11) pos, normal, uv, tangent
    indices: np.ndarray = None          # (m,) uint16
    material_id: int = 0
    layer: int = 0
    lod_in: float = 0.0
    lod_out: float = 0.0
    renderable: bool = True


@dataclass
class Kn5:
    version: int = 5
    textures: list = field(default_factory=list)  # (name, bytes)
    materials: list = field(default_factory=list)
    root: Node = None


class _R:
    def __init__(self, data):
        self.d, self.p = data, 0

    def take(self, n):
        b = self.d[self.p:self.p + n]
        self.p += n
        return b

    def i32(self):
        return struct.unpack("<i", self.take(4))[0]

    def u8(self):
        return self.take(1)[0]

    def f32(self, n=1):
        v = struct.unpack("<%df" % n, self.take(4 * n))
        return v[0] if n == 1 else v

    def s(self):
        return self.take(self.i32()).decode("latin-1")


def read(path):
    r = _R(open(path, "rb").read())
    assert r.take(6) == b"sc6969", "not a kn5"
    k = Kn5(version=r.i32())
    if k.version > 5:
        r.i32()
    for _ in range(r.i32()):
        r.i32()
        name = r.s()
        k.textures.append((name, r.take(r.i32())))
    for _ in range(r.i32()):
        m = Material(r.s(), r.s(), r.u8(), bool(r.u8()), r.i32())
        for _ in range(r.i32()):
            m.props.append((r.s(), r.f32(), r.f32(2), r.f32(3), r.f32(4)))
        for _ in range(r.i32()):
            m.maps.append((r.s(), r.i32(), r.s()))
        k.materials.append(m)

    def node():
        cls, name, nch = r.i32(), r.s(), r.i32()
        n = Node(name, cls, bool(r.u8()))
        if cls == 1:
            n.matrix = np.array(r.f32(16), dtype=np.float32).reshape(4, 4)
        elif cls == 2:
            n.cast_shadows, n.visible, n.transparent = bool(r.u8()), bool(r.u8()), bool(r.u8())
            nv = r.i32()
            n.verts = np.frombuffer(r.take(nv * 44), dtype="<f4").reshape(nv, 11).copy()
            ni = r.i32()
            n.indices = np.frombuffer(r.take(ni * 2), dtype="<u2").copy()
            n.material_id, n.layer = r.i32(), r.i32()
            n.lod_in, n.lod_out = r.f32(), r.f32()
            r.f32(4)
            n.renderable = bool(r.u8())
        elif cls == 3:
            # skinned mesh (AC's driver models): its bones (name, inverse bind matrix) and vertices with
            # four weights and four bone indices after the static mesh's eleven floats
            n.cast_shadows, n.visible, n.transparent = bool(r.u8()), bool(r.u8()), bool(r.u8())
            n.bones = [(r.s(), np.array(r.f32(16), dtype=np.float32).reshape(4, 4)) for _ in range(r.i32())]
            nv = r.i32()
            n.verts = np.frombuffer(r.take(nv * 76), dtype="<f4").reshape(nv, 19).copy()
            ni = r.i32()
            n.indices = np.frombuffer(r.take(ni * 2), dtype="<u2").copy()
            n.material_id, n.layer = r.i32(), r.i32()
            n.lod_in, n.lod_out = r.f32(), r.f32()       # (no bounding sphere or renderable flag after them)
        else:
            raise ValueError("unsupported node class %d (%s)" % (cls, name))
        n.children = [node() for _ in range(nch)]
        return n

    k.root = node()
    assert r.p == len(r.d), "trailing bytes: %d" % (len(r.d) - r.p)
    return k


def write(k, path):
    out = bytearray()
    i32 = lambda v: out.extend(struct.pack("<i", v))
    u8 = lambda v: out.extend(struct.pack("<B", int(v)))
    f32 = lambda *v: out.extend(struct.pack("<%df" % len(v), *v))

    def s(v):
        b = v.encode("latin-1")
        i32(len(b))
        out.extend(b)

    out.extend(b"sc6969")
    i32(5)
    i32(len(k.textures))
    for name, data in k.textures:
        i32(1)
        s(name)
        i32(len(data))
        out.extend(data)
    i32(len(k.materials))
    for m in k.materials:
        s(m.name)
        s(m.shader)
        u8(m.blend)
        u8(m.alpha_tested)
        i32(m.depth_mode)
        i32(len(m.props))
        for name, a, b, c, d in m.props:
            s(name)
            f32(a)
            f32(*b)
            f32(*c)
            f32(*d)
        i32(len(m.maps))
        for name, slot, tex in m.maps:
            s(name)
            i32(slot)
            s(tex)

    def node(n):
        i32(n.cls)
        s(n.name)
        i32(len(n.children))
        u8(n.active)
        if n.cls == 1:
            m = n.matrix if n.matrix is not None else np.eye(4)
            f32(*np.asarray(m, dtype=np.float32).reshape(16).tolist())
        else:
            u8(n.cast_shadows)
            u8(n.visible)
            u8(n.transparent)
            v = np.ascontiguousarray(n.verts, dtype="<f4")
            assert len(v) <= 65535, "%s has too many vertices (%d)" % (n.name, len(v))
            i32(len(v))
            out.extend(v.tobytes())
            idx = np.ascontiguousarray(n.indices, dtype="<u2")
            i32(len(idx))
            out.extend(idx.tobytes())
            i32(n.material_id)
            i32(n.layer)
            f32(n.lod_in, n.lod_out)
            p = v[:, 0:3]
            c = (p.min(0) + p.max(0)) / 2
            rad = float(np.sqrt(((p - c) ** 2).sum(1)).max()) if len(p) else 0.0
            f32(*c.tolist(), rad)
            u8(n.renderable)
        for c in n.children:
            node(c)

    node(k.root)
    open(path, "wb").write(bytes(out))


def walk(n, depth=0, parent=np.eye(4)):
    """Yield (node, depth, world matrix)."""
    world = (n.matrix @ parent) if n.cls == 1 else parent
    yield n, depth, world
    for c in n.children:
        yield from walk(c, depth + 1, world)
