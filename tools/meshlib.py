"""Small procedural mesh toolkit (numpy only).

Kart build coordinates: x = right, y = forward, z = up, metres, origin on the
ground at mid-wheelbase. Exporters convert to each game's axes.
"""
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Mesh:
    verts: np.ndarray                      # (n, 3)
    faces: list                            # list of index tuples (tris / quads / ngons)
    uvs: list = None                       # optional: per face, tuple of (u, v) per corner
    smooth: bool = True
    sharp_angle: float = 40.0              # degrees, for split normals
    modifiers: list = field(default_factory=list)  # e.g. [("subsurf", 2), ("bevel", 0.004, 2)]

    def copy(self):
        return Mesh(self.verts.copy(), list(self.faces),
                    None if self.uvs is None else list(self.uvs),
                    self.smooth, self.sharp_angle, list(self.modifiers))

    def transformed(self, m):
        """Apply a 4x4 (column-vector) matrix."""
        out = self.copy()
        v = np.c_[self.verts, np.ones(len(self.verts))] @ np.asarray(m).T
        out.verts = v[:, :3]
        if np.linalg.det(np.asarray(m)[:3, :3]) < 0:   # mirrored: flip winding
            out.faces = [tuple(reversed(f)) for f in out.faces]
            if out.uvs is not None:
                out.uvs = [tuple(reversed(u)) for u in out.uvs]
        return out

    def mirrored_x(self):
        return self.transformed(np.diag([-1.0, 1, 1, 1]))


def merge(meshes):
    vs, fs, uvs, off = [], [], [], 0
    has_uv = any(m.uvs is not None for m in meshes)
    for m in meshes:
        vs.append(m.verts)
        fs += [tuple(i + off for i in f) for f in m.faces]
        if has_uv:
            uvs += m.uvs if m.uvs is not None else [tuple((0.0, 0.0) for _ in f) for f in m.faces]
        off += len(m.verts)
    first = meshes[0]
    return Mesh(np.vstack(vs), fs, uvs if has_uv else None, first.smooth, first.sharp_angle,
                list(first.modifiers))


# --------------------------------------------------------------------- matrices
def translate(t):
    m = np.eye(4)
    m[:3, 3] = t
    return m


def rot_axis(axis, deg):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    x, y, z = a
    r = np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                  [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                  [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])
    m = np.eye(4)
    m[:3, :3] = r
    return m


def frame(origin, x_axis, y_axis):
    """Matrix mapping local (x, y, z) into a frame given by origin and two axes."""
    x = np.asarray(x_axis, float)
    x /= np.linalg.norm(x)
    y = np.asarray(y_axis, float)
    y = y - x * (y @ x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, origin
    return m


# --------------------------------------------------------------------- paths
def fillet(points, radius, steps=6):
    """Round the interior corners of a polyline with circular arcs."""
    p = [np.asarray(q, float) for q in points]
    if len(p) < 3 or radius <= 0:
        return np.array(p)
    out = [p[0]]
    for i in range(1, len(p) - 1):
        a, b, c = p[i - 1], p[i], p[i + 1]
        d1, d2 = a - b, c - b
        l1, l2 = np.linalg.norm(d1), np.linalg.norm(d2)
        d1, d2 = d1 / l1, d2 / l2
        cosang = np.clip(d1 @ d2, -1, 1)
        ang = math.acos(cosang)
        if ang > math.radians(179) or ang < 1e-3:
            out.append(b)
            continue
        t = radius / math.tan(ang / 2)
        t = min(t, l1 * 0.45, l2 * 0.45)
        r_eff = t * math.tan(ang / 2)
        p1, p2 = b + d1 * t, b + d2 * t
        bis = d1 + d2
        bis /= np.linalg.norm(bis)
        center = b + bis * (r_eff / math.sin(ang / 2))
        v1, v2 = p1 - center, p2 - center
        sweep = math.acos(np.clip((v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
        axis = np.cross(v1, v2)
        axis /= np.linalg.norm(axis)
        for k in range(steps + 1):
            m = rot_axis(axis, math.degrees(sweep * k / steps))[:3, :3]
            out.append(center + m @ v1)
    out.append(p[-1])
    return np.array(out)


def catmull(points, steps=8, closed=False):
    """Catmull-Rom spline through points."""
    p = np.asarray(points, float)
    n = len(p)
    out = []
    rng = range(n) if closed else range(n - 1)
    for i in rng:
        p0 = p[(i - 1) % n] if (closed or i > 0) else 2 * p[0] - p[1]
        p1 = p[i]
        p2 = p[(i + 1) % n]
        p3 = p[(i + 2) % n] if (closed or i + 2 < n) else 2 * p[-1] - p[-2]
        for s in range(steps):
            t = s / steps
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    if not closed:
        out.append(p[-1])
    return np.array(out)


def _frames(path, closed=False):
    """Rotation-minimising frames along a path: returns tangents, normals, binormals."""
    n = len(path)
    tang = np.zeros((n, 3))
    for i in range(n):
        if closed:
            d = path[(i + 1) % n] - path[i - 1]
        elif i == 0:
            d = path[1] - path[0]
        elif i == n - 1:
            d = path[-1] - path[-2]
        else:
            d = path[i + 1] - path[i - 1]
        tang[i] = d / np.linalg.norm(d)
    ref = np.array([0, 0, 1.0]) if abs(tang[0][2]) < 0.9 else np.array([1.0, 0, 0])
    nrm = np.zeros((n, 3))
    nrm[0] = np.cross(tang[0], ref)
    nrm[0] /= np.linalg.norm(nrm[0])
    for i in range(1, n):
        v = nrm[i - 1] - tang[i] * (nrm[i - 1] @ tang[i])
        ln = np.linalg.norm(v)
        nrm[i] = v / ln if ln > 1e-9 else nrm[i - 1]
    bin_ = np.cross(tang, nrm)
    return tang, nrm, bin_


def sweep(path, profile, closed_path=False, caps=True, scale=None):
    """Sweep a closed 2D profile [(a, b), ...] along a 3D path."""
    path = np.asarray(path, float)
    prof = np.asarray(profile, float)
    t, n, b = _frames(path, closed_path)
    k = len(prof)
    verts = []
    for i, p in enumerate(path):
        s = 1.0 if scale is None else scale[i]
        for a, c in prof:
            verts.append(p + (n[i] * a + b[i] * c) * s)
    verts = np.array(verts)
    faces, uvs = [], []
    rings = len(path)
    seg = rings if closed_path else rings - 1
    length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    for i in range(seg):
        i2 = (i + 1) % rings
        for j in range(k):
            j2 = (j + 1) % k
            faces.append((i * k + j, i * k + j2, i2 * k + j2, i2 * k + j))
            v1 = length[i] * 4
            v2 = length[i2] * 4 if i2 else length[-1] * 4
            uvs.append(((j / k, v1), ((j + 1) / k, v1), ((j + 1) / k, v2), (j / k, v2)))
    if caps and not closed_path:
        c0 = len(verts)
        verts = np.vstack([verts, path[0], path[-1]])
        for j in range(k):
            j2 = (j + 1) % k
            faces.append((c0, j2, j))
            faces.append((c0 + 1, (rings - 1) * k + j, (rings - 1) * k + j2))
            uvs += [((0.5, 0.5),) * 3, ((0.5, 0.5),) * 3]
    m = Mesh(verts, faces, uvs)
    return orient_outward(m, verts.mean(0)) if closed_path or caps else m


def circle(r, segs=12):
    return [(r * math.cos(2 * math.pi * i / segs), r * math.sin(2 * math.pi * i / segs))
            for i in range(segs)]


def tube(points, radius, segs=12, bend=0.0, caps=True, closed=False):
    pts = fillet(points, bend) if bend > 0 and not closed else np.asarray(points, float)
    return sweep(pts, circle(radius, segs), closed_path=closed, caps=caps)


def lathe(profile, segs=32, origin=(0, 0, 0), axis=(1, 0, 0), up=(0, 0, 1), closed_ends=False):
    """Revolve a profile [(r, a), ...] around an axis (a = distance along the axis)."""
    ax = np.asarray(axis, float)
    ax /= np.linalg.norm(ax)
    u = np.asarray(up, float)
    if abs(u @ ax) / np.linalg.norm(u) > 0.9:
        u = np.array([1.0, 0, 0]) if abs(ax[0]) < 0.9 else np.array([0, 1.0, 0])
    u = u - ax * (u @ ax)
    u /= np.linalg.norm(u)
    w = np.cross(ax, u)
    o = np.asarray(origin, float)
    k = len(profile)
    verts, faces, uvs = [], [], []
    for i in range(segs):
        ang = 2 * math.pi * i / segs
        d = u * math.cos(ang) + w * math.sin(ang)
        for r, a in profile:
            verts.append(o + ax * a + d * r)
    for i in range(segs):
        i2 = (i + 1) % segs
        for j in range(k - 1):
            faces.append((i * k + j, i * k + j + 1, i2 * k + j + 1, i2 * k + j))
            uvs.append(((i / segs, j / (k - 1)), (i / segs, (j + 1) / (k - 1)),
                        ((i + 1) / segs, (j + 1) / (k - 1)), ((i + 1) / segs, j / (k - 1))))
    verts = np.array(verts)
    if closed_ends:
        for j, sign in ((0, 1), (k - 1, -1)):
            if profile[j][0] > 1e-6:
                c = len(verts)
                verts = np.vstack([verts, o + ax * profile[j][1]])
                for i in range(segs):
                    i2 = (i + 1) % segs
                    f = (c, i2 * k + j, i * k + j) if sign > 0 else (c, i * k + j, i2 * k + j)
                    faces.append(f)
                    uvs.append(((0.5, 0.5),) * 3)
    return orient_outward(Mesh(verts, faces, uvs), o)


def signed_volume(mesh, center):
    """Sum of tetra volumes (center, triangle); > 0 when faces point away from center."""
    v = mesh.verts - np.asarray(center, float)
    vol = 0.0
    for f in mesh.faces:
        for k in range(1, len(f) - 1):
            vol += v[f[0]] @ np.cross(v[f[k]], v[f[k + 1]])
    return vol / 6.0


def orient_outward(mesh, center):
    """Flip winding (and UV order) if the surface faces towards center."""
    if signed_volume(mesh, center) < 0:
        mesh.faces = [tuple(reversed(f)) for f in mesh.faces]
        if mesh.uvs is not None:
            mesh.uvs = [tuple(reversed(u)) for u in mesh.uvs]
    return mesh


def box(center, size, rot=None):
    cx, cy, cz = center
    sx, sy, sz = [s / 2 for s in size]
    v = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
    if rot is not None:
        v = v @ np.asarray(rot)[:3, :3].T
    v += [cx, cy, cz]
    # vertex index = xi*4 + yi*2 + zi
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    uvs = [((0, 0), (1, 0), (1, 1), (0, 1))] * 6
    return Mesh(v, faces, uvs, smooth=False)


def loft(sections, closed=True, cap_start=False, cap_end=False):
    """Skin a list of sections, each an (k, 3) array with the same k."""
    secs = [np.asarray(s, float) for s in sections]
    k = len(secs[0])
    verts = np.vstack(secs)
    faces, uvs = [], []
    ns = len(secs)
    cols = k if closed else k - 1
    for i in range(ns - 1):
        for j in range(cols):
            j2 = (j + 1) % k
            faces.append((i * k + j, i * k + j2, (i + 1) * k + j2, (i + 1) * k + j))
            uvs.append(((j / k, i / ns), ((j + 1) / k, i / ns), ((j + 1) / k, (i + 1) / ns), (j / k, (i + 1) / ns)))
    if cap_start:
        faces.append(tuple(reversed(range(k))))
        uvs.append(tuple((0.5, 0.5) for _ in range(k)))
    if cap_end:
        faces.append(tuple(range((ns - 1) * k, ns * k)))
        uvs.append(tuple((0.5, 0.5) for _ in range(k)))
    m = Mesh(verts, faces, uvs)
    # a closed solid faces out whatever way round its sections run: the games draw one side only
    return orient_outward(m, verts.mean(0)) if closed and cap_start and cap_end else m


def superellipse(cx, cz, rx, rz, n=4.0, k=24, y=0.0, start=0.0, end=360.0, plane="xz"):
    """Points of a superellipse in the xz (constant y) or yz (constant x) plane."""
    pts = []
    full = abs(end - start) >= 360
    cnt = k if full else k + 1
    for i in range(cnt):
        t = math.radians(start + (end - start) * i / (k if full else k))
        c, s = math.cos(t), math.sin(t)
        a = cx + rx * math.copysign(abs(c) ** (2 / n), c)
        b = cz + rz * math.copysign(abs(s) ** (2 / n), s)
        pts.append((a, y, b) if plane == "xz" else (y, a, b))
    return np.array(pts)


def project_uv(mesh, origin, u_axis, v_axis, u_range, v_range, rect):
    """Planar UV projection into an atlas rectangle rect=(u0, v0, u1, v1).

    Panel coords (metres): pu = (p - origin) . u_axis, pv = (p - origin) . v_axis;
    u_range / v_range map panel coords onto the rectangle.
    """
    o = np.asarray(origin, float)
    ua, va = np.asarray(u_axis, float), np.asarray(v_axis, float)
    u0, v0, u1, v1 = rect

    def f(p):
        pu = (p - o) @ ua
        pv = (p - o) @ va
        return (u0 + (pu - u_range[0]) / (u_range[1] - u_range[0]) * (u1 - u0),
                v0 + (pv - v_range[0]) / (v_range[1] - v_range[0]) * (v1 - v0))

    mesh.uvs = [tuple(f(mesh.verts[i]) for i in face) for face in mesh.faces]
    return mesh
