"""Blender (bpy) bridge.

The editable *source model* of each variant is a .blend (plus an .fbx copy)
holding one object per part, parented to empties that carry the game node
names (BODY, STEER_HR, WHEEL_LF, SUSP_LF, ...). Materials are named after
the keys in materials.py. Both game converters read the source through
collect(), so edits made in Blender reach AC and BeamNG alike.
"""
import json
import math
import os
from dataclasses import dataclass

import bpy
import numpy as np
from mathutils import Matrix, Vector

from materials import MATERIALS

INNER_BLACK = ("body_livery", "pod_stripe")      # painted shells: their solidified inner side is plain black


@dataclass
class Baked:
    name: str
    node: str
    material: str
    pos: np.ndarray      # (n, 3) node-local
    nrm: np.ndarray      # (n, 3)
    uv: np.ndarray       # (n, 2)  OpenGL convention
    tris: np.ndarray     # (m, 3) int


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


# ----------------------------------------------------------------- materials
def _faces_inward(m):
    """True if an open shell's faces mostly look in, towards the kart's middle (area-weighted)."""
    v = np.asarray(m.verts, float)
    s = 0.0
    for f in m.faces:
        p = v[list(f)]
        c = p.mean(0)
        n = np.zeros(3)
        for i in range(len(f)):                       # Newell: area-weighted normal of the polygon
            a, b = p[i], p[(i + 1) % len(f)]
            n += np.cross(a, b)
        s += n @ (c - (0.0, 0.0, 0.15))
    return s < 0


def _material(name, tex_dir):
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    spec = MATERIALS.get(name, dict(color=(0.5, 0.5, 0.5), metal=0.0, rough=0.5))
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*spec["color"], 1)
    bsdf.inputs["Metallic"].default_value = spec["metal"]
    bsdf.inputs["Roughness"].default_value = spec["rough"]
    tex = spec.get("texture")
    if tex and tex_dir and os.path.exists(os.path.join(tex_dir, tex)):
        img = bpy.data.images.load(os.path.abspath(os.path.join(tex_dir, tex)))
        tn = nt.nodes.new("ShaderNodeTexImage")
        tn.image = img
        nt.links.new(tn.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


# ----------------------------------------------------------------- source creation
def create_source(sc, tex_dir):
    """Create the editable source objects for a kart_model.Scene in the current file."""
    coll = bpy.context.scene.collection
    empties = {}
    for n in sc.nodes:
        e = bpy.data.objects.new(n.name, None)
        e.empty_display_type = "ARROWS" if n.name != "BODY" else "PLAIN_AXES"
        e.empty_display_size = 0.08
        coll.objects.link(e)
        if n.parent:
            e.parent = empties[n.parent]
        e.matrix_local = Matrix(n.matrix.tolist())
        empties[n.name] = e
    for part in sc.parts:
        m = part.mesh
        if part.material in INNER_BLACK and any(md[0] == "solidify" for md in m.modifiers) and _faces_inward(m):
            # the solidify grows the shell behind the faces: the faces must look out (the nose's looked in, so its
            # black inner shell came out on the outside)
            m = m.copy()
            m.faces = [tuple(reversed(f)) for f in m.faces]
            if m.uvs is not None:
                m.uvs = [tuple(reversed(u)) for u in m.uvs]
        me = bpy.data.meshes.new(part.name)
        me.from_pydata(m.verts.tolist(), [], [list(f) for f in m.faces])
        uvl = me.uv_layers.new(name="UVMap")
        if m.uvs is not None:
            uvl.data.foreach_set("uv", [c for face in m.uvs for uv in face for c in uv])
        me.validate()
        if m.smooth:
            me.shade_smooth()
            me.set_sharp_from_angle(angle=math.radians(m.sharp_angle))
        else:
            me.shade_flat()
        me.materials.append(_material(part.material, tex_dir))
        obj = bpy.data.objects.new(part.name, me)
        coll.objects.link(obj)
        obj.parent = empties[part.node]
        obj.matrix_parent_inverse = Matrix.Identity(4)
        obj["sharp_angle"] = float(m.sharp_angle)
        if getattr(part, "hidden", False):
            # an alternative part (e.g. the other air cleaner): in the file to switch on, never exported
            obj.hide_render = obj.hide_viewport = True
            obj["alternative"] = True
        for mod in m.modifiers:
            kind = mod[0]
            if kind == "subsurf":
                md = obj.modifiers.new("subsurf", "SUBSURF")
                md.levels = md.render_levels = mod[1]
                md.uv_smooth = "PRESERVE_BOUNDARIES"
            elif kind == "solidify":
                md = obj.modifiers.new("solidify", "SOLIDIFY")
                md.thickness = mod[1]
                md.offset = -1.0
                if part.material in INNER_BLACK:
                    # the livery is on the outside only: the inner shell and the rim in plain black plastic
                    # (they shared the outside's UVs, so the inside showed the livery mirrored)
                    me.materials.append(_material("black_plastic", tex_dir))
                    md.material_offset = md.material_offset_rim = 1
            elif kind == "bevel":
                md = obj.modifiers.new("bevel", "BEVEL")
                md.width = mod[1]
                md.segments = mod[2]
                md.limit_method = "ANGLE"
    bpy.context.scene["kart_meta"] = json.dumps(dict(variant=sc.variant, panels=sc.panels))


def save_blend(path):
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(path), check_existing=False)


def export_fbx(path):
    try:
        bpy.ops.preferences.addon_enable(module="io_scene_fbx")
    except Exception:
        pass
    bpy.ops.export_scene.fbx(filepath=os.path.abspath(path), use_selection=False, apply_unit_scale=True,
                             axis_forward="Y", axis_up="Z", use_mesh_modifiers=True, mesh_smooth_type="FACE",
                             add_leaf_bones=False, bake_anim=False, path_mode="COPY", embed_textures=False)


def open_source(path):
    if path.lower().endswith(".blend"):
        bpy.ops.wm.open_mainfile(filepath=os.path.abspath(path))
    else:
        reset()
        try:
            bpy.ops.preferences.addon_enable(module="io_scene_fbx")
        except Exception:
            pass
        bpy.ops.import_scene.fbx(filepath=os.path.abspath(path), axis_forward="Y", axis_up="Z")
    meta = bpy.context.scene.get("kart_meta")
    return json.loads(meta) if meta else {}


# ----------------------------------------------------------------- collect / bake
def _mat_name(obj, idx):
    if idx < len(obj.material_slots) and obj.material_slots[idx].material:
        name = obj.material_slots[idx].material.name
        return name.split(".")[0]          # "steel.001" -> "steel"
    return "black_plastic"


def _evaluate(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    return bpy.data.meshes.new_from_object(obj.evaluated_get(dg))


def _evaluated_mesh(obj, lod):
    """The object's mesh with its modifiers applied. For a LOD (dict(ratio, min_tris, drop, subsurf, bevel)), its
    Subdivision modifiers are held to lod["subsurf"] levels, its Bevels left out unless lod["bevel"], and it is
    decimated to lod["ratio"] of its triangles (never under lod["min_tris"]) by a Decimate modifier added for the
    moment before its Solidify and Bevel, so a shell keeps two even sides. The object is left as it was."""
    if not lod:
        return _evaluate(obj)
    saved = []
    for m in obj.modifiers:
        if m.type == "SUBSURF" and m.levels > lod["subsurf"]:
            saved.append((m, "levels", m.levels))
            m.levels = lod["subsurf"]
        elif m.type == "BEVEL" and not lod["bevel"] and m.show_viewport:
            saved.append((m, "show_viewport", True))
            m.show_viewport = False
    try:
        me = _evaluate(obj)
        me.calc_loop_triangles()
        nt = len(me.loop_triangles)
        ratio = max(lod["ratio"], lod["min_tris"] / max(nt, 1))
        if ratio >= 0.999 or len(obj.data.polygons) <= 4:     # Decimate needs more than 3 faces
            return me
        bpy.data.meshes.remove(me)
        mod = obj.modifiers.new("_lod", "DECIMATE")
        mod.decimate_type = "COLLAPSE"
        mod.ratio = ratio
        mod.use_collapse_triangulate = True
        shell = [i for i, m in enumerate(obj.modifiers) if m.type in ("SOLIDIFY", "BEVEL")]
        if shell:
            obj.modifiers.move(len(obj.modifiers) - 1, shell[0])
        try:
            return _evaluate(obj)
        finally:
            obj.modifiers.remove(mod)
    finally:
        for m, attr, value in saved:
            setattr(m, attr, value)


def _large_parts(tri_verts, co, drop):
    """Mask of the triangles that belong to loose parts (connected pieces of the mesh) at least `drop` metres
    across: the bolts of a bolt set, the nuts on a rim, too small to see from a LOD's distance."""
    n = len(co)
    label = np.arange(n)
    while True:                                       # connected components: spread the lowest vertex index
        low = label[tri_verts].min(axis=1)
        new = label.copy()
        for k in range(3):
            np.minimum.at(new, tri_verts[:, k], low)
        new = new[new]                                # pointer jumping, so long tubes don't take one pass per vertex
        if np.array_equal(new, label):
            break
        label = new
    lo = np.full((n, 3), np.inf)
    hi = np.full((n, 3), -np.inf)
    np.minimum.at(lo, label, co)
    np.maximum.at(hi, label, co)
    size = (hi - lo).max(axis=1)
    return size[label[tri_verts[:, 0]]] >= drop


def collect(lod=None):
    """Read nodes and baked meshes from the open source file.

    Returns (nodes, baked): nodes = {name: (parent_name, local 4x4 np)},
    baked = list of Baked in node-local coordinates (one per object+material).
    lod = dict(ratio, min_tris, drop, loose, subsurf, bevel) bakes a lower-detail copy: objects whose largest size is
    under `drop` metres and loose pieces under `loose` are
    left out and the rest decimated (_evaluated_mesh). The source file is left as it was.
    """
    nodes, baked = {}, []
    for obj in bpy.data.objects:
        if obj.type == "EMPTY":
            par = obj.parent.name if obj.parent else None
            m = (obj.parent.matrix_world.inverted() @ obj.matrix_world) if obj.parent else obj.matrix_world
            nodes[obj.name] = (par, np.array(m))
    if "BODY" not in nodes:
        nodes["BODY"] = (None, np.eye(4))
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.hide_render or obj.name.startswith("preview_"):   # render()'s floor
            continue
        if lod and max(obj.dimensions) < lod["drop"]:
            continue
        node = obj.parent.name if (obj.parent and obj.parent.type == "EMPTY") else "BODY"
        node_world = bpy.data.objects[node].matrix_world if node in bpy.data.objects else Matrix.Identity(4)
        to_local = np.array(node_world.inverted() @ obj.matrix_world)
        me = _evaluated_mesh(obj, lod)
        me.calc_loop_triangles()
        nt = len(me.loop_triangles)
        if nt == 0:
            bpy.data.meshes.remove(me)
            continue
        loops = np.zeros(nt * 3, dtype=np.int32)
        me.loop_triangles.foreach_get("loops", loops)
        tmat = np.zeros(nt, dtype=np.int32)
        me.loop_triangles.foreach_get("material_index", tmat)
        nl = len(me.loops)
        vidx = np.zeros(nl, dtype=np.int32)
        me.loops.foreach_get("vertex_index", vidx)
        co = np.zeros(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3) @ to_local[:3, :3].T + to_local[:3, 3]
        cn = np.zeros(nl * 3)
        me.corner_normals.foreach_get("vector", cn)
        rot = to_local[:3, :3]
        cn = cn.reshape(-1, 3) @ np.linalg.inv(rot)            # normal matrix = inverse-transpose
        cn /= np.maximum(np.linalg.norm(cn, axis=1), 1e-9)[:, None]
        uv = np.zeros(nl * 2)
        if me.uv_layers.active:
            me.uv_layers.active.data.foreach_get("uv", uv)
        uv = uv.reshape(-1, 2)
        if lod:
            keep = _large_parts(vidx[loops.reshape(-1, 3)], co, lod["loose"])
            loops, tmat = loops.reshape(-1, 3)[keep].reshape(-1), tmat[keep]
            if not len(tmat):
                bpy.data.meshes.remove(me)
                continue
        flip = np.linalg.det(rot) < 0
        for mi in np.unique(tmat):
            sel = loops.reshape(-1, 3)[tmat == mi]
            if flip:
                sel = sel[:, ::-1]
            sel = sel.reshape(-1)
            corner = np.c_[co[vidx[sel]], cn[sel], uv[sel]]
            uniq, inv = np.unique(np.round(corner, 5), axis=0, return_inverse=True)
            first = np.zeros(len(uniq), dtype=np.int64)
            first[inv[::-1]] = np.arange(len(inv))[::-1]
            data = corner[first]
            name = obj.name if len(np.unique(tmat)) == 1 else "%s_%d" % (obj.name, mi)
            baked.append(Baked(name, node, _mat_name(obj, mi), data[:, :3], data[:, 3:6], data[:, 6:8],
                               inv.reshape(-1, 3).astype(np.int32)))
        bpy.data.meshes.remove(me)
    return nodes, baked


def baked_object(name, b, material=None):
    """A new mesh object (not linked to a scene) holding a Baked mesh in its own coordinates, with its normals and UVs,
    and `material` (a bpy material; by default the scene's material named b.material, if there is one)."""
    me = bpy.data.meshes.new(name)
    me.from_pydata(b.pos.tolist(), [], b.tris.tolist())
    me.validate()                                     # decimation leaves a few degenerate triangles
    me.update()
    loops = np.zeros(len(me.loops), dtype=np.int32)
    me.loops.foreach_get("vertex_index", loops)
    uv = me.uv_layers.new(name="UVMap")
    uv.data.foreach_set("uv", b.uv[loops].reshape(-1))
    me.normals_split_custom_set(b.nrm[loops].tolist())
    mat = material or bpy.data.materials.get(b.material)
    if mat:
        me.materials.append(mat)
    return bpy.data.objects.new(name, me)


def node_world(nodes, name):
    m = np.eye(4)
    while name is not None and name in nodes:
        parent, local = nodes[name]
        m = local @ m
        name = parent
    return m


# ----------------------------------------------------------------- previews
def render(path, cam_pos, target, lens=50, size=(1280, 800), samples=48, ortho=None):
    sc = bpy.context.scene
    for o in bpy.data.objects:
        if o.type == "EMPTY":
            o.hide_render = True
    if "preview_cam" not in bpy.data.objects:
        world = bpy.data.worlds.new("w")
        world.use_nodes = True
        world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.85, 0.86, 0.88, 1)
        world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.9
        sc.world = world
        sc.render.engine = "CYCLES"
        sc.cycles.device = "CPU"
        sc.cycles.use_denoising = True
        sun = bpy.data.lights.new("sun", "SUN")
        sun.energy = 3.5
        so = bpy.data.objects.new("preview_sun", sun)
        so.rotation_euler = (math.radians(40), math.radians(15), math.radians(30))
        sc.collection.objects.link(so)
        floor = bpy.data.meshes.new("floor")
        floor.from_pydata([(-20, -20, 0), (20, -20, 0), (20, 20, 0), (-20, 20, 0)], [], [(0, 1, 2, 3)])
        fm = bpy.data.materials.new("preview_floor")
        fm.use_nodes = True
        fm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1)
        floor.materials.append(fm)
        fo = bpy.data.objects.new("preview_floor", floor)
        sc.collection.objects.link(fo)
        cam = bpy.data.cameras.new("cam")
        co = bpy.data.objects.new("preview_cam", cam)
        sc.collection.objects.link(co)
        sc.camera = co
    sc.cycles.samples = samples
    # "Standard", like a photo: the default AgX washed the lemon number plates out to beige
    sc.view_settings.view_transform = "Standard"
    sc.render.resolution_x, sc.render.resolution_y = size
    cam = bpy.data.objects["preview_cam"]
    cam.data.type = "ORTHO" if ortho else "PERSP"
    if ortho:
        cam.data.ortho_scale = ortho
    cam.data.lens = lens
    cam.location = Vector(cam_pos)
    cam.rotation_euler = (Vector(target) - Vector(cam_pos)).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = os.path.abspath(path)
    bpy.ops.render.render(write_still=True)
