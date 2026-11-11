"""Minimal Collada 1.4.1 writer (Z-up, metres) for BeamNG meshes.

meshes: list of (name, [(material, pos(n,3), nrm(n,3), uv(n,2), tris(m,3)), ...], matrix4x4 or None)
Each mesh becomes one node with one geometry; sub-meshes share the vertex
arrays and get one <triangles> block per material.
"""
from xml.sax.saxutils import escape

import numpy as np


def _floats(a):
    return " ".join("%.6g" % v for v in np.asarray(a, float).reshape(-1))


def _accessor(sid, arr, params):
    return (f'<source id="{sid}"><float_array id="{sid}-a" count="{arr.size}">{_floats(arr)}</float_array>'
            f'<technique_common><accessor source="#{sid}-a" count="{len(arr)}" stride="{len(params)}">'
            + "".join(f'<param name="{p}" type="float"/>' for p in params) + "</accessor></technique_common></source>")


def write(path, meshes, materials):
    geo, nodes = [], []
    for name, subs, mat in meshes:
        gid = name + "-mesh"
        P = np.vstack([s[1] for s in subs])
        N = np.vstack([s[2] for s in subs])
        U = np.vstack([s[3] for s in subs])
        blocks, off = [], 0
        for m, p, n, uv, t in subs:
            idx = (np.asarray(t, int) + off).reshape(-1)
            blocks.append(f'<triangles material="{escape(m)}" count="{len(t)}">'
                          f'<input semantic="VERTEX" source="#{gid}-vtx" offset="0"/>'
                          f'<input semantic="NORMAL" source="#{gid}-nrm" offset="0"/>'
                          f'<input semantic="TEXCOORD" source="#{gid}-uv" offset="0" set="0"/>'
                          f'<p>{" ".join(map(str, idx))}</p></triangles>')
            off += len(p)
        geo.append(f'<geometry id="{gid}" name="{escape(name)}"><mesh>'
                   + _accessor(gid + "-pos", P, "XYZ") + _accessor(gid + "-nrm", N, "XYZ")
                   + _accessor(gid + "-uv", U, "ST")
                   + f'<vertices id="{gid}-vtx"><input semantic="POSITION" source="#{gid}-pos"/></vertices>'
                   + "".join(blocks) + "</mesh></geometry>")
        m4 = np.eye(4) if mat is None else np.asarray(mat)
        binds = "".join(f'<instance_material symbol="{escape(m)}" target="#{escape(m)}-mat"/>'
                        for m in dict.fromkeys(s[0] for s in subs))
        nodes.append(f'<node id="{escape(name)}" name="{escape(name)}" type="NODE">'
                     f'<matrix sid="transform">{_floats(m4)}</matrix>'
                     f'<instance_geometry url="#{gid}"><bind_material><technique_common>{binds}'
                     f'</technique_common></bind_material></instance_geometry></node>')
    mats = "".join(f'<material id="{escape(m)}-mat" name="{escape(m)}"><instance_effect url="#{escape(m)}-fx"/></material>'
                   for m in materials)
    fx = "".join(f'<effect id="{escape(m)}-fx"><profile_COMMON><technique sid="common"><lambert>'
                 f'<diffuse><color>0.8 0.8 0.8 1</color></diffuse></lambert></technique></profile_COMMON></effect>'
                 for m in materials)
    doc = ('<?xml version="1.0" encoding="utf-8"?>\n'
           '<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">\n'
           '<asset><unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>\n'
           f'<library_effects>{fx}</library_effects>\n<library_materials>{mats}</library_materials>\n'
           '<library_geometries>\n' + "\n".join(geo) + '\n</library_geometries>\n'
           '<library_visual_scenes><visual_scene id="Scene" name="Scene">\n' + "\n".join(nodes)
           + '\n</visual_scene></library_visual_scenes>\n<scene><instance_visual_scene url="#Scene"/></scene>\n</COLLADA>\n')
    open(path, "w", encoding="utf-8").write(doc)
