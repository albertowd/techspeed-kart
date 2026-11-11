"""Write tools/driver_80_skeleton.json from Assetto Corsa's own driver model.

    python tools/dev/driver_skeleton.py [<assettocorsa folder>]

AC's driver_80 (the driver driver3d.ini names) is a skinned mesh hanging from a tree of bone nodes in its
rest pose: a seated driver, knees up, hands forward. The posture file (.knh) and the steering animation the
build writes (tools/ac_driver.py) set those same bones, so the build needs their names, tree and rest
transforms; this keeps them in the repository so the build doesn't need the game installed. Only the bones
are kept, not the mesh-holder nodes (helmet, face, gloves, suit), which the game takes from the model.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import kn5  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
AC = sys.argv[1] if len(sys.argv) > 1 else r"D:\SteamLibrary\steamapps\common\assettocorsa"
OUT = os.path.join(ROOT, "tools", "driver_80_skeleton.json")


def main():
    k = kn5.read(os.path.join(AC, "content", "driver", "driver_80.kn5"))
    used = set()
    for n in _all(k.root):
        if n.cls == 3:
            used.update(b for b, _ in n.bones)
    bones = []

    def walk(n, parent):
        if n.cls != 1:
            return
        # a bone: skinned meshes weight to it, or it's the DRIVER root / a bone's ancestor
        if n.name.startswith("DRIVER:") and (n.name in used or n.name == "DRIVER:DRIVER" or _has_bone(n, used)):
            bones.append(dict(name=n.name, parent=parent, local=[round(float(v), 7) for v in n.matrix.ravel()]))
            parent = len(bones) - 1
        for c in n.children:
            walk(c, parent)

    walk(k.root, -1)
    json.dump(dict(source="content/driver/driver_80.kn5", bones=bones), open(OUT, "w"), indent=1)
    print("wrote %s: %d bones" % (OUT, len(bones)))


def _all(n):
    yield n
    for c in n.children:
        yield from _all(c)


def _has_bone(n, used):
    return any(c.name in used for c in _all(n) if c is not n)


if __name__ == "__main__":
    main()
