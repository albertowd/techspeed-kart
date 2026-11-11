"""Paint the liveries a build carries into build/skins/<variant>/ (textures.skins_dir).

    python tools/paint_skins.py [variant ...] [--skins 01_techspeed,07_techspeed]

Without --skins it paints cars.built_skins: the whole fleet in a release build (TECHSPEED_RELEASE=1), the first
livery locally, and clears the variant's folder first so a build never ships stale skins. With --skins it adds just
those (render_docs uses it for its showcase karts).
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kart_model  # noqa: E402
import textures  # noqa: E402
from cars import BUILD_VARIANTS, built_skins  # noqa: E402


def paint(variant, names=None):
    out = textures.skins_dir(variant)
    if names is None:
        names = built_skins(textures.skins_for(variant))
        if os.path.isdir(out):
            shutil.rmtree(out)
    sc = kart_model.build(variant)
    textures.paint_skins(sc, names, out)
    print("painted %d %s liveries" % (len(names), variant))


if __name__ == "__main__":
    args = sys.argv[1:]
    names = None
    if "--skins" in args:
        i = args.index("--skins")
        names = args[i + 1].split(",")
        del args[i:i + 2]
    for v in args or list(BUILD_VARIANTS):
        paint(v, names)
