"""Generate the editable source models (stage 1).

    python tools/generate_source.py [racing|rental ...] [--force]

Writes source/<variant>/<variant>.blend, <variant>.fbx and textures/.
An existing .blend is NOT overwritten unless --force is given, so manual
edits made in Blender are kept; converters always read the .blend.
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blender_bridge as bb  # noqa: E402
import kart_model  # noqa: E402
import textures  # noqa: E402
from cars import BUILD_VARIANTS  # noqa: E402

ROOT = os.path.dirname(HERE)


def generate(variant, force=False):
    out = os.path.join(ROOT, "source", variant)
    blend = os.path.join(out, variant + ".blend")
    if os.path.exists(blend) and not force:
        print("keep existing", blend, "(use --force to regenerate)")
        return blend
    os.makedirs(out, exist_ok=True)
    sc = kart_model.build(variant)
    tex_dir = os.path.join(out, "textures")
    if os.path.isdir(tex_dir):
        shutil.rmtree(tex_dir)
    textures.generate(sc, tex_dir)
    bb.reset()
    bb.create_source(sc, tex_dir)
    bb.save_blend(blend)
    bb.export_fbx(os.path.join(out, variant + ".fbx"))
    print("wrote", blend)
    return blend


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for v in args or list(BUILD_VARIANTS):
        generate(v, force="--force" in sys.argv)
