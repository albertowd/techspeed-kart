"""Full pipeline.

    python tools/build_all.py [--regen-source] [--no-previews] [--release]

A local build carries one livery per kart (the first of each fleet); --release (the GitHub release workflow)
carries the whole fleet, 36 per version, and stops if the AC sound banks (reference/GoKart_0.5c) are missing,
so a release never ships silent karts.

1. source/<variant>/<variant>.blend is generated only if missing (or with
   --regen-source), so edits made in Blender survive rebuilds.
2. the liveries the build carries, painted into build/skins/<variant>/ (tools/paint_skins.py; they are not kept in
   the source), and their preview renders (AC preview.jpg, BeamNG thumbnails).
3. AC cars -> build/ac/content/cars/<car_id>/
4. BeamNG vehicle -> build/beamng/vehicles/techspeed_kart/
5. archives -> build/dist/, named with the version from overtake/OVERTAKE.md:
     techspeed-kart-<version>-ac.7z       content/cars/<4 karts>      (extract into the assettocorsa folder)
     techspeed-kart-<version>-beamng.zip  vehicles/techspeed_kart/... (drop into BeamNG's mods folder as is)
Nothing is copied into the games; install the archives by hand.
"""
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
sys.path.insert(0, HERE)
from cars import BUILD_VARIANTS  # noqa: E402
SEVEN_ZIP = shutil.which("7z") or r"C:\Program Files\7-Zip\7z.exe"


def run(*args):
    print(">", " ".join(args))
    subprocess.run([PY, *args], check=True, cwd=ROOT)


def version():
    m = re.search(r"^Version:\s*(\S+)", open(os.path.join(ROOT, "overtake", "OVERTAKE.md"), encoding="utf-8").read(), re.M)
    return m.group(1) if m else "dev"


def pack(src_root, rel, out):
    """Archive the folder src_root/rel (7z or zip by extension), keeping rel as the top of the archive."""
    if os.path.exists(out):
        os.remove(out)
    kind = "-tzip" if out.endswith(".zip") else "-t7z"
    subprocess.run([SEVEN_ZIP, "a", kind, "-mx=9", out, rel], check=True, cwd=src_root,
                   stdout=subprocess.DEVNULL)
    print("packed", out, "%.1f MB" % (os.path.getsize(out) / 1e6))


def main():
    flags = set(sys.argv[1:])
    if "--release" in flags:
        os.environ["TECHSPEED_RELEASE"] = "1"          # read by every step (cars.built_skins)
        banks = os.path.join(ROOT, "reference", "GoKart_0.5c", "gokart390", "sfx")
        if not os.path.isdir(banks):
            sys.exit("release build: the AC sound banks are missing (%s); see CONTRIBUTING.md, Setting up" % banks)
    else:
        os.environ.pop("TECHSPEED_RELEASE", None)
    for v in BUILD_VARIANTS:
        args = [os.path.join(HERE, "generate_source.py"), v]
        if "--regen-source" in flags:
            args.append("--force")
        run(*args)
    run(os.path.join(HERE, "paint_skins.py"))
    if "--no-previews" not in flags:
        for v in BUILD_VARIANTS:
            run(os.path.join(HERE, "render_previews.py"), v)
    run(os.path.join(HERE, "export_ac.py"))
    run(os.path.join(HERE, "export_beamng.py"))
    run(os.path.join(HERE, "render_previews.py"), "--beamng-thumbs")

    ver = version()
    dist = os.path.join(ROOT, "build", "dist")
    os.makedirs(dist, exist_ok=True)
    ac_out = "techspeed-kart-%s-ac.7z" % ver
    bng_out = "techspeed-kart-%s-beamng.zip" % ver
    for old in os.listdir(dist):                      # keep only this build's two archives
        if old not in (ac_out, bng_out):
            os.remove(os.path.join(dist, old))
    pack(os.path.join(ROOT, "build", "ac"), "content", os.path.join(dist, ac_out))
    pack(os.path.join(ROOT, "build", "beamng"), "vehicles", os.path.join(dist, bng_out))


if __name__ == "__main__":
    main()
