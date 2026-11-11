"""Side render of a source model laid over a reference side photo at the photo's scale, to check the kart's
proportions. The scale comes from the tyres (real sizes, so it holds whatever the model's wheelbase), the
render is lined up on the rear axle, and the output is the photo and render blended plus a side-by-side.

The photo is a perspective shot from the kart's right: parts on the centre line (seat, steering wheel,
nose) sit ~0.6 m further from the camera than the near tyres and look ~10% smaller there than in the
orthographic render.

    python tools/dev/photo_overlay.py [rental] [front]   -> build/renders/<variant>_overlay[_front].jpg
                                                        (front: lined up on the front axle)
"""
import os
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import blender_bridge as bb  # noqa: E402
import kart_model as km  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "build", "renders")
# photo, rear and front axle pixels, px per metre at the wheel plane (from the tyre diameters, 11" rear /
# 10" front: 295 and 269 px). Axles 1181 px apart: a 1.12 m wheelbase.
PHOTOS = {"rental": ("references/rental/side-right.jpg", (370, 889), (1551, 904), 1052.0)}


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "rental"
    front = "front" in sys.argv[2:]
    photo_path, rear_px, front_px, ppm = PHOTOS[variant]
    ax, ay = front_px if front else rear_px
    photo = Image.open(os.path.join(ROOT, photo_path)).convert("RGB")
    w, h = photo.size
    p = km.VARIANTS[variant]
    r = p["tyre_f" if front else "tyre_r"][0]
    # image centre in model coordinates (y forward, z up; ground at z = 0)
    yc = (km.Y_FRONT if front else km.Y_REAR) + (w / 2 - ax) / ppm
    zc = r + (ay - h / 2) / ppm
    bb.open_source(os.path.join(ROOT, "source", variant, variant + ".blend"))
    tmp = os.path.join(OUT, "_%s_overlay.png" % variant)
    os.makedirs(OUT, exist_ok=True)
    bb.render(tmp, (6.0, yc, zc), (0.0, yc, zc), size=(w, h), samples=24, ortho=w / ppm)
    ren = Image.open(tmp).convert("RGB")
    os.remove(tmp)
    blend = Image.blend(photo, ren, 0.5)
    tag = "_front" if front else ""
    out = os.path.join(OUT, "%s_overlay%s.jpg" % (variant, tag))
    blend.save(out, quality=90)
    pair = Image.new("RGB", (w, 2 * h))
    pair.paste(photo, (0, 0))
    pair.paste(ren, (0, h))
    pair.resize((w // 2, h)).save(os.path.join(OUT, "%s_overlay%s_pair.jpg" % (variant, tag)), quality=88)
    print("overlay", out)


if __name__ == "__main__":
    main()
