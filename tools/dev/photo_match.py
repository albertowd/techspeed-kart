"""Render a source model from the cameras of the reference photos, to compare the two side by side.

Each view below lists landmarks: pixels in the photo (measured on a copy scaled to 1200 px wide) and the
model point they show, mostly wheel and hub centres and tyre edges (the parts of known size). A least-squares
fit finds the camera (position, aim, focal length) that puts the model points on the pixels; the model is
rendered from it at the photo's size.

    python tools/dev/photo_match.py [variant] [view ...] [--samples N] [--tag name] [--set key=value ...]
        -> build/renders/match/<tag>/<view>_pair.jpg   photo over render
           build/renders/match/<tag>/<view>_blend.jpg  the two blended half and half
           build/renders/match/<tag>/sheet.jpg         every view's pair on one sheet
    --set hide=<prefix>,...  hides objects whose names start with a prefix (e.g. hide=air_cleaner)
    --set show=<prefix>,...  shows hidden alternatives (e.g. show=air_filter with the box air cleaner exported)
    --set outline=<prefix>,...|<prefix>,...  also writes <view>_outline.jpg: each group's silhouette edge drawn
                             over the photo in its own colour (green, magenta, cyan, orange, yellow, red)
"""
import glob
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "build", "renders", "match")
GRID_W = 1200.0          # landmark pixels are measured on the photo scaled to this width

# wheel landmarks (model, ground at z = 0): rear rims' outer face at x 0.675, front 0.625; rear tyre radius 0.14,
# front 0.128; axles at y -0.56 / 0.56
RR, RF = (0.675, -0.56, 0.14), (0.625, 0.56, 0.128)
VIEWS = {
    # studio side, from the kart's right (the far front tyre, seen through the frame, sets the camera's height)
    "side": dict(photo="514414380", guess=((4.0, 0.0, 0.5), (0.0, 0.0, 0.25)), pts=[
        ((222, 515), RR), ((217, 600), (0.675, -0.56, 0.0)), ((217, 433), (0.675, -0.56, 0.28)),
        ((128, 516), (0.675, -0.70, 0.14)), ((303, 516), (0.675, -0.42, 0.14)),
        ((910, 524), RF), ((911, 605), (0.62, 0.56, 0.0)), ((911, 450), (0.62, 0.56, 0.256)),
        ((833, 525), (0.62, 0.432, 0.128)), ((988, 525), (0.62, 0.688, 0.128)),
        ((858, 417), (-0.56, 0.56, 0.256)), ((808, 467), (-0.56, 0.432, 0.128))]),
    # studio, straight ahead and a little above: the protector's tubes at the middle and round the front corners
    # (their heights checked on the side shot), the side hoops' tops, and the front tyres' bottoms (mostly hidden
    # behind the nose, so barely weighted). (It was fitted to the tyres alone with a guessed focal length and was off
    # by ~40 px over the nose.)
    "front": dict(photo="514401612", guess=((0.0, 4.0, 0.8), (0.0, 0.0, 0.25)), pts=[
        ((591, 636), (0.0, 1.05, 0.114)), ((591, 677), (0.0, 1.05, 0.070)),
        ((55, 590), (0.702, 0.66, 0.114)), ((1098, 590), (-0.702, 0.66, 0.114)),
        ((53, 628), (0.702, 0.66, 0.070)), ((1098, 628), (-0.702, 0.66, 0.070)),
        ((135, 421), (0.69, -0.18, 0.291)), ((993, 421), (-0.69, -0.18, 0.291)),
        ((180, 685), (0.56, 0.60, 0.0)), ((980, 685), (-0.56, 0.60, 0.0))],
        weights=[1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.1, 0.1]),
    # studio, over the right rear looking down into the cockpit
    "cockpit": dict(photo="515935974", guess=((1.0, -1.5, 1.6), (0.0, 0.3, 0.2)), pts=[
        ((418, 240), (-0.50, 0.56, 0.128)), ((1052, 345), RF), ((560, 182), (0.0, 0.29, 0.475)),
        ((812, 190), (0.18, 0.79, 0.203)), ((306, 230), (-0.624, 0.56, 0.128))],
        weights=[1, 1, 1, 0.5, 0.5]),
    # outdoors (kart 14 in the pit lane), from the right and a little above
    "outdoor": dict(photo="514273209_24310826518543682", guess=((4.0, 0.0, 1.5), (0.0, 0.0, 0.25)), pts=[
        ((222, 607), RR), ((222, 673), (0.675, -0.56, 0.0)), ((222, 520), (0.675, -0.56, 0.28)),
        ((153, 600), (0.675, -0.70, 0.14)),
        ((876, 604), RF), ((877, 674), (0.62, 0.56, 0.0)), ((877, 532), (0.62, 0.56, 0.256)),
        ((802, 604), (0.62, 0.432, 0.128)), ((953, 604), (0.62, 0.688, 0.128)),
        ((899, 468), (-0.50, 0.56, 0.128))]),
    # studio, from behind and above on the right
    "rear3q": dict(photo="514248494", guess=((2.5, -3.0, 2.0), (0.0, 0.0, 0.2)), pts=[
        ((833, 622), RR), ((1042, 318), RF), ((600, 228), (-0.50, 0.56, 0.128)),
        ((712, 160), (0.0, 0.29, 0.475)), ((398, 500), (0.0, -0.761, 0.305)),
        ((355, 590), (-0.045, -0.815, 0.178)), ((400, 598), (0.045, -0.815, 0.178))]),
    # studio, straight behind, low
    "rear": dict(photo="514395578", guess=((0.0, -4.0, 0.6), (0.0, 0.0, 0.25)), pts=[
        ((125, 560), (-0.679, -0.56, 0.14)), ((1068, 560), (0.679, -0.56, 0.14)),
        ((180, 680), (-0.585, -0.56, 0.0)), ((1010, 680), (0.585, -0.56, 0.0)),
        ((597, 300), (0.0, 0.29, 0.475)), ((575, 575), (-0.045, -0.815, 0.178)),
        ((632, 575), (0.045, -0.815, 0.178)), ((595, 480), (0.0, -0.761, 0.305))]),
}
VIEW_FILE = os.path.join(HERE, "photo_match_views.json")
if os.path.exists(VIEW_FILE):
    with open(VIEW_FILE, encoding="utf-8") as fh:
        for k, v in json.load(fh).items():
            VIEWS[k] = v


def _rot(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def _look(cam, target):
    """Camera rotation (columns: camera x, y, z in the world) looking from cam at target, z up."""
    d = np.asarray(target, float) - cam
    d /= np.linalg.norm(d)
    x = np.cross(d, (0, 0, 1.0))
    x /= np.linalg.norm(x)
    y = np.cross(x, d)
    return np.c_[x, y, -d]


def _from_R(R):
    """Rotation vector of a rotation matrix."""
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return v * th


def project(params, R0, X, size):
    C, w, lf = params[:3], params[3:6], params[6]
    R = R0 @ _rot(w)
    pc = (X - C) @ R
    W, H = size
    f = math.exp(lf)
    return np.c_[W / 2 + f * pc[:, 0] / -pc[:, 2], H / 2 - f * pc[:, 1] / -pc[:, 2]]


def solve(view, size):
    px = np.array([p for p, _ in view["pts"]], float)
    X = np.array([x for _, x in view["pts"]], float)
    wts = np.array([w for w in view.get("weights", [1.0] * len(px))], float)
    fixed_f = view.get("f")
    best = None
    cam0, tgt0 = (np.array(v, float) for v in view["guess"])
    for f0 in ([fixed_f] if fixed_f else [800.0, 1400.0, 2400.0, 4000.0]):
        R0 = _look(cam0, tgt0)
        prm = np.r_[cam0, np.zeros(3), math.log(f0)]
        lam = 1e-2
        free = np.arange(6 if fixed_f else 7)

        def res(q):
            return ((project(q, R0, X, size) - px) * wts[:, None]).ravel()

        r = res(prm)
        for _ in range(400):
            J = np.zeros((len(r), len(free)))
            for j, k in enumerate(free):
                d = np.zeros(7)
                d[k] = 1e-6
                J[:, j] = (res(prm + d) - r) / 1e-6
            A = J.T @ J
            g = J.T @ r
            step = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), g)
            q = prm.copy()
            q[free] += step
            r2 = res(q)
            if r2 @ r2 < r @ r:
                prm, r, lam = q, r2, lam * 0.4
                if np.abs(step).max() < 1e-9:
                    break
            else:
                lam *= 5
                if lam > 1e8:
                    break
        rms = math.sqrt(r @ r / len(px))
        if best is None or rms < best[0]:
            best = (rms, prm, R0)
    rms, prm, R0 = best
    R = R0 @ _rot(prm[3:6])
    return dict(C=prm[:3], R=R, f=math.exp(prm[6]), rms=rms,
                resid=np.linalg.norm(project(prm, R0, X, size) - px, axis=1))


def render_view(name, view, out_dir, samples, hide, outline=None):
    import bpy
    from mathutils import Matrix
    import blender_bridge as bb
    path = glob.glob(os.path.join(ROOT, "references", "rental", view["photo"] + "*.jpg"))[0]
    photo = Image.open(path).convert("RGB")
    s = GRID_W / photo.size[0]
    size = (int(GRID_W), int(round(photo.size[1] * s)))
    photo = photo.resize(size, Image.LANCZOS)
    cam = solve(view, size)
    print("%-8s rms %.1f px  f %.0f px  cam %s  worst %.1f" % (name, cam["rms"], cam["f"],
                                                          np.round(cam["C"], 2), cam["resid"].max()))
    tmp = os.path.join(out_dir, "_%s.png" % name)
    hide, show = hide
    for o in bpy.data.objects:
        if o.type == "MESH" and not o.name.startswith("preview_"):
            alt = bool(o.get("alternative")) and not any(o.name.startswith(h) for h in show)
            o.hide_render = alt or any(o.name.startswith(h) for h in hide)
    # bb.render sets up the world, sun, floor and camera (a thumbnail), then the fitted camera takes over
    bb.render(tmp, tuple(cam["C"]), tuple(cam["C"] - cam["R"][:, 2]), size=(32, 20), samples=1)
    sc = bpy.context.scene
    sc.render.resolution_x, sc.render.resolution_y = size
    sc.render.resolution_percentage = 100
    sc.cycles.samples = samples
    co = bpy.data.objects["preview_cam"]
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = cam["R"], cam["C"]
    co.matrix_world = Matrix(M.tolist())
    co.data.sensor_fit = "HORIZONTAL"
    co.data.sensor_width = 36.0
    co.data.lens = cam["f"] * 36.0 / size[0]
    bpy.context.scene.render.filepath = os.path.abspath(tmp)
    bpy.ops.render.render(write_still=True)
    ren = Image.open(tmp).convert("RGB")
    os.remove(tmp)
    # landmarks: photo pixel (red) and where the fitted camera puts the model point (green)
    mark = photo.copy()
    d = ImageDraw.Draw(mark)
    for (p, _), e in zip(view["pts"], cam["resid"]):
        d.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], outline=(255, 0, 0), width=2)
    pair = Image.new("RGB", (size[0], 2 * size[1]))
    pair.paste(photo, (0, 0))
    pair.paste(ren, (0, size[1]))
    pair.save(os.path.join(out_dir, name + "_pair.jpg"), quality=88)
    Image.blend(mark, ren, 0.5).save(os.path.join(out_dir, name + "_blend.jpg"), quality=88)
    if outline:
        outline_view(name, photo, size, out_dir, outline)
    return pair


OUTLINE_COLOURS = [(0, 255, 0), (255, 0, 255), (0, 220, 255), (255, 170, 0), (255, 255, 0), (255, 60, 60)]


def outline_view(name, photo, size, out_dir, groups):
    """Silhouette edges of groups of parts (object name prefixes), each rendered alone from the fitted camera, drawn
    over the photo in its own colour: <view>_outline.jpg. Parts in front of a group don't hide it."""
    import bpy
    from PIL import ImageFilter
    sc = bpy.context.scene
    keep = {o.name: o.hide_render for o in bpy.data.objects}
    sc.render.film_transparent = True
    samples, denoise = sc.cycles.samples, sc.cycles.use_denoising
    sc.cycles.samples, sc.cycles.use_denoising = 1, False
    over = photo.copy()
    for k, group in enumerate(groups):
        for o in bpy.data.objects:
            if o.type == "MESH":
                o.hide_render = not (any(o.name.startswith(g) for g in group) and not keep[o.name])
        tmp = os.path.join(out_dir, "_%s_mask.png" % name)
        sc.render.filepath = os.path.abspath(tmp)
        bpy.ops.render.render(write_still=True)
        a = Image.open(tmp).split()[-1].point(lambda v: 255 if v > 100 else 0)
        os.remove(tmp)
        edge = Image.fromarray((np.asarray(a.filter(ImageFilter.MaxFilter(3))).astype(int)
                                - np.asarray(a.filter(ImageFilter.MinFilter(3))).astype(int)).clip(0, 255).astype(np.uint8))
        over.paste(Image.new("RGB", size, OUTLINE_COLOURS[k % len(OUTLINE_COLOURS)]), (0, 0), edge)
    for o in bpy.data.objects:
        o.hide_render = keep[o.name]
    sc.render.film_transparent = False
    sc.cycles.samples, sc.cycles.use_denoising = samples, denoise
    over.save(os.path.join(out_dir, name + "_outline.jpg"), quality=90)


def main():
    args = sys.argv[1:]
    samples, tag, sets = 24, "latest", {}
    rest = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--samples":
            samples = int(args[i + 1])
            i += 2
        elif a == "--tag":
            tag = args[i + 1]
            i += 2
        elif a == "--set":
            k, v = args[i + 1].split("=", 1)
            sets[k] = v
            i += 2
        else:
            rest.append(a)
            i += 1
    variant = rest[0] if rest and rest[0] not in VIEWS else "rental"
    names = [a for a in rest if a in VIEWS] or list(VIEWS)
    hide = ([h for h in sets.get("hide", "").split(",") if h], [h for h in sets.get("show", "").split(",") if h])
    out_dir = os.path.join(OUT, tag)
    os.makedirs(out_dir, exist_ok=True)
    if "--solve-only" in args:
        for n in names:
            v = VIEWS[n]
            path = glob.glob(os.path.join(ROOT, "references", "rental", v["photo"] + "*.jpg"))[0]
            w, h = Image.open(path).size
            cam = solve(v, (int(GRID_W), int(round(h * GRID_W / w))))
            print(n, "rms %.1f" % cam["rms"], "f %.0f" % cam["f"], np.round(cam["C"], 2), np.round(cam["resid"], 1))
        return
    import blender_bridge as bb
    bb.open_source(os.path.join(ROOT, "source", variant, variant + ".blend"))
    outline = [g.split(",") for g in sets["outline"].split("|")] if "outline" in sets else None
    pairs = [render_view(n, VIEWS[n], out_dir, samples, hide, outline) for n in names]
    # contact sheet: pairs in two columns at half size
    tw = 600
    ims = [p.resize((tw, int(p.size[1] * tw / p.size[0])), Image.LANCZOS) for p in pairs]
    cols = 2 if len(ims) > 1 else 1
    rows = (len(ims) + cols - 1) // cols
    hmax = max(i.size[1] for i in ims)
    sheet = Image.new("RGB", (cols * tw, rows * hmax), "white")
    for k, im in enumerate(ims):
        sheet.paste(im, ((k % cols) * tw, (k // cols) * hmax))
    sheet.save(os.path.join(out_dir, "sheet.jpg"), quality=88)
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
