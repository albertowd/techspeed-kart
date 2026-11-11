"""Procedural textures: livery atlas per skin, tyre, carbon, radiator, badges.

UV convention: OpenGL (v = 0 at the bottom of the image). The kn5 exporter
flips v for DirectX.
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from kart_model import ATLAS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts")
LOGO_FONT = os.path.join(FONT_DIR, "ariblk.ttf")
TEXT_FONT = os.path.join(FONT_DIR, "arialbd.ttf")
NUM_FONT = os.path.join(FONT_DIR, "impact.ttf")

SIZE = 2048

SKINS = {
    "racing": {
        "01_techspeed_red":   dict(number="1", base=(14, 14, 16), main=(200, 16, 22), accent=(232, 150, 52), plate=(250, 232, 20), num=(10, 10, 10)),
        "07_techspeed_blue":  dict(number="7", base=(12, 16, 40), main=(25, 70, 190), accent=(240, 190, 40), plate=(250, 250, 250), num=(12, 16, 40)),
        "23_techspeed_white": dict(worn=True, number="23", base=(235, 235, 235), main=(200, 16, 22), accent=(20, 20, 22), plate=(250, 232, 20), num=(10, 10, 10)),
        "44_techspeed_green": dict(worn=True, number="44", base=(14, 14, 16), main=(20, 140, 60), accent=(230, 230, 230), plate=(250, 232, 20), num=(10, 10, 10)),
    },
}


class Panel:
    """Maps panel coordinates (metres) of one atlas rectangle to pixels."""

    def __init__(self, img, rect, u_range, v_range):
        self.img, self.draw = img, ImageDraw.Draw(img)
        self.rect, self.ur, self.vr = rect, u_range, v_range

    def px(self, u, v):
        u0, v0, u1, v1 = self.rect
        fu = (u - self.ur[0]) / (self.ur[1] - self.ur[0])
        fv = (v - self.vr[0]) / (self.vr[1] - self.vr[0])
        return ((u0 + fu * (u1 - u0)) * SIZE, (1 - (v0 + fv * (v1 - v0))) * SIZE)

    def scale(self):
        """pixels per metre along u and v."""
        u0, v0, u1, v1 = self.rect
        return ((u1 - u0) * SIZE / (self.ur[1] - self.ur[0]), (v1 - v0) * SIZE / (self.vr[1] - self.vr[0]))

    def fill(self, color):
        u0, v0, u1, v1 = self.rect
        self.draw.rectangle([u0 * SIZE, (1 - v1) * SIZE, u1 * SIZE, (1 - v0) * SIZE], fill=color)

    def poly(self, pts, color):
        self.draw.polygon([self.px(u, v) for u, v in pts], fill=color)

    def band(self, curve_a, curve_b, color):
        """Fill between two curves given as lists of (u, v)."""
        self.poly(list(curve_a) + list(reversed(curve_b)), color)

    def text(self, s, u, v, height_m, color, font=LOGO_FONT, shear=0.18, width_m=None, outline=None):
        """Draw text centred at (u, v), height in metres (along v)."""
        su, sv = self.scale()
        h_px = max(8, int(height_m * sv))
        f = ImageFont.truetype(font, h_px)
        l, t, r, b = f.getbbox(s)
        w, h = r - l, b - t
        pad = int(h * 0.4) + 4
        layer = Image.new("RGBA", (w + 2 * pad + int(h * shear), h + 2 * pad), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        if outline:
            d.text((pad - l, pad - t), s, font=f, fill=outline, stroke_width=max(2, h_px // 12), stroke_fill=outline)
        d.text((pad - l, pad - t), s, font=f, fill=color)
        if shear:
            layer = layer.transform(layer.size, Image.AFFINE, (1, shear, -shear * layer.size[1] * 0.5, 0, 1, 0),
                                    resample=Image.BICUBIC)
        # stretch u so that the text has the requested width (or keeps aspect in metres)
        target_w = int((width_m * su) if width_m else layer.size[0] * su / sv)
        layer = layer.resize((max(1, target_w), layer.size[1]), Image.LANCZOS)
        cx, cy = self.px(u, v)
        self.img.alpha_composite(layer, (int(cx - layer.size[0] / 2), int(cy - layer.size[1] / 2)))

    def rounded(self, u0, v0, u1, v1, radius_m, color):
        a, b = self.px(u0, v1), self.px(u1, v0)
        r = int(radius_m * self.scale()[0])
        self.draw.rounded_rectangle([min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])], r, fill=color)


def _swoosh(n, u_start, u_end, v_lo, v_hi, bend, thick):
    """Curved stripe as a polygon: n points along u."""
    top, bot = [], []
    for i in range(n + 1):
        t = i / n
        u = u_start + (u_end - u_start) * t
        v = v_lo + (v_hi - v_lo) * t + bend * math.sin(math.pi * t)
        w = thick * math.sin(math.pi * min(1, t * 1.2 + 0.05)) ** 0.6
        top.append((u, v + w / 2))
        bot.append((u, v - w / 2))
    return top + list(reversed(bot))


def livery(sc, skin, out_path):
    style = skin.get("style")
    if style:                                     # the rental fleets
        img = Image.new("RGBA", (SIZE, SIZE), BLACK + (255,))
        _nose_fleet(img, sc, style == "v1")
        _panel_fleet(img, sc, skin, style == "v1")
        if style == "v2":
            _pods_fleet(img, sc, skin)
            _rear_fleet_v2(img, sc, skin)
        else:
            _rear(img, sc, skin)                  # the v1's own cover (onboard frames)
        img = img.filter(ImageFilter.GaussianBlur(0.6))
        img.convert("RGB").save(out_path)
        return
    img = Image.new("RGBA", (SIZE, SIZE), skin["base"] + (255,))
    base, main, acc = skin["base"], skin["main"], skin["accent"]
    white = (245, 245, 245)
    rental = sc.family == "rental"

    # ---------------- nose (u = viewer's left->right from the front, v rows along the profile)
    pn = sc.panels["nose"]
    P = Panel(img, ATLAS["nose"], pn["u"], pn["v"])
    rows = pn["rows"]
    P.fill(base)
    vtop0, vtop1, vfront = rows[3], rows[6], rows[9]
    for s in (-1, 1):
        # red side wings with an orange swoosh
        P.poly([(s * 0.20, rows[0]), (s * 0.52, rows[0]), (s * 0.52, rows[-1]), (s * 0.30, rows[-1]),
                (s * 0.24, vfront), (s * 0.20, vtop1)], main)
        P.poly([(s * u, v) for u, v in _swoosh(24, 0.24, 0.50, vtop1 + 0.01, rows[2], 0.03, 0.035)], acc)
        P.poly([(s * u, v) for u, v in _swoosh(24, 0.28, 0.52, vfront - 0.01, vtop1 + 0.02, 0.02, 0.018)], base)
    P.text("TECHSPEED", 0.0, (rows[6] + rows[8]) / 2 + 0.005, 0.042, white, width_m=0.36)
    P.text("RACING EQUIPMENT", 0.0, (rows[7] + rows[9]) / 2 - 0.012, 0.018, white, font=TEXT_FONT, shear=0, width_m=0.27)

    # ---------------- front panel (u = viewer left->right, v from the bottom up)
    pp = sc.panels["panel"]
    P = Panel(img, ATLAS["panel"], pp["u"], pp["v"])
    P.fill(base)
    total = pp["v"][1]
    plate_lo, plate_hi = total * 0.60, total * 0.93
    P.poly([(u, v) for u, v in _swoosh(20, -0.02, 0.10, 0.02, plate_lo - 0.02, 0.04, 0.05)], main)
    P.poly([(u, v) for u, v in _swoosh(20, 0.00, 0.07, 0.06, plate_lo - 0.05, 0.03, 0.02)], acc)
    P.poly([(u, v) for u, v in _swoosh(20, 0.02, -0.12, 0.10, plate_lo - 0.01, -0.03, 0.035)], main)
    P.rounded(-0.135, plate_lo, 0.135, plate_hi, 0.04, skin["plate"])
    P.text(skin["number"], 0.0, (plate_lo + plate_hi) / 2, (plate_hi - plate_lo) * 0.72, skin["num"], font=NUM_FONT,
           shear=0.12)
    P.text("TECHSPEED", 0.0, (plate_hi + total) / 2 + 0.004, 0.018, white, width_m=0.15)

    # ---------------- side pods (u: metres, panel 'front' tells which end is the front)
    for key in ("pod_L", "pod_R"):
        if key not in sc.panels:              # the rental v1 has no side pods
            continue
        pd = sc.panels[key]
        P = Panel(img, ATLAS[key], pd["u"], pd["v"])
        P.fill(base)
        f = pd["front"]                       # +1: front at +u (right pod), -1: front at -u (left pod)
        tops = pd["top"]
        if pd.get("kind") == "rental":
            def Uy(y):
                return f * y
            # red band under the top edge, sloping up towards the rear, thin orange line under it
            P.poly([(Uy(y), z - 0.075) for y, z in tops] + [(Uy(y), 0.47) for y, z in reversed(tops)], main)
            P.poly([(Uy(y), z - 0.083) for y, z in tops] + [(Uy(y), z - 0.075) for y, z in reversed(tops)], acc)
            # swoosh from the front bottom
            P.poly([(Uy(y), v) for y, v in _swoosh(30, 0.30, -0.10, 0.13, 0.21, 0.03, 0.05)], main)
            a, b = sorted((Uy(0.14), Uy(0.28)))
            P.rounded(a, 0.135, b, 0.27, 0.02, skin["plate"])
            P.text(skin["number"], (a + b) / 2, 0.20, 0.11, skin["num"], font=NUM_FONT, shear=0.12)
            if key == "pod_L":
                P.text("VELOPARK", Uy(-0.04), 0.17, 0.045, acc, width_m=0.22)
            continue

        def U(t):                             # t = 0 rear end .. 1 front end
            return f * (-0.34 + 0.67 * t)

        # top edge stripe in main colour (the pod top is smeared by the side projection)
        P.poly([(U((y + 0.34) / 0.67), z - 0.035) for y, z in tops] +
               [(U((y + 0.34) / 0.67), 0.30) for y, z in reversed(tops)], main)
        # swooshes running forward from the plate
        P.poly([(U(t), v) for t, v in _swoosh(30, 0.20, 1.02, 0.07, 0.10, 0.06, 0.07)], main)
        P.poly([(U(t), v) for t, v in _swoosh(30, 0.24, 1.0, 0.125, 0.12, 0.05, 0.03)], acc)
        P.poly([(U(t), v) for t, v in _swoosh(30, 0.35, 1.0, 0.035, 0.06, 0.02, 0.025)], main)
        a, b = sorted((U(0.02), U(0.21)))
        P.rounded(a, 0.055, b, 0.235, 0.02, skin["plate"])
        P.text(skin["number"], (a + b) / 2, 0.145, 0.15, skin["num"], font=NUM_FONT, shear=0.12)
        P.text("TECHSPEED", U(0.55), 0.165, 0.040, white, width_m=0.30)
        P.text("RACING EQUIPMENT", U(0.57), 0.125, 0.016, white, font=TEXT_FONT, shear=0, width_m=0.20)

    _rear(img, sc, skin)

    img = img.filter(ImageFilter.GaussianBlur(0.6))
    img.convert("RGB").save(out_path)


TYRE_LINE_V = 0.43     # the new-tyre red line: on the tread, a little towards the inner side


def _rear(img, sc, skin):
    """The rear cover's print for the racing kart and the rental v1."""
    white = (245, 245, 245)
    base = skin.get("base", BLACK)
    rental = sc.family == "rental"
    # ---------------- rear bumper (u = x seen from behind)
    pr = sc.panels["rear"]
    P = Panel(img, ATLAS["rear"], pr["u"], pr["v"])
    rows = pr["rows"]
    P.fill((22, 22, 24) if not rental else base)
    if pr.get("kind") == "rental_cover_v1":
        # the v1's cover (onboard frames): dark plastic, red over the rounded top of the rear, an orange reflector
        # on each side of the rear face, and "CURTA VENCER" in yellow across the straight rear face; nothing else
        # rows (kart_model._rear_cover_rental, v1): 2-4 the top panel to its rear edge, 5-6 the vertical step in
        # the middle / the slope at the sides, 7-8 the shelf, 9-11 the lower face
        P.fill((16, 16, 18))                                  # the same black as the rest of the kart
        P.rounded(-0.43, rows[5] - 0.004, 0.43, rows[4] + 0.030, 0.024, (196, 44, 36))     # red round the top edge
        P.text("CURTA VENCER", 0.0, (rows[5] + rows[6]) / 2, 0.030, (240, 200, 30), font=TEXT_FONT, shear=0,
               width_m=0.38)                                  # on the vertical step, under the red edge
        # round orange decals (as the v2's) on the side slopes curving up over the tyres, facing back and up
        su, sv = P.scale()
        for x in (-0.50, 0.50):
            cx, cy = P.px(x, rows[6])
            r = 0.034
            P.draw.ellipse([cx - r * su, cy - r * sv, cx + r * su, cy + r * sv], fill=(238, 96, 40))
        # TECHSPEED on the top deck behind the seat
        P.text("TECHSPEED", 0.0, (rows[2] + rows[3]) / 2, 0.024, (235, 235, 235), width_m=0.24)
    elif rental:
        black = (16, 16, 18)
        P.fill(black)                                         # the rental cover is plain black plastic
        # yellow number plate on the rear face, over the exhaust tips (rear.jpg)
        vb0, vb1 = rows[6] - 0.01, rows[5] + 0.025
        P.rounded(-0.085, vb0, 0.085, vb1, 0.02, skin["plate"])
        P.text(skin["number"], 0.0, (vb0 + vb1) / 2, (vb1 - vb0) * 0.7, skin["num"], font=NUM_FONT, shear=0.12)
        # web address across the deck behind the seat (rear-3q.jpg)
        P.text("WWW.TECHSPEEDKART.COM", 0.0, (rows[3] + rows[4]) / 2, 0.026, white, font=TEXT_FONT, shear=0,
               width_m=0.42)
        # four round orange stickers on the rear face, two each side (decals, not lights)
        su, sv = P.scale()
        vc = (rows[5] + rows[6]) / 2
        for x in (-0.46, -0.36, 0.36, 0.46):
            cx, cy = P.px(x, vc)
            r = 0.038
            P.draw.ellipse([cx - r * su, cy - r * sv, cx + r * su, cy + r * sv], fill=(235, 70, 20))
            if abs(x) > 0.4:                                  # the outer ones carry a V mark
                P.poly([(x - 0.022, vc + 0.012), (x, vc - 0.018), (x + 0.022, vc + 0.012), (x + 0.012, vc + 0.012),
                        (x, vc - 0.004), (x - 0.012, vc + 0.012)], (250, 170, 60))
    else:
        P.text("TECHSPEED", 0.0, (rows[5] + rows[6]) / 2, 0.03, white, width_m=0.30)


# ---------------------------------------------------------------- rental fleet liveries
# The rental fleets as photographed, numbered 1 to FLEET_SIZE: the same livery on every kart, only the number
# changes. v2 (steel protector, front.jpg, front-panel-nose.jpg, side-right.jpg, sidepod-ballast-fuel.jpg,
# rear.jpg, rear-3q.jpg): gloss black (the bluish sheen in the photos is the sky reflected) with red-to-orange
# flames, a pale orange hook and gold pinstripes, lemon plates with a serif number, Velopark logos. v1 (rubber band, protector-v1-top.jpg,
# protector-v1-side-3q.jpg): gloss black with red and gold, a condensed number.
FLEET_SIZE = 36
WORN_EVERY = 3            # every third kart (3, 6, 9...) runs worn tyres, in both games
BLACK = (16, 16, 18)
RED = (222, 34, 26)
ORANGE = (246, 122, 30)
PALE = (250, 184, 96)     # the pale orange inside the flames
GOLD = (214, 164, 70)
LEMON = (255, 236, 0)
WHITE = (245, 245, 245)


def _font(name, fallback):
    path = os.path.join(FONT_DIR, name)
    return path if os.path.exists(path) else fallback


SERIF_NUM = _font("COOPBL.TTF", NUM_FONT)          # the v2's plates: a fat serif figure
NARROW_NUM = _font("ARIALNB.TTF", NUM_FONT)        # the v1's: tall and condensed


def _fleet(style):
    return {"%02d_techspeed" % n: dict(number=str(n), style=style, worn=n % WORN_EVERY == 0, plate=LEMON,
                                       num=(12, 12, 12)) for n in range(1, FLEET_SIZE + 1)}


def skins_for(variant):
    """The skins of a variant: its own list, else its family's (rental_v1 has its own fleet livery)."""
    from kart_model import family
    return SKINS.get(variant) or SKINS[family(variant)]


def _mix(a, b, t):
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def _grad_poly(P, pts, c0, c1, u0, u1):
    """Fill a polygon with a colour running from c0 at u0 to c1 at u1."""
    x0, x1 = P.px(u0, 0)[0], P.px(u1, 0)[0]
    mask = Image.new("L", P.img.size, 0)
    ImageDraw.Draw(mask).polygon([P.px(u, v) for u, v in pts], fill=255)
    box = mask.getbbox()
    if not box:
        return
    w = box[2] - box[0]
    xs = np.arange(box[0], box[2])
    t = np.clip((xs - x0) / ((x1 - x0) or 1), 0, 1)
    row = np.array([_mix(c0, c1, k) for k in t], np.uint8)
    grad = Image.fromarray(np.repeat(row[None], box[3] - box[1], 0), "RGB").convert("RGBA")
    P.img.paste(grad, box[:2], mask.crop(box))


def _curve(n, f):
    return [f(i / n) for i in range(n + 1)]


VELOPARK_LOGO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "source", "logos", "velopark.png")


def _velopark_px(img, cx, cy, h, aspect=1.0):
    """The Velopark logo (source/logos/velopark.png: the orange V with chequered flags and the VELOPARK
    badge, from the track's site) centred at pixel (cx, cy), h pixels tall (aspect: x pixels per y pixel)."""
    logo = Image.open(VELOPARK_LOGO).convert("RGBA")
    w = max(1, int(round(h * logo.size[0] / logo.size[1] * aspect)))
    logo = logo.resize((w, max(1, int(round(h)))), Image.LANCZOS)
    img.alpha_composite(logo, (int(cx - logo.size[0] / 2), int(cy - logo.size[1] / 2)))


def _velopark(P, u, v, size):
    """The Velopark logo on a panel, size (m) tall."""
    su, sv = P.scale()
    cx, cy = P.px(u, v)
    _velopark_px(P.img, cx, cy, size * sv, su / sv)


def _roundel(P, u, v, r, logo):
    """A round sticker: red-orange, with the Velopark logo (or plain)."""
    su, sv = P.scale()
    cx, cy = P.px(u, v)
    P.draw.ellipse([cx - r * su, cy - r * sv, cx + r * su, cy + r * sv], fill=(238, 78, 30))
    if logo:
        _velopark(P, u, v, r * 1.6)


def _plate(P, u0, v0, u1, v1, r, skin, font, scale=0.72, shear=0.10):
    P.rounded(u0, v0, u1, v1, r, skin["plate"])
    P.text(skin["number"], (u0 + u1) / 2, (v0 + v1) / 2, (v1 - v0) * scale, skin["num"], font=font, shear=shear)


def _nose_fleet(img, sc, v1):
    pn = sc.panels["nose"]
    P = Panel(img, ATLAS["nose"], pn["u"], pn["v"])
    rows = pn["rows"]
    r = lambda k: float(np.interp(k, range(len(rows)), rows))       # noqa: E731  (fractional profile row)
    P.fill(BLACK)
    for s in (-1, 1):
        if v1:
            # red over each fin's top and front, a black slash across it and two gold chevrons
            P.poly([(s * 0.21, r(2.6)), (s * 0.59, r(2.2)), (s * 0.59, r(8.6)), (s * 0.40, r(8.6)),
                    (s * 0.30, r(6.0)), (s * 0.21, r(4.4))], RED)
            P.poly([(s * 0.34, r(3.0)), (s * 0.40, r(3.0)), (s * 0.50, r(8.6)), (s * 0.44, r(8.6))], BLACK)
            for k, off in enumerate((0.0, 0.035)):
                a = 0.47 + off
                P.poly([(s * a, r(3.4)), (s * (a + 0.02), r(3.4)), (s * (a + 0.06), r(5.6)), (s * (a + 0.02), r(7.6)),
                        (s * a, r(7.6)), (s * (a + 0.04), r(5.6))], GOLD)
            continue
        # v2: a red band across the fin's top, rising from low beside the panel's foot to the fin's tip (front.jpg),
        # black behind it and in front of it, a gold pinstripe along its back edge and a pale orange stripe in
        # its lower part hooking down at the inner end; the fin's front face black with the logo
        top = _curve(16, lambda t: (s * (0.17 + 0.42 * t), r(5.0 - 1.7 * t ** 0.8)))
        bot = _curve(16, lambda t: (s * (0.17 + 0.42 * t), r(6.1 - 0.8 * t)))
        _grad_poly(P, top + bot[::-1], RED, (236, 58, 24), s * 0.17, s * 0.59)
        P.band(_curve(16, lambda t: (s * (0.19 + 0.40 * t), r(4.85 - 1.7 * t ** 0.8))),
               _curve(16, lambda t: (s * (0.19 + 0.40 * t), r(4.97 - 1.7 * t ** 0.8))), GOLD)
        hook = (_curve(14, lambda t: (s * (0.58 - 0.31 * t), r(4.55 + 0.75 * t))) +
                [(s * 0.25, r(5.6)), (s * 0.245, r(5.95)), (s * 0.27, r(5.75))] +
                _curve(14, lambda t: (s * (0.28 + 0.30 * t), r(5.55 - 0.65 * t))))
        P.poly(hook, PALE)
        _velopark(P, s * 0.46, r(6.9), 0.07)
    P.text("TECHSPEED", 0.0, (r(6) + r(8)) / 2, 0.040, WHITE, font=LOGO_FONT, shear=0, width_m=0.34)


def _panel_fleet(img, sc, skin, v1):
    pp = sc.panels["panel"]
    P = Panel(img, ATLAS["panel"], pp["u"], pp["v"])
    total = pp["v"][1]
    ws = np.array(pp["width"])
    w = lambda v: float(np.interp(v, ws[:, 0], ws[:, 1]))          # noqa: E731  half width up the panel
    P.fill(BLACK)
    lo, hi = total * 0.61, total * 0.93
    # red down the right edge beside the plate, then flames from under the plate down across the waist
    # (it stops short of the top: the panel's sides and back take their colour from its top right corner)
    P.poly([(w(hi) * 0.80, total * 0.955), (w(hi) * 1.1, total * 0.955), (w(lo) * 1.1, lo - 0.02), (w(lo) * 0.80, lo)], RED)
    end = total * (0.16 if v1 else 0.12)

    def flame(c0, c1, half0, half1, t_bend=0.5):
        a = _curve(18, lambda t: (w(lo - (lo - end) * t) * (c0 + (c1 - c0) * t - half0 * (1 - t) - half1 * t),
                                  lo - (lo - end) * t))
        b = _curve(18, lambda t: (w(lo - (lo - end) * t) * (c0 + (c1 - c0) * t + half0 * (1 - t) + half1 * t),
                                  lo - (lo - end) * t ** 1.15))
        return a + b[::-1]

    P.poly(flame(0.25, -0.30, 0.80, 0.10), RED)
    P.poly(flame(0.45, 0.05, 0.28, 0.04), GOLD if v1 else PALE)
    P.poly(flame(-0.08, -0.50, 0.06, 0.02), BLACK)
    P.poly(flame(-0.62, -0.85, 0.10, 0.02), RED)
    if v1:
        _plate(P, -w(hi) * 0.62, lo + 0.01, w(hi) * 0.62, hi, 0.02, skin, NARROW_NUM, 0.80, 0.0)
    else:
        _plate(P, -w(hi) * 0.80, lo, w(hi) * 0.72, hi, 0.035, skin, SERIF_NUM, 0.70, 0.06)
    P.text("TECHSPEED", 0.0, (hi + total) / 2 + 0.003, 0.018, WHITE, font=LOGO_FONT, shear=0, width_m=0.16)


def _pods_fleet(img, sc, skin):
    for key in ("pod_L", "pod_R"):
        if key not in sc.panels:
            continue
        pd = sc.panels[key]
        P = Panel(img, ATLAS[key], pd["u"], pd["v"])
        P.fill(BLACK)
        f = pd["front"]
        tops = pd["top"]
        y0, y1 = tops[0][0], tops[-1][0]                      # rear end, front end
        Uy = lambda y: f * y                                   # noqa: E731
        top = lambda y: float(np.interp(y, [t[0] for t in tops], [t[1] for t in tops]))  # noqa: E731
        # black along the top edge; under it a narrow flame band, red at the rear turning orange at the front, its lower
        # edge dropping a little towards the front, a gold pinstripe under it, and black below with VELOPARK in orange
        # (the studio side shot, 514252952, 515336126). The yellow plate low at the pod's front on the right, at its rear,
        # just ahead of the rear tyre, on the left. (The band was twice as tall and hid the lettering, with a pale tongue
        # and gold chevrons the real pods don't have, and the left plate was at the front.)
        P.poly([(Uy(y), z + 0.05) for y, z in tops] + [(Uy(y), z - 0.022) for y, z in reversed(tops)], BLACK)
        ys = np.linspace(y0, y1, 24)
        bottom = lambda y: float(np.interp(y, [b[0] for b in pd["bottom"]], [b[1] for b in pd["bottom"]]))  # noqa: E731
        lower = lambda y: max(top(y) - 0.075 - 0.02 * (y - y0) / (y1 - y0), bottom(y) + 0.10)          # noqa: E731
        band = [(Uy(y), top(y) - 0.022) for y in ys] + [(Uy(y), lower(y)) for y in ys[::-1]]
        _grad_poly(P, band, RED, ORANGE, Uy(y0), Uy(y1))
        P.band([(Uy(y), lower(y)) for y in ys], [(Uy(y), lower(y) - 0.007) for y in ys], GOLD)
        if f > 0:                                              # right: the plate at the front, the lettering behind it
            pa, pb, tv = 0.12, 0.27, 0.03
        else:                                                  # left: the plate at the rear, the lettering ahead of it
            pa, pb, tv = y0 + 0.03, y0 + 0.17, y0 + 0.30
        a, b = sorted((Uy(pa), Uy(pb)))
        _plate(P, a, 0.12, b, 0.26, 0.02, skin, SERIF_NUM, 0.70, 0.06)
        P.text("VELOPARK", Uy(tv), 0.17, 0.036, ORANGE, font=LOGO_FONT, shear=0.15, width_m=0.20)


def _rear_fleet_v2(img, sc, skin):
    pr = sc.panels["rear"]
    P = Panel(img, ATLAS["rear"], pr["u"], pr["v"])
    rows = pr["rows"]
    P.fill(BLACK)                                          # the rear cover is plain black plastic
    mk = pr["marks"]                                       # kart_model.REAR_ROWS
    # (the number plate over the exhaust is a flat plate of its own showing the front panel's plate: kart_model
    # rear_plate; the cover behind it stays black)
    # the recess in the middle of the face in its own shadow, darkest at the back (rear.jpg): a wedge from the
    # crease down to the exhaust, matching kart_model's pocket widths row by row, softened
    from kart_model import REAR_ROWS
    shade = Image.new("L", img.size, 0)
    sd = ImageDraw.Draw(shade)
    cen = min(pr["cols"], key=lambda c: abs(c[0]))[1]
    # the recess's trapezoid: wide at its top edge, the sides sloping in to the tunnel round the tips
    sd.polygon([P.px(-0.25, cen[9] - 0.004), P.px(0.25, cen[9] - 0.004), P.px(0.17, cen[12]), P.px(0.145, cen[13]),
                P.px(-0.145, cen[13]), P.px(-0.17, cen[12])], fill=210)
    # the vent slots either side of the recess, level with the plate's bottom (rear.jpg)
    for sx in (-1, 1):
        for x in (0.215, 0.245):
            c0 = min(pr["cols"], key=lambda c: abs(c[0] - sx * x))[1]
            sd.rounded_rectangle([P.px(sx * x - 0.009, c0[11] + 0.012), P.px(sx * x + 0.009, c0[11] - 0.010)], 4, fill=255)
    shade = shade.filter(ImageFilter.GaussianBlur(6))
    img.paste(Image.new("RGBA", img.size, (2, 2, 3, 255)), (0, 0), shade)
    P.text("WWW.TECHSPEEDKART.COM", 0.0, (mk["deck_front"] + mk["dish"]) / 2, 0.026, WHITE, font=TEXT_FONT, shear=0,
           width_m=0.42)
    for x in (-0.49, -0.40, 0.40, 0.49):                   # a plain red dot inboard, a Velopark roundel outboard
        # at their height on the face, 21 cm over the frame, 8 cm across (rear.jpg measured on the left: the dots'
        # centres at -0.49 / -0.40, level with the plate's bottom; they were at +-0.33 / +-0.44, placed by row)
        cv = min(pr["cols"], key=lambda c: abs(c[0] - x))[1]
        cz = min(pr["colz"], key=lambda c: abs(c[0] - x))[1]
        k = list(range(8, 15))
        vc = float(np.interp(0.21, [cz[i] for i in k][::-1], [cv[i] for i in k][::-1]))
        _roundel(P, x, vc, 0.040, abs(x) > 0.45)


def _pod_top_fleet(out_path):
    """The v2 pods' top (u front->rear, v inner->outer): black, a long red decal with the Velopark logo and a
    gold pinstripe (sidepod-ballast-fuel.jpg)."""
    w, h = 1024, 128
    img = Image.new("RGBA", (w, h), BLACK + (255,))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([60, 18, w - 120, h - 30], 22, fill=RED)
    d.line([(40, h - 16), (w - 60, h - 16)], fill=GOLD, width=3)
    _velopark_px(img, w * 0.62, h * 0.46, 92, 1.6)
    img.convert("RGB").save(out_path)


SKINS["rental"] = _fleet("v2")
SKINS["rental_v1"] = _fleet("v1")


def tyre(out_path, used=False):
    """Kart slick (the rentals' MG tyres: steering-pedals-front-axle.jpg, side-right.jpg). u runs once round
    the tyre, v across it from the inner bead (0) over the tread (~0.30-0.70) to the outer bead (1), the outer
    sidewall ~0.80-0.93 (kart_model.tyre_mesh). Satin black rubber; on the outer sidewall two cream MG badges
    near the shoulder, a small white barcode label by the rim and moulded wear marks on the shoulder. New
    tyres carry one thin red line on the tread, slightly to the inner side; it wears off, so used tyres have
    none, and a scuffed grey tread with streaks of picked-up rubber (protector-v2-rear-row.jpg)."""
    w, h = 2048, 512
    rng = np.random.default_rng(1)
    a = np.full((h, w), 27.0) + rng.normal(0, 2.0, (h, w))
    img = np.stack([a, a, a], -1)
    row = lambda v: int(round((1 - v) * h))                                        # noqa: E731
    # the same rubber all over (the tread was a darker band with a hard edge, the sidewalls lighter); worn
    # tyres get a light scuff on the tread fading out over the shoulders, no band
    vs = 1 - (np.arange(h) + 0.5) / h
    tread = np.clip(np.minimum(vs - 0.28, 0.72 - vs) / 0.06, 0, 1)[:, None]      # 0 on the sidewalls, 1 on the tread
    if used:
        streaks = rng.normal(0, 3.0, (h, 1)) * np.ones((1, w))
        img += (tread * (7 + streaks))[..., None]
        blobs = (rng.random((h, w)) < 0.003) & (tread[:, 0:1] > 0.5)
        img[blobs] = 16
    else:
        y = row(TYRE_LINE_V)
        img[y - 2:y + 2, :, :] = (150, 22, 20)
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7))
    d = ImageDraw.Draw(im)
    # wear indicator holes along both shoulders, a moulded triangle beside every fourth
    for k in range(48):
        x = (k + 0.5) / 48 * w
        for v in (0.69, 0.31):
            d.ellipse([x - 2, row(v) - 2, x + 2, row(v) + 2], fill=(12, 12, 12))
        if k % 4 == 0:
            y = row(0.74)
            d.polygon([(x - 7, y + 6), (x + 7, y + 6), (x, y - 7)], outline=(52, 52, 52))
    # the outer sidewall: two MG badges and a barcode label. The text's top points out of the tyre (towards
    # the shoulder, down the image), so it is drawn upside down.
    badge_h = row(0.800) - row(0.885)
    for u in (0.25, 0.75):
        bw, bh = int(badge_h * 2.3), badge_h
        lab = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        ld = ImageDraw.Draw(lab)
        cream = (236, 226, 204) if not used else (196, 188, 172)
        ld.rounded_rectangle([0, 0, bw - 1, bh - 1], bh // 2, fill=cream + (255,))
        f = ImageFont.truetype(LOGO_FONT, int(bh * 0.78))
        ld.text((bw / 2, bh / 2), "MG", font=f, fill=(196, 28, 30, 255), anchor="mm")
        lab = lab.transform(lab.size, Image.AFFINE, (1, -0.25, 0.25 * bh / 2, 0, 1, 0), resample=Image.BICUBIC)
        lab = lab.rotate(180)
        im.paste(lab, (int(u * w - bw / 2), row(0.885)), lab)
        bx = int((u + 0.06) * w)
        d.rectangle([bx, row(0.925), bx + 26, row(0.895)], fill=(226, 226, 222))
        for i in range(9):
            if (i * 7) % 3:
                d.line([(bx + 3 + i * 2.5, row(0.92)), (bx + 3 + i * 2.5, row(0.90))], fill=(30, 30, 30), width=1)
    im.save(out_path)


def carbon(out_path):
    n = 512
    y, x = np.mgrid[0:n, 0:n]
    cell = 32
    cx, cy = (x // cell) % 2, (y // cell) % 2
    phase = np.where(cx ^ cy, np.sin((x % cell) / cell * np.pi), np.sin((y % cell) / cell * np.pi))
    v = 18 + 26 * phase
    Image.fromarray(np.stack([v, v, v * 1.05], -1).clip(0, 255).astype(np.uint8)).save(out_path)


def radiator(out_path):
    n = 256
    img = np.full((n, n, 3), 150, np.uint8)
    img[:, ::4] = 95
    img[::24] = 70
    Image.fromarray(img).save(out_path)


def shroud(out_path):
    img = Image.new("RGBA", (512, 256), (190, 18, 22, 255))
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(LOGO_FONT, 42)
    d.text((256, 128), "TECHSPEED", font=f, fill=(245, 245, 245), anchor="mm")
    img.convert("RGB").save(out_path)


def badge(out_path):
    img = Image.new("RGB", (256, 170), (10, 10, 10))
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(LOGO_FONT, 34)
    d.text((128, 85), "TECHSPEED", font=f, fill=(230, 30, 30), anchor="mm")
    img.save(out_path)


def pedal_plate(out_path):
    img = Image.new("RGB", (256, 256), (14, 14, 16))
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(LOGO_FONT, 64)
    d.text((128, 128), "TECH", font=f, fill=(235, 235, 235), anchor="mm")
    img.save(out_path)


def pod_stripe(out_path):
    """The rental v2 pods' tops (u front->rear, v inner->outer): see _pod_top_fleet."""
    _pod_top_fleet(out_path)


def recoil_cover(out_path):
    """Red recoil starter cover seen face on: radial slots in two rings, black centre badge."""
    n = 512
    img = Image.new("RGB", (n, n), (185, 16, 20))
    d = ImageDraw.Draw(img)
    c = n / 2
    for ring, (r0, r1, count, wdeg) in enumerate(((0.30, 0.44, 36, 4.0), (0.18, 0.28, 24, 5.0))):
        for k in range(count):
            a = 2 * math.pi * k / count
            da = math.radians(wdeg) / 2
            pts = [(c + n * r0 * math.cos(a - da), c + n * r0 * math.sin(a - da)),
                   (c + n * r1 * math.cos(a - da * 0.6), c + n * r1 * math.sin(a - da * 0.6)),
                   (c + n * r1 * math.cos(a + da * 0.6), c + n * r1 * math.sin(a + da * 0.6)),
                   (c + n * r0 * math.cos(a + da), c + n * r0 * math.sin(a + da))]
            d.polygon(pts, fill=(25, 12, 12))
    d.ellipse([c - n * 0.13, c - n * 0.13, c + n * 0.13, c + n * 0.13], fill=(18, 18, 18))
    img.save(out_path)


def tread_plate(out_path, size=1024, lugs=8):
    """Aluminium tread ("diamond") plate, tileable (the onboard frames of the rental's floor): brushed silver
    with raised lugs on a square grid, alternate ones turned 90 degrees, each lit from the top left with a
    shadow on its lower right. One tile is kart_model.TREAD_TILE across."""
    import numpy as np
    rng = np.random.default_rng(7)
    base = 168 + rng.normal(0, 6, (size, 1)) + rng.normal(0, 3, (size, size))       # brushed along u
    base = np.clip(base, 0, 255)
    img = Image.fromarray(np.dstack([base, base + 2, base + 5]).clip(0, 255).astype("uint8"), "RGB")
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    d = ImageDraw.Draw(img)
    pitch = size / lugs
    half, w = pitch * 0.34, max(4, int(pitch * 0.17))

    def lug(cx, cy, ang, off, col, width):
        dx, dy = math.cos(ang) * half, math.sin(ang) * half
        for ox in (-size, 0, size):                      # wrap round the tile's edges
            for oy in (-size, 0, size):
                a = (cx - dx + off + ox, cy - dy + off + oy)
                b = (cx + dx + off + ox, cy + dy + off + oy)
                d.line([a, b], fill=col, width=width)
                for q in (a, b):
                    d.ellipse([q[0] - width / 2, q[1] - width / 2, q[0] + width / 2, q[1] + width / 2], fill=col)

    for i in range(lugs):
        for j in range(lugs):
            cx, cy = (i + 0.5) * pitch, (j + 0.5) * pitch
            ang = math.radians(45 if (i + j) % 2 == 0 else -45)
            lug(cx, cy, ang, 3, (96, 98, 102), w)          # shadow, lower right
            lug(cx, cy, ang, -1.5, (232, 234, 238), w)     # lit edge, upper left
            lug(cx, cy, ang, 0, (196, 198, 202), w - 3)    # lug top
    img.save(out_path)


def skins_dir(variant):
    """Where a build's liveries are painted: build/skins/<variant>/<skin>/ (body_livery.png, the worn liveries'
    tyre.png, the preview.jpg). They are not kept in the source: tools/paint_skins.py paints them from SKINS on
    every build, only the ones the build carries (cars.built_skins)."""
    return os.path.join(ROOT, "build", "skins", variant)


def paint_skins(sc, names, out_dir):
    """Paint the named skins of sc's variant into out_dir/<skin>/. Returns {skin_name: livery path}."""
    every = skins_for(sc.variant)
    skins = {}
    for name in names:
        skin = every[name]
        skin_dir = os.path.join(out_dir, name)
        os.makedirs(skin_dir, exist_ok=True)
        path = os.path.join(skin_dir, "body_livery.png")
        livery(sc, skin, path)
        if skin.get("worn"):                        # these liveries come on worn tyres
            tyre(os.path.join(skin_dir, "tyre.png"), used=True)
        skins[name] = path
    return skins


def generate(sc, out_dir):
    """Write the source textures of one variant; body_livery.png is the fleet's first livery, the one the .blend
    shows (the build paints the others, see skins_dir)."""
    os.makedirs(out_dir, exist_ok=True)
    tyre(os.path.join(out_dir, "tyre.png"))
    tyre(os.path.join(out_dir, "tyre_used.png"), used=True)
    carbon(os.path.join(out_dir, "carbon.png"))
    radiator(os.path.join(out_dir, "radiator.png"))
    shroud(os.path.join(out_dir, "radiator_shroud.png"))
    badge(os.path.join(out_dir, "logo_badge.png"))
    pedal_plate(os.path.join(out_dir, "pedal_plate.png"))
    pod_stripe(os.path.join(out_dir, "pod_stripe.png"))
    recoil_cover(os.path.join(out_dir, "recoil_cover.png"))
    tread_plate(os.path.join(out_dir, "tread_plate.png"))
    first = sorted(skins_for(sc.variant))[0]
    livery(sc, skins_for(sc.variant)[first], os.path.join(out_dir, "body_livery.png"))
