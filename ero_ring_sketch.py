"""
ERO Ring - front-view line sketch, meant as a SHAPE reference for Vizcom / Midjourney.

Writes two files into renders/:
  ero_ring_sketch.png          clean black-on-white line art, no text (feed this to the image tool)
  ero_ring_sketch_labeled.png  same drawing with part names and rough sizes (for people)

All numbers are millimetres. Ring centre is (0, 0), +y is up; the grip is the top of the loop.
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

S = 16          # drawing scale, px per mm (image is downsampled 2x afterwards)

# ----------------------------------------------------------------------------- shape
HOLE_R = 44.0           # inner opening radius at the sides/bottom
TOP_FLAT = 6.0          # how much the top of the opening is flattened (straighter grip)
BAND_T = 10.0           # soft band thickness
GRIP_T = 17.0           # extra thickness of the grip on top of the band
PALM_SWELL = 3.0        # extra bulge under the palm
GRIP_A0, GRIP_A1 = 46.0, 98.0     # degrees from the top: full grip inside A0, pure band beyond A1
FINGER_PITCH = 27.5     # degrees per finger (about 20 mm each)
FINGER_DEPTH = 3.8
THUMB_A = 47.0          # thumb rest position, degrees from the top (index-finger end)
SEAM_A = 80.0


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def ring_radii(th_deg):
    th = np.radians(th_deg)
    a_s = ((90.0 - th_deg + 180.0) % 360.0) - 180.0        # signed angle from the top
    a = np.abs(a_s)
    base = HOLE_R - TOP_FLAT * np.maximum(np.sin(th), 0) ** 2
    w = smoothstep(GRIP_A1, GRIP_A0, a)
    wg = smoothstep(61.0, 55.0, a)
    groove = 0.5 + 0.5 * np.cos(2 * np.pi * (a_s - 0.5 * FINGER_PITCH) / FINGER_PITCH)
    r_in = base + wg * (FINGER_DEPTH * groove - 0.6)
    thick = BAND_T + GRIP_T * w + PALM_SWELL * w * np.exp(-(((a_s + 8.0) / 28.0) ** 2))
    return r_in, base + thick


def polar(r, th_deg):
    th = np.radians(th_deg)
    return np.stack([r * np.cos(th), r * np.sin(th)], -1)


def stadium(cx, cy, half, rad, n=40):
    t = np.linspace(0, np.pi, n)
    top = np.stack([cx + rad * np.cos(t), cy + half + rad * np.sin(t)], -1)
    bot = np.stack([cx - rad * np.cos(t), cy - half - rad * np.sin(t)], -1)
    return np.concatenate([top, bot])


def ellipse(cx, cy, ra, rb, rot_deg, n=80):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(np.radians(rot_deg)), np.sin(np.radians(rot_deg))
    x, y = ra * np.cos(t), rb * np.sin(t)
    return np.stack([cx + c * x - s * y, cy + s * x + c * y], -1)


# ----------------------------------------------------------------------------- drawing
class Canvas:
    def __init__(self, x0, x1, y0, y1):
        self.x0, self.y1 = x0, y1
        self.im = Image.new("RGB", (int((x1 - x0) * S), int((y1 - y0) * S)), "white")
        self.d = ImageDraw.Draw(self.im)

    def px(self, pts):
        pts = np.atleast_2d(np.asarray(pts, float))
        return [((x - self.x0) * S, (self.y1 - y) * S) for x, y in pts]

    def line(self, pts, w=0.9, closed=False, fill="black"):
        p = self.px(pts)
        if closed:
            p = p + [p[0], p[1]]
        self.d.line(p, fill=fill, width=max(1, int(w * S)), joint="curve")

    def poly(self, pts, w=0.9, fill="white"):
        self.d.polygon(self.px(pts), fill=fill)
        self.line(pts, w, closed=True)

    def dot(self, x, y, r, fill="black"):
        (cx, cy), = self.px([(x, y)])
        self.d.ellipse([cx - r * S, cy - r * S, cx + r * S, cy + r * S], fill=fill)

    def text(self, x, y, s, font, anchor="lm"):
        (cx, cy), = self.px([(x, y)])
        self.d.text((cx, cy), s, fill="black", font=font, anchor=anchor)

    def save(self, path):
        im = self.im.resize((self.im.width // 2, self.im.height // 2), Image.LANCZOS)
        im.save(path)
        return im.size


def round_rect(x0, y0, x1, y1, r, n=10):
    t = np.linspace(0, np.pi / 2, n)
    pts = []
    for cx, cy, a0 in ((x1 - r, y1 - r, 0), (x0 + r, y1 - r, np.pi / 2), (x0 + r, y0 + r, np.pi), (x1 - r, y0 + r, 1.5 * np.pi)):
        pts += [(cx + r * np.cos(a0 + u), cy + r * np.sin(a0 + u)) for u in t]
    return np.array(pts)


def draw_ring(c):
    th = np.linspace(0, 360, 1441)[:-1]
    r_in, r_out = ring_radii(th)
    c.line(polar(r_out, th), 1.0, closed=True)
    c.line(polar(r_in, th), 1.0, closed=True)

    # seams between the rigid grip and the soft band
    for a in (SEAM_A, -SEAM_A):
        ri, ro = ring_radii(np.array([90.0 - a]))
        c.line([polar(ri[0] + 0.3, 90.0 - a), polar(ro[0] - 0.3, 90.0 - a)], 0.45)

    # thumb rest + raised button, on the face of the grip at the index-finger end
    ri, ro = ring_radii(np.array([90.0 - THUMB_A]))
    tc = polar(ri[0] + 0.50 * (ro[0] - ri[0]), 90.0 - THUMB_A)
    c.line(ellipse(tc[0], tc[1], 10.0, 7.5, 90.0 - THUMB_A + 90.0), 0.45, closed=True)
    c.line(ellipse(tc[0], tc[1], 4.6, 4.6, 0), 0.7, closed=True)

    # tiny status light near the other end
    ri2, ro2 = ring_radii(np.array([90.0 + 54.0]))
    lc = polar(ri2[0] + 0.6 * (ro2[0] - ri2[0]), 90.0 + 54.0)
    c.dot(lc[0], lc[1], 1.1)

    # connector pod on the band, boot, rope, ferrule, swivel, snap hook
    pod_y = -(HOLE_R + BAND_T / 2)
    c.poly(round_rect(-11, pod_y - 12, 11, pod_y + 10, 6), 0.9)
    c.line([(-11, pod_y - 4), (11, pod_y - 4)], 0.45)
    by = pod_y - 12
    c.poly([(-7.5, by), (7.5, by), (5.2, by - 8), (-5.2, by - 8)], 0.8)
    ry0, ry1 = by - 8, by - 50
    c.line([(-4.5, ry0), (-4.5, ry1)], 0.9)
    c.line([(4.5, ry0), (4.5, ry1)], 0.9)
    y = ry0
    while y - 4.5 > ry1:
        c.line([(-4.5, y), (4.5, y - 4.5)], 0.35)
        c.line([(4.5, y), (-4.5, y - 4.5)], 0.35)
        y -= 4.5
    c.poly(round_rect(-6.5, ry1 - 16, 6.5, ry1 + 1, 1.5), 0.9)
    for g in (5.0, 9.0):
        c.line([(-6.5, ry1 - g), (6.5, ry1 - g)], 0.4)
    c.poly([(-3, ry1 - 16), (3, ry1 - 16), (3, ry1 - 22), (-3, ry1 - 22)], 0.8)
    hy = ry1 - 22 - 9 - 11
    c.line(stadium(0, hy, 11, 11.0), 0.9, closed=True)
    c.line(stadium(0, hy, 11, 6.2), 0.9, closed=True)
    for gy in (hy + 9, hy - 8):
        c.line([(6.2, gy), (11.0, gy - 1.2)], 0.5)
    return dict(thumb=tc, light=lc, pod_y=pod_y, rope_y=(ry0 + ry1) / 2, ferrule_y=ry1 - 8, hook_y=hy,
                bottom=hy - 22)


def cjk_font(size_mm):
    for name in ("msyh.ttc", "msyhl.ttc", "simhei.ttf", "simsun.ttc"):
        p = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", name)
        if os.path.exists(p):
            return ImageFont.truetype(p, int(size_mm * S))
    raise SystemExit("No Chinese font found in Windows\\Fonts")


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "renders")
    os.makedirs(out, exist_ok=True)

    # 1) clean sketch, no text
    c = Canvas(-105, 105, -186, 84)
    info = draw_ring(c)
    p1 = os.path.join(out, "ero_ring_sketch.png")
    print("saved", p1, c.save(p1))

    # 2) labelled sketch
    c = Canvas(-120, 215, -186, 100)
    info = draw_ring(c)
    f, fs = cjk_font(5.2), cjk_font(4.2)
    LX = 96

    def label(target, ly, title, sub=None):
        c.line([target, (LX - 3, ly)], 0.25)
        c.dot(target[0], target[1], 0.8)
        c.text(LX, ly, title, f)
        if sub:
            c.text(LX, ly - 7, sub, fs)

    _, ro = ring_radii(np.array([98.0]))
    label(tuple(polar(ro[0], 98.0)), 90, "掌心鼓包", "握把最厚处约 30 mm")
    label(tuple(info["thumb"]), 66, "拇指托 + 略凸的圆按键", "在食指这一端，拇指自然落点")
    ri, ro = ring_radii(np.array([90.0 - SEAM_A]))
    label(tuple(polar(0.5 * (ri[0] + ro[0]), 90.0 - SEAM_A)), 40, "硬握把与软圈的分界", "手握的一段约 90 mm")
    label(tuple(polar(HOLE_R + BAND_T, -30.0)), -27, "软细圈", "粗约 10 mm，可压扁")
    label((11, info["pod_y"] - 2), -58, "连接件", "转环，拉力从这里进握把")
    label((4.5, info["rope_y"]), -86, "弹力编织绳", "直径约 9 mm")
    label((6.5, info["ferrule_y"]), -116, "铝合金套管")
    label((11, info["hook_y"]), -146, "挂钩", "扣在现有牵引绳的手环上")

    c.text(0, 20, "四个指槽", f, anchor="mm")
    c.text(0, 12, "每个宽约 20 mm", fs, anchor="mm")
    c.line([(2, 26), (10, 39.5)], 0.25)
    c.text(0, -10, "内孔约 88 × 82 mm", fs, anchor="mm")
    lc = info["light"]
    c.line([tuple(lc), (-78, 62)], 0.25)
    c.text(-80, 65, "状态灯", f, anchor="rm")
    _th = np.linspace(0, 360, 1441)[:-1]
    _o = polar(ring_radii(_th)[1], _th)
    _w, _h = np.ptp(_o[:, 0]), np.ptp(_o[:, 1])
    c.text(-116, -176, f"整体约 {_w:.0f} × {_h:.0f} mm（不含绳和挂钩）", fs)
    print(f"ring outline {_w:.1f} x {_h:.1f} mm")
    p2 = os.path.join(out, "ero_ring_sketch_labeled.png")
    print("saved", p2, c.save(p2))


if __name__ == "__main__":
    main()
