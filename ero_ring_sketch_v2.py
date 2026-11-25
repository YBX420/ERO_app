"""
ERO Ring v2 - front-view line sketch. Changes from ero_ring_sketch.py:
  * the leash is a flat tape stored INSIDE the hollow ring (wound flat in layers round the loop)
    that pulls out of a rotating outlet at the bottom, so the whole ring is rigid and thicker;
  * a fixed length of tape (about 1.2 m, drawn shortened) always hangs out, held by a stopper,
    and ends in a swivel dog clip that goes straight onto the harness;
  * the control is an index-finger trigger on the inside of the grip: locked by default,
    squeeze to let tape out or back in.

Writes into renders/:
  ero_ring_v3_sketch.png          clean line art, no text, no hidden lines (feed to the image tool)
  ero_ring_v3_sketch_labeled.png  with part names, rough sizes and the stored tape as dashed lines
(the ero_ring_v2_sketch*.png files are the earlier thin-cord version of this drawing)
"""
import os

import numpy as np

from ero_ring_sketch import Canvas, polar, stadium, round_rect, smoothstep, cjk_font

HOLE_R = 44.0           # inner opening radius at the sides/bottom
TOP_FLAT = 6.0          # flattening of the top of the opening (straighter grip)
BAND_T = 16.0           # hollow ring thickness (has to hold the wound rope)
GRIP_T = 12.0           # extra thickness of the grip on top of that
PALM_SWELL = 3.0
GRIP_A0, GRIP_A1 = 46.0, 98.0
FINGER_PITCH, FINGER_DEPTH = 27.5, 3.8
TRIG_A0, TRIG_A1 = 28.0, 56.0     # trigger zone = index finger, degrees from the top
TRIG_OUT = 7.5          # how far the trigger tips reach into the opening
TAPE_W = 10.0           # flat tape width (about 1 mm thick, wound flat in layers inside the ring)
TAPE_LEN = 70.0         # how much tape is drawn hanging out (the real fixed length is about 1.2 m)


def radii(th_deg):
    th = np.radians(th_deg)
    a_s = ((90.0 - th_deg + 180.0) % 360.0) - 180.0
    a = np.abs(a_s)
    base = HOLE_R - TOP_FLAT * np.maximum(np.sin(th), 0) ** 2
    w = smoothstep(GRIP_A1, GRIP_A0, a)
    wg = smoothstep(61.0, 55.0, a) * smoothstep(TRIG_A0 + 1.0, TRIG_A0 - 5.0, a_s)
    groove = 0.5 + 0.5 * np.cos(2 * np.pi * (a_s - 0.5 * FINGER_PITCH) / FINGER_PITCH)
    r_in = base + wg * (FINGER_DEPTH * groove - 0.6)
    thick = BAND_T + GRIP_T * w + PALM_SWELL * w * np.exp(-(((a_s + 8.0) / 28.0) ** 2))
    return base, r_in, base + thick


def dashed(c, pts, w=0.3, on=12, off=9):
    i = 0
    while i < len(pts) - 1:
        c.line(pts[i:i + on + 1], w)
        i += on + off


def draw(c, hidden=False, broken=False):
    th = np.linspace(0, 360, 1441)[:-1]
    base, r_in, r_out = radii(th)
    c.line(polar(r_out, th), 1.0, closed=True)
    c.line(polar(r_in, th), 1.0, closed=True)

    if hidden:  # rope wound inside the hollow ring
        for off in (5.0, 7.5, 10.0):
            dashed(c, polar(base + off, th))
        dashed(c, np.array([(0.0, y) for y in np.linspace(-(HOLE_R + 5.5), -(HOLE_R + BAND_T), 12)]), on=4, off=3)

    # index-finger trigger on the inside of the grip
    a = np.linspace(TRIG_A0, TRIG_A1, 60)
    u = (a - TRIG_A0) / (TRIG_A1 - TRIG_A0)
    tb, _, _ = radii(90.0 - a)
    front = polar(tb - TRIG_OUT + 3.0 * np.sin(np.pi * u), 90.0 - a)
    back = polar(tb + 1.2, 90.0 - a)[::-1]
    c.poly(np.concatenate([front, back]), 0.8)
    pv = polar(radii(np.array([90.0 - TRIG_A1 + 2.5]))[0][0] + 3.5, 90.0 - TRIG_A1 + 2.5)
    c.line(polar(1.3, np.linspace(0, 360, 24)) + pv, 0.4, closed=True)

    # tiny status light at the other end of the grip
    lb, _, lo = radii(np.array([90.0 + 54.0]))
    lc = polar(lb[0] + 0.6 * (lo[0] - lb[0]), 90.0 + 54.0)
    c.dot(lc[0], lc[1], 1.1)

    # tape outlet at the bottom, stopper block, flat tape (leaves edge-on, turns face-on), dog clip
    yb = -(HOLE_R + BAND_T)
    c.poly([(-7.5, yb + 2.5), (7.5, yb + 2.5), (5.5, yb - 7.0), (-5.5, yb - 7.0)], 0.9)
    c.line([(-6.1, yb - 4.5), (6.1, yb - 4.5)], 0.4)
    stop_y = yb - 10.5
    c.poly(round_rect(-3.4, stop_y - 2.6, 3.4, stop_y + 2.6, 1.6), 0.8)
    ry0, ry1 = stop_y - 2.6, stop_y - 2.6 - TAPE_LEN
    ys = np.linspace(ry0, ry1, 120)
    hw = 0.7 + (TAPE_W / 2 - 0.7) * smoothstep(0.0, 28.0, ry0 - ys)
    brk = ry0 - 46.0
    for sgn in (-1, 1):
        edge = np.stack([sgn * hw, ys], -1)
        if broken:
            c.line(edge[ys > brk + 1.5], 0.8)
            c.line(edge[ys < brk - 1.5], 0.8)
        else:
            c.line(edge, 0.8)
        inset = np.stack([sgn * (hw - 1.1), ys], -1)[hw > 3.2]
        if broken:
            c.line(inset[inset[:, 1] > brk + 1.5], 0.3)
            c.line(inset[inset[:, 1] < brk - 1.5], 0.3)
        else:
            c.line(inset, 0.3)
    if broken:
        for dy in (1.5, -1.5):
            c.line([(-TAPE_W / 2 - 2, brk + dy - 1.2), (TAPE_W / 2 + 2, brk + dy + 1.2)], 0.4)
    h = TAPE_W / 2
    c.poly([(-h, ry1 + 13), (h, ry1 + 13), (h, ry1), (-h, ry1)], 0.8)          # stitched fold
    c.line([(-h, ry1 + 13), (h, ry1)], 0.3)
    c.line([(h, ry1 + 13), (-h, ry1)], 0.3)
    ring_pts = polar(1.0, np.linspace(0, 360, 48))
    c.poly(ring_pts * 5.4 + (0, ry1 - 3.5), 0.8)                                 # swivel eye
    c.poly(ring_pts * 3.0 + (0, ry1 - 3.5), 0.6)
    c.poly(round_rect(-3.3, ry1 - 17, 3.3, ry1 - 8.5, 1.2), 0.8)                # swivel barrel
    c.line([(-3.3, ry1 - 12.5), (3.3, ry1 - 12.5)], 0.35)
    hy = ry1 - 33
    c.line(stadium(0, hy, 9, 9.0), 0.9, closed=True)
    c.line(stadium(0, hy, 9, 5.0), 0.9, closed=True)
    for gy in (hy + 7, hy - 6):
        c.line([(5.0, gy), (9.0, gy - 1.1)], 0.5)
    trig_mid = polar(radii(np.array([48.0]))[0][0] - TRIG_OUT + 3.0, 48.0)
    return dict(light=lc, trig=trig_mid, outlet_y=yb - 3, stop_y=stop_y, tape_y=ry0 - 34, brk_y=brk, hook_y=hy)


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "renders")
    os.makedirs(out, exist_ok=True)

    c = Canvas(-105, 105, -206, 84)
    draw(c)
    p1 = os.path.join(out, "ero_ring_v3_sketch.png")
    print("saved", p1, c.save(p1))

    c = Canvas(-120, 215, -206, 100)
    info = draw(c, hidden=True, broken=True)
    f, fs = cjk_font(5.2), cjk_font(4.2)
    LX = 96

    def label(target, ly, title, sub=None):
        c.line([target, (LX - 3, ly)], 0.25)
        c.dot(target[0], target[1], 0.8)
        c.text(LX, ly, title, f)
        if sub:
            c.text(LX, ly - 7, sub, fs)

    th = np.linspace(0, 360, 1441)[:-1]
    base, r_in, r_out = radii(th)
    o = polar(r_out, th)
    w_mm, h_mm = np.ptp(o[:, 0]), np.ptp(o[:, 1])
    label(tuple(polar(radii(np.array([98.0]))[2][0], 98.0)), 90, "掌心鼓包",
          f"握把最厚处约 {BAND_T + GRIP_T + PALM_SWELL:.0f} mm")
    label(tuple(info["trig"]), 64, "扳机", "扣住才能放长 / 收回，松手即锁")
    label(tuple(polar(HOLE_R + BAND_T, -30.0)), -27, "空心硬圈", f"粗约 {BAND_T:.0f} mm，虚线是一层层绕在里面的扁带")
    label((6.5, info["outlet_y"]), -56, "出带口", "可转动，扁带从这里拉出和收回")
    label((3.4, info["stop_y"]), -76, "止挡块", "卡住定长段，防止带头缩进圈里")
    label((TAPE_W / 2, info["tape_y"]), -100, "扁带", f"宽约 {TAPE_W:.0f} mm，厚约 1 mm")
    label((TAPE_W / 2 + 2, info["brk_y"]), -124, "露在外面的定长段", "约 1.2 m，图中截短")
    label((9, info["hook_y"]), -176, "带转环的狗扣", "直接扣在胸背的环上")

    c.text(-8, 18, "三个指槽", f, anchor="mm")
    c.text(-8, 10, "中指、无名指、小指", fs, anchor="mm")
    c.line([(-8, 24), (-6, 38)], 0.25)
    c.text(0, -12, f"内孔约 {2 * HOLE_R:.0f} × {2 * HOLE_R - TOP_FLAT:.0f} mm", fs, anchor="mm")
    lc = info["light"]
    c.line([tuple(lc), (-78, 62)], 0.25)
    c.text(-80, 65, "状态灯", f, anchor="rm")
    c.text(-116, -198, f"握把圈约 {w_mm:.0f} × {h_mm:.0f} mm（不含扁带和狗扣）", fs)
    p2 = os.path.join(out, "ero_ring_v3_sketch_labeled.png")
    print("saved", p2, c.save(p2))


if __name__ == "__main__":
    main()
