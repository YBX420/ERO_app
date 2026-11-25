"""
ERO Grip v3 - "hand-hook" D-loop concept. Reuses the render engine, rope and snap hook from
ero_grip_v2.py; only the handle geometry and materials are new.

Idea: the fingers go *through* a closed loop and wrap a soft bar (like a kettlebell or a
traffic handle). The dog's pull lands in the finger crease, so the grip cannot slide out of the
hand however hard the dog lunges, and the owner can also just hook it on relaxed fingers, the
forearm or the wrist. The aluminium arch has a spring gate on one side, so the whole handle
clips straight onto a belt loop, bag strap or a railing.

    python ero_grip_v3.py               # charcoal, 1080x1350
    python ero_grip_v3.py --color all   # four colourways + contact sheet
    python ero_grip_v3.py --preview

Local frame: the bar runs along x at y = 0, the loop hangs below it in the x-y plane,
+z faces the user's thumb / the camera.
"""
import argparse
import os
import time

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

import ero_grip_v2 as eng
from ero_grip_v2 import (F32, COLORS, TEAL, nrm, len2, smoothstep, smin, smax, sd_box2,
                         sd_profile, sd_capsule3, rot_y, rot_z, weave, fiber_albedo, env, aces,
                         LKEY, LFILL, LRIM, KEY_COL, FILL_COL)

# ----------------------------------------------------------------------------- design (mm)
D3 = dict(
    bar_half=52.0,        # bar half-length (104 mm: room for four fingers + margins)
    bar_r=14.0,           # bar radius at the ends
    bar_bulge=1.2,        # extra radius in the middle (barrel shape fills the palm)
    end_round=9.0,
    finger_pitch=21.0, finger_depth=0.9,      # soft finger scallops on the underside
    btn_x=-30.0, btn_r=5.2, btn_dir=(0.0, 0.72, 0.69),
    recess_R=28.0, recess_depth=1.6, recess_stretch=0.55,
    ring_x=43.0, ring_h=0.8, ring_inset=0.6,  # light-guide ring round the bar, outside the hand
    arch_hw=54.0, arch_hh=38.0, arch_corner=26.0, arch_r=4.8,   # aluminium U below the bar
    gate_y0=-22.0, gate_y1=-49.0, gate_r=4.2,                  # spring gate on the -x side
    pod_r=10.0, pod_h=11.0,                                    # sensor pod at the loop apex
    boot_len=10.0, boot_r=5.7,
)
P = D3
APEX_Y = -2 * P["arch_hh"]
POD_C = APEX_Y - 3.0
ROPE_OFF = POD_C - P["pod_h"]      # v2's rope/hook start at y=0; shift them under the pod


def sd_round_rect(px, py, bx, by, r):
    return sd_box2(px, py, bx - r, by - r) - r


# ----------------------------------------------------------------------------- bar
BTN_DIR = nrm(P["btn_dir"])


def bar_radius(x):
    t = np.clip(x / P["bar_half"], -1, 1)
    return P["bar_r"] + P["bar_bulge"] * (1 - t * t)


def btn_frame(x, y, z):
    rb = bar_radius(np.float32(P["btn_x"]))
    cx, cy, cz = P["btn_x"], BTN_DIR[1] * rb, BTN_DIR[2] * rb
    qx, qy, qz = x - cx, y - cy, z - cz
    along = qy * BTN_DIR[1] + qz * BTN_DIR[2]
    py, pz = qy - along * BTN_DIR[1], qz - along * BTN_DIR[2]
    return qx, py, pz, along


def sd_bar(x, y, z):
    ryz = len2(y, z)
    under = smoothstep(0.0, -0.8, y / np.maximum(ryz, 1e-3))
    span = smoothstep(P["bar_half"] - 6, P["bar_half"] - 12, np.abs(x))
    phase = 2 * np.pi * (x - 0.5 * P["finger_pitch"]) / P["finger_pitch"]
    groove = P["finger_depth"] * under * span * (0.5 + 0.5 * np.cos(phase)) ** 2
    d = 0.8 * sd_profile(ryz + groove, x, bar_radius(x), -P["bar_half"], P["bar_half"], P["end_round"])

    # thumb recess + flush button on the top-front of the bar
    qx, py, pz, along = btn_frame(x, y, z)
    R_, dep = P["recess_R"], P["recess_depth"]
    dish = np.sqrt(P["recess_stretch"] * qx * qx + py * py + pz * pz + (along - (R_ - dep)) ** 2) - R_
    d = smax(d, -dish, 1.2)
    rad = np.sqrt(qx * qx + py * py + pz * pz)
    groove_b = len2(rad - P["btn_r"], along + dep - P["btn_r"] ** 2 / (2 * R_)) - 0.28
    d = np.maximum(d, -np.where(along > -6, groove_b, 1e3))

    # light-guide ring
    rr = bar_radius(np.float32(P["ring_x"]))
    ring = np.maximum(np.abs(x - P["ring_x"]) - P["ring_h"], (rr - P["ring_inset"]) - ryz)
    return np.maximum(d, -ring)


# ----------------------------------------------------------------------------- arch with gate
def sd_arch(x, y, z):
    hw, hh = P["arch_hw"], P["arch_hh"]
    o = sd_round_rect(x, y + hh, hw, hh, P["arch_corner"])
    d = len2(np.abs(o), z) - P["arch_r"]
    g0, g1 = P["gate_y0"], P["gate_y1"]
    d = np.maximum(d, -sd_box2(x + hw, y - 0.5 * (g0 + g1), 9.0, 0.5 * (g0 - g1)))
    gate = sd_capsule3(x, y, z, (-hw, g0 + 2.0, 0.0), (-hw + 0.4, g1 + 2.2, 0.0), P["gate_r"])
    # small aluminium knuckles where the gate hinges / latches
    hinge = sd_capsule3(x, y, z, (-hw - 1.0, g0 + 0.5, 0.0), (-hw - 1.0, g0 + 3.5, 0.0), P["arch_r"] + 0.6)
    return np.minimum(np.minimum(d, gate), hinge)


# ----------------------------------------------------------------------------- sensor pod + boot
def sd_pod(x, y, z):
    rxz = len2(x, z)
    pod = sd_profile(rxz, y, P["pod_r"], POD_C - P["pod_h"], POD_C + P["pod_h"], P["pod_r"] - 0.5)
    yb = y - ROPE_OFF
    rb = P["boot_r"] + 3.0 * smoothstep(-P["boot_len"], 2.0, yb)
    boot = 0.85 * sd_profile(rxz, yb, rb, -P["boot_len"], 4.0, 1.0)
    return smin(pod, boot, 2.0)


def sd_rope3(x, y, z):
    return eng.sd_rope(x, y - ROPE_OFF, z)


def sd_hook3(x, y, z):
    return eng.sd_hook(x, y - ROPE_OFF, z)


# ----------------------------------------------------------------------------- scene setup (patch the v2 engine)
def _shift(lo, hi):
    off = np.array([0, ROPE_OFF, 0], F32)
    return lo + off, hi + off


_rope_lo, _rope_hi = _shift(*[c for c in eng.COMPONENTS if c[0] == "rope"][0][2:])
_hook_lo, _hook_hi = _shift(*[c for c in eng.COMPONENTS if c[0] == "metal"][0][2:])
_bh, _br = P["bar_half"], P["bar_r"] + P["bar_bulge"] + 1
_aw, _ar = P["arch_hw"] + P["arch_r"] + 2, P["arch_r"] + 2

COMPONENTS = [
    ("bar", sd_bar, np.array([-_bh - 1, -_br, -_br], F32), np.array([_bh + 1, _br, _br], F32)),
    ("arch", sd_arch, np.array([-_aw, APEX_Y - _ar, -_ar], F32), np.array([_aw, 2, _ar], F32)),
    ("pod", sd_pod, np.array([-12, ROPE_OFF - P["boot_len"] - 1, -12], F32),
     np.array([12, POD_C + P["pod_h"] + 1, 12], F32)),
    ("rope", sd_rope3, _rope_lo, _rope_hi),
    ("metal", sd_hook3, _hook_lo, _hook_hi),
]
eng.COMPONENTS = COMPONENTS
eng.LO_ALL = np.min([c[2] for c in COMPONENTS], 0)
eng.HI_ALL = np.max([c[3] for c in COMPONENTS], 0)
eng.LC = (0.5 * (eng.LO_ALL + eng.HI_ALL)).astype(F32)
eng.R = rot_y(-18.0) @ rot_z(-7.0)
eng.FRAME_SCALE = 1.22

ARCH_COL = [0.36, 0.37, 0.39]   # graphite anodised aluminium; [0.91,0.915,0.92] for natural silver

MATS = {
    "bar": (0.58, False), "button": (0.38, False), "ring": (0.5, False), "pod": (0.58, False),
    "boot": (0.7, False), "arch": (0.32, True), "rope": (0.75, False), "metal": (0.24, True),
}


# ----------------------------------------------------------------------------- shading
def shade(g, color, ring_on=True):
    W, H, ss, N = g["W"], g["H"], g["ss"], g["N"]
    img = np.repeat(g["bg"][:, None], 3, 1).astype(F32)
    emis = np.zeros((N, 3), F32)
    fidx, fshade = g["floor"]
    img[fidx] *= fshade[:, None]

    p, n, ids, pl = g["p"], g["n"], g["ids"], g["pl"]
    x, y, z = pl[:, 0], pl[:, 1], pl[:, 2]
    mat = np.array([c[0] for c in COMPONENTS], dtype=object)[ids]
    isb = mat == "bar"
    qx, py, pz, along = btn_frame(x, y, z)
    btn = isb & (np.sqrt(qx * qx + py * py + pz * pz) < P["btn_r"] - 0.1) & (along > -6)
    ring = isb & (np.abs(x - P["ring_x"]) < P["ring_h"] + 0.05) & (len2(y, z) < bar_radius(x) - 0.25)
    boot = (mat == "pod") & (y < ROPE_OFF + 1.0)
    mat[btn], mat[ring], mat[boot] = "button", "ring", "boot"

    base = np.array(COLORS[color], F32)
    alb = np.repeat(base[None], len(p), 0)
    alb[mat == "button"] = base * 1.08
    alb[mat == "ring"] = 0.55
    alb[mat == "boot"] = [0.018, 0.019, 0.020]
    mr = mat == "rope"
    if mr.any():
        _, psi, s = eng.rope_coords(x[mr], y[mr] - ROPE_OFF, z[mr])
        h, strand = weave(psi, s, 8, eng.D["rope_r"])
        a = fiber_albedo(strand, np.array([0.020, 0.021, 0.023], F32), [0.03, 0.10, 0.095])
        alb[mr] = a * (0.55 + 0.45 * h)[:, None]
    alb[mat == "metal"] = [0.91, 0.915, 0.92]
    alb[mat == "arch"] = ARCH_COL

    rough = np.array([MATS[m][0] for m in mat], F32)
    metal = np.array([MATS[m][1] for m in mat])
    v = -g["rd"][g["obj"]]
    nv = np.clip(np.sum(n * v, 1), 1e-3, 1)
    r = 2 * nv[:, None] * n - v
    sh, ao = g["shadow"], g["ao"]
    key = np.clip((n @ LKEY + 0.15) / 1.15, 0, 1) * sh
    E = ((0.22 + 0.12 * n[:, 1]) * ao)[:, None] + 2.3 * key[:, None] * KEY_COL \
        + 0.45 * np.clip(n @ LFILL, 0, 1)[:, None] * FILL_COL + 0.9 * np.clip(n @ LRIM, 0, 1)[:, None]
    fres = (1 - nv) ** 5
    spec_occ = (ao * (0.55 + 0.45 * sh))[:, None]
    envr = env(r, rough)
    col = np.zeros_like(alb)
    di = ~metal
    Fd = 0.04 + (np.maximum(1 - rough[di], 0.04) - 0.04) * fres[di]
    col[di] = alb[di] * E[di] + Fd[:, None] * envr[di] * spec_occ[di]
    Fm = alb[metal] + (1 - alb[metal]) * fres[metal, None]
    col[metal] = Fm * envr[metal] * spec_occ[metal] + 0.02 * E[metal]

    if ring_on:
        rg = mat == "ring"
        e = TEAL * 1.6
        col[rg] = col[rg] * 0.3 + e
        em = np.zeros((len(p), 3), F32)
        em[rg] = e
        emis[g["obj"]] = em
    img[g["obj"]] = col

    Wr, Hr = W * ss, H * ss
    img = img.reshape(Hr, Wr, 3)
    if ring_on:
        em = emis.reshape(Hr, Wr, 3)
        bloom = np.stack([gaussian_filter(em[..., c], 5 * ss) for c in range(3)], -1)
        bloom += 0.5 * np.stack([gaussian_filter(em[..., c], 16 * ss) for c in range(3)], -1)
        img += 0.55 * bloom
    out = aces(img * 1.05) ** (1 / 2.2)
    out = out.reshape(H, ss, W, ss, 3).mean((1, 3))
    out += (np.random.default_rng(1).random(out.shape) - 0.5) / 255.0
    return Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--color", default="charcoal", choices=list(COLORS) + ["all"])
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--ss", type=int, default=2)
    ap.add_argument("--ring", default="on", choices=["on", "off"])
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "renders"))
    a = ap.parse_args()
    if a.preview:
        a.width, a.ss = 540, 1
    W, H = a.width, int(round(a.width * 5 / 4))
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    print(f"geometry pass {W}x{H} @ {a.ss}x{a.ss}")
    g = eng.render_geometry(W, H, a.ss)
    colors = list(COLORS) if a.color == "all" else [a.color]
    ims = []
    for c in colors:
        im = shade(g, c, a.ring == "on")
        path = os.path.join(a.out, f"ero_grip_v3_{c}.png")
        im.save(path)
        ims.append(im)
        print(f"  saved {path}")
    if len(ims) > 1:
        sheet = Image.new("RGB", (W * 2, H * 2))
        for i, im in enumerate(ims):
            sheet.paste(im, ((i % 2) * W, (i // 2) * H))
        path = os.path.join(a.out, "ero_grip_v3_colorways.png")
        sheet.save(path)
        print(f"  saved {path}")
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
