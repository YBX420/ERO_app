"""
ERO Ring - "signet ring" handle concept. Reuses the render engine, rope and snap hook from
ero_grip_v2.py; only the handle geometry and materials are new.

Shape: a closed ring the hand goes through. The top third swells into a rigid grip arc
(battery, sensors, haptics, thumb button, light line, magnetic dock/charge puck on the crown);
the rest is a thin soft band with a load-bearing core. The shock-cord rope hangs from a swivel
pod at the bottom of the band.

    python ero_grip_ring.py               # charcoal, 1080x1350
    python ero_grip_ring.py --color all   # four colourways + contact sheet
    python ero_grip_ring.py --preview

Local frame: the ring lies in the x-y plane centred on the origin, grip arc at the top (+y),
+z is the face with the thumb button. Flip the ring over for the other hand.
"""
import argparse
import os
import time

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

import ero_grip_v2 as eng
from ero_grip_v2 import (F32, COLORS, TEAL, len2, smoothstep, sd_profile, rot_y, rot_z,
                         weave, fiber_albedo, env, aces, LKEY, LFILL, LRIM, KEY_COL, FILL_COL)

# ----------------------------------------------------------------------------- design (mm)
RING = dict(
    hole_r=45.0,          # inner opening radius (90 mm: four fingers, or a wrist)
    band_r=5.5,           # soft band tube radius
    grip_r=13.0,          # grip arc tube radius (26 mm across)
    flat=1.15,            # >1 flattens the cross-section front-to-back
    grip_a0=44.0, grip_a1=66.0,   # degrees from the top: fully grip inside a0, fully band beyond a1
    seam_a=57.0,          # hairline seam between rigid arc and soft band
    finger_pitch=22.0, finger_depth=0.9,
    btn_a=34.0, btn_r=5.0, recess_R=28.0, recess_depth=1.4,
    light_a=42.0, light_h=0.8, light_inset=0.6,
    puck_r=5.5,           # magnetic dock / charging contact on the crown
    pod_r=8.5, pod_h=10.0,
    boot_len=10.0, boot_r=5.7,
)
G = RING
POD_Y = -(G["hole_r"] + G["band_r"])          # band centreline at the bottom
ROPE_OFF = POD_Y - G["pod_h"]                  # v2's rope/hook start at y=0; shift them here
TOP_Y = G["hole_r"] + 2 * G["grip_r"]


def ring_coords(x, y, z):
    """Angle from the top (signed, degrees), radial position, tube radius, centreline radius."""
    rho = len2(x, y)
    a_s = ((90.0 - np.degrees(np.arctan2(y, x)) + 180.0) % 360.0) - 180.0
    a = np.abs(a_s)
    w = smoothstep(G["grip_a1"], G["grip_a0"], a)
    rt = G["band_r"] + (G["grip_r"] - G["band_r"]) * w
    return a_s, a, rho, rt, G["hole_r"] + rt


BTN_TH = np.deg2rad(90.0 - G["btn_a"])
BTN_RC = G["hole_r"] + G["grip_r"]
BTN_X, BTN_Y = BTN_RC * np.cos(BTN_TH), BTN_RC * np.sin(BTN_TH)
BTN_ZC = G["grip_r"] / G["flat"] + G["recess_R"] - G["recess_depth"]


def sd_ring(x, y, z):
    a_s, a, rho, rt, rc = ring_coords(x, y, z)
    zz = z * G["flat"]
    dr = rho - rc
    tube = len2(dr, zz)

    # finger scallops on the inside of the grip arc
    inner = smoothstep(-0.2, -0.8, dr / np.maximum(tube, 1e-3))
    wg = smoothstep(50.0, 42.0, a)
    fp = G["finger_pitch"]
    groove = G["finger_depth"] * inner * wg * (0.5 + 0.5 * np.cos(2 * np.pi * (a_s - 0.5 * fp) / fp)) ** 2
    d = tube - rt + groove

    # seam between the rigid arc and the soft band
    arc = np.abs(a - G["seam_a"]) * (np.pi / 180.0) * rho
    d = np.maximum(d, -np.maximum(arc - 0.3, (rt - 0.45) - tube))

    # light line along the outer rim of the grip arc
    arc_l = (a - G["light_a"]) * (np.pi / 180.0) * rho
    light = np.maximum(np.maximum(np.abs(z) - G["light_h"], (rc + rt - G["light_inset"]) - rho), arc_l)
    d = np.maximum(d, -light)

    # thumb recess + flush button on the +z face
    qx, qy = x - BTN_X, y - BTN_Y
    q = qx * qx + qy * qy
    dish = np.sqrt(q + (z - BTN_ZC) ** 2) - G["recess_R"]
    d = np.maximum(d, -dish)
    zs = BTN_ZC - np.sqrt(np.maximum(G["recess_R"] ** 2 - q, 0))
    gb = len2(np.sqrt(q) - G["btn_r"], z - zs) - 0.28
    d = np.maximum(d, -np.where(z > 0, gb, 1e3))
    return 0.72 * d


def sd_puck(x, y, z):
    return sd_profile(len2(x, z), y, G["puck_r"], TOP_Y - 4.0, TOP_Y + 0.5, 0.5)


def sd_pod(x, y, z):
    rxz = len2(x, z)
    pod = sd_profile(rxz, y, G["pod_r"], POD_Y - G["pod_h"], POD_Y + G["pod_h"] - 1.0, 3.5)
    pod = np.maximum(pod, -(len2(rxz - G["pod_r"], y - (POD_Y - 5.5)) - 0.3))
    yb = y - ROPE_OFF
    rb = G["boot_r"] + 2.6 * smoothstep(-G["boot_len"], 2.0, yb)
    boot = 0.85 * sd_profile(rxz, yb, rb, -G["boot_len"], 2.0, 1.0)
    return np.minimum(pod, boot)


def sd_rope(x, y, z):
    return eng.sd_rope(x, y - ROPE_OFF, z)


def sd_hook(x, y, z):
    return eng.sd_hook(x, y - ROPE_OFF, z)


# ----------------------------------------------------------------------------- scene setup (patch the v2 engine)
def _shifted(name):
    lo, hi = [c for c in eng.COMPONENTS if c[0] == name][0][2:]
    off = np.array([0, ROPE_OFF, 0], F32)
    return lo + off, hi + off


_ro = G["hole_r"] + 2 * G["grip_r"] + 1.5
_rz = G["grip_r"] / G["flat"] + 1.5
COMPONENTS = [
    ("ring", sd_ring, np.array([-_ro, -_ro, -_rz], F32), np.array([_ro, _ro, _rz], F32)),
    ("puck", sd_puck, np.array([-6.5, TOP_Y - 5, -6.5], F32), np.array([6.5, TOP_Y + 1.5, 6.5], F32)),
    ("pod", sd_pod, np.array([-10, ROPE_OFF - G["boot_len"] - 1, -10], F32),
     np.array([10, POD_Y + G["pod_h"] + 1, 10], F32)),
    ("rope", sd_rope, *_shifted("rope")),
    ("metal", sd_hook, *_shifted("metal")),
]
eng.COMPONENTS = COMPONENTS
eng.LO_ALL = np.min([c[2] for c in COMPONENTS], 0)
eng.HI_ALL = np.max([c[3] for c in COMPONENTS], 0)
eng.LC = (0.5 * (eng.LO_ALL + eng.HI_ALL)).astype(F32)
eng.R = rot_y(-6.0) @ rot_z(-7.0)
eng.FRAME_SCALE = 1.12

MATS = {  # roughness, metallic
    "grip": (0.58, False), "band": (0.72, False), "button": (0.38, False), "light": (0.5, False),
    "puck": (0.24, True), "pod": (0.24, True), "boot": (0.7, False),
    "rope": (0.75, False), "metal": (0.24, True),
}


def shade(g, color, light_on=True):
    W, H, ss, N = g["W"], g["H"], g["ss"], g["N"]
    img = np.repeat(g["bg"][:, None], 3, 1).astype(F32)
    emis = np.zeros((N, 3), F32)
    fidx, fshade = g["floor"]
    img[fidx] *= fshade[:, None]

    p, n, ids, pl = g["p"], g["n"], g["ids"], g["pl"]
    x, y, z = pl[:, 0], pl[:, 1], pl[:, 2]
    mat = np.array([c[0] for c in COMPONENTS], dtype=object)[ids]
    isr = mat == "ring"
    a_s, a, rho, rt, rc = ring_coords(x, y, z)
    mat[isr] = "grip"
    mat[isr & (a > G["seam_a"])] = "band"
    mat[isr & (len2(x - BTN_X, y - BTN_Y) < G["btn_r"] - 0.1) & (z > 0)] = "button"
    mat[isr & (a < G["light_a"] + 0.5) & (np.abs(z) < G["light_h"] + 0.05)
        & (rho > rc + rt - 1.6) & (rho < rc + rt - 0.25)] = "light"
    mat[(mat == "pod") & (y < ROPE_OFF - 0.2)] = "boot"

    base = np.array(COLORS[color], F32)
    alb = np.repeat(base[None], len(p), 0)
    alb[mat == "band"] = base * 0.62
    alb[mat == "button"] = base * 1.08
    alb[mat == "light"] = 0.55
    alb[mat == "boot"] = [0.018, 0.019, 0.020]
    mr = mat == "rope"
    if mr.any():
        _, psi, s = eng.rope_coords(x[mr], y[mr] - ROPE_OFF, z[mr])
        h, strand = weave(psi, s, 8, eng.D["rope_r"])
        fa = fiber_albedo(strand, np.array([0.020, 0.021, 0.023], F32), [0.03, 0.10, 0.095])
        alb[mr] = fa * (0.55 + 0.45 * h)[:, None]
    for m in ("puck", "pod", "metal"):
        alb[mat == m] = [0.91, 0.915, 0.92]

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

    if light_on:
        lg = mat == "light"
        e = TEAL * 1.6
        col[lg] = col[lg] * 0.3 + e
        em = np.zeros((len(p), 3), F32)
        em[lg] = e
        emis[g["obj"]] = em
    img[g["obj"]] = col

    Wr, Hr = W * ss, H * ss
    img = img.reshape(Hr, Wr, 3)
    if light_on:
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
    ap.add_argument("--light", default="on", choices=["on", "off"])
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
        im = shade(g, c, a.light == "on")
        path = os.path.join(a.out, f"ero_ring_{c}.png")
        im.save(path)
        ims.append(im)
        print(f"  saved {path}")
    if len(ims) > 1:
        sheet = Image.new("RGB", (W * 2, H * 2))
        for i, im in enumerate(ims):
            sheet.paste(im, ((i % 2) * W, (i // 2) * H))
        path = os.path.join(a.out, "ero_ring_colorways.png")
        sheet.save(path)
        print(f"  saved {path}")
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
