"""
ERO Grip v2 - procedural product render (SDF ray marching, numpy only).

What changed from v1 (ero_grip_render.py):
  * the bellows damper is replaced by a real braided shock-cord rope (9 mm, kernmantle braid,
    teal tracer strands) that curves out of a soft rubber boot into a crimped aluminium ferrule;
  * anti-slip: the top end flares into a pommel (the dog pulls the grip *through* the fist toward
    the hook end, so the stop has to be at the top), ribbed grip band on the finger side, and a
    braided wrist cord threaded through a metal bail on the top cap;
  * the same top bail is the hands-free / waist attachment point (carabiner or girth-hitch the
    wrist cord round a belt) - the load path stays through the grip, so sensing still works.

    python ero_grip_v2.py                 # charcoal hero, 1080x1350
    python ero_grip_v2.py --color all     # four colourways + contact sheet
    python ero_grip_v2.py --preview       # quick low-res check

Dimensions are millimetres, all in DESIGN. Local frame: +y along the grip axis (top cap at
y = body_len), +z is the thumb side.
"""
import argparse
import os
import time

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

F32 = np.float32

# ----------------------------------------------------------------------------- design (mm)
DESIGN = dict(
    body_len=112.0,      # grip body length, boot excluded
    body_r=15.8,         # grip radius (31.6 mm diameter where the fingers wrap)
    waist=1.0, waist_y=45.0, waist_w=25.0,
    pommel_r=4.6,        # how much the top flares out (anti-slip stop)
    pommel_y0=86.0, pommel_y1=103.0,
    end_round=5.5,
    rib_y0=10.0, rib_y1=76.0, rib_pitch=3.2, rib_depth=0.32,   # grip ribs on the finger side
    btn_y=80.0, btn_r=5.4, recess_R=30.0, recess_depth=1.7, recess_stretch=0.55,
    ring_y=106.0, ring_h=0.8, ring_inset=0.6,                    # light-guide ring on the pommel
    boot_len=11.0, boot_r=5.7,                                     # rubber strain-relief boot
    rope_r=4.5, rope_straight=12.0, rope_bend_R=70.0, rope_bend_deg=28.0,
    collar_r=6.4, collar_len=16.0,
    hook_cy=-40.0, hook_half=10.0, hook_R=8.0, hook_wire=2.5, gate_r=1.7,
    bail_R=5.5, bail_wire=1.6,                                     # top bail (wrist cord / waist clip)
    cord_r=2.0, loop_len=70.0, loop_r=18.0, loop_lean=-0.18,       # wrist cord teardrop
)
D = DESIGN

COLORS = {
    "charcoal": (0.022, 0.024, 0.026),
    "chalk":    (0.58, 0.57, 0.54),
    "moss":     (0.070, 0.095, 0.058),
    "sand":     (0.40, 0.335, 0.25),
}
TEAL = np.array([0.16, 0.52, 0.46], F32)


# ----------------------------------------------------------------------------- helpers
def nrm(v):
    v = np.asarray(v, F32)
    return v / np.linalg.norm(v)


def len2(a, b):
    return np.sqrt(a * a + b * b)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def sd_box2(px, py, bx, by):
    qx = np.abs(px) - bx
    qy = np.abs(py) - by
    return len2(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0)


def sd_profile(rxz, y, r, y0, y1, rr):
    """Rounded capped cylinder along y; r may vary with y."""
    h = 0.5 * (y1 - y0)
    yc = y - 0.5 * (y0 + y1)
    dx = rxz - r + rr
    dy = np.abs(yc) - h + rr
    return np.minimum(np.maximum(dx, dy), 0) + len2(np.maximum(dx, 0), np.maximum(dy, 0)) - rr


def sd_capsule3(x, y, z, a, b, r):
    a = np.asarray(a, F32)
    ba = np.asarray(b, F32) - a
    px, py, pz = x - a[0], y - a[1], z - a[2]
    h = np.clip((px * ba[0] + py * ba[1] + pz * ba[2]) / ba.dot(ba), 0, 1)
    return np.sqrt((px - ba[0] * h) ** 2 + (py - ba[1] * h) ** 2 + (pz - ba[2] * h) ** 2) - r


def sd_uneven_capsule(px, py, r1, r2, h):
    px = np.abs(px)
    b = (r1 - r2) / h
    a = np.sqrt(1 - b * b)
    k = -b * px + a * py
    return np.where(k < 0, len2(px, py) - r1,
                    np.where(k > a * h, len2(px, py - h) - r2, a * px + b * py - r1))


def rot_y(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], F32)


def rot_z(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], F32)


def weave(psi, s, n, r):
    """Plain-weave braid on a tube. Returns height 0..1 and the id of the strand on top."""
    k = n / r
    a = (n * psi + k * s) / np.pi
    b = (n * psi - k * s) / np.pi
    ia, ib = np.floor(a), np.floor(b)
    h1, h2 = np.sin(np.pi * (a - ia)), np.sin(np.pi * (b - ib))
    top1 = ((ia + ib) % 2) == 0
    h = np.where(top1, h1, h2) * 0.8 + 0.2 * h1 * h2
    strand = np.where(top1, ia, ib + 1000)
    return h, strand


# ----------------------------------------------------------------------------- body
def body_radius(y):
    return (D["body_r"]
            - D["waist"] * np.exp(-(((y - D["waist_y"]) / D["waist_w"]) ** 2))
            + D["pommel_r"] * smoothstep(D["pommel_y0"], D["pommel_y1"], y))


def sd_body(x, y, z):
    L = D["body_len"]
    rxz = len2(x, z)
    r = body_radius(y)

    # finger-side ribs (fade out on the thumb side and at both ends of the band)
    side = smoothstep(0.55, 0.10, z / np.maximum(rxz, 1e-3))
    band = smoothstep(D["rib_y0"], D["rib_y0"] + 6, y) * smoothstep(D["rib_y1"], D["rib_y1"] - 6, y)
    rib = D["rib_depth"] * side * band * (0.5 + 0.5 * np.cos(2 * np.pi * y / D["rib_pitch"])) ** 2
    d = 0.8 * sd_profile(rxz + rib, y, r, 0.0, L, D["end_round"])

    # rubber boot where the rope leaves the body
    bl = D["boot_len"]
    rb = D["boot_r"] + 3.6 * smoothstep(-bl, 2.0, y)
    boot = 0.85 * sd_profile(rxz, y, rb, -bl, 4.0, 1.0)
    d = smin(d, boot, 2.0)

    # thumb recess + flush button
    zc = body_radius(np.float32(D["btn_y"])) + D["recess_R"] - D["recess_depth"]
    by = y - D["btn_y"]
    q = x * x + D["recess_stretch"] * by * by
    dish = np.sqrt(q + (z - zc) ** 2) - D["recess_R"]
    d = smax(d, -dish, 1.2)
    zs = zc - np.sqrt(np.maximum(D["recess_R"] ** 2 - q, 0))
    groove = len2(len2(x, by) - D["btn_r"], z - zs) - 0.28
    d = np.maximum(d, -np.where(z > 0, groove, 1e3))

    # light-guide ring on the pommel
    rr = body_radius(np.float32(D["ring_y"]))
    ring = np.maximum(np.abs(y - D["ring_y"]) - D["ring_h"], (rr - D["ring_inset"]) - rxz)
    return np.maximum(d, -ring)


# ----------------------------------------------------------------------------- rope
Y0 = -D["rope_straight"] + 4.0          # where the straight part ends and the bend starts
TH = np.deg2rad(D["rope_bend_deg"])
RB = D["rope_bend_R"]
ROPE_END = np.array([RB - RB * np.cos(TH), Y0 - RB * np.sin(TH), 0.0], F32)


def rope_coords(x, y, z):
    """Distance to the rope centreline, plus (psi, s) braid coordinates."""
    # straight part: from y=4 (inside the body) down to Y0
    ys = np.clip(y, Y0, 4.0)
    d_st = np.sqrt(x * x + (y - ys) ** 2 + z * z)
    s_st = Y0 - ys
    psi_st = np.arctan2(z, -x)
    # bend: arc centred at (RB, Y0), sweeping toward +x as it goes down
    qx, qy = x - RB, y - Y0
    th = np.clip(np.arctan2(-qy, -qx), 0.0, TH)
    cx, cy = -np.cos(th), -np.sin(th)
    dx, dy = qx - RB * cx, qy - RB * cy
    d_arc = np.sqrt(dx * dx + dy * dy + z * z)
    psi_arc = np.arctan2(z, dx * cx + dy * cy)
    use_arc = d_arc < d_st
    return (np.where(use_arc, d_arc, d_st), np.where(use_arc, psi_arc, psi_st),
            np.where(use_arc, RB * th, s_st))


def sd_rope(x, y, z):
    d, psi, s = rope_coords(x, y, z)
    h, _ = weave(psi, s, 8, D["rope_r"])
    return 0.75 * (d - D["rope_r"] - 0.38 * h)


# ----------------------------------------------------------------------------- metal: ferrule + snap hook (hook frame) + top bail
def to_hook(x, y, z):
    c, s = np.cos(TH), np.sin(TH)
    qx, qy = x - ROPE_END[0], y - ROPE_END[1]
    return c * qx + s * qy, -s * qx + c * qy, z


def from_hook(pts):
    c, s = np.cos(TH), np.sin(TH)
    pts = np.asarray(pts, F32)
    x = c * pts[:, 0] - s * pts[:, 1] + ROPE_END[0]
    y = s * pts[:, 0] + c * pts[:, 1] + ROPE_END[1]
    return np.stack([x, y, pts[:, 2]], 1)


def sd_hook(x, y, z):
    hx, hy, hz = to_hook(x, y, z)
    rxz = len2(hx, hz)
    cl = D["collar_len"]
    collar = sd_profile(rxz, hy, D["collar_r"], -cl + 3.0, 3.0, 1.8)
    collar = np.maximum(collar, -(len2(rxz - D["collar_r"], hy + 5.0) - 0.3))
    collar = np.maximum(collar, -(len2(rxz - D["collar_r"], hy + 8.0) - 0.3))
    neck = sd_profile(rxz, hy, 2.8, -cl - 8.0, -cl + 4.0, 0.8)
    metal = smin(collar, neck, 1.0)
    yy = hy - D["hook_cy"]
    hh = D["hook_half"]
    loop = len2(len2(hx, yy - np.clip(yy, -hh, hh)) - D["hook_R"], hz) - D["hook_wire"]
    loop = np.maximum(loop, -sd_box2(hx - D["hook_R"], yy - 1.5, 5.0, 7.5))
    gate = sd_capsule3(hx, hy, hz, (D["hook_R"], D["hook_cy"] + hh, 0.0),
                       (D["hook_R"] - 0.3, D["hook_cy"] - 3.8, 0.0), D["gate_r"])
    return np.minimum(np.minimum(metal, loop), gate)


BAIL_Y = D["body_len"] + 0.5


def sd_bail(x, y, z):
    # wire loop standing on the top cap, in the y-z plane (axis along x)
    return len2(len2(y - BAIL_Y, z) - D["bail_R"], x) - D["bail_wire"]


# ----------------------------------------------------------------------------- wrist cord
CORD_A = np.array([0, BAIL_Y + D["bail_R"], 0], F32)          # wraps round the top of the bail wire
CORD_N = nrm([0.15, 0.0, 1.0])                                  # loop plane normal (faces camera)
CORD_U = nrm(np.array([D["loop_lean"], 1.0, 0.0]) - CORD_N * np.dot([D["loop_lean"], 1.0, 0.0], CORD_N))
CORD_V = np.cross(CORD_N, CORD_U)
CORD_R1 = D["bail_wire"] + D["cord_r"]


def cord_coords(x, y, z):
    qx, qy, qz = x - CORD_A[0], y - CORD_A[1], z - CORD_A[2]
    a = qx * CORD_V[0] + qy * CORD_V[1] + qz * CORD_V[2]
    b = qx * CORD_U[0] + qy * CORD_U[1] + qz * CORD_U[2]
    c = qx * CORD_N[0] + qy * CORD_N[1] + qz * CORD_N[2]
    o = sd_uneven_capsule(a, b, CORD_R1, D["loop_r"], D["loop_len"])
    psi = np.arctan2(c, o)
    s = np.arctan2(a, b - 0.6 * D["loop_len"]) * 40.0
    return len2(o, c), psi, s


def sd_cord(x, y, z):
    d, psi, s = cord_coords(x, y, z)
    h, _ = weave(psi, s, 6, D["cord_r"])
    return 0.75 * (d - D["cord_r"] - 0.16 * h)


# ----------------------------------------------------------------------------- scene assembly
def _aabb(pts, pad):
    pts = np.asarray(pts, F32)
    return pts.min(0) - pad, pts.max(0) + pad


def _box_corners(lo, hi):
    return np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])], F32)


_th = np.linspace(0, TH, 24)
_rope_pts = np.concatenate([
    np.array([[0, 4.0, 0], [0, Y0, 0]], F32),
    np.stack([RB - RB * np.cos(_th), Y0 - RB * np.sin(_th), 0 * _th], 1)])
_hook_box = _box_corners((-11.5, -63.0, -7.5), (11.5, 5.0, 7.5))
_cord_pts = np.array([CORD_A + a * CORD_V + b * CORD_U + c * CORD_N
                      for a in (-D["loop_r"] - 3, D["loop_r"] + 3)
                      for b in (-CORD_R1 - 3, D["loop_len"] + D["loop_r"] + 3)
                      for c in (-3, 3)], F32)
RMAX = D["body_r"] + D["pommel_r"] + 1.0

COMPONENTS = [  # name, fn, aabb (local)
    ("body", sd_body, *_aabb([[-RMAX, -D["boot_len"] - 1, -RMAX], [RMAX, D["body_len"] + 1, RMAX]], 0)),
    ("rope", sd_rope, *_aabb(_rope_pts, D["rope_r"] + 1.0)),
    ("metal", sd_hook, *_aabb(from_hook(_hook_box), 0.5)),
    ("bail", sd_bail, *_aabb([[-2, BAIL_Y - 8, -8], [2, BAIL_Y + 8, 8]], 0)),
    ("cord", sd_cord, *_aabb(_cord_pts, 0)),
]

LO_ALL = np.min([c[2] for c in COMPONENTS], 0)
HI_ALL = np.max([c[3] for c in COMPONENTS], 0)
LC = (0.5 * (LO_ALL + HI_ALL)).astype(F32)           # object centre (local)
R = rot_y(15.0) @ rot_z(16.0)                       # world = R @ (local - LC)
BOX_MARGIN = 30.0
FRAME_SCALE = 1.0   # >1 zooms out


def sd_aabb(p, lo, hi):
    c = 0.5 * (lo + hi)
    h = 0.5 * (hi - lo)
    q = np.abs(p - c) - h
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)


def to_local(p):
    return (p @ R + LC).astype(F32)


def scene(p, want_id=False):
    pl = to_local(p)
    best = np.full(len(p), 1e9, F32)
    ids = np.zeros(len(p), np.int8)
    for i, (_, fn, lo, hi) in enumerate(COMPONENTS):
        d = sd_aabb(pl, lo, hi)
        m = d < BOX_MARGIN
        if m.any():
            q = pl[m]
            d[m] = fn(q[:, 0], q[:, 1], q[:, 2])
        better = d < best
        best = np.where(better, d, best)
        ids[better] = i
    return (best, ids, pl) if want_id else best


# ----------------------------------------------------------------------------- lighting
LKEY = nrm([-0.45, 0.85, 0.40])
LFILL = nrm([0.90, 0.15, 0.45])
LRIM = nrm([0.55, 0.45, -0.75])
KEY_COL = np.array([1.0, 0.98, 0.95], F32)
FILL_COL = np.array([0.92, 0.96, 1.0], F32)


def panel(d, L, sa, sb, inten, w):
    ux = nrm(np.cross([0, 1, 0], L))
    vy = np.cross(L, ux)
    f = d @ L
    fs = np.maximum(f, 1e-3)
    a = np.abs(d @ ux) / fs
    b = np.abs(d @ vy) / fs
    return inten * (f > 0) * smoothstep(sa + w, sa - w, a) * smoothstep(sb + w, sb - w, b)


def env(d, rough):
    y = d[:, 1]
    base = np.where(y > 0, 0.30 + 0.30 * y, 0.50 - 0.15 * np.clip(-y * 3, 0, 1))
    w = 0.015 + 0.45 * rough * rough
    k = 1.0 / (1.0 + 6.0 * rough * rough)
    e = (panel(d, LKEY, 0.32, 0.60, 5.0, w) + panel(d, LRIM, 0.05, 0.90, 3.5, w)
         + panel(d, LFILL, 0.45, 0.45, 1.0, w) + 1.5 * smoothstep(0.75 - w, 0.95 + w, y))
    return (base + k * e)[:, None]


def aces(x):
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


# ----------------------------------------------------------------------------- marching
def march(ro, rd, t, tmax, steps=260, eps=0.02, step=0.8):
    t = t.astype(F32).copy()
    hit = np.zeros(len(t), bool)
    act = np.arange(len(t))
    for _ in range(steps):
        if act.size == 0:
            break
        d = scene(ro[act] + rd[act] * t[act, None])
        h = d < eps
        hit[act[h]] = True
        act, d = act[~h], d[~h]
        t[act] += d * step
        act = act[t[act] < tmax[act]]
    return hit, t


def soft_shadow(ro, rd, k, tmax=320.0, steps=80):
    res = np.ones(len(ro), F32)
    t = np.full(len(ro), 0.6, F32)
    act = np.arange(len(ro))
    for _ in range(steps):
        if act.size == 0:
            break
        d = scene(ro[act] + rd[act] * t[act, None])
        res[act] = np.minimum(res[act], k * d / t[act])
        t[act] += np.clip(d, 0.25, 10.0)
        act = act[(res[act] > 0.002) & (t[act] < tmax)]
    res = np.clip(res, 0, 1)
    return res * res * (3 - 2 * res)


def normals(p, e=0.02):
    ks = np.array([[1, -1, -1], [-1, -1, 1], [-1, 1, -1], [1, 1, 1]], F32)
    n = np.zeros_like(p)
    for k in ks:
        n += k * scene(p + e * k)[:, None]
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def ambient_occlusion(p, n):
    occ = np.zeros(len(p), F32)
    wsum = 0.0
    for i in range(5):
        h = 0.5 + 2.6 * i
        w = 0.75 ** i
        occ += w * np.clip((h - scene(p + n * h)) / h, 0, 1)
        wsum += w
    return np.clip(1 - 1.4 * occ / wsum, 0, 1)


# ----------------------------------------------------------------------------- render
def render_geometry(W, H, ss):
    t0 = time.time()
    Wr, Hr = W * ss, H * ss
    corners = (_box_corners(LO_ALL, HI_ALL) - LC) @ R.T
    C = 0.5 * (corners.min(0) + corners.max(0))
    rad = float(np.linalg.norm(HI_ALL - LO_ALL) * 0.5) + 2.0

    target = C + np.array([0, -4.0, 0], F32)
    az, el, dist = np.deg2rad(30), np.deg2rad(14), 1100.0
    half_h = 0.5 * (corners[:, 1].max() - corners[:, 1].min()) * FRAME_SCALE
    fov = 2 * np.arctan(half_h / dist)
    cam = (target + dist * np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])).astype(F32)
    fwd = nrm(target - cam)
    right = nrm(np.cross(fwd, [0, 1, 0]))
    up = np.cross(right, fwd)
    th = np.tan(fov / 2)
    jj, ii = np.meshgrid(np.arange(Wr, dtype=F32), np.arange(Hr, dtype=F32))
    sx = ((jj + 0.5) / Wr * 2 - 1) * th * (W / H)
    sy = (1 - (ii + 0.5) / Hr * 2) * th
    rd = fwd + sx[..., None] * right + sy[..., None] * up
    rd = (rd / np.linalg.norm(rd, axis=2, keepdims=True)).reshape(-1, 3).astype(F32)
    N = len(rd)
    ro = np.broadcast_to(cam, (N, 3))

    nx, ny = sx.reshape(-1) / th, sy.reshape(-1) / th
    bg = 0.86 * (1 - 0.10 * ((nx * 0.8) ** 2 + (ny - 0.15) ** 2))

    oc = cam - C
    b = rd @ oc
    disc = b * b - (oc @ oc - rad * rad)
    cand = np.where(disc > 0)[0]
    sq = np.sqrt(disc[cand])
    hit_c, t_c = march(ro[cand], rd[cand], np.maximum(-b[cand] - sq, 0), -b[cand] + sq)
    obj = cand[hit_c]
    print(f"  primary rays: {len(cand):,} marched, {len(obj):,} hit  ({time.time()-t0:.1f}s)")

    g = dict(W=W, H=H, ss=ss, N=N, rd=rd, bg=bg.astype(F32), obj=obj)
    p = cam + rd[obj] * t_c[hit_c][:, None]
    n = normals(p)
    _, ids, pl = scene(p, want_id=True)
    g.update(p=p, n=n, ids=ids, pl=pl)
    g["shadow"] = soft_shadow(p + n * 0.15, np.broadcast_to(LKEY, p.shape), k=10.0)
    g["ao"] = ambient_occlusion(p, n)
    print(f"  surface: normals, shadow, AO  ({time.time()-t0:.1f}s)")

    YG = float(corners[:, 1].min()) - 6.0
    fl = rd[:, 1] < 0
    fl[obj] = False
    fidx = np.where(fl)[0]
    tg = (YG - cam[1]) / rd[fidx, 1]
    pg = cam + rd[fidx] * tg[:, None]
    near = np.hypot(pg[:, 0] - C[0], pg[:, 2] - C[2]) < rad * 2.6
    fidx, pg = fidx[near], pg[near]
    vis = np.ones(len(pg), F32)
    oc2 = pg - C
    b2 = oc2 @ LKEY
    reach = (b2 * b2 - (np.sum(oc2 * oc2, 1) - rad * rad)) > 0
    vis[reach] = soft_shadow(pg[reach] + [0, 0.3, 0], np.broadcast_to(LKEY, pg[reach].shape), k=4.0, tmax=600)
    occ = np.zeros(len(pg), F32)
    for h, w in ((4, 1.0), (12, 0.8), (25, 0.6), (45, 0.45), (70, 0.3)):
        occ += w * np.clip((h - scene(pg + [0, h, 0])) / h, 0, 1)
    occ /= 3.15
    g["floor"] = (fidx, (1 - 0.50 * (1 - vis)) * (1 - 0.65 * occ))
    print(f"  floor shadow  ({time.time()-t0:.1f}s)")
    return g


MATS = {  # roughness, metallic
    "body": (0.58, False), "button": (0.38, False), "ring": (0.5, False), "boot": (0.7, False),
    "rope": (0.75, False), "metal": (0.24, True), "bail": (0.24, True), "cord": (0.8, False),
}


def fiber_albedo(strand, base, tracer):
    alb = np.repeat(base[None], len(strand), 0)
    v = 0.85 + 0.3 * ((strand * 7919) % 13) / 13.0          # per-strand tone variation
    alb *= v[:, None]
    t = (strand % 8) == 0
    alb[t] = tracer
    return alb


def shade(g, color, ring_on=True):
    W, H, ss, N = g["W"], g["H"], g["ss"], g["N"]
    img = np.repeat(g["bg"][:, None], 3, 1).astype(F32)
    emis = np.zeros((N, 3), F32)
    fidx, fshade = g["floor"]
    img[fidx] *= fshade[:, None]

    p, n, ids, pl = g["p"], g["n"], g["ids"], g["pl"]
    x, y, z = pl[:, 0], pl[:, 1], pl[:, 2]
    rxz = len2(x, z)
    mat = np.array([c[0] for c in COMPONENTS], dtype=object)[ids]
    isb = mat == "body"
    btn = isb & (len2(x, y - D["btn_y"]) < D["btn_r"] - 0.1) & (z > 8)
    ring = isb & (np.abs(y - D["ring_y"]) < D["ring_h"] + 0.05) & (rxz < body_radius(y) - 0.25)
    boot = isb & (y < 1.0)
    mat[btn], mat[ring], mat[boot] = "button", "ring", "boot"

    base = np.array(COLORS[color], F32)
    alb = np.repeat(base[None], len(p), 0)
    alb[mat == "button"] = base * 1.08
    alb[mat == "ring"] = 0.55
    alb[mat == "boot"] = [0.018, 0.019, 0.020]
    mr = mat == "rope"
    if mr.any():
        _, psi, s = rope_coords(x[mr], y[mr], z[mr])
        h, strand = weave(psi, s, 8, D["rope_r"])
        a = fiber_albedo(strand, np.array([0.020, 0.021, 0.023], F32), [0.03, 0.10, 0.095])
        alb[mr] = a * (0.55 + 0.45 * h)[:, None]
    mc = mat == "cord"
    if mc.any():
        _, psi, s = cord_coords(x[mc], y[mc], z[mc])
        h, strand = weave(psi, s, 6, D["cord_r"])
        a = fiber_albedo(strand, np.array([0.020, 0.021, 0.023], F32), [0.03, 0.10, 0.095])
        alb[mc] = a * (0.6 + 0.4 * h)[:, None]
    mm = (mat == "metal") | (mat == "bail")
    alb[mm] = [0.91, 0.915, 0.92]

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
    g = render_geometry(W, H, a.ss)
    colors = list(COLORS) if a.color == "all" else [a.color]
    ims = []
    for c in colors:
        im = shade(g, c, a.ring == "on")
        path = os.path.join(a.out, f"ero_grip_v2_{c}.png")
        im.save(path)
        ims.append(im)
        print(f"  saved {path}")
    if len(ims) > 1:
        sheet = Image.new("RGB", (W * 2, H * 2))
        for i, im in enumerate(ims):
            sheet.paste(im, ((i % 2) * W, (i // 2) * H))
        path = os.path.join(a.out, "ero_grip_v2_colorways.png")
        sheet.save(path)
        print(f"  saved {path}")
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
