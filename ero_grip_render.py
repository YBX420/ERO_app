"""
ERO Grip - procedural product render (signed-distance-field ray marching, numpy only).

    python ero_grip_render.py                  # charcoal hero, 1080x1350
    python ero_grip_render.py --color all      # charcoal / chalk / moss / sand + contact sheet
    python ero_grip_render.py --preview        # quick low-res check
    python ero_grip_render.py --ring off       # light-guide ring unlit (matte white)

Every dimension is in millimetres and lives in DESIGN below; change a number and re-run.
Local frame of the object: +y runs along the grip axis (top cap at y = body_len, hook below 0),
+z is the thumb side (button + recess).
"""
import argparse
import os
import time

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# ----------------------------------------------------------------------------- design (mm)
DESIGN = dict(
    body_len=120.0,      # grip body length
    body_r=17.0,         # radius at the ends  (34 mm diameter)
    waist=2.2,           # how much the waist pulls in
    waist_pos=0.45,      # waist position along the body (0 = bottom, 1 = top)
    waist_w=0.22,        # waist width (fraction of length)
    end_round=7.0,       # end-cap rounding radius
    btn_y=88.0,          # thumb button centre height
    btn_r=5.6,           # button radius
    recess_R=30.0,       # thumb recess sphere radius (bigger = shallower dish)
    recess_depth=1.8,    # recess depth
    recess_stretch=0.55, # <1 elongates the recess along the axis
    ring_y=104.0,        # light-guide ring height
    ring_h=0.9,          # ring half-height
    ring_inset=0.6,      # ring recess depth
    damper_top=1.0, damper_bot=-23.0, damper_r=8.0, bellows_amp=0.8, bellows_pitch=4.5,
    collar_top=-22.0, collar_bot=-35.0, collar_r=8.5,
    hook_cy=-60.0, hook_half=12.0, hook_R=8.5, hook_wire=2.6, gate_r=1.8,
    strap_y=-28.5, strap_len=14.0, strap_loop_r=3.5, strap_w=10.0, strap_t=1.0,
    strap_az=-20.0,      # direction the webbing loop sticks out (deg, from +z toward +x)
)
D = DESIGN

# colourways: linear-space albedo of the soft-touch body
COLORS = {
    "charcoal": (0.022, 0.024, 0.026),
    "chalk":    (0.58, 0.57, 0.54),
    "moss":     (0.070, 0.095, 0.058),
    "sand":     (0.40, 0.335, 0.25),
}
TEAL = np.array([0.16, 0.52, 0.46], np.float32)  # low-saturation teal, linear

F32 = np.float32


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


def sd_round_rect(px, py, bx, by, r):
    return sd_box2(px, py, bx - r, by - r) - r


def sd_profile(rxz, y, r, y0, y1, rr):
    """Rounded capped cylinder along y whose radius r may vary with y."""
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


def rot_y(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], F32)


def rot_z(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], F32)


# ----------------------------------------------------------------------------- components (local frame)
def body_radius(y):
    t = np.clip(y / D["body_len"], 0, 1)
    return D["body_r"] - D["waist"] * np.exp(-(((t - D["waist_pos"]) / D["waist_w"]) ** 2))


def recess_geom():
    zc = body_radius(np.float32(D["btn_y"])) + D["recess_R"] - D["recess_depth"]
    return zc


def sd_body(x, y, z):
    L = D["body_len"]
    rxz = len2(x, z)
    d = 0.9 * sd_profile(rxz, y, body_radius(y), 0.0, L, D["end_round"])

    # thumb recess: a shallow, axis-elongated dish
    zc = recess_geom()
    by = y - D["btn_y"]
    q = x * x + D["recess_stretch"] * by * by
    dish = np.sqrt(q + (z - zc) ** 2) - D["recess_R"]
    d = smax(d, -dish, 1.2)

    # hairline groove around the flush button, following the dish surface
    zs = zc - np.sqrt(np.maximum(D["recess_R"] ** 2 - q, 0))
    groove = len2(len2(x, by) - D["btn_r"], z - zs) - 0.28
    d = np.maximum(d, -np.where(z > 0, groove, 1e3))

    # recessed light-guide band
    rr = body_radius(np.float32(D["ring_y"]))
    band = np.maximum(np.abs(y - D["ring_y"]) - D["ring_h"], (rr - D["ring_inset"]) - rxz)
    d = np.maximum(d, -band)

    # USB-C rubber cover outline on the top cap
    u = sd_round_rect(x, z, 4.6, 1.9, 1.5)
    usb = np.maximum(np.abs(u) - 0.2, np.abs(y - L) - 0.45)
    d = np.maximum(d, -usb)
    return d


def braid(x, y, z):
    a = np.arctan2(z, x)
    h1 = np.sin(14 * a + 1.9 * y)
    h2 = np.sin(14 * a - 1.9 * y)
    return 0.10 * (h1 * h1 + h2 * h2)


def sd_damper(x, y, z):
    rxz = len2(x, z)
    r = D["damper_r"] + D["bellows_amp"] * np.cos(2 * np.pi * y / D["bellows_pitch"])
    return 0.6 * sd_profile(rxz - braid(x, y, z), y, r, D["damper_bot"], D["damper_top"], 1.0)


def strap_frame(x, y, z):
    a = np.deg2rad(D["strap_az"])
    hx, hz = np.sin(a), np.cos(a)
    u = x * hx + z * hz
    w = x * hz - z * hx
    v = (y - D["strap_y"]) + 0.22 * np.maximum(u, 0)  # slight droop
    return u, v, w


def sd_strap(x, y, z):
    u, v, w = strap_frame(x, y, z)
    ring = len2(u - np.clip(u, 0, D["strap_len"]), v) - D["strap_loop_r"]
    return 0.85 * (sd_box2(ring, w, D["strap_t"] / 2, D["strap_w"] / 2 - 0.25) - 0.25)


def sd_metal(x, y, z):
    rxz = len2(x, z)
    collar = sd_profile(rxz, y, D["collar_r"], D["collar_bot"], D["collar_top"], 1.6)
    collar = np.maximum(collar, -(len2(rxz - D["collar_r"], y + 26.5) - 0.3))  # fine groove
    # slot where the webbing exits
    slot = np.maximum(sd_strap(x, y, z) - 0.45, (D["collar_r"] - 1.5) - rxz)
    collar = np.maximum(collar, -slot)
    neck = sd_profile(rxz, y, 3.2, -42.0, D["collar_bot"] + 1.0, 0.8)
    metal = smin(collar, neck, 1.0)

    # snap hook: stadium loop in the x-y plane with an opening on the +x side
    yy = y - D["hook_cy"]
    hh = D["hook_half"]
    dseg = len2(x, yy - np.clip(yy, -hh, hh))
    loop = len2(dseg - D["hook_R"], z) - D["hook_wire"]
    opening = sd_box2(x - D["hook_R"], yy - 2.0, 5.0, 9.0)
    loop = np.maximum(loop, -opening)
    gate = sd_capsule3(x, y, z, (D["hook_R"], D["hook_cy"] + 12.0, 0.0),
                       (D["hook_R"] - 0.3, D["hook_cy"] - 4.8, 0.0), D["gate_r"])
    return np.minimum(np.minimum(metal, loop), gate)


L_ = D["body_len"]
COMPONENTS = [  # (name, fn, aabb lo, aabb hi) in local coords
    ("body", sd_body, (-18, -1, -18), (18, L_ + 1, 18)),
    ("damper", sd_damper, (-9.6, -24.5, -9.6), (9.6, 2.5, 9.6)),
    ("metal", sd_metal, (-12.5, -84.5, -9.0), (12.5, -21.0, 9.0)),
    ("strap", sd_strap, (-12.5, -38.5, -6.5), (7.5, -22.5, 20.5)),
]
COMPONENTS = [(n, f, np.array(lo, F32), np.array(hi, F32)) for n, f, lo, hi in COMPONENTS]

# object placement: world = R @ (local - LC)
LC = np.array([0, 18.0, 0], F32)
R = rot_y(15.0) @ rot_z(12.0)


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
        m = d < BOX_MARGIN  # box distance is only a lower bound; soft shadows/AO need the exact SDF nearby
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
    up = np.array([0, 1, 0], F32)
    ux = nrm(np.cross(up, L))
    vy = np.cross(L, ux)
    f = d @ L
    fs = np.maximum(f, 1e-3)
    a = np.abs(d @ ux) / fs
    b = np.abs(d @ vy) / fs
    m = (f > 0) * smoothstep(sa + w, sa - w, a) * smoothstep(sb + w, sb - w, b)
    return inten * m


def env(d, rough):
    """Studio environment: two tall softboxes, a rim strip, a top scrim, white floor."""
    y = d[:, 1]
    base = np.where(y > 0, 0.30 + 0.30 * y, 0.50 - 0.15 * np.clip(-y * 3, 0, 1))
    w = 0.015 + 0.45 * rough * rough
    k = 1.0 / (1.0 + 6.0 * rough * rough)
    e = (panel(d, LKEY, 0.32, 0.60, 5.0, w)
         + panel(d, LRIM, 0.05, 0.90, 3.5, w)
         + panel(d, LFILL, 0.45, 0.45, 1.0, w)
         + 1.5 * smoothstep(0.75 - w, 0.95 + w, y))
    return (base + k * e)[:, None]


def aces(x):
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


# ----------------------------------------------------------------------------- marching
STEP = 0.8
BOX_MARGIN = 30.0
TEXTURE = 0.0  # fine concentric ripple on the body; only use with --ss 3 or more (aliases otherwise)


def march(ro, rd, t, tmax, steps=220, eps=0.02):
    t = t.astype(F32).copy()
    hit = np.zeros(len(t), bool)
    act = np.arange(len(t))
    for _ in range(steps):
        if act.size == 0:
            break
        p = ro[act] + rd[act] * t[act, None]
        d = scene(p)
        h = d < eps
        hit[act[h]] = True
        act, d = act[~h], d[~h]
        t[act] += d * STEP
        act = act[t[act] < tmax[act]]
    return hit, t


def soft_shadow(ro, rd, k, tmax=260.0, steps=72):
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
        d = scene(p + n * h)
        occ += w * np.clip((h - d) / h, 0, 1)
        wsum += w
    return np.clip(1 - 1.4 * occ / wsum, 0, 1)


# ----------------------------------------------------------------------------- render
def render_geometry(W, H, ss):
    t0 = time.time()
    Wr, Hr = W * ss, H * ss
    target = np.array([0, 2.0, 0], F32)
    az, el, dist, fov = np.deg2rad(30), np.deg2rad(14), 950.0, np.deg2rad(15.5)
    cam = target + dist * np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)], F32)
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

    # background: seamless white sweep with a gentle falloff
    nx, ny = sx.reshape(-1) / th, sy.reshape(-1) / th
    bg = 0.86 * (1 - 0.10 * ((nx * 0.8) ** 2 + (ny - 0.15) ** 2))

    # primary rays vs bounding sphere of the object
    C = (R @ (np.array([0, 18.5, 0], F32) - LC)).astype(F32)
    rad = 110.0
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

    # floor: same tone as the sweep, darkened only by the object's shadow + contact occlusion
    YG = -112.0
    fl = np.zeros(N, bool)
    fl[rd[:, 1] < 0] = True
    fl[obj] = False
    fidx = np.where(fl)[0]
    tg = (YG - cam[1]) / rd[fidx, 1]
    pg = cam + rd[fidx] * tg[:, None]
    near = np.hypot(pg[:, 0] - C[0], pg[:, 2] - C[2]) < 320
    fidx, pg = fidx[near], pg[near]
    vis = np.ones(len(pg), F32)
    # only rays toward the key light that can reach the object need a shadow march
    oc2 = pg - C
    b2 = oc2 @ LKEY
    reach = (b2 * b2 - (np.sum(oc2 * oc2, 1) - rad * rad)) > 0
    vis[reach] = soft_shadow(pg[reach] + [0, 0.3, 0], np.broadcast_to(LKEY, pg[reach].shape), k=4.0, tmax=400)
    occ = np.zeros(len(pg), F32)
    for h, w in ((4, 1.0), (12, 0.8), (25, 0.6), (45, 0.45), (70, 0.3)):
        d = scene(pg + [0, h, 0])
        occ += w * np.clip((h - d) / h, 0, 1)
    occ /= 3.15
    g["floor"] = (fidx, (1 - 0.50 * (1 - vis)) * (1 - 0.65 * occ))
    print(f"  floor shadow  ({time.time()-t0:.1f}s)")
    return g


MATS = {  # roughness, metallic
    "body": (0.58, False), "button": (0.38, False), "ring": (0.5, False), "usb": (0.75, False),
    "damper": (0.72, False), "metal": (0.24, True), "strap": (0.85, False),
}


def shade(g, color, ring_on=True):
    W, H, ss, N = g["W"], g["H"], g["ss"], g["N"]
    img = np.repeat(g["bg"][:, None], 3, 1).astype(F32)
    emis = np.zeros((N, 3), F32)

    fidx, fshade = g["floor"]
    img[fidx] *= fshade[:, None]

    p, n, ids, pl = g["p"], g["n"].copy(), g["ids"], g["pl"]
    x, y, z = pl[:, 0], pl[:, 1], pl[:, 2]
    rxz = len2(x, z)
    names = np.array([c[0] for c in COMPONENTS])[ids]
    mat = names.astype(object)
    isb = names == "body"
    # sub-materials on the body
    btn = isb & (len2(x, y - D["btn_y"]) < D["btn_r"] - 0.1) & (z > 8)
    ring = isb & (np.abs(y - D["ring_y"]) < D["ring_h"] + 0.05) & (rxz < body_radius(y) - 0.25)
    usb = isb & (y > D["body_len"] - 1.0) & (sd_round_rect(x, z, 4.6, 1.9, 1.5) < 0)
    mat[btn], mat[ring], mat[usb] = "button", "ring", "usb"

    base = np.array(COLORS[color], F32)
    alb = np.zeros((len(p), 3), F32)
    alb[:] = base
    alb[mat == "button"] = base * 1.08
    alb[mat == "usb"] = base * 0.75
    alb[mat == "ring"] = 0.55
    md = mat == "damper"
    alb[md] = 0.016 * (0.6 + 0.4 * braid(x[md], y[md], z[md]) / 0.2)[:, None]
    ms = mat == "strap"
    _, _, w = strap_frame(x[ms], y[ms], z[ms])
    alb[ms] = 0.022 * (0.85 + 0.15 * np.sin(2 * np.pi * w / 0.9))[:, None]
    mm = mat == "metal"
    alb[mm] = [0.91, 0.915, 0.92]

    # fine concentric texture on the soft-touch body: tiny normal ripple along the axis
    bodyonly = mat == "body"
    axis = R @ np.array([0, 1, 0], F32)
    n[bodyonly] += (TEXTURE * np.cos(2 * np.pi * y[bodyonly] / 0.9))[:, None] * axis
    n /= np.linalg.norm(n, axis=1, keepdims=True)

    rough = np.array([MATS[m][0] for m in mat], F32)
    metal = np.array([MATS[m][1] for m in mat])

    v = -g["rd"][g["obj"]]
    nv = np.clip(np.sum(n * v, 1), 1e-3, 1)
    r = 2 * nv[:, None] * n - v
    sh, ao = g["shadow"], g["ao"]

    key = np.clip((n @ LKEY + 0.15) / 1.15, 0, 1) * sh
    E = ((0.22 + 0.12 * n[:, 1]) * ao)[:, None] \
        + 2.3 * key[:, None] * KEY_COL \
        + 0.45 * np.clip(n @ LFILL, 0, 1)[:, None] * FILL_COL \
        + 0.9 * np.clip(n @ LRIM, 0, 1)[:, None]
    fres = (1 - nv) ** 5
    spec_occ = (ao * (0.55 + 0.45 * sh))[:, None]
    envr = env(r, rough)

    col = np.zeros_like(alb)
    di = ~metal
    F0 = 0.04
    Fd = F0 + (np.maximum(1 - rough[di], F0) - F0) * fres[di]
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
    rng = np.random.default_rng(1)
    out += (rng.random(out.shape) - 0.5) / 255.0
    return Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--color", default="charcoal", choices=list(COLORS) + ["all"])
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--ss", type=int, default=2, help="supersampling factor per axis")
    ap.add_argument("--ring", default="on", choices=["on", "off"])
    ap.add_argument("--preview", action="store_true", help="540x675, no supersampling")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "renders"))
    a = ap.parse_args()
    if a.preview:
        a.width, a.ss = 540, 1
    W = a.width
    H = int(round(W * 5 / 4))
    os.makedirs(a.out, exist_ok=True)

    t0 = time.time()
    print(f"geometry pass {W}x{H} @ {a.ss}x{a.ss} supersampling")
    g = render_geometry(W, H, a.ss)
    colors = list(COLORS) if a.color == "all" else [a.color]
    ims = []
    for c in colors:
        im = shade(g, c, a.ring == "on")
        path = os.path.join(a.out, f"ero_grip_{c}.png")
        im.save(path)
        ims.append(im)
        print(f"  saved {path}")
    if len(ims) > 1:
        sheet = Image.new("RGB", (W * 2, H * 2))
        for i, im in enumerate(ims):
            sheet.paste(im, ((i % 2) * W, (i // 2) * H))
        path = os.path.join(a.out, "ero_grip_colorways.png")
        sheet.save(path)
        print(f"  saved {path}")
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
