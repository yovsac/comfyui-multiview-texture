# Made by Yovan Vujacic
"""Corner fix for the four-view projection texture (MultiviewCornerPlan / MultiviewCornerApply).

Where a lamp wraps a corner of the car, the two views that meet there each paint part of it and the parts don't line up
(doubled or stepped lamp). Chain of texturer jobs 5-9, tested on 6 cars:
  plan
    1. lines (job 6): rear-glass lower edge and tail-lamp top/bottom of the back and side views, lifted to the mesh and paired
       per column at the rear corners. The same >= 1 cm step at BOTH rear corners -> smooth vertical warp of the back view
       (pinned at the rear-glass top and 1.5 lamp heights below the lamps, rear half only), then the photos are projected again.
    2. split lamps (job 5, job 9 detector): a lamp blob BOTH adjacent views draw at a corner, with >= 108 cm2 of it inside the
       hand-over band -> the corner is rendered from its best angle (crop of a 3072 px render) for an image edit; 3 crops 0-2
       render px apart (jitter).
    3. a rear lamp step the lines can't pair (job 9): detected and reported only.
  image edit (in the graph): Qwen-Image-Edit-2511 joins the two halves of the light in each crop.
  apply
    4. design guard (jobs 7, 9): a redraw is kept only if, on the MEAN of its 3 crops, >= 60 % of the head-on lamp's darkest and
       brightest tone classes survive; the crop with the best min(recall) is pasted (ring-only tone match, z-tested). A rejected
       FRONT corner whose side-view lamp piece is a separate, side-facing strip gets that piece cut out of the side photo
       instead (ownership); otherwise it is left as projected.
    5. the glass material follows the warped glass.
Nothing to fix = the inputs come back unchanged.

Bit-faithful to the tested offline chain (analysis/corner_v1/corner_touchup.py, job 9): incoming float images are stored as
8-bit the way SaveImage / the GLB writer store them (floor), and every 8-bit PNG / fp16 round trip of that chain is mirrored in
memory. Deterministic: the planner seeds torch with 0 inside a forked RNG (the global RNG state is restored).
Cameras: the native four-view rig (front az 0, left 90, back 180, right 270, level, fov 20); 1 rig unit ~ 430 cm (a car)."""
import copy
import json
import logging
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree

from . import core as C

NAMES = ["front", "left", "back", "right"]
AZ = {"front": 0.0, "left": 90.0, "back": 180.0, "right": 270.0}
CORNERS = {"front-left": (45, "front", "left"), "rear-left": (135, "left", "back"), "rear-right": (225, "back", "right"),
           "front-right": (315, "right", "front")}
FOV = 20.0                      # four-view rig
CM = 430.0                      # rig units -> cm (car length ~ 1 rig unit)
R2 = 2048                       # input view resolution
# ---- pre-registered parameters (jobs 5-7) ----
GROW, PAIR = 0.006, 0.007       # region grow (~2.6 cm), two views' blobs closer than ~3 cm = one lamp drawn twice
SIG = 60.0                      # ring tone-match blur (px at native crop)
WARP_MIN_CM, WARP_MIN_COLS, KNOT_MIN_COLS, PIN_BELOW = 1.0, 15, 10, 1.5
GUARD_REC, GUARD_TOL = 0.6, 4
JITTER = [(0, 0), (2, -1), (-1, 2)]   # job 9: render-px offsets of the 3 crops per guarded redraw
OWN_T_CM, OWN_COS, SEP_TOUCH = 2.0, 0.7, 0.2
# ---- job 9 (chosen on 6 cars) ----
SPLIT_MIN_CM2 = 108.0           # redraw gate: a paired lamp must put >= this much area (cm2, either view) into the hand-over band
STEP_PAIR_CM, STEP_MIN_DZ_CM, STEP_MIN_BAND_CM2 = 15.0, 2.0, 20.0   # rear step: side piece in the band, back piece <= 15 cm, |dz| >= 2 cm
MESH_PEAK_MIN, MESH_PEAK_RATIO = 1.3, 1.25   # mesh verdict (reported only)
REAR = (("rear-left", "left", 135.0, 1.0), ("rear-right", "right", 225.0, -1.0))
BAND_FR = dict(glass=(0.0, 0.6), lamp=(0.2, 0.6), glass_top=(-0.1, 0.6))
dev = C._dev()


def log(*a):
    logging.info("[MultiviewCorner] " + " ".join(str(x) for x in a))


# ------------------------------------------------------------------ 8-bit helpers (the offline chain's PNG round trips)
def to8(t):
    """float image 0..1 -> uint8 as SaveImage / the GLB writer store it (floor)"""
    a = t.detach().float().cpu().numpy() if torch.is_tensor(t) else np.asarray(t, np.float32)
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def q8(t):
    """float -> uint8 rounded (the offline chain's save_png)"""
    a = t.clamp(0, 1).cpu().numpy() if torch.is_tensor(t) else np.clip(t, 0, 1)
    return (a * 255 + 0.5).astype(np.uint8)


def f8(a):
    return a.astype(np.float32) / 255.0


def gray_f(a8, size=None):
    """uint8 mask -> float 0..1 tensor, LANCZOS-resized to size if needed (the offline chain's img(gray=True, size=...))"""
    im = Image.fromarray(a8)
    if size and im.size != (size, size):
        im = im.resize((size, size), Image.LANCZOS)
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)


def mask2k(a8):
    return np.asarray(Image.fromarray(a8).resize((R2, R2), Image.BILINEAR), np.float32) / 255 > 0.5


class Scene:
    """mesh + inputs (8-bit), = the offline chain's Capture"""

    def __init__(self, v, f, uv, base8, views8, glass8, A):
        self.v, self.f, self.uv = v.float().cpu(), f.long().cpu(), uv.float().cpu()
        self.base = torch.from_numpy(f8(base8))
        self.views8, self.glass8 = list(views8), list(glass8)
        self.views = [(torch.from_numpy(f8(x)), AZ[n], 0.0) for n, x in zip(NAMES, self.views8)]
        self.glass_t = [torch.from_numpy(f8(g)) for g in self.glass8]
        self.A = int(A)
        self.P0 = C.yup_to_rig(self.v.to(dev)); self.Fd = self.f.to(dev)


# ------------------------------------------------------------------ re-projection (= node 700 on defaults)
def reproject(sc, views, glass):
    """-> (atlas uint8 rounded, per-view weights fp16 (4,A,A), glass map fp16 (A,A))"""
    tex, dbg = C.texture_mesh(sc.v, sc.f, sc.uv, sc.base, views, FOV, {"mode": "blend", "keep_views": True}, log=lambda *a: None, glass_masks=glass)
    A = tex.shape[0]; T = dbg["tex"]
    W = np.zeros((4, A * A), np.float16)
    for k in range(4):
        W[k, T.cpu().numpy()] = dbg["w_t"][k].cpu().numpy()
    return q8(tex), W.reshape(4, A, A), dbg["glass"].numpy().astype(np.float16)


def texel_area(sc, A):
    """cm2 of mesh surface per occupied texel (same order as C._texel_geometry's t_idx)"""
    Fd, uvd = sc.Fd, sc.uv.to(dev)
    fid, _ = C.uv_raster(uvd, Fd, A, C.DEFAULTS["tol"]); f = fid[torch.nonzero(fid >= 0).squeeze(1)]
    a3 = 0.5 * torch.linalg.cross(sc.P0[Fd[:, 1]] - sc.P0[Fd[:, 0]], sc.P0[Fd[:, 2]] - sc.P0[Fd[:, 0]]).norm(dim=1) * CM ** 2
    e1, e2 = uvd[Fd[:, 1]] - uvd[Fd[:, 0]], uvd[Fd[:, 2]] - uvd[Fd[:, 0]]
    auv = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).abs() * A * A
    return (a3 / auv.clamp(min=1e-6))[f].clamp(max=1.0)


# ------------------------------------------------------------------ split-lamp plan (job 5 round 2 + job 9 detector)
def plan_corners(sc, tex8, W16, lamp8, corners, PL):
    """-> plan [dict], info {corner: ...}, crops {corner: uint8 (S,S,3)}, masks {corner: uint8 (S,S)}"""
    v, f, uv = sc.v, sc.f, sc.uv
    torch.manual_seed(0)
    tex = torch.from_numpy(f8(tex8))
    A = tex.shape[0]
    t_idx, P, N, P0, Fd = C._texel_geometry(v, f, uv, A, dev)
    W = torch.from_numpy(W16.reshape(4, -1).astype(np.float32)).to(dev)[:, t_idx]
    lamp = torch.zeros(len(t_idx), device=dev)
    blob = {}
    for k, n in enumerate(NAMES):
        m = gray_f(lamp8[n], 2048).numpy() > 0.5
        lab2, nb = ndimage.label(m)
        vis, cos, xy = C._visible(C.Cam(AZ[n], 0.0, FOV, 2048), P, N, P0, Fd, 2048)
        ok = (vis > 0) & (cos > 0.2)
        px = xy.round().long().clamp(0, 2047)
        b = torch.from_numpy(lab2.astype(np.int64)).to(dev)[px[:, 1], px[:, 0]] * ok.long()
        blob[n] = b; lamp = torch.maximum(lamp, (b > 0).float())
    plan, info, crops, masks = [], {}, {}, {}
    tarea = None
    for corner in corners:
        az_c, va, vb = CORNERS[corner]
        ia, ib = NAMES.index(va), NAMES.index(vb)
        a = np.radians(az_c)
        quad = (torch.sign(P[:, 0]) == float(np.sign(np.sin(a)))) & (torch.sign(P[:, 1]) == float(np.sign(-np.cos(a))))
        hand = (W[ia] > 0.02) & (W[ib] > 0.02) & quad
        pts = {}
        for vv in (va, vb):
            for j in torch.unique(blob[vv][quad]).tolist():
                if j == 0: continue
                idx = torch.nonzero((blob[vv] == j) & quad).squeeze(1)
                if len(idx) >= 200: pts[(vv, j)] = idx
        sub = lambda idx: P[idx[torch.randperm(len(idx), device=dev)[:3000]]]
        near_hand = lambda idx: int(hand[idx].sum())
        A_ = [k for k in pts if k[0] == va]; B_ = [k for k in pts if k[0] == vb]
        S_ = {k: sub(pts[k]) for k in pts}

        def link(ka, kb):
            d = torch.cdist(S_[ka], S_[kb])
            return (round(float(d.min()), 3), -float((d.min(1).values < PAIR).float().mean() + (d.min(0).values < PAIR).float().mean()))
        Lk = {(ka, kb): link(ka, kb) for ka in A_ for kb in B_}
        keep = set()
        for ka in A_:
            for kb in B_:
                l = Lk[(ka, kb)]
                mutual = l == min(Lk[(ka, x)] for x in B_) and l == min(Lk[(x, kb)] for x in A_)
                if mutual and l[0] < PAIR and min(near_hand(pts[ka]), near_hand(pts[kb])) >= 20:
                    keep |= {ka, kb}
        info[corner] = dict(blobs={va: len(A_), vb: len(B_)}, pair=bool(keep))
        if not keep:
            continue
        # job 9 split detector (texture space, no RNG): cm2 of each view's PAIRED lamp blobs inside the hand-over band; the max of the two
        tarea = texel_area(sc, A) if tarea is None else tarea
        band_cm2 = {}
        for vv in (va, vb):
            m = torch.zeros(len(t_idx), dtype=torch.bool, device=dev)
            for k in keep:
                if k[0] == vv:
                    m[pts[k]] = True
            band_cm2[vv] = round(float((tarea * (m & hand).float()).sum()), 2)
        info[corner].update(band_cm2=band_cm2, split_cm2=max(band_cm2.values()), split=max(band_cm2.values()) >= SPLIT_MIN_CM2)
        rs = torch.unique(torch.cat([pts[k] for k in keep]))
        frac = {k: near_hand(pts[k]) / len(pts[k]) for k in keep}
        minor = [max(pr, key=lambda k: frac[k]) for pr in [(ka, kb) for ka in A_ for kb in B_ if ka in keep and kb in keep
                 and Lk[(ka, kb)] == min(Lk[(ka, x)] for x in B_) and Lk[(ka, kb)] == min(Lk[(x, kb)] for x in A_)]]
        mi = torch.unique(torch.cat([pts[k] for k in minor]))
        head = "front" if corner.startswith("front") else "back"
        hb = [pts[k] for k in keep if k[0] == head] or [pts[k] for k in keep]
        anchor = P[torch.cat(hb)].median(0).values.cpu().tolist()
        rp = P[rs]
        cand = torch.nonzero(quad & (((P - rp.mean(0)).norm(dim=1)) < (rp - rp.mean(0)).norm(dim=1).max() + GROW)).squeeze(1)
        grown = torch.zeros(len(t_idx), dtype=torch.bool, device=dev); grown[rs] = True
        sub = rp[torch.randperm(len(rp), device=dev)[:4000]]
        for ch in torch.split(cand, 20000):
            grown[ch] |= torch.cdist(P[ch], sub).min(1).values < GROW
        rs = torch.nonzero(grown).squeeze(1)
        best = None
        for el in (5.0, 10.0, 15.0, 20.0, 25.0):
            for az in range(az_c - 30, az_c + 31, 5):
                vis, cos, _ = C._visible(C.Cam(float(az), el, FOV, 512), P[rs], N[rs], P0, Fd, 512)
                sc_ = float((vis * cos.clamp(min=0)).sum())
                if best is None or sc_ > best[0]:
                    best = (sc_, float(az % 360), el)
        _, az, el = best
        amask = torch.zeros(A * A, device=dev); amask[t_idx[rs]] = 1.0
        R = C.TOUCHUP_RES
        mimg, _ = C.render_texture(v, f, uv, amask.view(A, A, 1), az, el, FOV, R)
        m = (mimg[..., 0] > 0.5).cpu().numpy(); ys, xs = np.nonzero(m)
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        side = int(np.clip(max(xs.max() - xs.min(), ys.max() - ys.min()) * 1.8, 384, R))
        x0 = int(np.clip(cx - side / 2, 0, R - side)); y0 = int(np.clip(cy - side / 2, 0, R - side))
        rimg, _ = C.render_texture(v, f, uv, tex, az, el, FOV, R)
        crop = F.interpolate(rimg[y0:y0 + side, x0:x0 + side].permute(2, 0, 1)[None], size=(side, side), mode="bilinear", align_corners=False, antialias=True)
        mc = F.interpolate(torch.tensor(m[y0:y0 + side, x0:x0 + side], dtype=torch.float32)[None, None], size=(side, side), mode="area")[0, 0].numpy() > 0.3
        mc = ndimage.binary_fill_holes(ndimage.binary_closing(mc, iterations=8))
        mc = ndimage.binary_dilation(mc, iterations=max(6, side // 170))
        mc = ndimage.gaussian_filter(mc.astype(np.float32), max(3, side / 340))
        crops[corner] = q8(crop[0].permute(1, 2, 0))
        masks[corner] = (mc * 255).astype(np.uint8)
        _ = P[mi][torch.randperm(len(mi), device=dev)[:4000]]        # kept for the RNG call order (job 5's misplaced-piece mask)
        plan.append(dict(corner=corner, az=az, el=el, fov=FOV, x0=x0, y0=y0, side=side, texels=int(len(rs)),
                         blobs=sorted([list(k) for k in keep]), anchor=anchor))
        log(PL, corner, "split lamp ->", {k: plan[-1][k] for k in ("az", "el", "side", "texels", "blobs")})
    return plan, info, crops, masks


def jitter_crop(sc, tex8, r, mask8, dx, dy):
    """the same corner crop moved by (dx, dy) render px -> (crop uint8, mask uint8, moved plan dict)"""
    R, S = C.TOUCHUP_RES, r["side"]
    x0, y0 = int(np.clip(r["x0"] + dx, 0, R - S)), int(np.clip(r["y0"] + dy, 0, R - S))
    rimg, _ = C.render_texture(sc.v, sc.f, sc.uv, torch.from_numpy(f8(tex8)), r["az"], r["el"], FOV, R)
    crop = F.interpolate(rimg[y0:y0 + S, x0:x0 + S].permute(2, 0, 1)[None], size=(S, S), mode="bilinear", align_corners=False, antialias=True)
    mj = np.zeros_like(mask8)
    sx, sy = x0 - r["x0"], y0 - r["y0"]
    mj[max(0, -sy):S - max(0, sy), max(0, -sx):S - max(0, sx)] = mask8[max(0, sy):S - max(0, -sy), max(0, sx):S - max(0, -sx)]
    return q8(crop[0].permute(1, 2, 0)), mj, dict(r, x0=x0, y0=y0, jitter=[sx, sy])


# ------------------------------------------------------------------ line alignment (job 6)
def pmap(sc, az, el=0.0, res=R2):
    cam = C.Cam(az, el, FOV, res); xy, z = cam.project(sc.P0)
    fid, bary, _ = C.screen_raster(xy, z, sc.Fd, res, res)
    P = torch.zeros((res * res, 3), device=dev); hit = fid >= 0
    P[hit] = (sc.P0[sc.Fd[fid[hit]]] * bary[hit][:, :, None]).sum(1)
    return P.view(res, res, 3).cpu().numpy(), hit.view(res, res).cpu().numpy()


def bands(sc):
    """image-row bands (2K) where the lines are searched, relative to the back view's mesh silhouette"""
    _, hit = pmap(sc, 180.0)
    cols = np.nonzero(hit.any(0))[0]; c0, c1 = np.percentile(cols, 25), np.percentile(cols, 75); cols = cols[(cols >= c0) & (cols <= c1)]
    top = float(np.median(hit[:, cols].argmax(0))); h = float(np.median(R2 - 1 - hit[::-1, cols].argmax(0))) - top
    return {k: (int(round(top + a * h)), int(round(top + b * h))) for k, (a, b) in BAND_FR.items()}


def measure_lines(sc, views, glass8, lamp8, bd):
    curves, sil = {}, {}
    g0, g1 = bd["glass"]; l0, l1 = bd["lamp"]
    for k, n in enumerate(NAMES):
        if n == "front":
            continue
        im = views[k][0].numpy(); az = views[k][1]
        P, hit = pmap(sc, az)
        cut = ndimage.binary_fill_holes(ndimage.binary_closing(im.max(2) > 0.01, iterations=3))
        sil[n] = round(float((hit & cut).sum() / (hit | cut).sum()), 4)
        g = mask2k(glass8[k]); lm = mask2k(lamp8[n])
        cv = {"glass_bottom": [], "lamp_top": [], "lamp_bottom": []}
        for x in range(R2):
            ys = np.nonzero(g[g0:g1, x])[0]
            if len(ys) >= 6:
                y = g0 + ys.max()
                if hit[y, x] and hit[min(y + 3, R2 - 1), x]:
                    cv["glass_bottom"].append([int(x), int(y)] + P[y, x].tolist())
            ys = np.nonzero(lm[l0:l1, x])[0]
            if len(ys) >= 4:
                y = l0 + ys.min()
                if hit[y, x]:
                    cv["lamp_top"].append([int(x), int(y)] + P[y, x].tolist())
                y = l0 + ys.max()
                if hit[y, x]:
                    cv["lamp_bottom"].append([int(x), int(y)] + P[y, x].tolist())
        curves[n] = cv
    return curves, sil


def dz_corner(cur, side, feat, az, sx):
    """column-paired (level corner camera, 4-px columns, closest z pair) dz = z_side - z_back (rig units) + back points used"""
    cam = C.Cam(az, 0.0, FOV, R2)
    s = np.asarray(cur[side][feat], np.float64).reshape(-1, 5); b = np.asarray(cur["back"][feat], np.float64).reshape(-1, 5)
    s = s[s[:, 2] * sx > 0]; b = b[b[:, 2] * sx > 0]
    if not len(s) or not len(b):
        return None, None, 0
    cs = cam.project(torch.from_numpy(s[:, 2:5]).float().to(dev))[0][:, 0].cpu().numpy() // 4
    cb = cam.project(torch.from_numpy(b[:, 2:5]).float().to(dev))[0][:, 0].cpu().numpy() // 4
    vals, used = [], []
    for c in np.intersect1d(cs, cb):
        bi = np.nonzero(cb == c)[0]; d = s[cs == c, 4][:, None] - b[bi, 4][None, :]
        k = np.abs(d).argmin(); vals.append(d.flat[k]); used.append(bi[k % len(bi)])
    if not vals:
        return None, None, 0
    return float(np.median(vals)), b[np.asarray(used)], len(vals)


def line_steps(cur):
    out = {}
    for corner, side, az, sx in REAR:
        out[corner] = {}
        for feat in ("glass_bottom", "lamp_top", "lamp_bottom"):
            d, _, n = dz_corner(cur, side, feat, az, sx)
            out[corner][feat] = dict(dz_cm=None if d is None else round(d * CM, 2), cols=n)
    return out


def warp_rule(steps):
    votes = []
    for feat in ("glass_bottom", "lamp_top"):
        ds = [steps[c][feat] for c, *_ in REAR]
        if all(x["dz_cm"] is not None and x["cols"] >= WARP_MIN_COLS and abs(x["dz_cm"]) >= WARP_MIN_CM for x in ds) \
                and np.sign(ds[0]["dz_cm"]) == np.sign(ds[1]["dz_cm"]):
            votes.append(feat)
    return votes


def fit_warp(sc, cur, lamp_back8, bd):
    """job 6 final warp. -> (ok, info, {"back": uint8 2K view, "glass": uint8 glass mask (its own size), "lamp": uint8 2K}|None)"""
    camB = C.Cam(180.0, 0.0, FOV, R2)
    knots = {}
    for feat in ("glass_bottom", "lamp_top", "lamp_bottom"):
        per = []
        for corner, side, az, sx in REAR:
            d, bu, n = dz_corner(cur, side, feat, az, sx)
            if d is None or n < KNOT_MIN_COLS:
                per = None; break
            depth = float(np.median((bu[:, 2:5] - camB.pos.cpu().numpy()) @ camB.fwd.cpu().numpy()))
            per.append(dict(corner=side, dz_cm=round(d * CM, 2), n=n, row=float(np.median(bu[:, 1])), depth=round(depth, 3),
                            shift_px=round(d * camB.f / depth, 1)))
        if per:
            knots[feat] = dict(row=float(np.mean([p["row"] for p in per])), shift=float(np.mean([p["shift_px"] for p in per])), per=per)
    if "lamp_top" not in knots or "lamp_bottom" not in knots:
        return False, dict(why="no tail-lamp top/bottom line at both rear corners (pin below the lamps impossible)", knots=knots), None
    PB, hit = pmap(sc, 180.0)
    im = sc.views[2][0].numpy()
    cut = ndimage.binary_fill_holes(ndimage.binary_closing(im.max(2) > 0.01, iterations=3))
    cols = np.nonzero(hit.any(0) & cut.any(0))[0]; c0, c1 = np.percentile(cols, 25), np.percentile(cols, 75)
    cols = cols[(cols >= c0) & (cols <= c1)]
    top_i = cut[:, cols].argmax(0); bot_i = R2 - 1 - cut[::-1, cols].argmax(0)
    knots["sil_top"] = dict(row=float(np.median(top_i)), shift=0.0)          # the outline keeps its registration
    knots["sil_bottom"] = dict(row=float(np.median(bot_i)), shift=0.0)
    gmk = np.asarray(Image.fromarray(sc.glass8[2]).resize((R2, R2), Image.BILINEAR)) > 127
    t0_, t1_ = bd["glass_top"]; tops = []
    for x in cols:
        ys = np.nonzero(gmk[t0_:t1_, x])[0] + t0_
        if len(ys) < 6:
            continue
        run_top = ys[-1]
        while run_top - 1 in set(ys[ys < run_top]) or (run_top - 1 >= t0_ and gmk[run_top - 1, x]):
            run_top -= 1
        tops.append(run_top)
    if not tops:
        return False, dict(why="no rear-glass top found (the spoiler pin is mandatory)", knots=knots), None
    knots["glass_top"] = dict(row=float(np.median(tops)), shift=0.0)
    lt, lb = knots["lamp_top"]["row"], knots["lamp_bottom"]["row"]
    knots["pin_below"] = dict(row=float(lb + PIN_BELOW * (lb - lt)), shift=0.0)
    K = sorted((k["row"], k["shift"]) for name, k in knots.items() if name != "sil_bottom")
    ys_k = np.array([k[0] for k in K]); ds_k = np.array([k[1] for k in K])
    if len(np.unique(ys_k)) < len(ys_k):
        return False, dict(why="two knots on the same row", knots=knots), None
    pc = PchipInterpolator(ys_k, ds_k, extrapolate=False)
    yy = np.arange(R2, dtype=np.float64)
    delta = np.where(yy < ys_k[0], ds_k[0], np.where(yy > ys_k[-1], ds_k[-1], np.nan_to_num(pc(yy))))
    fwd = yy - delta
    slope = np.gradient(fwd)
    info = dict(knots=knots, knot_rows=ys_k.tolist(), knot_shift=ds_k.tolist(), max_shift=float(np.abs(delta).max()), min_slope=float(slope.min()))
    if slope.min() <= 0.2:
        return False, dict(info, why=f"warp would fold (min slope {slope.min():.2f})"), None
    src_of_dst = np.interp(yy, fwd, yy)
    dinv = src_of_dst - yy
    Aw = np.clip((PB[..., 1] + 0.05) / 0.10, 0, 1).astype(np.float32)
    idx = ndimage.distance_transform_edt(~hit, return_distances=False, return_indices=True)
    Aw = ndimage.gaussian_filter(Aw[idx[0], idx[1]], 4)
    mapy = (yy[:, None] + Aw * dinv[:, None]).astype(np.float32)
    mapx = np.tile(np.arange(R2, dtype=np.float32), (R2, 1))
    warp = lambda a: cv2.remap(a, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    back = warp((np.clip(im, 0, 1) * 255 + 0.5).astype(np.uint8))
    gm = np.asarray(Image.fromarray(sc.glass8[2]).resize((R2, R2), Image.BILINEAR))
    gh, gw = sc.glass8[2].shape[:2]
    glass = np.asarray(Image.fromarray(warp(gm)).resize((gw, gh), Image.BILINEAR))
    lm = np.asarray(Image.fromarray(lamp_back8).resize((R2, R2), Image.BILINEAR))
    return True, info, {"back": back, "glass": glass, "lamp": warp(lm)}


# ------------------------------------------------------------------ job 9 rule 2: the large rear lamp step (detect + report)
def _edge_map(sc, az, R=R2):
    cam = C.Cam(az, 0.0, FOV, R); xy, z = cam.project(sc.P0); fid, _, _ = C.screen_raster(xy, z, sc.Fd, R, R); hit = fid >= 0
    P0, Fd = sc.P0, sc.Fd
    fn = F.normalize(torch.linalg.cross(P0[Fd[:, 1]] - P0[Fd[:, 0]], P0[Fd[:, 2]] - P0[Fd[:, 0]]), dim=1)
    n = torch.zeros((R * R, 3), device=dev); n[hit] = fn[fid[hit]]; n = n.view(R, R, 3).cpu().numpy(); g = 0
    for c in range(3):
        a = ndimage.gaussian_filter(n[..., c], 1.0); g = g + np.hypot(ndimage.sobel(a, 0), ndimage.sobel(a, 1))
    return ndimage.gaussian_filter(g, 1.5), cam


def _outline_peak(E, m):
    ys, xs = [], []
    for x in np.nonzero(m.any(0))[0]:
        col = np.nonzero(m[:, x])[0]; ys += [col.min(), col.max()]; xs += [x, x]
    if len(ys) < 20:
        return None
    ys, xs = np.asarray(ys), np.asarray(xs)
    e = lambda s: float(E[np.clip(ys - s, 0, E.shape[0] - 1), xs].mean())
    return round(e(0) / max(np.mean([e(-12), e(-9), e(9), e(12)]), 1e-6), 3)


def rear_steps(sc, lamp8, W16, infoA):
    A = sc.A
    t_idx, P, N, P0, Fd = C._texel_geometry(sc.v, sc.f, sc.uv, A, dev)
    W = torch.from_numpy(W16.reshape(4, -1).astype(np.float32)).to(dev)[:, t_idx]
    tarea = texel_area(sc, A)
    blob = {}
    for n in ("left", "back", "right"):
        lab2, _ = ndimage.label(gray_f(lamp8[n], 2048).numpy() > 0.5)
        vis, cos, xy = C._visible(C.Cam(AZ[n], 0.0, FOV, 2048), P, N, P0, Fd, 2048)
        px = xy.round().long().clamp(0, 2047)
        blob[n] = torch.from_numpy(lab2.astype(np.int64)).to(dev)[px[:, 1], px[:, 0]] * ((vis > 0) & (cos > 0.2)).long()
    out = {}
    for corner, side, az, sx in REAR:
        if infoA.get(corner, {}).get("pair"):
            out[corner] = dict(step=False, why="planner pairs a split lamp here (redraw path)"); continue
        ia, ib = NAMES.index(side), NAMES.index("back"); a = np.radians(az)
        quad = (torch.sign(P[:, 0]) == float(np.sign(np.sin(a)))) & (torch.sign(P[:, 1]) == float(np.sign(-np.cos(a))))
        band = (W[ia] > 0.02) & (W[ib] > 0.02) & quad
        cand = {}
        for vv in (side, "back"):
            for j in torch.unique(blob[vv][quad]).tolist():
                if j == 0: continue
                m = (blob[vv] == j) & quad
                if int(m.sum()) >= 200: cand[(vv, j)] = m
        ss = [k for k in cand if k[0] == side]
        if not ss:
            out[corner] = dict(step=False, why="no side-view lamp piece at this corner"); continue
        ks = max(ss, key=lambda k: int((cand[k] & band).sum()))
        s_band = round(float((tarea * (cand[ks] & band).float()).sum()), 2)
        sp = P[cand[ks]]
        best = None
        for k in cand:
            if k[0] != "back": continue
            d = float(torch.cdist(sp[::7], P[cand[k]][::7]).min()) * CM
            if d <= STEP_PAIR_CM and (best is None or int(cand[k].sum()) > int(cand[best[0]].sum())):
                best = (k, d)
        if best is None or s_band < STEP_MIN_BAND_CM2:
            out[corner] = dict(step=False, side_band_cm2=s_band, why="no back-view lamp within 15 cm" if best is None else "side piece barely in the band"); continue
        kb, dmin = best
        zs, zb = sp[:, 2], P[cand[kb]][:, 2]
        q = lambda z, p: float(torch.quantile(z, p))
        dz = {k_: round((q(zs, p) - q(zb, p)) * CM, 2) for k_, p in (("top", 0.95), ("med", 0.5), ("bot", 0.05))}
        rec = dict(side_piece=list(ks), back_piece=list(kb), side_band_cm2=s_band, back_band_texels=int((cand[kb] & band).sum()),
                   dmin_cm=round(dmin, 2), dz_cm=dz, step=bool(abs(dz["med"]) >= STEP_MIN_DZ_CM))
        if rec["step"]:
            E, cam = _edge_map(sc, az)
            pk = {}
            for nm, key in (("side", ks), ("back", kb)):
                am = torch.zeros(A * A, device=dev); am[t_idx[cand[key]]] = 1
                mm, _ = C.render_texture(sc.v, sc.f, sc.uv, am.view(A, A, 1), az, 0.0, FOV, R2)
                pk[nm] = _outline_peak(E, mm[..., 0].cpu().numpy() > 0.5)
            ps, pb = pk["side"] or 0.0, pk["back"] or 0.0
            rec.update(mesh_peak=pk, mesh_backs_side=bool(ps >= MESH_PEAK_MIN and ps >= MESH_PEAK_RATIO * pb))
        out[corner] = rec
        log("rear step", corner, json.dumps(rec))
    return out


# ------------------------------------------------------------------ tone match + move gate (job 5)
def lowpass(x, w, s):
    k = int(s * 4) | 1
    return cv2.GaussianBlur(x * w[..., None], (k, k), s) / (cv2.GaussianBlur(w, (k, k), s)[..., None] + 1e-4)


def tone_match(crop8, mask8, edit8, S):
    """-> (tone-matched edit float (S,S,3), the same rounded to uint8 (what the guard reads), mask float (S,S), move gate)"""
    crop = f8(crop8)
    ed = np.asarray(Image.fromarray(edit8).resize((S, S), Image.LANCZOS), np.float32) / 255
    mask = f8(mask8)
    hp = lambda a: (lambda g: g - cv2.GaussianBlur(g, (0, 0), 6))(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY))
    win = (mask < 0.05).astype(np.float32) * cv2.createHanningWindow((S, S), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(hp(crop) * win, hp(ed) * win)
    car_c = ndimage.binary_erosion(crop.max(2) > 0.03, iterations=3); car_e = ndimage.binary_erosion(ed.max(2) > 0.05, iterations=3)
    w = (car_c & car_e & (mask < 0.05)).astype(np.float32)
    ratio = np.clip((lowpass(crop, w, SIG) + 0.02) / (lowpass(ed, w, SIG) + 0.02), 0.6, 1.6)
    edt = np.where(car_e[..., None], np.clip(ed * ratio, 0, 1), ed)
    return edt, (edt * 255 + 0.5).astype(np.uint8), mask, dict(shift_px=[round(dx, 2), round(dy, 2)], ok=bool(abs(dx) <= 4 and abs(dy) <= 4))


def component(mask, r, keep, anchor):
    """keep=None: whole mask; 'tail': the connected component holding the tail-lamp anchor; 'other': all the other components"""
    if keep is None:
        return mask
    lab, n = ndimage.label(mask > 0.02)
    cam = C.Cam(r["az"], r["el"], r["fov"], C.TOUCHUP_RES); c, _ = cam.project(torch.tensor(anchor, device=dev)[None].float())
    x, y = int(c[0, 0]) - r["x0"], int(c[0, 1]) - r["y0"]
    S = mask.shape[0]
    k = lab[min(max(y, 0), S - 1), min(max(x, 0), S - 1)]
    if k == 0:                                   # anchor off the mask: the nearest component
        d, (iy, ix) = ndimage.distance_transform_edt(lab == 0, return_indices=True)
        k = lab[iy[min(max(y, 0), S - 1), min(max(x, 0), S - 1)], ix[min(max(y, 0), S - 1), min(max(x, 0), S - 1)]]
    return mask * ((lab == k) if keep == "tail" else ((lab != k) & (lab > 0)))


# ------------------------------------------------------------------ design guard (job 7)
def kmeans3(x):
    c = np.percentile(x, [10, 50, 90]).astype(np.float64)
    for _ in range(30):
        lab = np.abs(x[:, None] - c[None]).argmin(1)
        c = np.array([x[lab == k].mean() if (lab == k).any() else c[k] for k in range(3)])
    c.sort(); return (c[0] + c[1]) / 2, (c[1] + c[2]) / 2


def guard(sc, corner, r, head, other, lamp_head8, W16, crop8, tm8):
    A = sc.A
    t_idx, P, N, P0, Fd = C._texel_geometry(sc.v, sc.f, sc.uv, A, dev)
    lum = lambda a: a @ np.array([0.299, 0.587, 0.114], np.float32)
    S = r["side"]
    W = W16.reshape(4, -1).astype(np.float32)
    ih, io_ = NAMES.index(head), NAMES.index(other)
    Wt = torch.from_numpy(W[[ih, io_]][:, t_idx.cpu().numpy()]).to(dev)
    share = Wt[1] / (Wt[0] + Wt[1]).clamp(min=1e-6)
    m2 = ndimage.binary_fill_holes(np.asarray(Image.fromarray(lamp_head8).resize((2048, 2048)), np.float32) > 127)
    vis, cos, xy = C._visible(C.Cam(AZ[head], 0.0, FOV, 2048), P, N, P0, Fd, 2048)
    px = xy.round().long().clamp(0, 2047)
    inm = torch.from_numpy(m2).to(dev)[px[:, 1], px[:, 0]]
    a = np.radians(CORNERS[corner][0])
    quad = (torch.sign(P[:, 0]) == float(np.sign(np.sin(a)))) & (torch.sign(P[:, 1]) == float(np.sign(-np.cos(a))))
    face = (vis > 0) & (cos > 0.7) & inm & (share < 0.1) & quad
    am = torch.zeros(A * A, device=dev); am[t_idx[face]] = 1
    mimg, _ = C.render_texture(sc.v, sc.f, sc.uv, am.view(A, A, 1), r["az"], r["el"], FOV, C.TOUCHUP_RES)
    M = (mimg[r["y0"]:r["y0"] + S, r["x0"]:r["x0"] + S, 0] > 0.5).cpu().numpy()
    M = ndimage.binary_fill_holes(ndimage.binary_closing(M, iterations=8, border_value=0))
    M = ndimage.binary_erosion(M, iterations=2)
    crop, ed = f8(crop8), f8(tm8)
    lc, le = lum(crop), lum(ed)
    g = dict(face_px=int(M.sum()))
    if M.sum() >= 1000:
        t1, t2 = kmeans3(lc[M])
        dk_c, dk_e, br_c, br_e = (lc < t1) & M, (le < t1), (lc > t2) & M, (le > t2)
        rec = lambda c_, e_: float((c_ & ndimage.binary_dilation(e_, iterations=GUARD_TOL)).sum() / max(c_.sum(), 1))
        g.update(dark_px=int(dk_c.sum()), bright_px=int(br_c.sum()), recall_dark=round(rec(dk_c, dk_e), 3), recall_bright=round(rec(br_c, br_e), 3))
        bad = [k for k in ("dark", "bright") if g[f"{k}_px"] >= 200 and g[f"recall_{k}"] < GUARD_REC]
        g["verdict"] = "REJECT (" + ",".join(bad) + ")" if bad else "accept"
    else:
        g["verdict"] = "accept (no head-on lamp face in the crop)"
    return g


# ------------------------------------------------------------------ ownership fallback (job 7)
def separate_strip(sc, lamp8, W16, corners):
    A = sc.A
    t_idx, P, N, P0, Fd = C._texel_geometry(sc.v, sc.f, sc.uv, A, dev)
    W = W16.reshape(4, -1)
    ymin = float(P0[:, 1].min()); out = {}
    mask = lambda a8: torch.from_numpy(ndimage.binary_fill_holes(np.asarray(Image.fromarray(a8).resize((2048, 2048)), np.float32) > 127)).to(dev)

    def blob_mask(view, sx):
        m = mask(lamp8[view]).cpu().numpy(); lab, n = ndimage.label(m)
        ids = [j for j in range(1, n + 1) if (lab == j).sum() > 300]
        if not ids:
            return torch.zeros((2048, 2048), dtype=torch.bool, device=dev)
        if view == "front":
            keep = [j for j in ids if (np.nonzero(lab == j)[1].mean() - 1024) * sx > 0]
        else:
            cam = C.Cam(AZ[view], 0.0, FOV, 2048); fil = float(cam.right @ torch.tensor([0.0, 1.0, 0.0], device=dev)) > 0
            cx = {j: np.nonzero(lab == j)[1].mean() for j in ids}; keep = [min(ids, key=lambda j: cx[j]) if fil else max(ids, key=lambda j: cx[j])]
        return torch.from_numpy(np.isin(lab, keep)).to(dev)
    for corner in corners:
        side = "left" if corner == "front-left" else "right"; sx = 1.0 if side == "left" else -1.0
        sel = torch.nonzero((torch.sign(P[:, 0]) == sx) & (P[:, 1] < ymin + 0.15)).squeeze(1)
        Ps, Ns = P[sel], N[sel]; ti = t_idx[sel].cpu().numpy()
        wf = torch.from_numpy(W[0, ti].astype(np.float32)).to(dev); ws = torch.from_numpy(W[NAMES.index(side), ti].astype(np.float32)).to(dev)
        res = {}
        for view, dom in (("front", wf > ws), (side, ws > wf)):
            vis, cos, xy = C._visible(C.Cam(AZ[view], 0.0, FOV, 2048), Ps, Ns, P0, Fd, 2048)
            px = xy.round().long().clamp(0, 2047); m = blob_mask(view, sx)
            res[view] = dom & (vis > 0) & m[px[:, 1], px[:, 0]] & (wf + ws > 0.02)
        FL, SL = Ps[res["front"]], Ps[res[side]]
        if len(SL) and len(FL):
            d = torch.cat([torch.cdist(ch, FL[::2]).min(1).values for ch in torch.split(SL, 4000)]) * CM
            touch = float((d < 1.0).float().mean()); gap_med = float(d.median())
        else:
            touch, gap_med = 1.0, 0.0
        out[corner] = dict(visible_front_lamp=int(len(FL)), visible_side_lamp=int(len(SL)), touching_share=round(touch, 3),
                           gap_cm_median=round(gap_med, 2), separate=bool(len(SL) >= 300 and touch < SEP_TOUCH))
        log("separate strip", corner, json.dumps(out[corner]))
    return out


def ownership(sc, corners, views, glass, lamp8, base8):
    """corners already passed guard REJECT + separate strip -> (atlas uint8 or None, info)"""
    R = R2; T = OWN_T_CM / CM
    views = list(views)
    PF, hitF = pmap(sc, 0.0)
    labF, _ = ndimage.label(mask2k(lamp8["front"]))

    def side_cos(cam):
        xy, z = cam.project(sc.P0); fid, bary, _ = C.screen_raster(xy, z, sc.Fd, R, R); h = fid >= 0
        P0, Fd = sc.P0, sc.Fd
        fn = F.normalize(torch.linalg.cross(P0[Fd[:, 1]] - P0[Fd[:, 0]], P0[Fd[:, 2]] - P0[Fd[:, 0]]), dim=1)
        Pp = (P0[Fd[fid[h]]] * bary[h][:, :, None]).sum(1)
        c = torch.zeros(R * R, device=dev); c[h] = (fn[fid[h]] * F.normalize(cam.pos[None] - Pp, dim=1)).sum(1)
        return c.view(R, R).cpu().numpy()
    cut3d, info = [], {}
    for corner in corners:
        side = "left" if corner == "front-left" else "right"; sx = 1.0 if side == "left" else -1.0; k = NAMES.index(side)
        ids = [j for j in range(1, labF.max() + 1) if (labF == j).sum() > 300 and np.median(PF[labF == j][:, 0]) * sx > 0]
        camS = C.Cam(AZ[side], 0.0, FOV, R); PS, hitS = pmap(sc, AZ[side])
        labS, _ = ndimage.label(mask2k(lamp8[side]))
        idsS = [j for j in range(1, labS.max() + 1) if (labS == j).sum() > 300]
        if not ids or not idsS:
            info[corner] = dict(side=side, cut=False, why="no head-lamp blob in the front or side view"); continue
        bF = labF == max(ids, key=lambda j: (labF == j).sum())
        front_is_left = float(camS.right @ torch.tensor([0.0, 1.0, 0.0], device=dev)) > 0
        cx = {j: np.nonzero(labS == j)[1].mean() for j in idsS}
        bS = labS == (min(idsS, key=lambda j: cx[j]) if front_is_left else max(idsS, key=lambda j: cx[j]))
        cos_med = float(np.median(side_cos(camS)[bS & hitS]))
        if cos_med <= OWN_COS:
            info[corner] = dict(side=side, cut=False, cos_side_median=round(cos_med, 3),
                                why=f"side piece is not side-facing (median cos_side {cos_med:.2f} <= {OWN_COS}): keep the texture"); continue
        tree = cKDTree(PF[bF & hitF])
        ys, xs = np.nonzero(bS & hitS)
        d, _ = tree.query(PS[ys, xs])
        far = np.zeros((R, R), bool); far[ys[d > T], xs[d > T]] = True
        near = np.zeros((R, R), bool); near[ys[d <= T], xs[d <= T]] = True
        im = views[k][0].numpy(); carpx = im.max(2) > 0.04
        cut = ndimage.binary_dilation(far, iterations=6) & ~ndimage.binary_dilation(near, iterations=1) & carpx
        img8 = (np.clip(im, 0, 1) * 255 + 0.5).astype(np.uint8)
        w = (~cut & carpx).astype(np.float32); fl = img8.astype(np.float32)
        s = 9.0; num = cv2.GaussianBlur(fl * w[..., None], (0, 0), s); den = cv2.GaussianBlur(w, (0, 0), s)[..., None]
        fill = num / np.maximum(den, 1e-4)
        for it in range(3):
            bad = (den[..., 0] < 0.05) & cut
            if not bad.any(): break
            s *= 2; num = cv2.GaussianBlur(fl * w[..., None], (0, 0), s); den2 = cv2.GaussianBlur(w, (0, 0), s)[..., None]
            fill = np.where(bad[..., None], num / np.maximum(den2, 1e-4), fill); den = np.maximum(den, den2)
        soft = cv2.GaussianBlur(cut.astype(np.float32), (0, 0), 1.5)[..., None]
        outi = np.clip(fl * (1 - soft) + fill * soft, 0, 255).astype(np.uint8)
        views[k] = (torch.from_numpy(outi.astype(np.float32) / 255), views[k][1], views[k][2])
        yy, xx = np.nonzero(cut)
        cut3d.append(PS[yy, xx][hitS[yy, xx]])
        info[corner] = dict(side=side, cut=True, cos_side_median=round(cos_med, 3), side_lamp_px=int(bS.sum()), far_px=int(far.sum()),
                            cut_px=int(cut.sum()), gap_cm_median=round(float(np.median(d)) * CM, 2))
        log("ownership", corner, json.dumps(info[corner]))
    if not cut3d:
        return None, info
    tex, dbg = C.texture_mesh(sc.v, sc.f, sc.uv, sc.base, views, FOV, {"mode": "blend"}, log=lambda *a: None, glass_masks=glass)
    new = tex.numpy(); A = new.shape[0]
    cur = f8(base8)
    t_idx, P, N, _, _ = C._texel_geometry(sc.v, sc.f, sc.uv, A, dev)
    c3 = torch.from_numpy(np.concatenate(cut3d)).float().to(dev)[::4]
    dist = torch.full((len(t_idx),), 9.0, device=dev)
    for ch in torch.split(torch.arange(len(t_idx), device=dev), 20000):
        dist[ch] = torch.cdist(P[ch], c3).min(1).values
    wt = 1 - C.smoothstep(dist, 3.0 / CM, 5.0 / CM)
    W = np.zeros(A * A, np.float32); W[t_idx.cpu().numpy()] = wt.cpu().numpy(); W = W.reshape(A, A, 1)
    info["outside_region_mean_abs_255"] = round(float(np.abs(new - cur)[W[..., 0] == 0].mean() * 255), 3)
    info["region_texels"] = int((W > 0).sum())
    return q8(cur * (1 - W) + new * W), info


# ------------------------------------------------------------------ compose (job 7 apply)
def compose(sc, base8, regions):
    """regions: [(plan dict, edt (S,S,3), mask (S,S), keep, anchor)] pasted z-tested, max-weight blend -> (atlas uint8, pasted)"""
    tex = torch.from_numpy(f8(base8))
    A = tex.shape[0]
    t_idx, P, N, P0, Fd = C._texel_geometry(sc.v, sc.f, sc.uv, A, dev)
    flat = tex.to(dev).reshape(-1, 3).clone()
    ws, cols, pasted = [], [], []
    for r, PL, edt, mask, keep, anchor in regions:
        S = r["side"]
        vis, cos, xy = C._visible(C.Cam(r["az"], r["el"], r["fov"], C.TOUCHUP_RES), P, N, P0, Fd, C.TOUCHUP_RES)
        lxy = xy - torch.tensor([r["x0"], r["y0"]], device=dev, dtype=torch.float32)
        inside = (lxy[:, 0] >= 0) & (lxy[:, 0] < S - 1) & (lxy[:, 1] >= 0) & (lxy[:, 1] < S - 1)
        e = torch.from_numpy(edt).to(dev).permute(2, 0, 1)[None]
        pm = component(mask, r, keep, anchor).astype(np.float32)
        if (pm > 0.5).sum() == 0:
            pasted.append(dict(corner=r["corner"], plan=PL, keep=keep, texels=0)); continue
        wt = (C.sample(torch.from_numpy(pm).to(dev)[None, None], lxy, S, S)[:, 0] * vis * C.smoothstep(cos, 0.15, 0.4) * inside.float()).clamp(0, 1)
        ws.append(wt); cols.append(C.sample(e, lxy, S, S))
        pasted.append(dict(corner=r["corner"], plan=PL, keep=keep, texels=int((wt > 0.05).sum())))
    if ws:
        W = torch.stack(ws); wmax, best = W.max(0)
        col = torch.stack(cols)[best, torch.arange(len(best), device=dev)]
        flat[t_idx] = flat[t_idx] * (1 - wmax[:, None]) + col * wmax[:, None]
    im = flat.view(A, A, 3).cpu().numpy()
    occ = np.zeros(A * A, bool); occ[t_idx.cpu().numpy()] = True
    idx = ndimage.distance_transform_edt(~occ.reshape(A, A), return_distances=False, return_indices=True)
    return q8(im[idx[0], idx[1]]), pasted


# ------------------------------------------------------------------ the two node steps
def plan(sc, rel8, W0, lamp8):
    """-> plan state (dict). rel8 = the projection atlas (uint8), W0 = its per-view weights (fp16 (4,A,A)), lamp8 {view: uint8 2K}"""
    tm, t0 = {}, time.time()
    D = dict(corners={c: {} for c in CORNERS})
    with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
        t = time.time()
        planA, infoA, cropsA, masksA = plan_corners(sc, rel8, W0, lamp8, list(CORNERS), "planA")
        tm["plan_release"] = round(time.time() - t, 1)
        t = time.time(); bd = bands(sc)
        cur, sil = measure_lines(sc, sc.views, sc.glass8, lamp8, bd)
        steps = line_steps(cur); votes = warp_rule(steps)
        D["lines"] = dict(bands=bd, silhouette_iou=sil, release_dz=steps, warp_votes=votes)
        warped, wimg = False, None
        if votes:
            warped, winfo, wimg = fit_warp(sc, cur, lamp8["back"], bd)
            D["lines"]["warp"] = winfo
            if not warped:
                log("warp rule fired but warp refused:", winfo["why"])
        tm["lines_warp"] = round(time.time() - t, 1)
        t = time.time(); rstep = {}
        if not warped:
            rstep = rear_steps(sc, lamp8, W0, infoA)
        D["rear_step"] = dict(corners=rstep, action="none (detect + report)")
        tm["rear_step"] = round(time.time() - t, 1)
        planB, infoB, cropsB, masksB, Ww, gw, texw8, lamp_w = [], {}, {}, {}, None, None, None, None
        views, glass, glass8 = list(sc.views), list(sc.glass_t), list(sc.glass8)
        if warped:
            t = time.time()
            views[2] = (torch.from_numpy(f8(wimg["back"])), views[2][1], views[2][2])
            glass[2] = torch.from_numpy(f8(wimg["glass"])); glass8[2] = wimg["glass"]
            texw8, Ww, gw = reproject(sc, views, glass)
            lamp_w = dict(lamp8, back=wimg["lamp"])
            cur_w, _ = measure_lines(sc, views, glass8, lamp_w, bd)
            D["lines"]["warped_dz"] = line_steps(cur_w)
            planB, infoB, cropsB, masksB = plan_corners(sc, texw8, Ww, lamp_w, ["rear-left", "rear-right"], "planB")
            tm["reproject_plan_warp"] = round(time.time() - t, 1)
        crops = {("planA", c): x for c, x in cropsA.items()}; crops.update({("planB", c): x for c, x in cropsB.items()})
        masks = {("planA", c): x for c, x in masksA.items()}; masks.update({("planB", c): x for c, x in masksB.items()})
        # regions (corner, plan dict, PL, keep, anchor) in the tested order (the max-weight blend breaks ties by order)
        pA = {r["corner"]: r for r in planA}; pB = {r["corner"]: r for r in planB}
        if not warped:
            regs = [(r["corner"], r, "planA", None, None) for r in planA]
        else:                                    # warped rear: tail lamp from the warped plan, other lamp components from the release plan
            regs = [(c, pA[c], "planA", None, None) for c in ("front-left", "front-right") if c in pA]
            others, tails = [], []
            for c in ("rear-left", "rear-right"):
                if c not in pB:
                    continue
                if c in pA:
                    if (component(f8(masks[("planA", c)]), pA[c], "other", pB[c]["anchor"]) > 0.5).sum() > 0:
                        others.append((c, pA[c], "planA", "other", pB[c]["anchor"]))
                tails.append((c, pB[c], "planB", "tail", pB[c]["anchor"]))
            regs += others + tails
        gate = {"planA": infoA, "planB": infoB}
        D["split_gate"] = {f"{PL}/{c}": dict(split_cm2=gate[PL].get(c, {}).get("split_cm2"), fires=bool(gate[PL].get(c, {}).get("split")))
                           for c, r, PL, k, an in regs}
        regs = [x for x in regs if gate[x[2]].get(x[0], {}).get("split")]     # job 9 rule 1: redraw only where the split detector fires
        jvar = {}
        for c, r, PL, k, an in regs:
            if k == "other":
                continue
            tex8 = texw8 if PL == "planB" else rel8
            jvar[f"{PL}/{c}"] = [[PL, r]]
            for i, (dx, dy) in enumerate(JITTER[1:], 1):
                cj, mj, rj = jitter_crop(sc, tex8, r, masks[(PL, c)], dx, dy)
                crops[(f"{PL}j{i}", c)] = cj; masks[(f"{PL}j{i}", c)] = mj
                jvar[f"{PL}/{c}"].append([f"{PL}j{i}", rj])
    items = [(PL, c) for c, r, PL, k, an in regs if k == "other"] + [(PLj, key.split("/")[1]) for key, vs in jvar.items() for PLj, _ in vs]
    tm["plan_total"] = round(time.time() - t0, 1)
    D["seconds_plan"] = tm
    D["redraw_corners"] = sorted({c for c, *_ in regs})
    D["edits_requested"] = len(items)
    log("plan:", len(items), "edits for", D["redraw_corners"] or "no corner", "| warped" if warped else "", tm)
    return dict(D=D, regs=regs, jvar=jvar, warped=warped, infoA=infoA, infoB=infoB, steps=steps, items=items,
                crops={k: crops[k] for k in items}, masks={k: masks[k] for k in items},
                rel8=rel8, W0=W0, Ww=Ww, gw=gw, texw8=texw8, lamp8=lamp8, lamp_w=lamp_w, wimg=wimg)


def apply(sc, S, edits8):
    """S = plan state, edits8 {(PL, corner): uint8 edit} -> (atlas uint8 or None if nothing changed, glass map fp16 or None, decisions)"""
    t0 = time.time(); tm = {}
    D, regs, jvar, warped = copy.deepcopy(S["D"]), S["regs"], S["jvar"], S["warped"]      # the plan output is cached by ComfyUI
    infoA, infoB, steps, masks, crops = S["infoA"], S["infoB"], S["steps"], S["masks"], S["crops"]
    lamp8, W0, Ww = S["lamp8"], S["W0"], S["Ww"]
    views, glass = list(sc.views), list(sc.glass_t)
    if warped:
        views[2] = (torch.from_numpy(f8(S["wimg"]["back"])), views[2][1], views[2][2]); glass[2] = torch.from_numpy(f8(S["wimg"]["glass"]))
    t = time.time()
    tmd = {}
    for c, r, PL, k, an in regs:
        if k == "other":
            tmd[(PL, c)] = tone_match(crops[(PL, c)], masks[(PL, c)], edits8[(PL, c)], r["side"])
    G_, chosen = {}, {}
    for c, r, PL, k, an in regs:
        if k == "other":
            continue
        head = "front" if c.startswith("front") else "back"
        other = [x for x in CORNERS[c][1:] if x != head][0]
        rear_w = warped and PL == "planB"
        res = []
        for PLj, rj in jvar.get(f"{PL}/{c}", [[PL, r]]):
            tmd[(PLj, c)] = tone_match(crops[(PLj, c)], masks[(PLj, c)], edits8[(PLj, c)], rj["side"])
            g = guard(sc, c, rj, head, other, S["lamp_w"]["back"] if rear_w else lamp8[head], Ww if rear_w else W0,
                      crops[(PLj, c)], tmd[(PLj, c)][1])
            log("guard", c, PLj, json.dumps(g))
            res.append(dict(g, plan=PLj, variant=PLj, jitter=rj.get("jitter", [0, 0]), move_ok=tmd[(PLj, c)][3]["ok"]))
        ok = [x for x in res if x["move_ok"]]
        agg = dict(variants=res)
        if not ok:
            agg["verdict"] = "REJECT (move gate)"
        elif all("recall_dark" not in x for x in ok):
            agg["verdict"] = "accept (no head-on lamp face in the crop)"; chosen[c] = ok[0]["variant"]
        else:
            mean = {kk: round(float(np.mean([x[f"recall_{kk}"] for x in ok])), 3) for kk in ("dark", "bright")}
            px = {kk: ok[0][f"{kk}_px"] for kk in ("dark", "bright")}
            bad = [kk for kk in ("dark", "bright") if px[kk] >= 200 and mean[kk] < GUARD_REC]
            agg.update(mean_recall=mean, verdict="REJECT (" + ",".join(bad) + ")" if bad else "accept")
            if not bad:
                chosen[c] = max(ok, key=lambda x: min(x["recall_dark"], x["recall_bright"]))["variant"]
        agg["chosen"] = chosen.get(c)
        G_[c] = agg; log("guard (jitter mean)", c, json.dumps({kk: v for kk, v in agg.items() if kk != "variants"}))
    rejected = {c for c, g in G_.items() if g["verdict"].startswith("REJECT")}
    tm["tone_match_guard"] = round(time.time() - t, 1)
    t = time.time()
    base8 = S["texw8"] if warped else S["rel8"]
    fronts_rej = [c for c in ("front-left", "front-right") if c in rejected]
    sep, own_info, own8 = {}, {}, None
    if fronts_rej:
        sep = separate_strip(sc, lamp8, Ww if warped else W0, fronts_rej)
        cand = [c for c in fronts_rej if sep[c]["separate"]]
        if cand:
            own8, own_info = ownership(sc, cand, views, glass, lamp8, base8)
    tm["ownership"] = round(time.time() - t, 1)
    t = time.time()
    paste = []
    for c, r, PL, k, an in regs:
        if k != "other" and c in chosen:         # the chosen jitter variant of the guarded redraw
            PL, r = next((PLj, rj) for PLj, rj in jvar.get(f"{PL}/{c}", [[PL, r]]) if PLj == chosen[c])
        edt, _, mask, mg = tmd[(PL, c)]
        D["corners"][c].setdefault("redraws", []).append(dict(plan=PL, keep=k, move_gate=mg, guard=G_.get(c, {}).get("verdict")))
        if c in rejected or not mg["ok"]:
            continue
        paste.append((r, PL, edt, mask, k, an))
    out8 = None
    if paste or warped or own8 is not None:
        out8, pasted = compose(sc, own8 if own8 is not None else base8, paste)
        D["pasted"] = pasted
    tm["compose"] = round(time.time() - t, 1)
    for c in CORNERS:
        d = D["corners"][c]
        d["split_lamp_release"] = infoA.get(c, {}).get("pair", False)
        if c.startswith("rear"):
            d["line_dz_cm"] = {k: v["dz_cm"] for k, v in steps[c].items()}
            if warped:
                d["line_dz_cm_after_warp"] = {k: v["dz_cm"] for k, v in D["lines"]["warped_dz"][c].items()}
                d["split_lamp_after_warp"] = infoB.get(c, {}).get("pair", False)
        for key, src in (("guard", G_), ("separate_strip", sep), ("ownership", own_info)):
            if c in src:
                d[key] = src[c]
        d["split_detector"] = {k.split("/")[0]: v for k, v in D.get("split_gate", {}).items() if k.endswith("/" + c)}
        if c.startswith("rear"):
            d["rear_step"] = D.get("rear_step", {}).get("corners", {}).get(c)
        tags = ["warped"] if c.startswith("rear") and warped else []
        if c in rejected:
            tags += ["guard-rejected", "owned" if own_info.get(c, {}).get("cut") else "untouched"]
        elif any(x["corner"] == c and x["texels"] > 0 for x in D.get("pasted", [])):
            tags.append("redrawn")
        elif any(not x["move_gate"]["ok"] for x in d.get("redraws", [])):
            tags.append("move-gate-rejected")
        d["decision"] = "+".join(tags) or "untouched"
        if c.startswith("rear") and (d.get("rear_step") or {}).get("step"):
            d["decision"] += " (rear step detected, left as release)"
    tm["apply_total"] = round(time.time() - t0, 1)
    D["seconds_apply"] = tm
    log("decisions", json.dumps({c: D["corners"][c]["decision"] for c in CORNERS}), tm)
    return out8, (S["gw"] if warped else None), D
