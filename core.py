# Made by Yovan Vujacic
"""Multiview projection texturing for the Pixal3D / Trellis2 four-view rig.

Conventions (ComfyUI native MESH): vertices Y-up (the frame Pixal3D outputs and SaveGLB writes),
UVs glTF-style (texel row = v * H, column = u * W), textures (H, W, 3) float 0..1, row 0 at top.
Internally positions are converted to the rig's Z-up frame: (x, y, z)_rig = (x, -z, y)_yup, where
the native Pixal3DMultiViewConditioning cameras live (azimuth 0 = front, camera on -Y).

Pipeline (texture_mesh):
  1. conservative UV raster -> texel -> (face, barycentrics)
  2. weld vertices by position (UV seams split them); smoothed welded normals
  3. per view: exact z-buffer of the welded mesh, texel visibility, smooth per-vertex weights
     (visibility x smoothstep(cos) x distance to the photo's silhouette), sharpened
  4. per-view colour gain from overlaps (exposure differences between photos)
  5. mode "label": one view per face (argmax of the smooth weights) + seam levelling (per-view
     per-vertex colour offsets solved so both sides of every seam agree);
     mode "blend": weighted blend of all views (v5 behaviour)
  6. hidden texels: smoothed generated colour + smooth offset toward the projected colour
  7. island padding
"""
import logging
import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse, ndimage
from scipy.sparse.linalg import splu

VIEW_AZ = {"front": 0.0, "left": 90.0, "back": 180.0, "right": 270.0}
VIEW_PAD = 1.1

DEFAULTS = dict(
    mode="blend",        # "blend" (default), "twoband" (broad colour blended, detail from best view; no gain on the estate corner, 4x slower) or "label"
    band_smooth=120,     # twoband: smoothing passes defining "broad colour" (larger = detail band holds bigger features)
    c0=0.35, c1=0.8,     # smoothstep on cos(view, normal); shallow views reflect sky, so trust them less
    edge_px=8.0,         # distance to the photo silhouette (px) at which a view gets full weight
    wsmooth=25,          # smoothing iterations on per-vertex view weights
    nsmooth=15,          # smoothing iterations on normals used for weighting
    sharp=3.0,           # blend mode: weight exponent
    gain=False,          # per-view colour gain compensation (exposure shifts between generated views)
    level=True,          # label mode: seam levelling
    level_lam=1.0,       # label mode: smoothness of the levelling offsets
    bsmooth=15,          # hidden fill: smoothing of the generated colour
    fill_lam=3.0,        # hidden fill: smoothness of the hue offset
    fill="transfer",     # hidden fill: "transfer" (per-material colour transfer) or "offset" (v5)
    fill_k=8,            # transfer: number of material clusters in the generated texture
    fill_decay=0.005,    # transfer: how fast the boundary offset fades into hidden areas
    fill_robust=0.2,     # boundary offsets larger than this (feature colours, e.g. a red light) don't spread
    glass_tint=-1.0,     # glass colour: -1 = darkest typical glass colour found in the photos, else a grey level
    glass_max_lum=0.10,  # cap on the glass colour's brightness
    glass_detail=0.0,    # 0 = uniform tint, 1 = keep the photos' glass pixels
    glass_c0=0.2,        # min cos(view, surface) for a view's glass label to count
    glass_spread_decay=0.3,  # how far glass extends into surface no view sees (higher = shorter; 0.02 bled onto bonnets)
    consensus=True,      # where views overlap and disagree, prefer the one closer to the 3D model's colour
    cons_sigma=0.08,
    prior_check=False,   # (off: overrules real trim and wheels too) single-view areas much darker than the 3D model's own colour (invented vents) -> fill
    prior_dl=0.18,       # luminance drop (0..1, large-scale) where the overrule starts; full at 1.8x
    prior_smooth=20,     # vertex smoothing passes for the large-scale comparison
    paint_guard=True,
    gen_agree=False,     # (off: misfires on windscreen bases, misses the estate corner) where the generator paints body colour and the photos are dark+grey, keep the generator
    agree_smooth=3,      # vertex smoothing before comparing (small: narrow vents must still register)
    agree_ds=0.25,       # saturation drop (generator - photo) where it starts
    agree_dl=0.08,       # luminance drop where it starts
    debug=False,    # prior check only where the generator paints the body colour
    tol=0.75,            # conservative raster tolerance (texels)
    keep_views=False,    # also return the per-view texel weights in dbg["w_t"] (corner fix)
)


def _dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def smoothstep(x, a, b):
    t = ((x - a) / (b - a)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def yup_to_rig(v):
    return torch.stack([v[:, 0], -v[:, 2], v[:, 1]], 1)


class Cam:
    def __init__(self, az, el, fov_deg, res, dist=None):
        dev = _dev()
        fov = math.radians(fov_deg)
        self.dist = dist if dist is not None else VIEW_PAD * 0.5 / math.tan(fov / 2)
        a, e = math.radians(az), math.radians(el)
        back = torch.tensor([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)], device=dev)
        self.pos = back * self.dist
        self.fwd = -back
        self.right = F.normalize(torch.linalg.cross(self.fwd, torch.tensor([0.0, 0, 1], device=dev)), dim=0)
        self.up = torch.linalg.cross(self.right, self.fwd)
        self.W = self.H = res
        self.f = (res / 2) / math.tan(fov / 2)

    def project(self, P):
        pc = P - self.pos
        z = pc @ self.fwd
        return torch.stack([self.W / 2 + self.f * (pc @ self.right) / z, self.H / 2 - self.f * (pc @ self.up) / z], 1), z


def screen_raster(xy, z, faces, W, H):
    """Exact z-buffer: front-most face per pixel. Returns fid (H*W,), bary (H*W,3), depth (H*W,)."""
    dev = xy.device
    T = xy[faces]
    Z = z[faces]
    lim = torch.tensor([W - 1, H - 1], device=dev)
    lo = torch.minimum(torch.floor(T.min(1).values - 0.5).long().clamp(min=0), lim)
    hi = torch.minimum(torch.ceil(T.max(1).values - 0.5).long().clamp(min=0), lim)
    size = (hi - lo + 1).max(1).values
    vis = (Z.min(1).values > 0) & (T.max(1).values >= 0).all(1) & (T.min(1).values[:, 0] < W) & (T.min(1).values[:, 1] < H)
    zb = torch.full((H * W,), float("inf"), device=dev)
    rec = []
    prev = 0
    for K in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048):
        sel = torch.nonzero(vis & (size <= K) & (size > prev)).squeeze(1)
        prev = K
        if len(sel) == 0:
            continue
        for chunk in torch.split(sel, max(1, 4_000_000 // (K * K))):
            ox, oy = torch.meshgrid(torch.arange(K, device=dev), torch.arange(K, device=dev), indexing="xy")
            px = lo[chunk, 0, None] + ox.reshape(1, -1)
            py = lo[chunk, 1, None] + oy.reshape(1, -1)
            cx, cy = px.float() + 0.5, py.float() + 0.5
            a, b, c = T[chunk, 0], T[chunk, 1], T[chunk, 2]
            v0, v1 = b - a, c - a
            d = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
            good = d.abs() > 1e-12
            d = torch.where(good, d, torch.ones_like(d))
            wx, wy = cx - a[:, 0, None], cy - a[:, 1, None]
            l1 = (wx * v1[:, 1, None] - v1[:, 0, None] * wy) / d[:, None]
            l2 = (v0[:, 0, None] * wy - wx * v0[:, 1, None]) / d[:, None]
            l0 = 1 - l1 - l2
            ins = (l0 >= 0) & (l1 >= 0) & (l2 >= 0) & good[:, None] & (px < W) & (py < H)
            zz = l0 * Z[chunk, 0, None] + l1 * Z[chunk, 1, None] + l2 * Z[chunk, 2, None]
            idx = (py * W + px)[ins]
            zi = zz[ins]
            zb.scatter_reduce_(0, idx, zi, reduce="amin")
            rec.append((idx, zi, chunk[:, None].expand_as(px)[ins], torch.stack([l0[ins], l1[ins], l2[ins]], 1)))
    fid = torch.full((H * W,), -1, dtype=torch.long, device=dev)
    bary = torch.zeros((H * W, 3), device=dev)
    for idx, zi, fi, bi in rec:
        top = zi <= zb[idx]
        fid[idx[top]] = fi[top]
        bary[idx[top]] = bi[top]
    return fid, bary, zb


def uv_raster(uv, faces, A, tol):
    """Conservative UV raster (glTF uv: row = v*A). Texels within `tol` texels of a triangle belong to
    the triangle they are most inside; barycentrics clamped. Returns fid (A*A,), bary (A*A,3)."""
    dev = uv.device
    fid = torch.full((A * A,), -1, dtype=torch.long, device=dev)
    bary = torch.zeros((A * A, 3), device=dev)
    best = torch.full((A * A,), -1e9, device=dev)
    T = uv[faces] * A
    lo = (torch.floor(T.min(1).values) - 1).long().clamp(0, A - 1)
    hi = (torch.ceil(T.max(1).values) + 1).long().clamp(0, A - 1)
    size = (hi - lo + 1).max(1).values
    prev = 0
    for K in (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
        sel = torch.nonzero((size <= K) & (size > prev)).squeeze(1)
        prev = K
        if len(sel) == 0:
            continue
        for chunk in torch.split(sel, max(1, 4_000_000 // (K * K))):
            ox, oy = torch.meshgrid(torch.arange(K, device=dev), torch.arange(K, device=dev), indexing="xy")
            px = lo[chunk, 0, None] + ox.reshape(1, -1)
            py = lo[chunk, 1, None] + oy.reshape(1, -1)
            cx, cy = px.float() + 0.5, py.float() + 0.5
            a, b, c = T[chunk, 0], T[chunk, 1], T[chunk, 2]
            v0, v1 = b - a, c - a
            d = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
            good = d.abs() > 1e-9
            d = torch.where(good, d, torch.ones_like(d))
            wx, wy = cx - a[:, 0, None], cy - a[:, 1, None]
            l1 = (wx * v1[:, 1, None] - v1[:, 0, None] * wy) / d[:, None]
            l2 = (v0[:, 0, None] * wy - wx * v0[:, 1, None]) / d[:, None]
            l0 = 1 - l1 - l2
            ad = d.abs()[:, None]
            d0 = l0 * ad / (c - b).norm(dim=1).clamp(min=1e-9)[:, None]
            d1 = l1 * ad / v1.norm(dim=1).clamp(min=1e-9)[:, None]
            d2 = l2 * ad / v0.norm(dim=1).clamp(min=1e-9)[:, None]
            score = torch.minimum(torch.minimum(d0, d1), d2)
            ins = (score >= -tol) & good[:, None] & (px < A) & (py < A)
            idx = (py * A + px)[ins]
            sc = score[ins]
            best.scatter_reduce_(0, idx, sc, reduce="amax")
            cand = torch.arange(len(idx), device=dev)
            top = sc >= best[idx]
            win = torch.full((A * A,), -1, dtype=torch.long, device=dev)
            win.scatter_reduce_(0, idx[top], cand[top], reduce="amax")
            keep = top & (win[idx] == cand)
            L = torch.stack([l0[ins], l1[ins], l2[ins]], 1)[keep].clamp(min=0)
            fid[idx[keep]] = chunk[:, None].expand_as(px)[ins][keep]
            bary[idx[keep]] = L / L.sum(1, keepdim=True).clamp(min=1e-9)
    return fid, bary


def sample(img_chw, xy, W, H):
    """Bilinear sample a (1,C,H,W) image at pixel coords xy (K,2) -> (K,C)."""
    g = torch.stack([xy[:, 0] / W * 2 - 1, xy[:, 1] / H * 2 - 1], 1)
    return F.grid_sample(img_chw, g[None, None], mode="bilinear", align_corners=False)[0, :, 0].T


def photo_mask_and_edge(img_hwc):
    """Foreground (subject on black) and distance (px) to the silhouette, as (1,1,H,W) tensors."""
    a = img_hwc.detach().cpu().numpy()
    m = a.max(2) > 0.04
    m = ndimage.binary_opening(m, iterations=1)
    dist = ndimage.distance_transform_edt(m).astype(np.float32)
    dev = img_hwc.device
    return torch.tensor(m, device=dev), torch.tensor(dist, device=dev)[None, None]


def texture_mesh(vertices_yup, faces, uvs, base_tex, views, fov=20.0, opts=None, log=print, glass_masks=None):
    """vertices_yup (N,3), faces (M,3), uvs (N,2) glTF convention in [0,1], base_tex (A,A,3) 0..1,
    views: list of (img (H,W,3) 0..1 subject on black, azimuth_deg, elevation_deg).
    glass_masks: optional list (same order as views) of (H,W) 0..1 masks of the glass in each view;
    glass becomes one uniform dark tint and dbg["glass"] (A,A) marks it for a glossy material.
    Returns (texture (A,A,3) float cpu, debug dict)."""
    o = dict(DEFAULTS, **(opts or {}))
    dev = _dev()
    A = int(base_tex.shape[0])
    V = vertices_yup.to(dev).float()
    Fc = faces.to(dev).long()
    UV = uvs.to(dev).float()
    base_full = base_tex.to(dev).float()

    # --- welded mesh ---------------------------------------------------------------------------
    vn = V.cpu().numpy()
    _, weld = np.unique(np.round(vn, 5), axis=0, return_inverse=True)
    weld = weld.ravel()
    nW = int(weld.max()) + 1
    fw = weld[Fc.cpu().numpy()]
    Pw_np = np.zeros((nW, 3), np.float32)
    Pw_np[weld] = vn
    Pw = yup_to_rig(torch.tensor(Pw_np, device=dev))
    fw_t = torch.tensor(fw, device=dev, dtype=torch.long)
    e = np.r_[fw[:, [0, 1]], fw[:, [1, 2]], fw[:, [2, 0]]]
    e = e[e[:, 0] != e[:, 1]]
    Adj = sparse.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(nW, nW)).tocsr()
    Adj.data[:] = 1
    deg = np.asarray(Adj.sum(1)).ravel()
    Wn = (sparse.diags(1 / np.maximum(deg, 1)) @ Adj).tocsr()
    Lap = (sparse.diags(deg) - Adj).tocsr()

    def smooth(x, it):
        for _ in range(it):
            x = 0.5 * x + 0.5 * (Wn @ x)
        return x

    P_np = Pw.cpu().numpy()
    fn = np.cross(P_np[fw[:, 1]] - P_np[fw[:, 0]], P_np[fw[:, 2]] - P_np[fw[:, 0]])
    Nw = np.stack([sum(np.bincount(fw[:, k], fn[:, c], minlength=nW) for k in range(3)) for c in range(3)], 1)
    Nw /= np.linalg.norm(Nw, axis=1, keepdims=True).clip(1e-12)
    Nw = smooth(Nw, o["nsmooth"])
    Nw /= np.linalg.norm(Nw, axis=1, keepdims=True).clip(1e-12)
    Nw_t = torch.tensor(Nw, dtype=torch.float32, device=dev)
    log(f"mesh {len(V)} verts -> {nW} welded, {len(Fc)} faces, atlas {A}")

    # --- texels --------------------------------------------------------------------------------
    fid, bary = uv_raster(UV, Fc, A, o["tol"])
    occ = fid >= 0
    tex = torch.nonzero(occ).squeeze(1)
    f = fid[tex]
    b = bary[tex]
    tw = fw_t[f]
    P = (Pw[tw] * b[..., None]).sum(1)
    Nt = F.normalize((Nw_t[tw] * b[..., None]).sum(1), dim=1)
    base = base_full.reshape(-1, 3)[tex]
    nT = len(tex)

    def to_vert(val, wt):
        num = torch.zeros((nW, val.shape[1]), device=dev)
        den = torch.zeros(nW, device=dev)
        for j in range(3):
            ww = b[:, j] * wt
            num.index_add_(0, tw[:, j], val * ww[:, None])
            den.index_add_(0, tw[:, j], ww)
        return num, den

    def to_tex(v):
        return (v[tw] * b).sum(1) if v.dim() == 1 else (v[tw] * b[..., None]).sum(1)

    ones = torch.ones(nT, device=dev)

    # --- per view weights and colours ------------------------------------------------------------
    nV = len(views)
    w_t = torch.zeros((nV, nT), device=dev)
    w_v_all = np.zeros((nV, nW), np.float32)
    cols = []
    vcols = []
    g_acc = torch.zeros(nT, device=dev)
    g_wsum = torch.zeros(nT, device=dev)
    glass_pix = []
    for k, (img, az, el) in enumerate(views):
        img = img.to(dev).float()
        H, W = int(img.shape[0]), int(img.shape[1])
        cam = Cam(az, el, fov, W)
        img_chw = img.permute(2, 0, 1)[None]
        _, edge = photo_mask_and_edge(img)
        xy, z = cam.project(P)
        ok = (xy[:, 0] >= 0) & (xy[:, 0] < W - 1) & (xy[:, 1] >= 0) & (xy[:, 1] < H - 1) & (z > 0)
        pix = xy[:, 1].long().clamp(0, H - 1) * W + xy[:, 0].long().clamp(0, W - 1)
        xw, zw = cam.project(Pw)
        _, _, zb = screen_raster(xw, zw, fw_t, W, H)
        cos_t = (Nt * F.normalize(cam.pos[None] - P, dim=1)).sum(1).clamp(min=0.05)
        eps = 0.002 + 0.8 * (z / cam.f) * ((1 - cos_t ** 2).sqrt() / cos_t).clamp(max=4.0)
        vis_t = (ok & (z <= zb[pix] + eps)).float()
        num, den = to_vert(vis_t[:, None], ones)
        vis_v = (num[:, 0] / den.clamp(min=1e-9)).cpu().numpy()
        cos_v = (Nw_t * F.normalize(cam.pos[None] - Pw, dim=1)).sum(1)
        edge_v = sample(edge, xw, W, H)[:, 0]
        wv = (smoothstep(cos_v, o["c0"], o["c1"]) * cos_v.clamp(min=0) ** 2 * (edge_v / o["edge_px"]).clamp(0, 1)).cpu().numpy() * vis_v
        vis_s = smooth(vis_v, 5)
        wv = smooth(wv, o["wsmooth"]) * np.clip((vis_s - 0.2) / 0.6, 0, 1)
        w_v_all[k] = wv
        w_t[k] = to_tex(torch.tensor(wv, dtype=torch.float32, device=dev)) * vis_t
        cols.append(sample(img_chw, xy, W, H))                       # (nT,3)
        if glass_masks is not None and glass_masks[k] is not None:
            gm = glass_masks[k].to(dev).float()
            if gm.shape[0] != H or gm.shape[1] != W:
                gm = F.interpolate(gm[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
            gw = vis_t * smoothstep(cos_t, o["glass_c0"], o["glass_c0"] + 0.2)   # shallow views mislabel the bonnet
            g_acc += sample(gm[None, None], xy, W, H)[:, 0] * gw
            g_wsum += gw
            sel = (gm > 0.5).reshape(-1)
            if int(sel.sum()) > 100:
                glass_pix.append(img.reshape(-1, 3)[sel])
        vcols.append(sample(img_chw, xw, W, H))                      # (nW,3) colour at each welded vertex
        log(f"view az{az:g} el{el:g}: {int((w_t[k] > 0.05).sum())} weighted texels")

    # --- gain compensation (robust: median luminance ratio in overlaps, log domain, capped) -----
    gains = torch.ones((nV, 3), device=dev)
    if o["gain"] and nV > 1:
        wmax = w_t.max(0).values.clamp(min=1e-6)
        rows = []
        for i in range(nV):
            for j in range(i + 1, nV):
                m = (w_t[i] > 0.3 * wmax) & (w_t[j] > 0.3 * wmax) & (w_t[i] > 0.05) & (w_t[j] > 0.05)
                n = int(m.sum())
                if n < 500:
                    continue
                li = cols[i][m].mean(1).clamp(min=0.02)
                lj = cols[j][m].mean(1).clamp(min=0.02)
                rows.append((i, j, float(torch.log(lj / li).median()), n))
        if rows:
            ntot = sum(r[3] for r in rows)
            Am = np.zeros((len(rows) + nV, nV))
            bm = np.zeros(len(rows) + nV)
            for r, (i, j, lr, n) in enumerate(rows):
                s_ = math.sqrt(n / ntot)
                Am[r, i], Am[r, j], bm[r] = s_, -s_, s_ * lr          # log g_i - log g_j = log(L_j / L_i)
            for i in range(nV):
                Am[len(rows) + i, i] = 1.0                            # strong pull toward no change
            lg = np.clip(np.linalg.lstsq(Am, bm, rcond=None)[0], math.log(0.8), math.log(1.25))
            gains[:] = torch.tensor(np.exp(lg), device=dev, dtype=torch.float32)[:, None]
            cols = [cols[k] * gains[k] for k in range(nV)]
            vcols = [vcols[k] * gains[k] for k in range(nV)]
        log("gains " + " ".join(f"{views[k][1]:g}:{gains[k].mean().item():.3f}" for k in range(nV)))

    wsum = w_t.sum(0)
    conf = smoothstep(wsum, 0.03, 0.25)

    # --- consensus against the generated colour (tie-break where overlapping views disagree) -----
    if o["consensus"] and nV > 1:
        ws0 = w_t ** o["sharp"]
        proj0 = sum(cols[k] * ws0[k][:, None] for k in range(nV)) / ws0.sum(0).clamp(min=1e-12)[:, None]
        ref = _material_transfer(base, proj0, smoothstep(wsum, 0.03, 0.25), o["fill_k"], dev, lambda *a: None)
        dev_k = []
        for k in range(nV):
            dk = (cols[k] - ref).abs().mean(1)
            num, den = to_vert(dk[:, None], (w_t[k] > 0.02).float())
            dv = smooth((num[:, 0] / den.clamp(min=1e-9)).cpu().numpy(), 3)
            dev_k.append(to_tex(torch.tensor(dv, dtype=torch.float32, device=dev)))
        dev_k = torch.stack(dev_k, 0)
        multi = (w_t > 0.05).sum(0) >= 2
        dmin = torch.where(w_t > 0.05, dev_k, torch.full_like(dev_k, 9.0)).min(0).values
        pen = torch.exp(-((dev_k - dmin[None] - 0.03).clamp(min=0) / o["cons_sigma"]) ** 2)
        w_t = torch.where(multi[None], w_t * pen, w_t)
        log(f"consensus: {int(multi.sum())} overlap texels, {int((multi & (pen.min(0).values < 0.5)).sum())} with a suppressed view")

    # --- projected colour ----------------------------------------------------------------------
    label_t = None
    if o["mode"] == "blend":
        ws = w_t ** o["sharp"]
        proj = sum(cols[k] * ws[k][:, None] for k in range(nV)) / ws.sum(0).clamp(min=1e-12)[:, None]
    elif o["mode"] == "twoband":
        # broad colour blended across views (no seams), fine detail from the single best view (no ghosts)
        ws = w_t ** o["sharp"]
        lows = []
        for k in range(nV):
            seen_k = (w_t[k] > 0.01).float()
            num, den = to_vert(cols[k], seen_k)
            nv = num.cpu().numpy(); dv = den.cpu().numpy()
            for _ in range(o["band_smooth"]):
                nv = 0.5 * nv + 0.5 * (Wn @ nv)
                dv = 0.5 * dv + 0.5 * (Wn @ dv)
            low_v = torch.tensor(nv / np.maximum(dv, 1e-9)[:, None], dtype=torch.float32, device=dev)
            lows.append(to_tex(low_v))
        low = sum(lows[k] * ws[k][:, None] for k in range(nV)) / ws.sum(0).clamp(min=1e-12)[:, None]
        win = w_t.argmax(0)
        ar = torch.arange(nT, device=dev)
        high = torch.stack(cols, 0)[win, ar] - torch.stack(lows, 0)[win, ar]
        proj = low + high
        log(f"two-band: detail views " + " ".join(f"{views[k][1]:g}:{int((win == k).sum())}" for k in range(nV)))
    else:
        # one view per face: argmax of face-averaged smooth weights
        wf = torch.tensor(w_v_all, device=dev)[:, fw_t].mean(2)        # (nV, nF)
        face_lab = wf.argmax(0)
        face_ok = wf.max(0).values > 0.02
        label_t = face_lab[f]
        proj = torch.stack(cols, 0)[label_t, torch.arange(nT, device=dev)]
        if o["level"]:
            proj = proj + _seam_levelling(fw, face_lab.cpu().numpy(), face_ok.cpu().numpy(),
                                          [vc.cpu().numpy() for vc in vcols], nW, o["level_lam"],
                                          f, b, tw, label_t, dev, log)

    # --- prior check: a lone view much darker than the 3D model over a large area is overruled ---
    if o["prior_check"]:
        ref2 = _material_transfer(base, proj, conf, o["fill_k"], dev, lambda *a: None)
        lum = lambda c: 0.299 * c[:, 0] + 0.587 * c[:, 1] + 0.114 * c[:, 2]
        num, den = to_vert(torch.stack([lum(proj), lum(ref2)], 1), conf)
        lv = smooth((num / den.clamp(min=1e-9)[:, None]).cpu().numpy(), o["prior_smooth"])
        dl = lv[:, 1] - lv[:, 0]                                      # >0: projection darker than model
        single = (w_v_all > 0.05).sum(0) <= 1
        sv = np.clip((dl - o["prior_dl"]) / (0.8 * o["prior_dl"]), 0, 1) * single
        if o["paint_guard"]:
            # only where the generator itself paints the body colour (dominant saturated hue): real
            # trim, wheels and glass are dark in the generator too, so they are never overruled
            mx, mn = base.max(1).values, base.min(1).values
            sat = (mx - mn) / mx.clamp(min=1e-4)
            r_, g_, b_ = base[:, 0], base[:, 1], base[:, 2]
            hue = torch.atan2(1.7320508 * (g_ - b_), 2 * r_ - g_ - b_)
            colourful = (sat > 0.3) & (mx > 0.2)
            if int(colourful.sum()) > 1000:
                hc = torch.polar(torch.ones_like(hue[colourful]), hue[colourful]).mean()
                dom = torch.angle(hc)
                dh = torch.remainder(hue - dom + math.pi, 2 * math.pi) - math.pi
                paint = (colourful & (dh.abs() < 0.35)).float()
            else:
                paint = torch.zeros_like(sat)
            num, den = to_vert(paint[:, None], ones)
            pv_ = smooth((num[:, 0] / den.clamp(min=1e-9)).cpu().numpy(), 2)
            sv = sv * np.clip((pv_ - 0.5) / 0.3, 0, 1)
        sv = smooth(sv, 5)
        st = to_tex(torch.tensor(sv, dtype=torch.float32, device=dev)).clamp(0, 1)
        log(f"prior check: {int((st > 0.5).sum())} texels overruled")
        if o.get("debug"):
            T = lambda x: to_tex(torch.tensor(np.asarray(x, np.float32), device=dev))
            prior_dbg = {"dl": T(dl), "single": T(single.astype(np.float32)), "paint": T(pv_) if o["paint_guard"] else None}
        conf = conf * (1 - st)

    # --- generator agreement: photos dark/grey where the generator paints body colour -> generator wins
    if o["gen_agree"]:
        def hsl(c):
            mx, mn = c.max(1).values, c.min(1).values
            return (mx - mn) / mx.clamp(min=1e-4), 0.299 * c[:, 0] + 0.587 * c[:, 1] + 0.114 * c[:, 2]
        num, den = to_vert(torch.cat([base, proj], 1), ones)
        bp = smooth((num / den.clamp(min=1e-9)[:, None]).cpu().numpy(), o["agree_smooth"])
        bpt = torch.tensor(bp, dtype=torch.float32, device=dev)
        s_g, l_g = hsl(bpt[:, :3])
        s_p, l_p = hsl(bpt[:, 3:])
        # body colour = the generator's dominant saturated hue
        mx = bpt[:, :3].max(1).values
        r_, g_, b_ = bpt[:, 0], bpt[:, 1], bpt[:, 2]
        hue = torch.atan2(1.7320508 * (g_ - b_), 2 * r_ - g_ - b_)
        colourful = (s_g > 0.3) & (mx > 0.2)
        if int(colourful.sum()) > 500:
            dom = torch.angle(torch.polar(torch.ones_like(hue[colourful]), hue[colourful]).mean())
            dh = (torch.remainder(hue - dom + math.pi, 2 * math.pi) - math.pi).abs()
            paint = colourful & (dh < 0.35)
        else:
            paint = torch.zeros_like(colourful)
        dis = (((s_g - s_p) - o["agree_ds"]) / 0.2).clamp(0, 1) * (((l_g - l_p) - o["agree_dl"]) / 0.15).clamp(0, 1) * paint.float()
        dis_v = smooth(dis.cpu().numpy(), 2)
        dt = to_tex(torch.tensor(dis_v, dtype=torch.float32, device=dev)).clamp(0, 1)
        log(f"generator agreement: {int((dt > 0.5).sum())} texels handed back to the generator")
        if o.get("debug"):
            agree_dbg = {"ds": to_tex(s_g - s_p), "dl": to_tex(l_g - l_p), "paint": to_tex(paint.float()), "sg": to_tex(s_g), "hue_d": to_tex(dh) if int(colourful.sum()) > 500 else None}
        conf = conf * (1 - dt)
        agree_t = dt
    else:
        agree_t = None

    # --- hidden fill ---------------------------------------------------------------------------
    if o["fill"] == "transfer":
        base = _material_transfer(base, proj, conf, o["fill_k"], dev, log)
    num, den = to_vert(base, ones)
    base_v = smooth((num / den.clamp(min=1e-9)[:, None]).cpu().numpy(), o["bsmooth"])
    num, den = to_vert(proj, conf)
    pv = (num / den.clamp(min=1e-9)[:, None]).cpu().numpy()
    numc, denc = to_vert(conf[:, None], ones)
    cv = (numc[:, 0] / denc.clamp(min=1e-9)).cpu().numpy()
    cw = np.clip((cv - 0.3) / 0.5, 0, 1)
    if o["fill"] == "transfer":
        dmag = np.abs(pv - base_v).mean(1)
        cw = cw * np.exp(-(dmag / o["fill_robust"]) ** 2)
    mu = o["fill_decay"] if o["fill"] == "transfer" else 0.0
    lu = splu((sparse.diags(cw + 1e-6 + mu) + o["fill_lam"] * Lap).tocsc())
    off = np.stack([lu.solve(cw * (pv[:, c] - base_v[:, c])) for c in range(3)], 1)
    fill = to_tex(torch.tensor(np.clip(base_v + off, 0, 1), dtype=torch.float32, device=dev))
    final = (proj * conf[:, None] + fill * (1 - conf[:, None])).clamp(0, 1)
    log(f"coverage conf>0.5 {float((conf > 0.5).float().mean()):.3f}")

    # --- glass: one uniform dark tint wherever the views' glass masks land ---------------------
    glass_t = None
    if glass_masks is not None:
        gseen = g_wsum > 0.02
        gp = torch.where(gseen, g_acc / g_wsum.clamp(min=1e-6), torch.zeros_like(g_acc))
        num, den = to_vert(gp[:, None], gseen.float())
        gv = (num[:, 0] / den.clamp(min=1e-9)).cpu().numpy()
        # spread a little into texels no view sees (edge of the windscreen seen edge-on), keep seen ones
        sv = (den.cpu().numpy() > 1e-6).astype(np.float64)
        lu_g = splu((sparse.diags(sv + 1e-6 + o["glass_spread_decay"]) + 1.0 * Lap).tocsc())
        gv = lu_g.solve(sv * gv)
        glass_t = smoothstep(to_tex(torch.tensor(gv, dtype=torch.float32, device=dev)), 0.35, 0.65)
        if o["glass_tint"] >= 0:
            tint = torch.full((3,), float(o["glass_tint"]), device=dev)
        elif glass_pix:
            px = torch.cat(glass_pix, 0)
            lum = px @ torch.tensor([0.299, 0.587, 0.114], device=dev)
            dark = px[lum <= torch.quantile(lum[:200000] if len(lum) > 200000 else lum, 0.25)]
            tint = dark.median(0).values
        else:
            tint = torch.full((3,), 0.05, device=dev)
        tl = float(tint @ torch.tensor([0.299, 0.587, 0.114], device=dev))
        if tl > o["glass_max_lum"]:
            tint = tint * (o["glass_max_lum"] / tl)
        gcol = tint[None] * (1 - o["glass_detail"]) + final * o["glass_detail"]
        final = final * (1 - glass_t[:, None]) + gcol * glass_t[:, None]
        log(f"glass: {int((glass_t > 0.5).sum())} texels, tint {[round(float(x), 3) for x in tint]}")

    # --- atlas + island padding ----------------------------------------------------------------
    atlas = torch.zeros((A * A, 3), device=dev)
    atlas[tex] = final
    img = atlas.view(A, A, 3).cpu().numpy()
    mask = occ.view(A, A).cpu().numpy()
    idx = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
    img = img[idx[0], idx[1]]
    dbg = {"gains": gains.cpu().numpy(), "conf": conf, "label": label_t, "tex": tex, "occ": occ}
    if o.get("debug") and o["prior_check"]:
        dbg["prior"] = prior_dbg
    dbg["agree"] = agree_t
    if o.get("keep_views"):
        dbg["w_t"] = w_t
    if o.get("debug") and o["gen_agree"]:
        dbg["agree_dbg"] = agree_dbg
    if glass_t is not None:
        ga = torch.zeros(A * A, device=dev)
        ga[tex] = glass_t
        ga = ga.view(A, A).cpu().numpy()[idx[0], idx[1]]
        dbg["glass"] = torch.tensor(ga)
    return torch.tensor(img), dbg


def _seam_levelling(fw, face_lab, face_ok, vcols, nW, lam, f, b, tw, label_t, dev, log):
    """Per (vertex, label) colour offsets g: at every seam vertex the corrected colours of its labels
    agree; within a label region offsets vary smoothly (Waechter-style global levelling)."""
    nF = len(fw)
    labs = np.where(face_ok, face_lab, -1)
    # (vertex, label) pairs used by labelled faces
    pv = fw.reshape(-1)
    pl = np.repeat(labs, 3)
    keep = pl >= 0
    key = pv[keep].astype(np.int64) * 16 + pl[keep]
    uk = np.unique(key)
    nvar = len(uk)
    kv = uk // 16
    kl = uk % 16
    # smoothness: edges of each labelled face, same label
    fl = labs[labs >= 0]
    ff = fw[labs >= 0].astype(np.int64)
    ids = np.stack([np.searchsorted(uk, ff[:, k] * 16 + fl) for k in range(3)], 1)
    ea = np.r_[ids[:, 0], ids[:, 1], ids[:, 2]]
    eb = np.r_[ids[:, 1], ids[:, 2], ids[:, 0]]
    # seam constraints: vertices carrying >1 label
    order = np.argsort(kv, kind="stable")
    kvs = kv[order]
    starts = np.r_[0, np.nonzero(np.diff(kvs))[0] + 1]
    counts = np.diff(np.r_[starts, len(kvs)])
    sa, sb = [], []
    for s, c in zip(starts[counts > 1], counts[counts > 1]):
        grp = order[s:s + c]
        for i in range(c):
            for j in range(i + 1, c):
                sa.append(grp[i]); sb.append(grp[j])
    sa = np.array(sa, np.int64); sb = np.array(sb, np.int64)
    log(f"levelling: {nvar} vars, {len(sa)} seam pairs, {len(ea)} smooth edges")
    if len(sa) == 0:
        return torch.zeros((len(f), 3), device=dev)
    # colour of each (vertex,label) var = that view's colour at the vertex
    vc = np.stack(vcols, 0)                                           # (nV, nW, 3)
    cvar = vc[kl, kv]                                                 # (nvar, 3)
    # least squares: seam rows (g_a - g_b = c_b - c_a), smooth rows sqrt(lam)(g_a - g_b) = 0, tiny prior
    nS, nE = len(sa), len(ea)
    r = np.r_[np.arange(nS), np.arange(nS), nS + np.arange(nE), nS + np.arange(nE)]
    cidx = np.r_[sa, sb, ea, eb]
    sl = math.sqrt(lam)
    data = np.r_[np.ones(nS), -np.ones(nS), sl * np.ones(nE), -sl * np.ones(nE)]
    Am = sparse.coo_matrix((data, (r, cidx)), shape=(nS + nE, nvar)).tocsr()
    AtA = (Am.T @ Am + 1e-4 * sparse.eye(nvar)).tocsc()
    lu = splu(AtA)
    g = np.zeros((nvar, 3), np.float32)
    for c in range(3):
        rhs = np.r_[cvar[sb, c] - cvar[sa, c], np.zeros(nE)]
        g[:, c] = lu.solve(Am.T @ rhs)
    log(f"levelling offsets: mean |g| {np.abs(g).mean():.4f}, max {np.abs(g).max():.3f}")
    # texel offset: interpolate the face's own-label vars
    gt = torch.tensor(g, device=dev)
    tw_np = tw.cpu().numpy().astype(np.int64)
    lab_np = label_t.cpu().numpy().astype(np.int64)
    vid = np.stack([np.searchsorted(uk, tw_np[:, k] * 16 + lab_np) for k in range(3)], 1)
    found = np.stack([uk[np.clip(vid[:, k], 0, nvar - 1)] == tw_np[:, k] * 16 + lab_np for k in range(3)], 1)
    vid = torch.tensor(np.clip(vid, 0, nvar - 1), device=dev)
    fnd = torch.tensor(found, device=dev).float()
    bw = b * fnd
    return (gt[vid] * bw[..., None]).sum(1) / bw.sum(1, keepdim=True).clamp(min=1e-6)


def _material_transfer(base, proj, conf, k, dev, log):
    """Map the generated colours to how each material looks in the photos: cluster the generated
    texture (Lab-ish, k-means), learn per cluster the median shift generated -> projected on texels
    the photos see well, apply it (soft cluster assignment) everywhere."""
    x = base
    # simple perceptual-ish space: luminance + 2 chroma axes
    def feat(c):
        L = 0.299 * c[:, 0] + 0.587 * c[:, 1] + 0.114 * c[:, 2]
        return torch.stack([L, (c[:, 0] - L) * 1.5, (c[:, 2] - L) * 1.5], 1)
    fx = feat(x)
    g = torch.Generator(device="cpu").manual_seed(0)
    samp = fx[torch.randint(0, len(fx), (min(200000, len(fx)),), generator=g).to(dev)]
    cent = samp[torch.randperm(len(samp), generator=g)[:k].to(dev)].clone()
    for _ in range(25):
        lab = torch.cdist(samp, cent).argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                cent[j] = samp[m].mean(0)
    d = torch.cdist(fx, cent)
    hard = d.argmin(1)
    seen = conf > 0.7
    shift = torch.zeros((k, 3), device=dev)
    for j in range(k):
        m = seen & (hard == j)
        if int(m.sum()) > 2000:
            shift[j] = (proj[m] - base[m]).median(0).values
    sig = d.min(1, keepdim=True).values.clamp(min=1e-3) + 0.05
    pw = torch.softmax(-(d / sig) ** 2, 1)
    log("material transfer: " + " ".join(f"{int((hard == j).sum())}:{shift[j].mean().item():+.2f}" for j in range(k)))
    return (base + pw @ shift).clamp(0, 1)


def smooth_normals(vertices, faces, iterations):
    """Area-weighted vertex normals on the position-welded mesh, Laplacian-smoothed, returned per
    original (split) vertex. Same frame as the input vertices."""
    vertices = np.asarray(vertices, np.float64)
    faces = np.asarray(faces, np.int64)
    _, weld = np.unique(np.round(vertices, 5), axis=0, return_inverse=True)
    weld = weld.ravel()
    nW = int(weld.max()) + 1
    fw = weld[faces]
    P = np.zeros((nW, 3))
    P[weld] = vertices
    fn = np.cross(P[fw[:, 1]] - P[fw[:, 0]], P[fw[:, 2]] - P[fw[:, 0]])
    N = np.stack([sum(np.bincount(fw[:, k], fn[:, c], minlength=nW) for k in range(3)) for c in range(3)], 1)
    N /= np.linalg.norm(N, axis=1, keepdims=True).clip(1e-12)
    if iterations > 0:
        e = np.r_[fw[:, [0, 1]], fw[:, [1, 2]], fw[:, [2, 0]]]
        e = e[e[:, 0] != e[:, 1]]
        Adj = sparse.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(nW, nW)).tocsr()
        Adj.data[:] = 1
        Wn = (sparse.diags(1 / np.maximum(np.asarray(Adj.sum(1)).ravel(), 1)) @ Adj).tocsr()
        for _ in range(iterations):
            N = 0.5 * N + 0.5 * (Wn @ N)
            N /= np.linalg.norm(N, axis=1, keepdims=True).clip(1e-12)
    return N[weld].astype(np.float32)


# --- touch-up: repaint painted spots from the angle that sees them best --------------------------
# The user paints a mask on any input view. Masked texels are found on the mesh, grouped into
# regions, and each region is rendered (flat, from the finished texture) through the camera that
# sees it most squarely. An image edit redraws only the masked pixels of that render; the edit is
# projected back onto the texture. Unchanged render pixels map back onto their own texels, so only
# the painted spot changes.

TOUCHUP_RES = 3072       # full render resolution the crops are cut from
TOUCHUP_OUT = 1024       # crop size handed to the image edit
TOUCHUP_MAX = 4          # regions per run (one image-edit branch each)


def _texel_geometry(vertices_yup, faces, uvs, A, dev):
    """Occupied texels of an A x A atlas -> (texel index, rig-frame position, face normal, mesh)."""
    Fd = faces.to(dev).long()
    P0 = yup_to_rig(vertices_yup.to(dev).float())
    fid, bary = uv_raster(uvs.to(dev).float(), Fd, A, DEFAULTS["tol"])
    t_idx = torch.nonzero(fid >= 0).squeeze(1)
    f = fid[t_idx]
    P = (P0[Fd[f]] * bary[t_idx][..., None]).sum(1)
    fn = F.normalize(torch.linalg.cross(P0[Fd[:, 1]] - P0[Fd[:, 0]], P0[Fd[:, 2]] - P0[Fd[:, 0]]), dim=1)
    return t_idx, P, fn[f], P0, Fd


def _visible(cam, P, N, P0, Fd, res):
    """Texels the camera sees (z-tested against the whole mesh), their cos to the camera, pixel coords."""
    xy, z = cam.project(P)
    xw, zw = cam.project(P0)
    _, _, zb = screen_raster(xw, zw, Fd, res, res)
    pix = xy[:, 1].long().clamp(0, res - 1) * res + xy[:, 0].long().clamp(0, res - 1)
    cos = (N * F.normalize(cam.pos[None] - P, dim=1)).sum(1)
    c = cos.clamp(0.05, 1)
    eps = 0.002 + 0.8 * (z / cam.f) * ((1 - c ** 2).sqrt() / c).clamp(max=4.0)
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < res - 1) & (xy[:, 1] >= 0) & (xy[:, 1] < res - 1) & (z > 0)
    return (inside & (z <= zb[pix] + eps)).float(), cos, xy


def render_texture(vertices_yup, faces, uvs, tex, az, el, fov, res):
    """Flat (unlit) render of the textured mesh through a rig camera -> (res, res, C) float, hit (res, res)."""
    dev = _dev()
    Fd = faces.to(dev).long()
    cam = Cam(az, el, fov, res)
    xy, z = cam.project(yup_to_rig(vertices_yup.to(dev).float()))
    fid, bary, _ = screen_raster(xy, z, Fd, res, res)
    hit = fid >= 0
    out = torch.zeros((res * res, tex.shape[-1]), device=dev)
    uvp = (uvs.to(dev).float()[Fd[fid[hit]]] * bary[hit][..., None]).sum(1)
    out[hit] = F.grid_sample(tex.to(dev).float().permute(2, 0, 1)[None], (uvp * 2 - 1)[None, None], mode="bilinear",
                             align_corners=False)[0, :, 0].T
    return out.view(res, res, -1), hit.view(res, res)


def touchup_plan(vertices_yup, faces, uvs, tex, view_masks, fov, log=print):
    """view_masks: list of (mask (H, W) 0..1 painted on an input view, azimuth_deg of that view).
    Returns one dict per painted region: camera (az, el), crop (x0, y0, side) inside a TOUCHUP_RES
    render, the crop image (TOUCHUP_OUT^2 x 3), its edit mask, and `src` = the view that contributed
    most of the region."""
    dev = _dev()
    A = int(tex.shape[0])
    t_idx, P, N, P0, Fd = _texel_geometry(vertices_yup, faces, uvs, A, dev)
    tm = torch.zeros(len(t_idx), device=dev)
    src = torch.zeros(len(t_idx), dtype=torch.long, device=dev)
    for k, (mask, az) in enumerate(view_masks):
        mask = mask.to(dev).float()
        if float(mask.max()) < 0.5:
            continue
        H, W = mask.shape
        vis, _, xy = _visible(Cam(az, 0.0, fov, W), P, N, P0, Fd, W)
        mk = sample(mask[None, None], xy, W, H)[:, 0] * vis
        src = torch.where(mk > tm, torch.full_like(src, k), src)
        tm = torch.maximum(tm, mk)
    sel = torch.nonzero(tm > 0.5).squeeze(1)
    if len(sel) < 50:
        log("touch-up: nothing painted")
        return []
    # group painted texels into regions: connected cells of a coarse 3D grid, per painted view (each view
    # carries its own instruction, so paint from different views is never merged into one edit)
    lo, hi = P0.min(0).values, P0.max(0).values
    cell = ((P[sel] - lo) / (hi - lo).max() * 127).long().clamp(0, 127).cpu().numpy()
    src_sel = src[sel].cpu().numpy()
    comps = []
    for k in np.unique(src_sel):
        ck = cell[src_sel == k]
        grid = np.zeros((128, 128, 128), bool)
        grid[ck[:, 0], ck[:, 1], ck[:, 2]] = True
        lab, n = ndimage.label(ndimage.binary_dilation(grid, iterations=2), structure=np.ones((3, 3, 3)))
        tl = lab[ck[:, 0], ck[:, 1], ck[:, 2]]
        idx_k = np.nonzero(src_sel == k)[0]
        comps += [(int((tl == j).sum()), int(k), idx_k[tl == j]) for j in range(1, n + 1) if (tl == j).sum() >= 50]
    comps.sort(key=lambda c: -c[0])
    if len(comps) > TOUCHUP_MAX:
        log(f"touch-up: {len(comps)} painted regions, keeping the {TOUCHUP_MAX} largest")
    regions = []
    for _, k, members in comps[:TOUCHUP_MAX]:
        rs = sel[torch.tensor(members, device=dev)]
        best = None
        for el in (0.0, 10.0, 20.0, 30.0):
            for az in range(0, 360, 10):
                vis, cos, _ = _visible(Cam(float(az), el, fov, 512), P[rs], N[rs], P0, Fd, 512)
                score = float((vis * cos.clamp(min=0)).sum())
                if best is None or score > best[0]:
                    best = (score, float(az), el)
        _, az, el = best
        # the region's texels rendered as a texture = its mask in the render
        amask = torch.zeros(A * A, device=dev)
        amask[t_idx[rs]] = tm[rs]
        mimg, _ = render_texture(vertices_yup, faces, uvs, amask.view(A, A, 1), az, el, fov, TOUCHUP_RES)
        m = (mimg[..., 0] > 0.5).cpu().numpy()
        ys, xs = np.nonzero(m)
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        side = int(np.clip(max(xs.max() - xs.min(), ys.max() - ys.min()) * 2.2, 384, TOUCHUP_RES))
        x0 = int(np.clip(cx - side / 2, 0, TOUCHUP_RES - side))
        y0 = int(np.clip(cy - side / 2, 0, TOUCHUP_RES - side))
        img, _ = render_texture(vertices_yup, faces, uvs, tex, az, el, fov, TOUCHUP_RES)
        crop = img[y0:y0 + side, x0:x0 + side].permute(2, 0, 1)[None]
        crop = F.interpolate(crop, size=(TOUCHUP_OUT, TOUCHUP_OUT), mode="bilinear", align_corners=False, antialias=True)
        mc = torch.tensor(m[y0:y0 + side, x0:x0 + side], dtype=torch.float32)[None, None]
        mc = F.interpolate(mc, size=(TOUCHUP_OUT, TOUCHUP_OUT), mode="area")[0, 0].numpy() > 0.3
        mc = ndimage.binary_fill_holes(ndimage.binary_closing(mc, iterations=8))   # texels hidden from this angle leave holes
        mc = ndimage.binary_dilation(mc, iterations=6)          # margin so the edit blends into its surroundings
        mc = ndimage.gaussian_filter(mc.astype(np.float32), 3)
        regions.append(dict(az=az, el=el, fov=fov, x0=x0, y0=y0, side=side, src=k,
                            image=crop[0].permute(1, 2, 0).clamp(0, 1).cpu(), mask=torch.tensor(mc)))
        log(f"touch-up region {len(regions)}: {len(rs)} texels, camera az {az:g} el {el:g}, crop {side}px")
    return regions


def touchup_apply(vertices_yup, faces, uvs, tex, regions, edits, log=print):
    """Project each region's edited crop (TOUCHUP_OUT^2 x 3) back onto the texture, inside its mask.
    Where two regions' blend margins overlap, each texel takes the region it belongs to most."""
    dev = _dev()
    A = int(tex.shape[0])
    t_idx, P, N, P0, Fd = _texel_geometry(vertices_yup, faces, uvs, A, dev)
    flat = tex.to(dev).float().reshape(-1, 3).clone()
    ws, cols = [], []
    for r, edit in zip(regions, edits):
        if edit is None:
            continue
        vis, cos, xy = _visible(Cam(r["az"], r["el"], r["fov"], TOUCHUP_RES), P, N, P0, Fd, TOUCHUP_RES)
        lxy = (xy - torch.tensor([r["x0"], r["y0"]], device=dev, dtype=torch.float32)) * (TOUCHUP_OUT / r["side"])
        inside = (lxy[:, 0] >= 0) & (lxy[:, 0] < TOUCHUP_OUT - 1) & (lxy[:, 1] >= 0) & (lxy[:, 1] < TOUCHUP_OUT - 1)
        e = edit.to(dev).float()[..., :3].permute(2, 0, 1)[None]
        if e.shape[-2:] != (TOUCHUP_OUT, TOUCHUP_OUT):
            e = F.interpolate(e, size=(TOUCHUP_OUT, TOUCHUP_OUT), mode="bilinear", align_corners=False)
        w = (sample(r["mask"].to(dev).float()[None, None], lxy, TOUCHUP_OUT, TOUCHUP_OUT)[:, 0]
             * vis * smoothstep(cos, 0.15, 0.4) * inside.float()).clamp(0, 1)
        ws.append(w)
        cols.append(sample(e, lxy, TOUCHUP_OUT, TOUCHUP_OUT))
        log(f"touch-up applied at az {r['az']:g} el {r['el']:g}: {int((w > 0.05).sum())} texels")
    if ws:
        W = torch.stack(ws)
        wmax, best = W.max(0)
        col = torch.stack(cols)[best, torch.arange(len(best), device=dev)]
        flat[t_idx] = flat[t_idx] * (1 - wmax[:, None]) + col * wmax[:, None]
    img = flat.view(A, A, 3).cpu().numpy()
    occ = np.zeros(A * A, bool)
    occ[t_idx.cpu().numpy()] = True
    idx = ndimage.distance_transform_edt(~occ.reshape(A, A), return_distances=False, return_indices=True)
    return torch.tensor(img[idx[0], idx[1]])


def _robust_box(m):
    """Cut-out box ignoring thin protrusions (antennas, mirrors' stalks, stray pixels): rows / columns with < 3 % of the fullest one."""
    b = m > 0.5
    rows, cols = b.sum(1), b.sum(0)
    if rows.max() == 0:
        return None
    r = np.nonzero(rows > 0.03 * rows.max())[0]; c = np.nonzero(cols > 0.03 * cols.max())[0]
    return c.min(), r.min(), c.max() + 1, r.max() + 1


def auto_frame(images, masks, fill=0.91, size=1024, match_height=True, mirror_missing_side=True):
    """Re-frame four photos so the object is big and centred, as the four-view rig expects.

    images: four (H, W, 3) uint8 arrays (front, left, back, right), any size (the ORIGINAL photos, so no detail is lost).
    masks:  four (T, T) float arrays from a ResizeAndPadImage(T, T, black) + RemoveBackground pre-pass.
    match_height: every view is scaled so the object has the SAME HEIGHT (a car is equally tall in all four level
      views). Fixes views framed one by one, e.g. Two-Input Multiview output (car drawn ~30 % bigger in front/back than
      in the side view -> mesh 32-39 % too wide, 2026-10-12). Off = one zoom shared by all four (relative scale as given).
    mirror_missing_side: an empty side slot (no object found, e.g. the black placeholder of an unused optional input)
      is filled with the mirrored other side.
    The widest normalised cut-out fills `fill` of the frame; each view is centred on its own cut-out box; outside the
    photo is black. A view with no cut-out keeps its frame centre. Returns four (size, size, 3) uint8 arrays and the
    shared crop side in mask-frame pixels (None = nothing found, plain fit-and-pad)."""
    from PIL import Image
    images, masks = list(images), list(masks)
    boxes = []
    for im, m in zip(images, masks):
        # An unused slot is not black: ComfyUI loads the dim "Drop Image here" placeholder, whose grey icon the
        # background remover keeps as an object. No real photo is that dark (99.5th pct luminance ~35 vs >150).
        if mirror_missing_side and np.percentile(np.asarray(im, np.float32).mean(axis=2), 99.5) < 80:
            boxes.append(None); continue
        if match_height or mirror_missing_side:
            n = int((m > 0.5).sum())
            boxes.append(_robust_box(m) if n >= max(64, 0.002 * m.size) else None)
        else:   # both options off = exactly the release framing (plain cut-out box, >= 64 px)
            ys, xs = np.nonzero(m > 0.5)
            boxes.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if len(xs) >= 64 else None)
    if mirror_missing_side:
        for a, b in ((1, 3), (3, 1)):
            if boxes[a] is None and boxes[b] is not None:
                images[a] = np.ascontiguousarray(images[b][:, ::-1]); masks[a] = np.ascontiguousarray(masks[b][:, ::-1])
                boxes[a] = _robust_box(masks[a])
                logging.info("[MultiviewAutoFrame] %s side empty -> mirrored %s side", ("left", "right")[a == 3], ("left", "right")[b == 3])
    found = [b for b in boxes if b is not None]
    if not found:
        span, scale = None, [1.0] * 4
    else:
        hs = [b[3] - b[1] for b in found]; h_ref = float(np.median(hs))
        scale = [(h_ref / (b[3] - b[1]) if (match_height and b is not None) else 1.0) for b in boxes]   # mask px -> common units
        span = max(max(b[2] - b[0], b[3] - b[1]) * k for b, k in zip(boxes, scale) if b is not None) / fill
        if match_height:
            logging.info("[MultiviewAutoFrame] height match: object heights %s px -> scale %s", [None if b is None else b[3] - b[1] for b in boxes], [round(k, 3) for k in scale])
    out = []
    for img, m, b, k in zip(images, masks, boxes, scale):
        T = m.shape[0]; H, W = img.shape[:2]
        s = min(T / W, T / H); nw, nh = int(W * s), int(H * s)   # ResizeAndPadImage geometry
        ox, oy = (T - nw) // 2, (T - nh) // 2; sx, sy = nw / W, nh / H
        sp = (span / k) if span else T                           # this view's crop side in its own mask pixels
        cx, cy = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) if b is not None else (T / 2, T / 2)
        x0, y0 = (cx - sp / 2 - ox) / sx, (cy - sp / 2 - oy) / sy
        box = (round(x0), round(y0), round(x0 + sp / sx), round(y0 + sp / sy))
        out.append(np.asarray(Image.fromarray(img).crop(box).resize((size, size), Image.LANCZOS)))
    return out, span
