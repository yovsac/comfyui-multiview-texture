# Made by Yovan Vujacic
"""Panel UVs for generated car meshes (no Blender): replaces the native UnwrapMesh in the multi-view graph.

1. visible shell: faces any of 42 cameras around the object sees (+4 rings of neighbours). Hidden faces (the inside
   wall of the remeshed shell, ~70% of the faces) all share one parked texel instead of three quarters of the atlas.
2. panels: the shell split by smoothed surface direction (18 directions, so no panel bends more than ~45 degrees),
   cleaned by majority vote, small pieces merged whole into their main neighbour.
3. per panel: cut non-manifold edges, carve out small tunnels (mesh-generator noise), flatten (mean-value Tutte start
   -> SLIM symmetric Dirichlet), accept only with no flips, no overlap and mean area stretch <= 1.25 (PartUV rule),
   else split by direction and retry; tiny leftovers on their best-fit plane.
4. pack with xatlas (MIT) as a UV mesh, so relative panel sizes are kept (underside at reduced density).
"""
import time
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla, torch
from scipy.sparse.csgraph import connected_components

TAU = 1.25          # PartUV: accepted panel = mean area distortion <= 1.25
OVERLAP = 0.002     # share of a panel's raster cells that may be covered twice (edge rounding)
REFACTOR = 8        # SLIM: fresh factorisation every N rounds, preconditioned CG in between
CAMS = [(az, el) for el in (-35, -12, 8, 30, 55) for az in range(0, 360, 45)] + [(0, 89), (180, -60)]
AXES = ([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        + [[a, b, 0] for a in (1, -1) for b in (1, -1)] + [[a, 0, b] for a in (1, -1) for b in (1, -1)]
        + [[0, a, b] for a in (1, -1) for b in (1, -1)])                      # y-up: AXES[3] = straight down = underside
UNDERSIDE = 3


# ---------------------------------------------------------------------------------------------- shell + panels
def visible_shell(Vw, Fw, fov=20.0, res=1536, rings=4):
    """face mask: seen by any of CAMS (rig cameras, same framing as the projection) plus `rings` of neighbours"""
    from . import core
    dev = core._dev()
    P = core.yup_to_rig(torch.tensor(Vw, dtype=torch.float32, device=dev)); Fd = torch.tensor(Fw, dtype=torch.long, device=dev)
    seen = torch.zeros(len(Fw), dtype=torch.bool, device=dev)
    for az, el in CAMS:
        cam = core.Cam(float(az), float(el), fov, res); xy, z = cam.project(P)
        fid, _, _ = core.screen_raster(xy, z, Fd, res, res); seen[fid[fid >= 0]] = True
    s = seen.cpu().numpy()
    Inc = sp.csr_matrix((np.ones(Fw.size, np.float32), (np.repeat(np.arange(len(Fw)), 3), Fw.ravel())), shape=(len(Fw), len(Vw)))
    for _ in range(rings):
        s = (Inc @ (Inc.T @ s.astype(np.float32))) > 0
    return s


def smoothed_normals(Vw, Fs, Adj, iters=60):
    p = Vw[Fs].astype(np.float64); n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    for _ in range(iters):
        n = n + Adj @ n; n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return n


def panels(Fs, n, Adj, vote=8, min_faces=400):
    """panel id per shell face (+ direction label) from smoothed normals n"""
    AX = np.array(AXES, np.float64); AX /= np.linalg.norm(AX, axis=1, keepdims=True); K = len(AX)
    lab = np.argmax(n @ AX.T, 1)
    for _ in range(vote):                                # majority vote with neighbours (self counts double)
        H = Adj @ np.eye(K)[lab] + 2 * np.eye(K)[lab]
        lab = np.argmax(H, 1)
    r, c = Adj.nonzero()
    def pieces(lab):
        keep = lab[r] == lab[c]
        return connected_components(sp.coo_matrix((np.ones(keep.sum()), (r[keep], c[keep])), shape=Adj.shape), directed=False)
    for _ in range(50):                                  # merge small pieces whole into the neighbour they share most edges with
        nc, comp = pieces(lab); size = np.bincount(comp)
        cr, cc_ = comp[r], comp[c]; x = cr != cc_
        S = sp.coo_matrix((np.ones(x.sum()), (cr[x], cc_[x])), shape=(nc, nc)).tocsr() @ sp.diags(1.0 + 1e6 * (size >= min_faces))
        small = np.nonzero((size < min_faces) & (np.asarray(S.sum(1)).ravel() > 0))[0]
        if len(small) == 0:
            break
        small = small[np.argsort(size[small])][: max(1, len(small) // 2)]
        tgt = np.asarray(S[small].argmax(1)).ravel()
        clab = np.zeros(nc, np.int64); clab[comp] = lab
        remap = clab.copy(); remap[small] = clab[tgt]; lab = remap[comp]
    nc, comp = pieces(lab)
    return comp, lab


# ---------------------------------------------------------------------------------------------- flattening (tested in uvtest/uvflat.py)
def tri_frames(P, F, nV):
    """P (|F|,3,3) rest positions of each face's corners -> area and gradient operators Dx, Dy (|F| x nV)
    in each triangle's own 2D frame"""
    p0, p1, p2 = P[:, 0], P[:, 1], P[:, 2]
    e1, e2 = p1 - p0, p2 - p0
    n = np.cross(e1, e2); a2 = np.linalg.norm(n, axis=1); A = a2 / 2
    x = e1 / np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-20)
    y = np.cross(n / np.maximum(a2[:, None], 1e-20), x)
    q = np.stack([np.zeros((len(F), 2)), np.c_[(e1 * x).sum(1), (e1 * y).sum(1)], np.c_[(e2 * x).sum(1), (e2 * y).sum(1)]], 1)
    g = np.zeros((len(F), 3, 2))                         # grad of barycentric i = perp(q[i+2] - q[i+1]) / 2A
    for i in range(3):
        d = q[:, (i + 2) % 3] - q[:, (i + 1) % 3]
        g[:, i] = np.c_[-d[:, 1], d[:, 0]] / np.maximum(a2[:, None], 1e-20)
    r = np.repeat(np.arange(len(F)), 3)
    Dx = sp.csr_matrix((g[:, :, 0].ravel(), (r, F.ravel())), shape=(len(F), nV))
    Dy = sp.csr_matrix((g[:, :, 1].ravel(), (r, F.ravel())), shape=(len(F), nV))
    return A, Dx, Dy


def tri_grads(P):
    """P (|F|,3,3) rest corners -> area (|F|,), barycentric gradients (|F|,3,2) in each triangle's own frame"""
    p0, p1, p2 = P[:, 0], P[:, 1], P[:, 2]; e1, e2 = p1 - p0, p2 - p0
    n = np.cross(e1, e2); a2 = np.linalg.norm(n, axis=1)
    x = e1 / np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-20); y = np.cross(n / np.maximum(a2[:, None], 1e-20), x)
    q = np.stack([np.zeros((len(P), 2)), np.c_[(e1 * x).sum(1), (e1 * y).sum(1)], np.c_[(e2 * x).sum(1), (e2 * y).sum(1)]], 1)
    g = np.zeros((len(P), 3, 2))
    for i in range(3):
        d = q[:, (i + 2) % 3] - q[:, (i + 1) % 3]; g[:, i] = np.c_[-d[:, 1], d[:, 0]] / np.maximum(a2[:, None], 1e-20)
    return a2 / 2, g


def signed_area(UV, F):
    a, b, c = UV[F[:, 0]], UV[F[:, 1]], UV[F[:, 2]]
    return ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])) / 2


def boundary_loops(F):
    """ordered boundary loops (lists of vertex ids), longest first; None if the boundary is not simple"""
    E = np.c_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]].reshape(-1, 2)
    key = np.minimum(E[:, 0], E[:, 1]).astype(np.int64) * (F.max() + 1) + np.maximum(E[:, 0], E[:, 1])
    u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    if (cnt > 2).any():
        return None
    b = E[cnt[inv] == 1]                                 # directed boundary half-edges
    nxt = {}
    for s, t in b:
        if s in nxt:
            return None                                  # pinched vertex
        nxt[s] = t
    loops, seen = [], set()
    for s in nxt:
        if s in seen: continue
        L, v = [], s
        while v not in seen:
            seen.add(v); L.append(v); v = nxt.get(v)
            if v is None: return None
        loops.append(L)
    return sorted(loops, key=len, reverse=True)


def make_manifold(F):
    """cut at non-manifold edges (3+ faces, or two faces with clashing winding) and split pinched vertices:
    one vertex per fan of faces around it. Returns (F', orig) with orig[new vertex] = old vertex."""
    nF = len(F); corner = np.arange(3 * nF).reshape(-1, 3)
    E = np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1).reshape(-1, 2)            # directed, per corner i -> i+1
    C = np.stack([corner[:, [0, 1]], corner[:, [1, 2]], corner[:, [2, 0]]], 1).reshape(-1, 2)
    key = np.minimum(E[:, 0], E[:, 1]).astype(np.int64) * (F.max() + 1) + np.maximum(E[:, 0], E[:, 1])
    o = np.argsort(key, kind="stable"); ks = key[o]
    u, start, cnt = np.unique(ks, return_index=True, return_counts=True)
    two = start[cnt == 2]; h1, h2 = o[two], o[two + 1]
    ok = E[h1, 0] == E[h2, 1]                                                              # opposite directions = consistent
    h1, h2 = h1[ok], h2[ok]
    # half-edge h1 = (a->b) with corners (ca, cb); h2 = (b->a) with corners (cb', ca'): join ca~ca', cb~cb'
    a = np.r_[C[h1, 0], C[h1, 1]]; b = np.r_[C[h2, 1], C[h2, 0]]
    G = sp.coo_matrix((np.ones(len(a)), (a, b)), shape=(3 * nF, 3 * nF))
    k, lab = connected_components(G, directed=False)
    orig = np.zeros(k, np.int64); orig[lab] = F.ravel()
    return lab.reshape(-1, 3), orig


def genus(F):
    """genus of a (manifold-cut) face set; -1 if it has no boundary (closed)"""
    loops = boundary_loops(F)
    if loops is None or not loops: return -1
    E = np.sort(np.c_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]].reshape(-1, 2), 1)
    nE = len(np.unique(E[:, 0].astype(np.int64) * (F.max() + 1) + E[:, 1]))
    return (2 - (len(np.unique(F)) - nE + len(F)) - len(loops)) // 2


def carve_handles(V, F, patch=150, rounds=3, seed=0):
    """small tunnels (mesh-generator noise) make a panel unflattenable. Cover the panel with ~patch-face
    patches (k-means on face centres, two offset tilings per round, growing size); patches that contain a
    tunnel on their own are marked. Returns a face mask of the carved faces (empty if none found)."""
    from scipy.cluster.vq import kmeans2
    cen = V[F].mean(1); mark = np.zeros(len(F), bool); rng = np.random.default_rng(seed)
    for r in range(rounds):
        size = patch * (2 ** r)
        for t in range(2):
            k = max(2, len(F) // size)
            _, lab = kmeans2(cen, k, minit="++", seed=int(rng.integers(1 << 30)))
            for j in np.unique(lab):
                fi = np.nonzero(lab == j)[0]
                Fm, orig = make_manifold(F[fi])
                if genus(Fm) > 0:
                    mark[fi] = True
        rest = np.nonzero(~mark)[0]
        if len(rest) == 0: break
        Fm, orig = make_manifold(F[rest])
        kk, lab = connected_components(face_adjacency(Fm), directed=False)
        if all(genus(Fm[lab == j]) <= 0 or (lab == j).sum() < 30 for j in range(kk)):
            break
    return mark


def tutte(V, F, loops):
    """flip-free starting layout: outer loop on an ellipse with the panel's own proportions (by arc length),
    holes fanned to their 3D centre, interior by mean-value weights (Floater). Returns (UV of all vertices
    incl. hole centres, faces incl. hole fans)."""
    nV = len(V); Fe = [F]; cen = []
    for L in loops[1:]:
        c = nV + len(cen); cen.append(V[L].mean(0))
        Fe.append(np.c_[np.full(len(L), c), np.roll(L, -1), L])
    Fa = np.concatenate(Fe); Vx = np.concatenate([V, np.array(cen).reshape(-1, 3)]).astype(np.float64); n = len(Vx)
    # mean-value weights: w_ij += tan(angle at i / 2) / |ij| for both triangle angles at i adjacent to ij
    P = Vx[Fa]; I, J, W = [], [], []
    for k in range(3):
        i, j, l = Fa[:, k], Fa[:, (k + 1) % 3], Fa[:, (k + 2) % 3]
        eij, eil = Vx[j] - Vx[i], Vx[l] - Vx[i]
        dij, dil = np.linalg.norm(eij, axis=1), np.linalg.norm(eil, axis=1)
        ang = np.arccos(np.clip((eij * eil).sum(1) / np.maximum(dij * dil, 1e-30), -1, 1)); t = np.tan(ang / 2)
        I += [i, i]; J += [j, l]; W += [t / np.maximum(dij, 1e-30), t / np.maximum(dil, 1e-30)]
    Wm = sp.coo_matrix((np.concatenate(W), (np.concatenate(I), np.concatenate(J))), shape=(n, n)).tocsr()
    L = sp.diags(np.asarray(Wm.sum(1)).ravel()) - Wm
    # outer loop -> ellipse with the panel's aspect (PCA of the loop), arc-length spaced, starting on the long axis
    out = np.array(loops[0]); q = Vx[out] - Vx[out].mean(0)
    ax = np.linalg.svd(q, full_matrices=False)[2]; pr = q @ ax[:2].T
    ra, rb = np.ptp(pr[:, 0]) / 2, max(np.ptp(pr[:, 1]) / 2, 1e-9)
    st = int(np.argmax(pr[:, 0])); out = np.roll(out, -st)
    seg = np.linalg.norm(Vx[np.roll(out, -1)] - Vx[out], axis=1); s = np.r_[0, np.cumsum(seg)[:-1]] / max(seg.sum(), 1e-20)
    th = np.linspace(0, 2 * np.pi, 2049); E = np.c_[ra * np.cos(th), rb * np.sin(th)]
    el = np.r_[0, np.cumsum(np.linalg.norm(np.diff(E, axis=0), axis=1))]; el /= el[-1]
    t = np.interp(s, el, th); UVb = np.c_[ra * np.cos(t), rb * np.sin(t)]
    # orientation: match the loop's own turning direction so faces come out positive
    free = np.setdiff1d(np.arange(n), out)
    UV = np.zeros((n, 2)); UV[out] = UVb
    if len(free):
        UV[free] = sla.splu(L[free][:, free].tocsc()).solve(-L[free][:, out] @ UVb)
    return UV, Fa


def lscm(P, F, nV, pins):
    """least-squares conformal map (Cauchy-Riemann residuals per face, area weighted), two pinned vertices"""
    A, Dx, Dy = tri_frames(P, F, nV); sA = sp.diags(np.sqrt(np.maximum(A, 1e-30)))
    M = sp.vstack([sp.hstack([sA @ Dx, -(sA @ Dy)]), sp.hstack([sA @ Dy, sA @ Dx])]).tocsc()   # u_x - v_y, u_y + v_x
    fixed = np.r_[pins, np.asarray(pins) + nV]; val = np.array([0.0, 1.0, 0.0, 0.0])
    free = np.setdiff1d(np.arange(2 * nV), fixed)
    Mf, Mc = M[:, free], M[:, fixed]
    x = np.zeros(2 * nV); x[fixed] = val
    x[free] = sla.spsolve((Mf.T @ Mf).tocsc(), -(Mf.T @ (Mc @ val)))
    return np.c_[x[:nV], x[nV:]]


def _svd2(J):
    """closed-form SVD of (F,2,2): J = R(phi) diag(s1, s2) R(theta); s2 < 0 if det < 0"""
    a, b, c, d = J[:, 0, 0], J[:, 0, 1], J[:, 1, 0], J[:, 1, 1]
    E, Fq, G, H = (a + d) / 2, (a - d) / 2, (c + b) / 2, (c - b) / 2
    Q, R = np.hypot(E, H), np.hypot(Fq, G)
    a1, a2 = np.arctan2(G, Fq), np.arctan2(H, E)
    return Q + R, Q - R, (a2 + a1) / 2, (a2 - a1) / 2, a2


def slim_fast(P, F, UV, wface, iters=150, tol=1e-4):
    """same SLIM as slim(), but: energy and 2x2 SVD in closed form, system assembled from a fixed
    per-triangle pattern (no sparse re-stacking), symmetric positive definite solve"""
    nV = len(UV); A, g = tri_grads(P)
    keep = A > 1e-14 * A.max(); A = A[keep] * wface[keep]; F = F[keep]; gx, gy = g[keep, :, 0], g[keep, :, 1]
    K = (gx[:, :, None] * gx[:, None, :] + gy[:, :, None] * gy[:, None, :]).reshape(-1, 9)   # stiffness per face
    rr = np.repeat(F, 3, axis=1).ravel(); cc = np.tile(F, (1, 3)).ravel()
    rows = np.r_[rr, rr, rr + nV, rr + nV, np.arange(2 * nV)]; cols = np.r_[cc, cc + nV, cc, cc + nV, np.arange(2 * nV)]
    key = cols.astype(np.int64) * (2 * nV) + rows; order = np.argsort(key, kind="stable")
    uk, inv = np.unique(key[order], return_inverse=True)
    indptr = np.searchsorted(uk // (2 * nV), np.arange(2 * nV + 1)); indices = (uk % (2 * nV)).astype(np.int32)
    def jac(x):
        u, v = x[F, 0], x[F, 1]
        return np.stack([np.stack([(u * gx).sum(1), (u * gy).sum(1)], 1), np.stack([(v * gx).sum(1), (v * gy).sum(1)], 1)], 1)
    def energy(J):
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        if (det <= 1e-24).any(): return np.inf
        n2 = (J ** 2).sum((1, 2)); return float((A * n2 * (1 + 1 / det ** 2)).sum())
    x = UV.astype(np.float64).copy(); J = jac(x); E = energy(J); lu = None
    if not np.isfinite(E): return None
    for it in range(iters):
        s1, s2, phi, th, rot = _svd2(J)
        w1 = np.sqrt((s1 ** 2 + 1) * (s1 + 1) / s1 ** 3); w2 = np.sqrt((s2 ** 2 + 1) * (s2 + 1) / s2 ** 3)
        cp, sp_ = np.cos(phi), np.sin(phi)                                  # U = R(phi); W^2 = U diag(w^2) U^T
        q1, q2 = w1 ** 2, w2 ** 2
        a = A * (cp * cp * q1 + sp_ * sp_ * q2); b = A * (cp * sp_ * (q1 - q2)); c = A * (sp_ * sp_ * q1 + cp * cp * q2)
        cr, sr = np.cos(rot), np.sin(rot)                                    # closest rotation R
        # (W^2 R): rows of W^2 = [[a, b], [b, c]] / A  times R = [[cr, -sr], [sr, cr]]
        t00, t01 = a * cr + b * sr, -a * sr + b * cr; t10, t11 = b * cr + c * sr, -b * sr + c * cr
        ru = np.bincount(F.ravel(), weights=(t00[:, None] * gx + t01[:, None] * gy).ravel(), minlength=nV)
        rv = np.bincount(F.ravel(), weights=(t10[:, None] * gx + t11[:, None] * gy).ravel(), minlength=nV)
        diag_scale = (np.bincount(F.ravel(), weights=np.repeat(a + c, 3), minlength=nV)).mean() * 1e-6 + 1e-30
        data = np.r_[(K * a[:, None]).ravel(), (K * b[:, None]).ravel(), (K * b[:, None]).ravel(), (K * c[:, None]).ravel(), np.full(2 * nV, diag_scale)]
        Mcsc = sp.csc_matrix((np.bincount(inv, weights=data[order]), indices, indptr), shape=(2 * nV, 2 * nV))
        rhs = np.r_[ru, rv] + diag_scale * np.r_[x[:, 0], x[:, 1]]
        if lu is None or it % REFACTOR == 0:                         # fresh factorisation every few rounds
            lu = sla.splu(Mcsc, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
            sol = lu.solve(rhs)
        else:                                                        # in between: CG preconditioned by the last one
            sol, _ = sla.cg(Mcsc, rhs, x0=np.r_[x[:, 0], x[:, 1]], rtol=1e-7, maxiter=30,
                            M=sla.LinearOperator(Mcsc.shape, matvec=lu.solve))
        d = np.c_[sol[:nV], sol[nV:]] - x
        a1_ = x[F[:, 1]] - x[F[:, 0]]; a2_ = x[F[:, 2]] - x[F[:, 0]]; b1 = d[F[:, 1]] - d[F[:, 0]]; b2 = d[F[:, 2]] - d[F[:, 0]]
        crs = lambda p, q: p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]
        qa, qb, qc = crs(b1, b2), crs(a1_, b2) + crs(b1, a2_), crs(a1_, a2_)
        with np.errstate(all="ignore"):
            disc = qb ** 2 - 4 * qa * qc; sq = np.sqrt(np.maximum(disc, 0))
            r1 = np.where(np.abs(qa) > 1e-30, (-qb - sq) / (2 * qa), -qc / qb); r2 = np.where(np.abs(qa) > 1e-30, (-qb + sq) / (2 * qa), np.inf)
            roots = np.r_[np.where((disc >= 0) & (r1 > 0), r1, np.inf), np.where((disc >= 0) & (r2 > 0), r2, np.inf)]
        t = min(1.0, 0.8 * float(roots.min())) if len(roots) else 1.0; En = np.inf
        for _ in range(20):
            Jn = jac(x + t * d); En = energy(Jn)
            if En <= E: break
            t *= 0.5
        if not En <= E: break
        x = x + t * d; J = Jn
        if (E - En) / max(E, 1e-20) < tol: break
        E = En
    return x


def flatten_chart(V, F):
    loops = boundary_loops(F)
    if not loops: return None
    nV, nE, nF = len(np.unique(F)), None, len(F)
    E = np.c_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]].reshape(-1, 2); E.sort(1)
    nE = len(np.unique(E[:, 0].astype(np.int64) * (F.max() + 1) + E[:, 1]))
    if nV - nE + nF != 2 - len(loops):                   # genus > 0 (handles): cannot flatten in one piece
        return None
    UVt, Fa = tutte(V, F, loops); nV = len(V)
    # hole caps (as Blender's "fill holes"): each hole's virtual centre sits at the 3D centre of its loop, so
    # the cap triangles have a real 3D rest shape and move with the panel; they keep holes from folding over
    Vx = np.concatenate([V, np.array([V[L].mean(0) for L in loops[1:]]).reshape(-1, 3)])
    P = Vx[Fa]; w = np.ones(len(Fa))
    A3 = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum() / 2
    out = np.array(loops[0]); s0 = out[np.argmax(np.linalg.norm(Vx[out] - Vx[out[0]], axis=1))]
    s1 = out[np.argmax(np.linalg.norm(Vx[out] - Vx[s0], axis=1))]
    UVa = None
    for init in ("lscm", "tutte"):                       # conformal start is far closer; Tutte is always flip-free
        U0 = lscm(P, Fa, len(Vx), [s0, s1]) if init == "lscm" else UVt.copy()
        sa = signed_area(U0, Fa)
        if (sa < 0).sum() > (sa > 0).sum(): U0[:, 1] *= -1; sa = -sa
        if (sa <= 0).any() or not np.isfinite(U0).all(): continue
        U0 *= np.sqrt(A3 / sa[:len(F)].sum()); UVa = U0; break
    if UVa is None: return None
    UVa = slim_fast(P, Fa, UVa, w)
    return None if UVa is None else UVa[:nV]


def raster_overlap(UV, F, cells=512):
    """share of raster cells covered by 2+ triangles (UV normalised to its bbox)"""
    lo = UV.min(0); s = cells / max((UV.max(0) - lo).max(), 1e-20); T = (UV - lo)[F] * s
    blo = np.floor(T.min(1)).astype(np.int64).clip(0, cells - 1); bhi = np.ceil(T.max(1)).astype(np.int64).clip(0, cells - 1)
    size = (bhi - blo + 1).max(1); prev = 0; I = []
    for K in (2, 4, 8, 16, 32, 64, 128, 256, 512):
        sel = np.nonzero((size <= K) & (size > prev))[0]; prev = K
        for ch in (np.array_split(sel, max(1, len(sel) * K * K // 2_000_000)) if len(sel) else []):
            ox, oy = np.meshgrid(np.arange(K), np.arange(K)); px = blo[ch, 0, None] + ox.reshape(1, -1); py = blo[ch, 1, None] + oy.reshape(1, -1)
            a, b, c = T[ch, 0], T[ch, 1], T[ch, 2]; v0, v1 = b - a, c - a
            d = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]; good = np.abs(d) > 1e-12; d = np.where(good, d, 1.0)
            wx, wy = px + 0.5 - a[:, 0, None], py + 0.5 - a[:, 1, None]
            l1 = (wx * v1[:, 1, None] - v1[:, 0, None] * wy) / d[:, None]; l2 = (v0[:, 0, None] * wy - wx * v0[:, 1, None]) / d[:, None]
            ins = (1 - l1 - l2 > 1e-9) & (l1 > 1e-9) & (l2 > 1e-9) & good[:, None] & (px < cells) & (py < cells)
            I.append((py * cells + px)[ins])
    if not I: return 0.0
    cnt = np.bincount(np.concatenate(I), minlength=cells * cells)
    return float((cnt > 1).sum()) / max(float((cnt > 0).sum()), 1.0)


def distortion(V, F, UV):
    a3 = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1) / 2
    a2 = np.abs(signed_area(UV, F)); r = (a2 / max(a2.sum(), 1e-20)) / np.maximum(a3 / max(a3.sum(), 1e-20), 1e-20)
    r = np.maximum(r, 1 / np.maximum(r, 1e-12))
    return float((r * a3).sum() / max(a3.sum(), 1e-20))


def accept(V, F, UV):
    if UV is None or not np.isfinite(UV).all(): return False, "flatten failed"
    sa = signed_area(UV, F)
    if (sa <= 0).any(): return False, "flips"
    ov = raster_overlap(UV, F, cells=int(np.clip(np.sqrt(len(F)) * 4, 64, 1024)))
    if ov > OVERLAP: return False, f"overlap {ov:.3f}"
    dist = distortion(V, F, UV)
    if dist > TAU: return False, f"distortion {dist:.2f}"
    return True, f"ok dist {dist:.2f}"


def planar(V, F):
    """last resort for tiny pieces: project on the best-fit plane"""
    c = V[F].reshape(-1, 3); c = c - c.mean(0)
    ax = np.linalg.svd(c[:: max(1, len(c) // 3000)], full_matrices=False)[2]
    return np.c_[V @ ax[0], V @ ax[1]]


def split2(V, F, fn):
    """two groups: by smoothed normal (2-means); if the normals barely differ, by position along the long axis"""
    cen = V[F].mean(1)
    spread = np.degrees(np.arccos(np.clip(fn @ (fn.mean(0) / max(np.linalg.norm(fn.mean(0)), 1e-12)), -1, 1)))
    if np.percentile(spread, 90) > 15:
        c0 = fn[np.argmax(spread)]; c1 = fn.mean(0)
        for _ in range(10):
            g = (fn @ c0) > (fn @ c1)
            if g.all() or (~g).all(): break
            c0, c1 = fn[g].mean(0), fn[~g].mean(0)
        if 0 < g.sum() < len(g): return g
    x = cen - cen.mean(0); ax = np.linalg.svd(x[:: max(1, len(x) // 5000)], full_matrices=False)[2][0]
    p = x @ ax; return p > np.median(p)


def face_adjacency(F):
    E = np.c_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]].reshape(-1, 2); E.sort(1)
    fe = np.repeat(np.arange(len(F)), 3); key = E[:, 0].astype(np.int64) * (F.max() + 1) + E[:, 1]
    o = np.argsort(key, kind="stable"); k2 = key[o]; same = np.nonzero(k2[1:] == k2[:-1])[0]
    a, b = fe[o][same], fe[o][same + 1]
    return sp.coo_matrix((np.ones(2 * len(a)), (np.r_[a, b], np.r_[b, a])), shape=(len(F), len(F))).tocsr()


def _process(item, Vw, Fs, fn, depth_max, min_faces):
    """one work item -> ('done', idx, uv_corner, kind) or ('push', [items], reason)"""
    idx, depth, carved, flat_only = item
    Fl, orig = make_manifold(Fs[idx]); Vl = Vw[orig]
    k, lab = connected_components(face_adjacency(Fl), directed=False)
    if k > 1:                                            # the seam cuts separated it into several pieces
        return ("push", [(idx[lab == j], depth, carved, flat_only) for j in range(k)], None)
    if flat_only or len(idx) < min_faces or depth >= depth_max:
        UV = planar(Vl, Fl); sa = signed_area(UV, Fl)
        if len(idx) > 1 and (sa > 0).any() and (sa < 0).any():       # the flat projection folds: split by direction
            gs = split2(Vl, Fl, fn[idx])
            if gs.all() or not gs.any(): gs = np.arange(len(idx)) < len(idx) // 2
            return ("push", [(idx[gs], depth + 1, carved, True), (idx[~gs], depth + 1, carved, True)], None)
        a3 = np.linalg.norm(np.cross(Vl[Fl[:, 1]] - Vl[Fl[:, 0]], Vl[Fl[:, 2]] - Vl[Fl[:, 0]]), axis=1).sum() / 2
        if len(idx) > 1 and np.abs(sa).sum() < 0.25 * a3:          # projection squashes it: one piece per triangle
            return ("push", [(np.array([f]), depth + 1, carved, True) for f in idx], None)
        return ("done", idx, UV[Fl], "planar")
    if genus(Fl) > 0 and not carved:                     # small tunnels: lift them out as their own pieces
        m = carve_handles(Vl, Fl)
        if m.any() and not m.all():
            return ("push", [(idx[m], depth + 1, True, False), (idx[~m], depth, True, False)], "carved")
    UV = flatten_chart(Vl, Fl); ok, why = accept(Vl, Fl, UV)
    if ok:
        return ("done", idx, UV[Fl], "flattened")
    gsplit = split2(Vl, Fl, fn[idx])
    if gsplit.all() or not gsplit.any():                 # cannot split: flat projection
        return ("done", idx, planar(Vl, Fl)[Fl], "planar")
    return ("push", [(idx[gsplit], depth + 1, False, False), (idx[~gsplit], depth + 1, False, False)], "split:" + why.split()[0])


def unwrap_charts(Vw, Fs, chart, fn, depth_max=10, min_faces=12, log=print, workers=None):
    """Vw welded verts, Fs shell faces, chart id per face, fn smoothed face normals.
    Per piece: cut non-manifold seams -> carve small tunnels -> flatten (Tutte -> SLIM) -> accept, else split.
    Pieces are independent and run in parallel threads (numpy / SuperLU release the GIL).
    Returns per-face-corner UV (|Fs|,3,2) in 3D units (each chart its own frame) and final chart ids."""
    import os
    from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
    Vw = Vw.astype(np.float64)
    out = np.zeros((len(Fs), 3, 2)); final = np.zeros(len(Fs), np.int64); nid = 0
    st = dict(flattened=0, planar=0, carved=0, splits=0, reasons={})
    items = sorted([(np.nonzero(chart == c)[0], 0, False, False) for c in np.unique(chart)], key=lambda t: -len(t[0]))
    # at most 16: Windows' OpenBLAS corrupts the heap past 24 concurrent callers (96-thread servers crashed)
    workers = workers or max(1, min(16, (os.cpu_count() or 2) - 2))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        running = {ex.submit(_process, it, Vw, Fs, fn, depth_max, min_faces) for it in items if len(it[0])}
        while running:
            done, running = wait(running, return_when=FIRST_COMPLETED)
            for fut in done:
                r = fut.result()
                if r[0] == "done":
                    _, idx, uv, kind = r; out[idx] = uv; final[idx] = nid; nid += 1; st[kind] += 1
                else:
                    _, new, why = r
                    if why == "carved": st["carved"] += 1
                    elif why: st["splits"] += 1; k = why.split(":")[1]; st["reasons"][k] = st["reasons"].get(k, 0) + 1
                    running |= {ex.submit(_process, it, Vw, Fs, fn, depth_max, min_faces) for it in new if len(it[0])}
    log(f"charts: {nid} ({st['flattened']} flattened, {st['planar']} planar); {st['carved']} tunnel carvings, {st['splits']} splits {st['reasons']}")
    return out, final


def pack(uv_corner, chart, Vw, Fs, scale_per_chart=None, resolution=4096, padding=4, brute=False, log=print):
    """scale each chart so UV area = 3D area (times scale_per_chart[c] in area), then xatlas add_uv_mesh
    (keeps relative chart sizes). Returns per-corner UV in [0,1], uniform scale on both axes."""
    import xatlas
    uvc = uv_corner.copy()
    p = Vw[Fs]; a3 = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1) / 2
    t = uvc; a2 = np.abs((t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1]) - (t[:, 2, 0] - t[:, 0, 0]) * (t[:, 1, 1] - t[:, 0, 1])) / 2
    A2 = np.bincount(chart, weights=a2); A3 = np.bincount(chart, weights=a3)
    L3 = np.bincount(chart, weights=np.linalg.norm(p - np.roll(p, -1, 1), axis=2).sum(1))
    L2 = np.bincount(chart, weights=np.linalg.norm(t - np.roll(t, -1, 1), axis=2).sum(1))
    kl = L3 / np.maximum(L2, 1e-30)                                  # scale by edge length (robust for slivers)
    k = np.where(A2 > 1e-24, np.sqrt(A3 / np.maximum(A2, 1e-30)), kl)
    k = np.minimum(k, 2.0 * kl)                                      # a squashed sliver must not blow up past its real size
    if scale_per_chart is not None: k = k * np.sqrt(scale_per_chart)
    ids = np.unique(chart); ext = np.zeros(chart.max() + 1)
    for c in ids:                                                    # scale each chart about its own corner
        m = chart == c; cc = uvc[m].reshape(-1, 2); uvc[m] = (uvc[m] - cc.min(0)) * k[c]; ext[c] = uvc[m].max()
    uvc = np.nan_to_num(uvc); assert np.isfinite(ext[ids]).all()
    # lay the charts out on a grid first: xatlas treats charts that overlap in its INPUT as one chart
    cell = ext.max() * 1.05; cols = int(np.ceil(np.sqrt(len(ids))))
    for j, c in enumerate(ids):
        m = chart == c; uvc[m] += np.array([(j % cols) * cell, (j // cols) * cell])
    # one vertex per (chart, corner position) so islands are separate in the UV mesh
    key = np.c_[np.repeat(chart, 3), np.round(uvc.reshape(-1, 2) * 1e7).astype(np.int64)]
    u, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.ravel()
    first = np.zeros(len(u), np.int64); first[inv] = np.arange(len(inv))
    U = uvc.reshape(-1, 2)[first]; U = (U - U.min(0)).astype(np.float32)
    Fi = inv.reshape(-1, 3).astype(np.uint32)
    tu = U[Fi]; S = np.abs((tu[:, 1, 0]-tu[:, 0, 0])*(tu[:, 2, 1]-tu[:, 0, 1])-(tu[:, 2, 0]-tu[:, 0, 0])*(tu[:, 1, 1]-tu[:, 0, 1])).sum() / 2
    tpu = float(np.sqrt(0.75 * resolution ** 2 / S))                  # texels per UV unit for ~75% fill of one sheet
    for _ in range(30):                                              # shrink until everything fits one sheet
        at = xatlas.Atlas(); at.add_uv_mesh(U, Fi)
        po = xatlas.PackOptions(); po.resolution = resolution; po.texels_per_unit = tpu; po.padding = padding
        po.rotate_charts = True; po.bruteForce = brute; po.bilinear = True
        at.generate(pack_options=po)
        if at.atlas_count == 1: break
        tpu *= 0.97
    log(f'xatlas charts {at.chart_count} (input charts {len(ids)}), atlases {at.atlas_count}, {at.width}x{at.height}, texels/unit {tpu:.0f}')
    vm, idx, uvo = at[0]
    tex = uvo * np.array([at.width, at.height], np.float32) / resolution       # true texels of a resolution^2 sheet
    out = tex[idx].reshape(-1, 3, 2).copy()
    # xatlas skips triangles too small to rasterise; vertices used only by those keep their raw INPUT
    # coordinates (unplaced). Snap such triangles onto a placed corner (they cover no texel anyway).
    placed = ~np.all(np.isclose(uvo * np.array([at.width, at.height], np.float32), U[vm], rtol=0, atol=1e-4), axis=1)
    ok = (vm[idx] == Fi) & placed[idx]
    bad = ~ok.all(1)
    if bad.any():
        anyok = ok.any(1)
        first = np.argmax(ok, 1)
        fallback = np.zeros((chart.max() + 1, 2), np.float32)       # a valid placed point per chart
        good_f = np.nonzero(ok.all(1))[0]; fallback[chart[good_f]] = out[good_f, 0]
        pt = np.where(anyok[:, None], out[np.arange(len(out)), first], fallback[chart])
        out[bad] = pt[bad][:, None, :]
    log(f'xatlas: {int(bad.sum())} sub-texel triangles with unplaced corners snapped onto a placed corner')
    return out, (at.width, at.height, at.utilization)


# ---------------------------------------------------------------------------------------------- entry point
def panel_unwrap(vertices, faces, resolution=4096, padding=4, underside=0.25, fov=20.0, log=print):
    """vertices (N,3) y-up, faces (M,3). Returns (vmapping (N',) into the input vertices, faces' (M,3) int32,
    uvs (N',2) float32 in the native MESH convention: like UnwrapMesh, xatlas v flipped)."""
    t0 = time.time()
    V = np.asarray(vertices, np.float64); F = np.asarray(faces, np.int64)
    ext = float(np.ptp(V, 0).max()) or 1.0
    _, weld = np.unique(np.round(V / ext, 5), axis=0, return_inverse=True); weld = weld.ravel()
    Vw = np.zeros((weld.max() + 1, 3)); Vw[weld] = V; Fw = weld[F]
    shell = np.nonzero(visible_shell(Vw, Fw, fov))[0]; Fs = Fw[shell]
    Adj = face_adjacency(Fs); n = smoothed_normals(Vw, Fs, Adj)
    chart, lab = panels(Fs, n, Adj)
    log(f"visible shell {len(shell)} of {len(F)} faces, {chart.max() + 1} panels ({time.time() - t0:.0f}s)")
    uvc, final = unwrap_charts(Vw, Fs, chart, n, log=log)
    under = np.bincount(final, weights=(lab == UNDERSIDE).astype(float)) / np.maximum(np.bincount(final), 1) > 0.5
    uv, (w, h, util) = pack(uvc, final, Vw, Fs, scale_per_chart=np.where(under, underside, 1.0),
                            resolution=int(resolution), padding=int(padding), log=log)
    # hidden faces park on one texel in the free corner; panels keep [0, 0.99]
    corner = np.full((len(F), 3, 2), 0.995, np.float32); corner[shell] = uv * 0.99
    key = np.c_[F.ravel(), np.round(corner.reshape(-1, 2) * (1 << 22)).astype(np.int64)]
    u, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.ravel()
    first = np.zeros(len(u), np.int64); first[inv] = np.arange(len(inv))
    vmap = F.ravel()[first]; uvs = corner.reshape(-1, 2)[first].copy()
    uvs[:, 1] = 1.0 - uvs[:, 1]                          # same flip as the native UnwrapMesh
    log(f"panel UVs: {final.max() + 1} pieces, xatlas utilization {util:.2f}, {len(vmap)} verts ({time.time() - t0:.0f}s)")
    return vmap, inv.reshape(-1, 3).astype(np.int32), uvs.astype(np.float32)
