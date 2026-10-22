# Made by Yovan Vujacic
"""ComfyUI node: project the input views onto a UV-unwrapped mesh as its base colour.

Sits between BakeTextureFromVoxel and ApplyTextureToMesh in the Pixal3D multi-view graph:
the generated base colour is kept only where no input view sees the surface (recoloured per
material to match the photos), everything the views do see takes the photos' own pixels.
Cameras are the native Pixal3D four-view rig (front/left/back/right, level, given fov).
"""
import json
import logging
import torch
from . import core
try:  # the corner fix needs opencv (cv2); without it the rest of the pack still loads
    from . import corner
except Exception as _e:  # pragma: no cover
    corner = None
    logging.warning("[comfyui-multiview-texture] corner fix nodes not loaded (needs opencv-python): %r", _e)

try:  # same UV normalisation the native bake/apply nodes use, so texels line up exactly
    from comfy_extras.nodes_mesh_postprocess import _normalize_uvs_to_unit
except Exception:  # pragma: no cover - outside ComfyUI
    _normalize_uvs_to_unit = None


def _view_hwc(img):
    if img is None:
        return None
    v = img[0]
    if v.shape[-1] == 4:
        v = v[..., :3] * v[..., 3:4]
    return v.float()


def _mesh_arrays(mesh):
    """First mesh of a native MESH batch -> vertices, faces, UVs normalised like the bake/apply nodes."""
    v = mesh.vertices[0]
    f = mesh.faces[0]
    uv = mesh.uvs[0]
    if getattr(mesh, "vertex_counts", None) is not None:
        n_v, n_f = int(mesh.vertex_counts[0]), int(mesh.face_counts[0])
        v, f, uv = v[:n_v], f[:n_f], uv[:n_v]
    if _normalize_uvs_to_unit is not None:
        uv = torch.from_numpy(_normalize_uvs_to_unit(uv.detach().cpu().numpy()))
    return v, f, uv


def _glass_material(metallic, roughness, g, glass_roughness):
    """Glass (A,A) map -> metallic 0 and roughness glass_roughness under it (shared by the projection and the corner fix)."""
    def _resized(gm, ref):
        if gm.shape[0] == ref.shape[1] and gm.shape[1] == ref.shape[2]:
            return gm
        return torch.nn.functional.interpolate(gm[None, None], size=(ref.shape[1], ref.shape[2]), mode="bilinear",
                                               align_corners=False)[0, 0]
    gm = _resized(g, metallic)[None, ..., None]
    met = metallic.float() * (1 - gm)
    gr = _resized(g, roughness)[None, ..., None]
    rou = roughness.float() * (1 - gr) + float(glass_roughness) * gr
    return met, rou


class MultiviewProjectTexture:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH", {"tooltip": "The UV-unwrapped mesh fed to BakeTextureFromVoxel."}),
                "base_color": ("IMAGE", {"tooltip": "Base colour from BakeTextureFromVoxel; used where no view sees the surface."}),
                "fov": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 170.0, "step": 0.01,
                                  "tooltip": "Horizontal fov of the views, same as the multi-view conditioning."}),
                "mode": (["blend", "label"], {"default": "blend",
                                              "tooltip": "blend: weighted mix of the views (best for glossy paint). "
                                                         "label: one view per surface patch with levelled seams (sharper, matte objects)."}),
            },
            "optional": {
                "front": ("IMAGE",), "left": ("IMAGE",), "back": ("IMAGE",), "right": ("IMAGE",),
                "glass_front": ("MASK", {"tooltip": "Glass in the front view (e.g. SAM3_Detect 'window:8'). Glass becomes one dark tint."}),
                "glass_left": ("MASK",), "glass_back": ("MASK",), "glass_right": ("MASK",),
                "metallic": ("IMAGE", {"tooltip": "Metallic map from BakeTextureFromVoxel; glass is set to 0."}),
                "roughness": ("IMAGE", {"tooltip": "Roughness map from BakeTextureFromVoxel; glass is set to glass_roughness."}),
                "glass_roughness": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01,
                                              "tooltip": "0.3 = soft glass reflection. Lower gets mirror-like and shows the surface's small bumps."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "MASK", "MV_VIEW_WEIGHTS")
    RETURN_NAMES = ("base_color", "coverage", "metallic", "roughness", "glass", "view_weights")
    FUNCTION = "run"
    CATEGORY = "3d/texturing"

    def run(self, mesh, base_color, fov, mode, front=None, left=None, back=None, right=None,
            glass_front=None, glass_left=None, glass_back=None, glass_right=None,
            metallic=None, roughness=None, glass_roughness=0.3):
        views_in = {"front": front, "left": left, "back": back, "right": right}
        names = [n for n in core.VIEW_AZ if views_in[n] is not None]
        if not names:
            raise ValueError("MultiviewProjectTexture needs at least one view")
        first = core.VIEW_AZ[names[0]]                     # the conditioning poses the mesh to its first view
        views = [(_view_hwc(views_in[n]), core.VIEW_AZ[n] - first, 0.0) for n in names]
        v, f, uv = _mesh_arrays(mesh)
        glass_in = {"front": glass_front, "left": glass_left, "back": glass_back, "right": glass_right}
        glass = None
        if any(glass_in[n] is not None for n in names):
            glass = [glass_in[n][0].float() if glass_in[n] is not None else None for n in names]
        tex, dbg = core.texture_mesh(v, f, uv, base_color[0], views, fov, {"mode": mode, "keep_views": True},
                                     log=lambda *a: logging.info("[MultiviewProjectTexture] " + " ".join(str(x) for x in a)),
                                     glass_masks=glass)
        A = tex.shape[0]
        cov = torch.zeros(A * A)
        cov[dbg["tex"].cpu()] = dbg["conf"].cpu()
        cov = cov.view(1, A, A, 1).repeat(1, 1, 1, 3)
        g = dbg.get("glass", torch.zeros((A, A))).float()
        met = metallic if metallic is not None else torch.zeros((1, A, A, 3))
        rou = roughness if roughness is not None else torch.ones((1, A, A, 3))
        met, rou = _glass_material(met, rou, g, glass_roughness)
        # per-view texel weights for the corner fix, half precision (4 x 4096^2 = 134 MB)
        import numpy as np
        W = np.zeros((len(names), A * A), np.float16)
        T = dbg["tex"].cpu().numpy()
        for k in range(len(names)):
            W[k, T] = dbg["w_t"][k].cpu().numpy()
        weights = {"names": names, "w": W.reshape(len(names), A, A)}
        return (tex[None].float().clamp(0, 1), cov, met, rou, g[None], weights)


class MeshShadingForPhotoTexture:
    """Smooth shading normals (welded, Laplacian-smoothed: hides small lumps of the generated surface
    that otherwise catch the light) and optional metallic / roughness overrides for photo textures,
    which already carry the photos' lighting."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mesh": ("MESH",),
            "smoothing": ("INT", {"default": 20, "min": 0, "max": 200,
                                  "tooltip": "Neighbour-averaging passes on the shading normals. 0 = plain smooth normals."}),
            "metallic": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01,
                                   "tooltip": "-1 keeps the texture's metallic map."}),
            "roughness": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01,
                                    "tooltip": "-1 keeps the texture's roughness map."}),
        }}

    RETURN_TYPES = ("MESH",)
    FUNCTION = "run"
    CATEGORY = "3d/mesh"

    def run(self, mesh, smoothing, metallic, roughness):
        import copy
        import numpy as np
        out = copy.copy(mesh)
        normals = torch.zeros_like(mesh.vertices, dtype=torch.float32)
        for i in range(mesh.vertices.shape[0]):
            v, f = mesh.vertices[i], mesh.faces[i]
            if getattr(mesh, "vertex_counts", None) is not None:
                v, f = v[:int(mesh.vertex_counts[i])], f[:int(mesh.face_counts[i])]
            n = core.smooth_normals(v.detach().cpu().numpy(), f.detach().cpu().numpy(), smoothing)
            normals[i, :n.shape[0]] = torch.from_numpy(n)
        out.normals = normals.to(mesh.vertices)
        mat = dict(mesh.material) if isinstance(getattr(mesh, "material", None), dict) else {}
        if metallic >= 0:
            mat["metallic_factor"] = float(metallic)
        if roughness >= 0:
            mat["roughness_factor"] = float(roughness)
        out.material = mat or getattr(mesh, "material", None)
        return (out,)


_SLOTS = range(1, core.TOUCHUP_MAX + 1)
TOUCHUP_PROMPT = ("Repaint only the masked area so it looks right: clean, continuous details that match the rest of "
                  "the object. Keep the same colours and lighting. Keep everything else identical, same black background.")


class MultiviewTouchUpPrepare:
    """Finds what the user painted on the input views, groups it into up to four spots on the model (paint
    from different views is never merged) and renders each spot from the angle that sees it best. Feed
    crop_N / mask_N / prompt_N to an image edit and the edited image to MultiviewTouchUpApply. Nothing
    painted = no spots, and the edit branches never run. Views are the four-view rig (front at azimuth 0)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "base_color": ("IMAGE", {"tooltip": "The finished base colour (e.g. MultiviewProjectTexture)."}),
                "fov": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 170.0, "step": 0.01}),
            },
            "optional": {
                **{f"mask_{n}": ("MASK", {"tooltip": f"Painted on the {n} view, same framing as the view fed to the 3D step."})
                   for n in core.VIEW_AZ},
                **{f"prompt_{n}": ("STRING", {"multiline": True, "default": TOUCHUP_PROMPT,
                                              "tooltip": f"What the edit should do where the {n} view is painted."})
                   for n in core.VIEW_AZ},
            },
        }

    RETURN_TYPES = tuple(t for _ in _SLOTS for t in ("IMAGE", "MASK", "STRING")) + ("TOUCHUP_PLAN",)
    RETURN_NAMES = tuple(n for i in _SLOTS for n in (f"crop_{i}", f"mask_{i}", f"prompt_{i}")) + ("plan",)
    FUNCTION = "run"
    CATEGORY = "3d/texturing"

    def run(self, mesh, base_color, fov, **kw):
        used = [n for n in core.VIEW_AZ if kw.get(f"mask_{n}") is not None]
        v, f, uv = _mesh_arrays(mesh)
        regions = core.touchup_plan(v, f, uv, base_color[0], [(kw[f"mask_{n}"][0], core.VIEW_AZ[n]) for n in used], fov,
                                    log=lambda *a: logging.info("[MultiviewTouchUp] " + " ".join(str(x) for x in a)))
        S = core.TOUCHUP_OUT
        out = []
        for i in _SLOTS:
            if i <= len(regions):
                r = regions[i - 1]
                out += [r["image"][None].float(), r["mask"][None].float(), kw.get(f"prompt_{used[r['src']]}", TOUCHUP_PROMPT)]
            else:
                out += [torch.zeros((1, S, S, 3)), torch.zeros((1, S, S)), ""]
        plan = {"regions": [{k: r[k] for k in ("az", "el", "fov", "x0", "y0", "side", "mask")} for r in regions]}
        return tuple(out) + (plan,)


class MultiviewTouchUpApply:
    """Projects the edited spot renders back onto the base colour. Only spots that exist pull their
    edit input (lazy), so with nothing painted the image-edit model is never loaded."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"mesh": ("MESH",), "base_color": ("IMAGE",), "plan": ("TOUCHUP_PLAN",)},
            "optional": {f"edit_{i}": ("IMAGE", {"lazy": True}) for i in _SLOTS},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("base_color",)
    FUNCTION = "run"
    CATEGORY = "3d/texturing"

    def check_lazy_status(self, mesh, base_color, plan, **edits):
        return [f"edit_{i}" for i in _SLOTS if i <= len(plan["regions"]) and edits.get(f"edit_{i}") is None]

    def run(self, mesh, base_color, plan, **edits):
        if not plan["regions"]:
            return (base_color,)
        v, f, uv = _mesh_arrays(mesh)
        tex = core.touchup_apply(v, f, uv, base_color[0], plan["regions"],
                                 [edits[f"edit_{i}"][0] if edits.get(f"edit_{i}") is not None else None
                                  for i in range(1, len(plan["regions"]) + 1)],
                                 log=lambda *a: logging.info("[MultiviewTouchUp] " + " ".join(str(x) for x in a)))
        return (tex[None].float().clamp(0, 1),)


class MultiviewCornerPlan:
    """Corner fix, step 1. Where a lamp wraps a corner of the car, the two views that meet there each paint part of it and
    the parts don't line up. Measures the rear lines and warps the back view when both rear corners show the same step,
    then finds lamps BOTH adjacent views draw across the hand-over (>= 108 cm2 in the band) and renders each such corner
    as 3 crops for an image edit (crops, a list: 0 items = nothing to redraw, the edit never runs). See corner.py."""

    @classmethod
    def INPUT_TYPES(cls):
        views = {n: ("IMAGE", {"tooltip": f"The {n} view fed to MultiviewProjectTexture."}) for n in core.VIEW_AZ}
        glass = {f"glass_{n}": ("MASK", {"tooltip": f"Glass in the {n} view (the mask fed to MultiviewProjectTexture)."})
                 for n in core.VIEW_AZ}
        lamps = {f"lamp_{n}": ("MASK", {"tooltip": f"Lights in the {n} view: SAM3_Detect 'headlight, tail light, brake light:8', threshold 0.45."})
                 for n in core.VIEW_AZ}
        return {"required": {
            "mesh": ("MESH", {"tooltip": "The UV-unwrapped mesh fed to MultiviewProjectTexture."}),
            "base_color": ("IMAGE", {"tooltip": "Generated base colour (BakeTextureFromVoxel), the fill where no view sees the surface."}),
            "projected": ("IMAGE", {"tooltip": "MultiviewProjectTexture base_color."}),
            "view_weights": ("MV_VIEW_WEIGHTS", {"tooltip": "MultiviewProjectTexture view_weights."}),
            "glass": ("MASK", {"tooltip": "MultiviewProjectTexture glass."}),
            **views, **glass, **lamps}}

    RETURN_TYPES = ("MV_CORNER_PLAN", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("plan", "crops", "back_view", "decisions")
    OUTPUT_IS_LIST = (False, True, False, False)
    FUNCTION = "run"
    CATEGORY = "3d/texturing"

    def run(self, mesh, base_color, projected, view_weights, glass, front, left, back, right,
            glass_front, glass_left, glass_back, glass_right, lamp_front, lamp_left, lamp_back, lamp_right):
        st = {"S": None, "projected": projected, "glass": glass, "error": None}
        try:
            if list(view_weights["names"]) != corner.NAMES:
                raise ValueError(f"needs the four views front/left/back/right, got {view_weights['names']}")
            v, f, uv = _mesh_arrays(mesh)
            rel8 = corner.to8(projected[0])
            sc = corner.Scene(v, f, uv, corner.to8(base_color[0]), [corner.to8(_view_hwc(x)) for x in (front, left, back, right)],
                              [corner.to8(m[0]) for m in (glass_front, glass_left, glass_back, glass_right)], rel8.shape[0])
            lamp8 = {n: corner.to8(m[0]) for n, m in zip(corner.NAMES, (lamp_front, lamp_left, lamp_back, lamp_right))}
            st["sc"] = sc
            st["S"] = corner.plan(sc, rel8, view_weights["w"], lamp8)
        except Exception as e:                    # the fix is optional: a failure leaves the texture as projected
            logging.exception("[MultiviewCorner] plan failed, texture left as projected")
            st["S"], st["error"] = None, repr(e)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        S = st["S"]
        crops = [torch.from_numpy(corner.f8(S["crops"][k]))[None] for k in S["items"]] if S else []
        back_view = torch.from_numpy(corner.f8(S["wimg"]["back"]))[None] if S and S["warped"] else back
        dec = S["D"] if S else {"error": st["error"]}
        return (st, crops, back_view, json.dumps(dec))


class MultiviewCornerApply:
    """Corner fix, step 2. Takes the edited crops (lazy: not requested when the plan has none, so the edit model never
    loads), keeps a redraw only if the head-on lamp's design survives on the mean of its 3 crops, pastes the best one
    (ring-only tone match, z-tested), cuts a separate side-facing lamp strip from the side photo when a front redraw is
    rejected, and moves the glass material with a warped back window. Nothing changed = base colour comes back as it was."""

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan": ("MV_CORNER_PLAN",),
            "metallic": ("IMAGE", {"tooltip": "Metallic map from BakeTextureFromVoxel (the one fed to MultiviewProjectTexture)."}),
            "roughness": ("IMAGE", {"tooltip": "Roughness map from BakeTextureFromVoxel (the one fed to MultiviewProjectTexture)."}),
            "glass_roughness": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "Same as MultiviewProjectTexture's glass_roughness."})},
            "optional": {"edits": ("IMAGE", {"lazy": True, "tooltip": "The plan's crops after the image edit, same order (a list)."})}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("base_color", "metallic", "roughness", "decisions")
    FUNCTION = "run"
    CATEGORY = "3d/texturing"

    def check_lazy_status(self, plan, metallic, roughness, glass_roughness, edits=None):
        S = plan[0]["S"]
        if S and S["items"] and (edits is None or any(e is None for e in edits)):
            return ["edits"]
        return []

    def run(self, plan, metallic, roughness, glass_roughness, edits=None):
        st = plan[0]
        S = st["S"]
        out8, gw, dec = None, None, {"error": st["error"]}
        if S is not None:
            try:
                e8 = {k: corner.to8(_view_hwc(e)) for k, e in zip(S["items"], edits)} if S["items"] else {}
                out8, gw, dec = corner.apply(st["sc"], S, e8)
            except Exception as e:                # the fix is optional: a failure leaves the texture as projected
                logging.exception("[MultiviewCorner] apply failed, texture left as projected")
                out8, gw, dec = None, None, {"error": repr(e)}
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        # an 8-bit atlas v is handed on as v/255: the GLB writer floors it back to exactly v
        base = st["projected"] if out8 is None else torch.from_numpy(corner.f8(out8))[None]
        g = st["glass"][0].float() if gw is None else torch.from_numpy(gw.astype("float32"))
        met, rou = _glass_material(metallic[0], roughness[0], g, glass_roughness[0])
        return (base, met, rou, json.dumps(dec))


class MultiviewPanelUnwrap:
    """Drop-in for UnwrapMesh on generated car meshes: UVs only for the surface the cameras can see, laid out as
    panels (sides, hood, roof, bumpers...) and packed with xatlas. Hidden faces share one texel. See panel_uv.py."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mesh": ("MESH",),
            "resolution": ("INT", {"default": 4096, "min": 512, "max": 8192, "step": 256,
                                   "tooltip": "Texture size the layout is packed for (same as the bake's texture size)."}),
            "padding": ("INT", {"default": 4, "min": 1, "max": 16, "tooltip": "Texels between panels."}),
            "underside_density": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 1.0, "step": 0.05,
                                            "tooltip": "Texel density of the underside (no input view sees it) relative to the rest."}),
            "fov": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 170.0, "step": 0.01,
                              "tooltip": "Rig fov, same as the projection; used to find the visible surface."}),
        }}

    RETURN_TYPES = ("MESH",)
    FUNCTION = "run"
    CATEGORY = "3d/texturing"

    def run(self, mesh, resolution, padding, underside_density, fov):
        from . import panel_uv
        from comfy_extras.nodes_mesh_postprocess import _pack_uv_meshes
        log = lambda *a: logging.info("[MultiviewPanelUnwrap] " + " ".join(str(x) for x in a))
        is_list = isinstance(mesh.vertices, list)
        bsz = len(mesh.vertices) if is_list else (mesh.vertices.shape[0] if mesh.vertices.ndim == 3 else 1)
        out_v, out_f, out_uv, out_c = [], [], [], []
        for i in range(bsz):
            v = mesh.vertices[i] if (is_list or mesh.vertices.ndim == 3) else mesh.vertices
            f = mesh.faces[i] if (is_list or mesh.faces.ndim == 3) else mesh.faces
            if getattr(mesh, "vertex_counts", None) is not None:
                v, f = v[:int(mesh.vertex_counts[i])], f[:int(mesh.face_counts[i])]
            vmap, faces, uvs = panel_uv.panel_unwrap(v.detach().cpu().numpy(), f.detach().cpu().numpy(), resolution,
                                                     padding, underside_density, fov, log=log)
            vm = torch.from_numpy(vmap).to(v.device)
            out_v.append(v[vm]); out_f.append(torch.from_numpy(faces).to(v.device)); out_uv.append(torch.from_numpy(uvs).to(v.device))
            vc = getattr(mesh, "vertex_colors", None)
            if vc is not None:
                ci = vc[i] if (isinstance(vc, list) or vc.ndim == 3) else vc
                out_c.append(ci[vm])
        out = _pack_uv_meshes(out_v, out_f, out_uv, out_c if out_c else None)
        if mesh.texture is not None:
            out.texture = mesh.texture
        return (out,)


class Save3DReturnPath:
    """Save3DAdvanced, but also returns the saved file's path relative to output/ (e.g. "3d/X_00001.glb").
    Workflow Studio only picks up a 3D result as a Display Any (rgthree) text "file:<path>"; the native
    save nodes return no path, so nothing in the graph could build that text."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model_3d": ("FILE_3D_GLB",),
                             "filename_prefix": ("STRING", {"default": "3d/ComfyUI"})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "3d"

    def run(self, model_3d, filename_prefix):
        from comfy_extras.nodes_save_3d import _save_file3d_to_output
        saved = _save_file3d_to_output(model_3d, filename_prefix)
        path = f"{saved.subfolder}/{saved.filename}" if saved.subfolder else saved.filename
        return {"ui": {"3d": [{"filename": saved.filename, "subfolder": saved.subfolder, "type": "output"}]},
                "result": (path,)}


class MultiviewAutoFrame:
    """Photos with backgrounds have the object small and low in the frame; Trellis then builds a boxier, blurrier
    model (S-01 on forest photos: side view 37% of the frame, width/length +10%). Crops each ORIGINAL photo around
    its cut-out, each view scaled so the object is equally tall in all four (match_height), object centred, widest view
    filling `fill` of the frame. Masks: ResizeAndPadImage(1024, black) + RemoveBackground pre-pass per view.
    Inputs that already fill the frame pass through unchanged."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"front": ("IMAGE",), "left": ("IMAGE",), "back": ("IMAGE",), "right": ("IMAGE",),
                             "front_mask": ("MASK",), "left_mask": ("MASK",), "back_mask": ("MASK",), "right_mask": ("MASK",),
                             "fill": ("FLOAT", {"default": 0.91, "min": 0.5, "max": 1.0, "step": 0.01}),
                             "size": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 8})},
                "optional": {"match_height": ("BOOLEAN", {"default": True, "tooltip": "Scale every view so the object is equally tall in all four (fixes views framed one by one, e.g. Two-Input Multiview output)."}),
                             "mirror_missing_side": ("BOOLEAN", {"default": True, "tooltip": "An empty side slot is filled with the mirrored other side."})}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("front", "left", "back", "right")
    FUNCTION = "run"
    CATEGORY = "3d/multiview"

    def run(self, front, left, back, right, front_mask, left_mask, back_mask, right_mask, fill, size, match_height=True, mirror_missing_side=True):
        import numpy as np
        ims = [(_view_hwc(i).cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8) for i in (front, left, back, right)]
        ms = [m[0].float().cpu().numpy() for m in (front_mask, left_mask, back_mask, right_mask)]
        out, span = core.auto_frame(ims, ms, fill, size, match_height, mirror_missing_side)
        logging.info("[MultiviewAutoFrame] shared crop %s px of the %d frame -> %d", "none" if span is None else round(span), ms[0].shape[0], size)
        return tuple(torch.from_numpy(o.astype(np.float32) / 255.0)[None] for o in out)


NODE_CLASS_MAPPINGS = {"MultiviewProjectTexture": MultiviewProjectTexture,
                       "MeshShadingForPhotoTexture": MeshShadingForPhotoTexture,
                       "MultiviewTouchUpPrepare": MultiviewTouchUpPrepare,
                       "MultiviewTouchUpApply": MultiviewTouchUpApply,
                       "MultiviewPanelUnwrap": MultiviewPanelUnwrap,
                       "Save3DReturnPath": Save3DReturnPath,
                       "MultiviewAutoFrame": MultiviewAutoFrame}
NODE_DISPLAY_NAME_MAPPINGS = {"MultiviewProjectTexture": "Multi-View Project Texture",
                              "MeshShadingForPhotoTexture": "Mesh Shading for Photo Texture",
                              "MultiviewTouchUpPrepare": "Multi-View Touch-Up (prepare)",
                              "MultiviewTouchUpApply": "Multi-View Touch-Up (apply)",
                              "MultiviewPanelUnwrap": "Multi-View Panel Unwrap (UVs)",
                              "Save3DReturnPath": "Save 3D (returns path)",
                              "MultiviewAutoFrame": "Multi-View Auto Frame"}
if corner is not None:
    NODE_CLASS_MAPPINGS.update(MultiviewCornerPlan=MultiviewCornerPlan, MultiviewCornerApply=MultiviewCornerApply)
    NODE_DISPLAY_NAME_MAPPINGS.update(MultiviewCornerPlan="Multi-View Corner Fix (plan)", MultiviewCornerApply="Multi-View Corner Fix (apply)")
