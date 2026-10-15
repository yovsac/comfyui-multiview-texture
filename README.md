<img src="icon.png" width="96" alt="Yovan Vujacic">

# comfyui-multiview-texture

ComfyUI custom nodes used by the **Multiview to 3D** tool (four views of a car → textured 3D model).
They paint the four input views onto the generated mesh, unwrap it into clean panels and fix the
lamps that come out split or doubled where two views meet at a corner of the car.

Tested on **ComfyUI 0.39.0** (Python 3.13, PyTorch 2.9.1 + CUDA 13.0). The Trellis 2 / Pixal3D,
SAM3, Qwen-Image-Edit and background-removal nodes the tool also uses are native to ComfyUI 0.39.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yovsac/comfyui-multiview-texture
pip install -r comfyui-multiview-texture/requirements.txt
```

Restart ComfyUI. The log shows a warning if `opencv` is missing: the corner-fix nodes are then
skipped and the rest of the pack still loads (the tool needs all of them).

Python dependencies: `xatlas==0.0.11`, `opencv-python-headless`, plus `torch`, `numpy`, `scipy`
and `Pillow`, which ComfyUI already installs.

## Nodes

| Node (class name) | Display name | What it does |
|---|---|---|
| `MultiviewAutoFrame` | Multi-View Auto Frame | Crops each photo around the car with one zoom shared by all four views, so they keep their relative scale. |
| `MultiviewPanelUnwrap` | Multi-View Panel Unwrap (UVs) | Drop-in for the native `UnwrapMesh`: texture space only for the surface the cameras see, laid out as low-stretch panels (CPU, ~1.5-2 min per car). |
| `MultiviewProjectTexture` | Multi-View Project Texture | Paints the four views onto the UV layout through the exact rig cameras; hidden surfaces keep the generated colour. |
| `MeshShadingForPhotoTexture` | Mesh Shading for Photo Texture | Smoothed shading normals + optional metallic/roughness overrides. |
| `MultiviewCornerPlan` | Multi-View Corner Fix (plan) | Finds lamps split across a corner and outputs crops to redraw (0 crops = nothing to fix). |
| `MultiviewCornerApply` | Multi-View Corner Fix (apply) | Pastes the redrawn crops back. Its edit input is lazy: with no crop the edit model is never loaded. |
| `Save3DReturnPath` | Save 3D (returns path) | Saves the model and returns its path. |
| `MultiviewTouchUpPrepare` / `MultiviewTouchUpApply` | Multi-View Touch-Up | Painted-mask touch-up (not used by the current tool). |

## Other custom nodes the tool uses

- `UnetLoaderGGUF` — https://github.com/city96/ComfyUI-GGUF
- `Seed (rgthree)`, `Display Any (rgthree)` — https://github.com/rgthree/rgthree-comfy
- `AppInfo` — https://github.com/shadowcz007/comfyui-mixlab-nodes

## Models the tool loads

| Folder (`ComfyUI/models/...`) | File | Used by |
|---|---|---|
| `clip_vision` | `dino_v3_L_naf_fp32.safetensors` | 3D stage |
| `vae` | `trellis_2_shape_vae_bf16.safetensors`, `trellis_2_texture_vae_bf16.safetensors` | 3D stage |
| `diffusion_models` (unet) | `pixal3d_multiview_int8_convrot.safetensors` | 3D stage |
| `background_removal` | `birefnet.safetensors` | view cut-outs |
| `upscale_models` | `RealESRGAN_x4.pth` | 2K views for the projection |
| `checkpoints/sam3` | `sam3.pt` | glass + lamp masks |
| `unet/QWEN` | `qwen-image-edit-2511-Q8_0.gguf` | corner fix (only when a corner needs redrawing) |
| `loras/qwen` | `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` | corner fix |
| `text_encoders` | `qwen_2.5_vl_7b_fp8_scaled.safetensors` | corner fix |
| `vae` | `qwen_image_vae.safetensors` | corner fix |

**Check the Qwen GGUF:** the 2511 Q8 file must contain **1934 tensors**, including
`__index_timestep_zero__`. A 1933-tensor copy loads without error but ComfyUI then treats the model as
2509, and edits come out ghosted.

```python
from gguf import GGUFReader
r = GGUFReader("qwen-image-edit-2511-Q8_0.gguf")
print(len(r.tensors), any(t.name == "__index_timestep_zero__" for t in r.tensors))  # 1934 True
```

Made by Yovan Vujacic.
