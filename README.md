# LTX-2.3 Image Upscaler — Apple Silicon / Metal (GGUF)

Run [JXL's **LTX-2.3 Image Upscaler**](https://civitai.com/models/2741107) on an Apple Silicon
Mac — no NVIDIA GPU, nothing leaves the machine. This is a **Metal/GGUF port** of the upstream
workflow, which ships as fp8 DiT + fp4 Gemma (both hit the MPS float8 wall and won't run on
Apple Silicon). Swapping those loaders for GGUF / split-file equivalents runs the whole
two-stage pipeline on MPS.

It's a real image super-resolver: it renders a blur→sharp fade micro-video with the `upscalify`
LoRA and keeps the sharpest frame, rebuilding fine detail (pores, hair, texture) while holding
identity. Base gen → latent upsample ×2 → refine → last frame.

> **Why this exists:** the upstream fp8 workflow won't load on Metal, and the LTX-2.3 two-stage
> `ManualSigmas + SamplerCustomAdvanced` pipeline has been [reported to NaN at VAE decode on
> MPS](https://lilting.ch/en/articles/ltx2-wan22-mac-local-video-gen). This port (using the
> **dev** GGUF) runs it end-to-end on Apple Silicon without NaN. As far as I can tell there's no
> other Mac/GGUF build of this image upscaler — corrections welcome.

## Example

Soft/low-res input → upscaled to ~1920px. Detail (skin pores, beard, hair, fabric) is rebuilt
while the face stays the same person.

| before | after |
|---|---|
| ![before](examples/before.png) | ![after](examples/after.png) |

## Quick start

```sh
python3 run.py shot.png                 # -> out/shot_up.png   (2.06 MP wide default)
python3 run.py shot.png --mp 2.75       # portrait  (-> ~1496x1920)
python3 run.py shot.png --launch        # auto-start ComfyUI on :8199 if not running
```
`--mp` = base-gen megapixels; final long side lands ~1920. **2.06** wide · **2.75** portrait · **3.68** square.

**Slow on Metal:** ~25–35 min/image (two-stage 22B dev; the ×2 refine is the cost). Fine for
one-offs where you want it local/free/private; for turnaround use an NVIDIA box.

## Setup

**ComfyUI + custom nodes** (all cross-platform, install into `ComfyUI/custom_nodes/`):
- [`city96/ComfyUI-GGUF`](https://github.com/city96/ComfyUI-GGUF) — `UnetLoaderGGUF`, `DualCLIPLoaderGGUF`
- [`kijai/ComfyUI-KJNodes`](https://github.com/kijai/ComfyUI-KJNodes) — `LTX2_NAG`, `LTXVChunkFeedForward`, resize/pad
- [`cubiq/ComfyUI_essentials`](https://github.com/cubiq/ComfyUI_essentials) — `SimpleMath+`, `ImageCrop+`
- [`rgthree/rgthree-comfy`](https://github.com/rgthree/rgthree-comfy) — `Power Lora Loader`

**Weights** — download and place in the standard ComfyUI folders:

| file | → folder | source |
|---|---|---|
| `LTX-2.3-dev-Q4_K_M.gguf` | `models/unet/` | [QuantStack/LTX-2.3-GGUF](https://huggingface.co/QuantStack/LTX-2.3-GGUF) · [unsloth/LTX-2.3-GGUF](https://huggingface.co/unsloth/LTX-2.3-GGUF) |
| `gemma-3-12b-it-Q4_K_M.gguf` | `models/text_encoders/` | [unsloth/gemma-3-12b-it-GGUF](https://huggingface.co/unsloth/gemma-3-12b-it-GGUF) |
| `ltx-2.3_text_projection_bf16.safetensors` | `models/text_encoders/` | [Kijai/LTX2.3_comfy](https://huggingface.co/Kijai/LTX2.3_comfy) |
| `LTX23_video_vae_bf16.safetensors` | `models/vae/` | [Kijai/LTX2.3_comfy](https://huggingface.co/Kijai/LTX2.3_comfy) |
| `LTX23_audio_vae_bf16.safetensors` | `models/vae/` **and** symlink into `models/checkpoints/` | [Kijai/LTX2.3_comfy](https://huggingface.co/Kijai/LTX2.3_comfy) |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | `models/latent_upscale_models/` | [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) |
| `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | `models/loras/` | [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) |
| `ltx23_upscalify_4psc4l1fy.safetensors` | `models/loras/` | [CivitAI 2741107](https://civitai.com/models/2741107) |

The **audio VAE** must be visible in `models/checkpoints/` (the `LTXVAudioVAELoader` node reads
from there):
```sh
ln -s ../vae/LTX23_audio_vae_bf16.safetensors models/checkpoints/LTX23_audio_vae_bf16.safetensors
```

Point `run.py` at your ComfyUI with `COMFYUI_DIR` if it isn't `~/ComfyUI`. `--launch` uses
`$COMFYUI_DIR/venv/bin/python` (override with `COMFYUI_PYTHON`), or just start ComfyUI yourself
and drop `--launch`.

## What was changed vs upstream

Only the three Metal-hostile loaders — everything else is kept verbatim, so behaviour matches
the original (see [`build_local_graph.py`](build_local_graph.py) for the exact transform):

| upstream (fp8, NVIDIA) | this port (Metal) |
|---|---|
| `CheckpointLoaderSimple(ltx-2.3-22b-dev-fp8)` → MODEL | `UnetLoaderGGUF(LTX-2.3-dev-Q4_K_M.gguf)` |
| …its VAE output | new `VAELoader(LTX23_video_vae_bf16)` |
| `LTXAVTextEncoderLoader(gemma fp4)` | `DualCLIPLoaderGGUF(gemma Q4 + text_projection, ltxv)` |
| `LTXVAudioVAELoader(dev-fp8)` | same node, repointed to standalone `LTX23_audio_vae_bf16` |

Tested on an M5 Pro (48 GB), ComfyUI 0.26, torch 2.11 / MPS. `--lowvram` keeps peak ~17 GB
(loads sequentially).

## Notes

- Clean input → use the default (no preblur). The upstream `--preblur` branch over-textures
  already-sharp skin; it's for genuinely degraded/low-res input.
- The GGUF loader logs `model_type FLUX` for the LTX dev checkpoint — cosmetic; output is valid.

## Credits

- **JXL** — the original [LTX-2.3 Image Upscaler workflow + `upscalify` LoRA](https://civitai.com/models/2741107). This repo is just a Mac/GGUF port of that work.
- **Lightricks** — LTX-2.3. **Kijai** — split VAEs / text projection + KJNodes. **city96** — ComfyUI-GGUF. **cubiq** — ComfyUI_essentials. **QuantStack / Unsloth** — GGUF quants.

## License

MIT for the scripts and the ported workflow JSON in this repo (see [LICENSE](LICENSE)). The
model weights and the `upscalify` LoRA are **not** included and carry their own licenses — get
them from the sources above and follow their terms.
