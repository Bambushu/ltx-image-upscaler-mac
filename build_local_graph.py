#!/usr/bin/env python3
"""
How the Metal/GGUF graph was derived from the upstream fp8 workflow — kept for transparency
and so it can be re-derived when the upstream workflow updates. You do NOT need to run this to
use the upscaler; `workflows/ltx23_image_upscaler_mac_gguf.json` is already built.

Input:  a FLATTENED (API-format) export of JXL's LTX-2.3 Image Upscaler workflow
        (CivitAI 2741107 -> JXL_LTX23_Upscaler_WF_V2a.json). Flatten the subgraphs first by
        loading the .json in the ComfyUI frontend and using app.graphToPrompt().
Output: workflows/ltx23_image_upscaler_mac_gguf.json

The only changes are the three Metal-hostile loaders. Every other node in the graph
(LTX2_NAG, LTXVChunkFeedForward, the LTX AV nodes, ManualSigmas, SimpleMath+, ImageCrop+) is
available on Apple Silicon via ComfyUI core + KJNodes + ComfyUI_essentials, so the sizing /
sampling / upscale / decode topology is kept verbatim.

  1. MODEL      CheckpointLoaderSimple(ltx-2.3-22b-dev-fp8)  -> UnetLoaderGGUF(LTX-2.3-dev-Q4_K_M.gguf)
     VIDEO VAE  ...its VAE output                            -> new VAELoader(LTX23_video_vae_bf16)
  2. TEXT ENC   LTXAVTextEncoderLoader(gemma fp4)            -> DualCLIPLoaderGGUF(gemma-3-12b-it-Q4_K_M + text_projection, type=ltxv)
  3. AUDIO VAE  LTXVAudioVAELoader — keep the node, repoint ckpt_name to the standalone
                LTX23_audio_vae_bf16 (symlink it into your ComfyUI checkpoints/ folder; its
                audio_vae./vocoder. key prefixes match this loader's filter).

Why GGUF: fp8/fp4 hit the MPS float8 wall on Apple Silicon; GGUF dequantizes to bf16 on the
fly and sidesteps it.
"""
import argparse, json, os

DEV_GGUF   = "LTX-2.3-dev-Q4_K_M.gguf"
GEMMA_GGUF = "gemma-3-12b-it-Q4_K_M.gguf"
TEXT_PROJ  = "ltx-2.3_text_projection_bf16.safetensors"
VIDEO_VAE  = "LTX23_video_vae_bf16.safetensors"
AUDIO_VAE  = "LTX23_audio_vae_bf16.safetensors"   # symlink into checkpoints/
VID_VAE_ID = "9001"

def build(src, dst):
    g = json.load(open(src))
    g["267:236"] = {"inputs": {"unet_name": DEV_GGUF}, "class_type": "UnetLoaderGGUF",
                    "_meta": {"title": "Unet GGUF (LTX-2.3 dev Q4)"}}
    g[VID_VAE_ID] = {"inputs": {"vae_name": VIDEO_VAE}, "class_type": "VAELoader",
                     "_meta": {"title": "Video VAE bf16"}}
    for node in g.values():
        for k, v in (node.get("inputs", {}) if isinstance(node, dict) else {}).items():
            if isinstance(v, list) and len(v) == 2 and v[0] == "267:236" and v[1] == 2:
                node["inputs"][k] = [VID_VAE_ID, 0]
    g["267:243"] = {"inputs": {"clip_name1": GEMMA_GGUF, "clip_name2": TEXT_PROJ, "type": "ltxv"},
                    "class_type": "DualCLIPLoaderGGUF", "_meta": {"title": "Gemma TE GGUF (ltxv)"}}
    g["267:221"]["inputs"]["ckpt_name"] = AUDIO_VAE
    g["297"]["inputs"]["value"] = "up_local"
    json.dump(g, open(dst, "w"), indent=1)
    print("wrote", dst, "|", len(g), "nodes")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="flattened (API) export of the upstream fp8 workflow")
    ap.add_argument("--dst", default=os.path.join(os.path.dirname(__file__),
                    "workflows", "ltx23_image_upscaler_mac_gguf.json"))
    a = ap.parse_args()
    build(a.src, a.dst)
