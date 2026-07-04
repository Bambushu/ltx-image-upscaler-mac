#!/usr/bin/env python3
"""
LTX-2.3 Image Upscaler — Apple Silicon / Metal (GGUF)

Runs JXL's LTX-2.3 Image Upscaler (CivitAI 2741107) fully on an Apple Silicon Mac via
ComfyUI-GGUF — no NVIDIA GPU, nothing leaves the machine. The upstream workflow ships as
fp8 DiT + fp4 Gemma, both of which die on Metal (the MPS float8 wall); this port swaps
those three loaders for GGUF / split-file equivalents so the whole pipeline runs on MPS.

Mechanism (unchanged from upstream): render a blur->sharp fade micro-video with the
`upscalify` LoRA, then keep the last (sharpest) frame — rebuilds real detail (pores, hair,
texture) while holding identity. Two stages: base gen -> latent upsample x2 -> refine.

    python3 run.py shot.png                 # -> out/shot_up.png   (2.06 MP wide default)
    python3 run.py shot.png --mp 2.75       # portrait  (-> ~1496x1920)
    python3 run.py shot.png --mp 3.68       # square
    python3 run.py shot.png --launch        # auto-start ComfyUI on :8199 if not running

`--mp` = base-gen megapixels; final long side lands ~1920. 2.06 wide / 2.75 portrait / 3.68 square.

Setup, weights, and the loader-swap table are in README.md. Slow on Metal (~25-35 min/image,
2-stage 22B dev); for turnaround use an NVIDIA box. See build_local_graph.py for how the graph
was derived from the upstream fp8 workflow.

Env overrides:
  COMFYUI_DIR     path to your ComfyUI (default ~/ComfyUI)
  COMFYUI_PYTHON  python used for --launch (default $COMFYUI_DIR/venv/bin/python)
"""
import argparse, json, os, shutil, subprocess, sys, time, urllib.request, urllib.error

HERE   = os.path.dirname(os.path.abspath(__file__))
GRAPH  = os.path.join(HERE, "workflows", "ltx23_image_upscaler_mac_gguf.json")
COMFY  = os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI"))
PYEXE  = os.environ.get("COMFYUI_PYTHON", os.path.join(COMFY, "venv", "bin", "python"))
INDIR  = os.path.join(COMFY, "input")
OUTDIR = os.path.join(HERE, "out")
PORT   = 8199

def up(port):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=3); return True
    except Exception:
        return False

def launch(port):
    if not os.path.exists(PYEXE):
        sys.exit(f"--launch needs a ComfyUI python at {PYEXE} (set COMFYUI_PYTHON), "
                 f"or start ComfyUI yourself and drop --launch")
    db = f"sqlite:////tmp/ltx_upscaler_{port}.db"
    log = "/tmp/comfy_ltx_upscaler.log"
    subprocess.Popen(
        [PYEXE, "main.py", "--lowvram", "--port", str(port),
         "--output-directory", OUTDIR, "--input-directory", INDIR, "--database-url", db],
        cwd=COMFY, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    sys.stderr.write(f"launching ComfyUI :{port} (log {log})\n")
    for _ in range(150):
        if up(port): return
        time.sleep(3)
    sys.exit("ComfyUI did not come up — check the log")

def main():
    ap = argparse.ArgumentParser(description="LTX-2.3 Image Upscaler on Apple Silicon (GGUF)")
    ap.add_argument("image")
    ap.add_argument("--mp", type=float, default=2.06, help="base-gen megapixels (2.06 wide / 2.75 portrait / 3.68 square)")
    ap.add_argument("--out", default=None, help="output png (default out/<name>_up.png)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--launch", action="store_true", help="auto-launch ComfyUI if not running")
    a = ap.parse_args()

    src = os.path.abspath(a.image)
    if not os.path.exists(src): sys.exit(f"no such image: {src}")
    os.makedirs(OUTDIR, exist_ok=True)

    if not up(a.port):
        if a.launch: launch(a.port)
        else: sys.exit(f"ComfyUI not up on :{a.port} — start it, or pass --launch")

    stem = os.path.splitext(os.path.basename(src))[0]
    staged = f"upin_{os.getpid()}_{stem}.png"
    shutil.copyfile(src, os.path.join(INDIR, staged))

    g = json.load(open(GRAPH))
    g["284"]["inputs"]["image"] = staged
    g["333"]["inputs"]["megapixels"] = a.mp
    g["357"]["inputs"]["megapixels"] = a.mp
    if a.seed is not None:
        g["267:216"]["inputs"]["noise_seed"] = a.seed
        g["267:237"]["inputs"]["noise_seed"] = a.seed

    payload = json.dumps({"prompt": g, "client_id": "ltx_upscale_mac"}).encode()
    try:
        r = urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{a.port}/prompt", data=payload,
            headers={"Content-Type": "application/json"}), timeout=30)
        pid = json.load(r)["prompt_id"]
    except urllib.error.HTTPError as e:
        sys.exit(f"submit failed {e.code}: {e.read().decode()[:800]}")

    t0 = time.time()
    sys.stderr.write(f"queued {pid} (mp={a.mp}); ~25-35 min on Metal ...\n")
    fn = None
    while True:
        time.sleep(10)
        try:
            h = json.load(urllib.request.urlopen(f"http://127.0.0.1:{a.port}/history/{pid}", timeout=10))
        except Exception:
            continue
        if pid in h:
            st = h[pid].get("status", {})
            for o in h[pid].get("outputs", {}).values():
                for im in o.get("images", []):
                    fn = (im.get("subfolder", ""), im["filename"])
            if st.get("status_str") == "error":
                sys.exit("render errored — check the ComfyUI log")
            break

    if not fn: sys.exit("no output image produced")
    produced = os.path.join(OUTDIR, fn[0], fn[1])
    out = a.out or os.path.join(OUTDIR, f"{stem}_up.png")
    if os.path.abspath(produced) != os.path.abspath(out):
        shutil.copyfile(produced, out)
    try: os.remove(os.path.join(INDIR, staged))
    except OSError: pass
    print(f"{out}  ({(time.time()-t0)/60:.1f} min)")

if __name__ == "__main__":
    main()
