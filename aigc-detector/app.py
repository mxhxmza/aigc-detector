#!/usr/bin/env python3
"""Local web interface: upload an image, get a calibrated p(AI-generated).

This is a thin front-end over `src.inference.Scorer` -- the same forward pass
`predict.py` uses. It runs entirely on your machine; nothing is uploaded
anywhere. Images are scored in memory and not written to disk.

Usage:
    python app.py                                   # http://127.0.0.1:8000
    python app.py --checkpoint checkpoints/full.pt --port 8000
    python app.py --host 0.0.0.0                     # expose on the LAN

The model (frozen CLIP backbone + detector head) loads once at startup and
stays in memory, so the first request is fast.
"""

from __future__ import annotations

import argparse
import io
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

from src.inference import Scorer

MAX_UPLOAD_BYTES = 25 * 1024 * 1024      # 25 MB is plenty for a single photo

app = FastAPI(title="AI-Generated Image Detector")

_scorer: Scorer | None = None
_scorer_lock = threading.Lock()          # torch forward passes are serialised
_settings: dict = {}


def get_scorer() -> Scorer:
    global _scorer
    if _scorer is None:
        with _scorer_lock:
            if _scorer is None:
                _scorer = Scorer(_settings["checkpoint"], device=_settings["device"],
                                 multicrop=_settings.get("multicrop", True))
    return _scorer


def verdict(p: float) -> tuple[str, str]:
    """(label, band) for a probability. The band drives the UI colour."""
    if p >= 0.85:
        return "Very likely AI-generated", "high"
    if p >= 0.60:
        return "Probably AI-generated", "mid-high"
    if p > 0.40:
        return "Uncertain", "mid"
    if p > 0.15:
        return "Probably authentic", "mid-low"
    return "Very likely authentic", "low"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE


@app.get("/api/info")
def info() -> JSONResponse:
    s = get_scorer()
    return JSONResponse({
        "checkpoint": str(s.checkpoint),
        "backbone": s.config["backbone"],
        "device": s.device,
        "uses_forensic_branch": s.config.get("use_forensic", True),
        "uses_degradation_gate": s.config.get("use_gate", True),
        "multicrop": s.multicrop,
        "temperature": round(s.temperature, 4),
        "total_parameters": s.params["total"],
    })


@app.post("/api/predict")
async def predict(image: UploadFile = File(...)) -> JSONResponse:
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "empty upload")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"image larger than {MAX_UPLOAD_BYTES // (1024*1024)} MB")

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
        img = img.convert("RGB")
    except Exception:
        raise HTTPException(415, "could not read that file as an image")

    scorer = get_scorer()
    with _scorer_lock:
        p = float(scorer.score_many([img])[0])

    label, band = verdict(p)
    return JSONResponse({
        "filename": image.filename,
        "p_ai_generated": round(p, 6),
        "p_authentic": round(1.0 - p, 6),
        "verdict": label,
        "band": band,
        "image_size": list(img.size),
    })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/full.pt"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--no-multicrop", action="store_true",
                    help="score the whole frame only (faster, less robust to "
                         "framed / foreground-heavy photos)")
    args = ap.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(
            f"error: checkpoint not found: {args.checkpoint}\n"
            "       Train one first (see the README), or pass --checkpoint."
        )

    _settings["checkpoint"] = args.checkpoint
    _settings["device"] = args.device
    _settings["multicrop"] = not args.no_multicrop

    print(f"loading model from {args.checkpoint} ...", flush=True)
    s = get_scorer()
    print(f"ready on {s.device}  |  backbone {s.config['backbone']}  |  "
          f"temperature {s.temperature:.3f}", flush=True)
    print(f"\n    open  http://{args.host}:{args.port}\n", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI-Generated Image Detector</title>
<style>
  :root {
    --bg: #f7f7f8; --card: #ffffff; --ink: #1a1a1e; --muted: #6b6b76;
    --line: #e4e4e8; --accent: #4f46e5;
    --low: #15803d; --mid: #b0b0b8; --high: #dc2626;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0e0e11; --card: #17171b; --ink: #ececf1; --muted: #9a9aa5;
      --line: #2a2a31; --accent: #7c73ff;
      --low: #22c55e; --mid: #52525b; --high: #f87171;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.55 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
  .wrap { max-width: 640px; margin: 0 auto; padding: 40px 20px 80px; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  p.sub { color: var(--muted); margin: 0 0 28px; font-size: 14px; }
  .card { background: var(--card); border: 1px solid var(--line);
    border-radius: 14px; padding: 22px; }
  #drop { border: 1.5px dashed var(--line); border-radius: 12px;
    padding: 38px 20px; text-align: center; cursor: pointer;
    transition: border-color .15s, background .15s; }
  #drop.hot { border-color: var(--accent); background: color-mix(in srgb, var(--accent) 8%, transparent); }
  #drop strong { color: var(--accent); }
  #drop small { display: block; color: var(--muted); margin-top: 6px; }
  input[type=file] { display: none; }
  .actions { display: flex; gap: 10px; margin-top: 12px; }
  button { flex: 1; font: inherit; padding: 10px 14px; border-radius: 10px;
    border: 1px solid var(--line); background: var(--accent); color: #fff;
    cursor: pointer; }
  button.ghost { background: transparent; color: var(--muted); }
  button:active { transform: translateY(1px); }
  #camera { margin-top: 16px; }
  #camera[hidden] { display: none; }
  #video { width: 100%; max-height: 340px; object-fit: contain; border-radius: 10px;
    border: 1px solid var(--line); background: #000; }
  #preview { margin-top: 18px; display: none; }
  #preview img { width: 100%; max-height: 320px; object-fit: contain;
    border-radius: 10px; border: 1px solid var(--line); background: #0003; }
  #result { margin-top: 18px; display: none; }
  .verdict { font-size: 19px; font-weight: 650; margin-bottom: 12px; }
  .bar { height: 30px; border-radius: 8px;
    background: linear-gradient(90deg, var(--low), var(--mid) 50%, var(--high));
    position: relative; border: 1px solid var(--line); margin-top: 4px; }
  .bar .needle { position: absolute; top: -5px; bottom: -5px; width: 3px;
    background: var(--ink); border-radius: 2px; transition: left .25s ease;
    box-shadow: 0 0 0 2px var(--card); }
  .scale { display: flex; justify-content: space-between; color: var(--muted);
    font-size: 12px; margin-top: 6px; }
  .nums { display: flex; gap: 10px; margin-top: 14px; }
  .num { flex: 1; border: 1px solid var(--line); border-radius: 10px;
    padding: 10px 12px; text-align: center; }
  .num b { display: block; font-size: 22px; margin-top: 2px; }
  .num span { color: var(--muted); font-size: 12px; text-transform: uppercase;
    letter-spacing: .04em; }
  .err { color: var(--high); margin-top: 14px; display: none; }
  .foot { color: var(--muted); font-size: 12.5px; margin-top: 22px; }
  .spin { display: inline-block; width: 15px; height: 15px; border: 2px solid var(--line);
    border-top-color: var(--accent); border-radius: 50%; animation: sp .7s linear infinite;
    vertical-align: -3px; margin-right: 7px; }
  @keyframes sp { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="wrap">
  <h1>AI-Generated Image Detector</h1>
  <p class="sub">Upload an image or take a photo. The model returns a calibrated probability that it was AI-generated. Runs locally &mdash; nothing leaves your machine.</p>

  <div class="card">
    <div id="drop">
      <strong>Click to choose an image</strong> or drop it here
      <small>JPEG, PNG, WebP &mdash; up to 25&nbsp;MB</small>
    </div>
    <div class="actions">
      <button id="camBtn" type="button">Take a photo</button>
    </div>
    <input type="file" id="file" accept="image/*">
    <input type="file" id="camFallback" accept="image/*" capture="environment">

    <div id="camera" hidden>
      <video id="video" playsinline muted></video>
      <div class="actions">
        <button id="shootBtn" type="button">Capture</button>
        <button id="cancelCam" type="button" class="ghost">Cancel</button>
      </div>
    </div>

    <div id="preview"><img id="img" alt="uploaded image preview"></div>

    <div id="busy" style="display:none;margin-top:16px"><span class="spin"></span>Scoring&hellip;</div>
    <div class="err" id="err"></div>

    <div id="result">
      <div class="verdict" id="verdict"></div>
      <div class="bar">
        <div class="needle" id="needle"></div>
      </div>
      <div class="scale"><span>authentic</span><span>uncertain</span><span>AI-generated</span></div>
      <div class="nums">
        <div class="num"><span>p(AI-generated)</span><b id="pAI">&mdash;</b></div>
        <div class="num"><span>p(authentic)</span><b id="pHuman">&mdash;</b></div>
      </div>
    </div>
  </div>

  <p class="foot" id="foot"></p>
</div>

<script>
const drop = document.getElementById('drop');
const file = document.getElementById('file');
const busy = document.getElementById('busy');
const err  = document.getElementById('err');
const result = document.getElementById('result');
const preview = document.getElementById('preview');
const camBtn = document.getElementById('camBtn');
const camFallback = document.getElementById('camFallback');
const camera = document.getElementById('camera');
const video = document.getElementById('video');
const shootBtn = document.getElementById('shootBtn');
const cancelCam = document.getElementById('cancelCam');

drop.addEventListener('click', () => file.click());
['dragover','dragenter'].forEach(e => drop.addEventListener(e, ev => {
  ev.preventDefault(); drop.classList.add('hot');
}));
['dragleave','drop'].forEach(e => drop.addEventListener(e, ev => {
  ev.preventDefault(); drop.classList.remove('hot');
}));
drop.addEventListener('drop', ev => {
  if (ev.dataTransfer.files.length) handle(ev.dataTransfer.files[0]);
});
file.addEventListener('change', () => { if (file.files.length) handle(file.files[0]); });
camFallback.addEventListener('change', () => { if (camFallback.files.length) handle(camFallback.files[0]); });

// Live camera. getUserMedia needs a secure context (localhost counts, a LAN
// IP over plain http does not) -- fall back to the OS camera picker, which on
// a phone still opens the camera directly.
let stream = null;
function stopCam() {
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  camera.hidden = true;
  video.srcObject = null;
}
camBtn.addEventListener('click', async () => {
  err.style.display = 'none';
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    camFallback.click(); return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
    video.srcObject = stream;
    await video.play();
    camera.hidden = false;
    camera.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (e) {
    camFallback.click();   // no permission / no camera / insecure context
  }
});
cancelCam.addEventListener('click', stopCam);
shootBtn.addEventListener('click', () => {
  const c = document.createElement('canvas');
  c.width = video.videoWidth; c.height = video.videoHeight;
  c.getContext('2d').drawImage(video, 0, 0);
  stopCam();
  c.toBlob(b => { if (b) handle(new File([b], 'camera.jpg', { type: 'image/jpeg' })); },
           'image/jpeg', 0.95);
});

fetch('/api/info').then(r => r.json()).then(d => {
  document.getElementById('foot').textContent =
    `model: ${d.checkpoint} · backbone ${d.backbone} · ${d.device} · ` +
    `temperature ${d.temperature}` + (d.multicrop ? ' · 6-crop median' : '') +
    ` · ${(d.total_parameters/1e6).toFixed(1)}M params`;
}).catch(() => {});

async function handle(f) {
  err.style.display = 'none';
  result.style.display = 'none';
  document.getElementById('img').src = URL.createObjectURL(f);
  preview.style.display = 'block';
  busy.style.display = 'block';

  const fd = new FormData();
  fd.append('image', f);
  try {
    const r = await fetch('/api/predict', { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'request failed');
    show(d);
  } catch (e) {
    err.textContent = e.message;
    err.style.display = 'block';
  } finally {
    busy.style.display = 'none';
  }
}

function show(d) {
  const p = d.p_ai_generated;
  document.getElementById('verdict').textContent = d.verdict;
  document.getElementById('pAI').textContent = (p * 100).toFixed(1) + '%';
  document.getElementById('pHuman').textContent = (d.p_authentic * 100).toFixed(1) + '%';
  document.getElementById('needle').style.left = `calc(${(p * 100).toFixed(1)}% - 1.5px)`;
  const v = document.getElementById('verdict');
  v.style.color = p >= 0.6 ? 'var(--high)' : p <= 0.4 ? 'var(--low)' : 'var(--ink)';
  result.style.display = 'block';
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
