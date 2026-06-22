# T2-lite Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the document-authentication model on **one real MIDV-500 document type** (instead of the synthetic toy data), then redeploy the checkpoint to the live Hugging Face Space — within a <2 GB free-disk and slow-M1 budget.

**Architecture:** A two-script data pipeline feeds the existing two-phase `Trainer`. `prepare_t2_authentic.py` downloads a single MIDV-500 zip, reads only the `.tif` frames *out of the zip* (never extracting the video), downscales them to ≤384 px, writes them as the authentic class, and deletes the zip immediately. `build_t2_dataset.py` turns each real authentic frame into a synthetic forgery via the existing `SyntheticForgeryGenerator`, producing a balanced set. Training and deployment reuse the existing `scripts/train_demo.py` and the Space push flow unchanged.

**Tech Stack:** Python 3.11, OpenCV (`cv2`), NumPy, the project's `SyntheticForgeryGenerator` and `Trainer`, the `midv500` link list (already installed via the `datasets` extra), Git LFS + Hugging Face CLI for redeploy.

## Global Constraints

- **Python:** 3.11. **Lint:** ruff (line-length 100, rules `E,F,I,UP`, `E501` ignored). **Types:** mypy runs on `src/`+`api/` only (scripts are not type-checked in CI but must still import cleanly).
- **Tests must be CI-safe:** no network, no dataset download, no GPU. Every test runs on synthetic in-memory images or `tmp_path`.
- **Disk budget <2 GB:** never extract the MIDV `.mov` video; read frames directly from the zip; delete the zip immediately after reading; downscale every frame to longest-side ≤ **384 px**; cap the dataset at **≤250 images total** (≈100 authentic + ≈100 forged).
- **MIDV type:** default `01_alb_id` (first link in `midv500_links`). One type only.
- **Honesty constraint (do not remove from README):** the forged class is **synthetic** (generated over authentic frames). T2-lite makes the authentic class and substrate real; it does **not** make this a validated real-forgery detector.
- **Checkpoint handling:** `*.pt` stays `.gitignore`d in GitHub; the trained checkpoint is shipped to the Space via Git LFS only. Generated data lives under `data/train/` (already gitignored).
- **Reuse, don't rewrite:** use the existing `SyntheticForgeryGenerator` (`src/data/augmentation.py`), `create_dataloaders` (`src/data/loader.py`), `Trainer` (`src/models/trainer.py`), and the existing Space deploy flow.

---

## File Structure

- `scripts/__init__.py` — **new**, empty. Makes `scripts/` importable so tests can import the helpers.
- `scripts/prepare_t2_authentic.py` — **new**. Downloads one MIDV-500 type, reads `.tif` frames from the zip, downscales, writes `data/train/authentic/auth_XXXX.jpg`, deletes the zip. Exposes `downscale_rgb(img, max_side)` for testing.
- `scripts/build_t2_dataset.py` — **new**. Reads `data/train/authentic/`, writes one synthetic forgery per frame to `data/train/forged/forged_XXXX.jpg`. Exposes `build_forgeries(authentic_dir, forged_dir, seed) -> int` for testing.
- `scripts/train_demo.py` — **modify**. Add a `--data-dir` argument (default `data/train`) so the same script trains on either the T1 or T2 dataset.
- `tests/test_t2_data.py` — **new**. CI-safe unit tests for `downscale_rgb` and `build_forgeries`.

---

### Task 1: Authentic-frame downloader

Downloads one MIDV-500 type and produces real, downscaled authentic frames in `data/train/authentic/`, reading frames directly from the zip and deleting the zip to respect the disk budget.

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/prepare_t2_authentic.py`
- Create: `tests/test_t2_data.py`

**Interfaces:**
- Consumes: nothing from earlier tasks. Uses `midv500.download_dataset.midv500_links` (a `list[str]` of FTP zip URLs) only at runtime, not at import.
- Produces: `downscale_rgb(img: np.ndarray, max_side: int = 384) -> np.ndarray` — returns an RGB array whose longest side is ≤ `max_side` (unchanged if already smaller). Used by Task 1's script and tested by Task 1's test.

- [ ] **Step 1: Create the scripts package marker**

Create `scripts/__init__.py` with a single line:

```python
"""Project automation scripts (importable for tests)."""
```

- [ ] **Step 2: Write the failing test for `downscale_rgb`**

Create `tests/test_t2_data.py`:

```python
"""CI-safe tests for the T2-lite data pipeline (no network, no dataset)."""

import cv2
import numpy as np

from scripts.prepare_t2_authentic import downscale_rgb


def test_downscale_caps_longest_side():
    img = np.zeros((1000, 600, 3), dtype=np.uint8)
    out = downscale_rgb(img, max_side=384)
    assert max(out.shape[:2]) == 384
    assert out.shape[2] == 3


def test_downscale_is_noop_when_already_small():
    img = np.zeros((200, 100, 3), dtype=np.uint8)
    out = downscale_rgb(img, max_side=384)
    assert out.shape == (200, 100, 3)
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_t2_data.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.prepare_t2_authentic'`

- [ ] **Step 4: Implement `scripts/prepare_t2_authentic.py`**

```python
#!/usr/bin/env python3
"""Download ONE MIDV-500 document type and produce real authentic frames (T2-lite).

Disk-safe by design: reads `.tif` frames directly out of the downloaded zip
(never extracting the .mov video), downscales each to longest-side <= max_side,
writes them as JPGs to data/train/authentic/, then deletes the zip.

Usage:
    uv run python scripts/prepare_t2_authentic.py --type 01_alb_id --n 100
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = (".tif", ".tiff")


def downscale_rgb(img: np.ndarray, max_side: int = 384) -> np.ndarray:
    """Downscale an RGB image so its longest side is <= max_side (no upscaling)."""
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img


def _zip_url(midv_type: str) -> str:
    from midv500.download_dataset import midv500_links

    for link in midv500_links:
        if link.rstrip("/").split("/")[-1] == f"{midv_type}.zip":
            return link
    available = ", ".join(sorted(link.split("/")[-1][:-4] for link in midv500_links))
    raise SystemExit(f"Unknown MIDV type {midv_type!r}. Available: {available}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", default="01_alb_id", help="MIDV-500 document type (zip basename)")
    ap.add_argument("--n", type=int, default=100, help="Number of frames to keep")
    ap.add_argument("--max-side", type=int, default=384, help="Longest side after downscale")
    ap.add_argument("--out", type=Path, default=Path("data/train/authentic"))
    args = ap.parse_args()

    out: Path = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    url = _zip_url(args.type)
    zip_path = Path("data") / f"{args.type}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, zip_path)
    print(f"Downloaded {zip_path.stat().st_size / 1e6:.0f} MB")

    saved = 0
    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = [
                m
                for m in zf.namelist()
                if "/images/" in m.lower() and m.lower().endswith(IMAGE_EXTS)
            ]
            members.sort()
            if not members:
                raise SystemExit("No image frames found under images/ in the zip.")
            step = max(1, len(members) // args.n)
            picked = members[::step][: args.n]
            for member in picked:
                raw = np.frombuffer(zf.read(member), dtype=np.uint8)
                bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = downscale_rgb(rgb, args.max_side)
                cv2.imwrite(str(out / f"auth_{saved:04d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                saved += 1
    finally:
        zip_path.unlink(missing_ok=True)  # reclaim disk no matter what

    print(f"Saved {saved} authentic frames to {out}/ ; removed {zip_path.name}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_t2_data.py -q`
Expected: PASS (2 passed)

- [ ] **Step 6: Lint the new files**

Run: `uv run ruff check scripts/prepare_t2_authentic.py scripts/__init__.py tests/test_t2_data.py && uv run ruff format --check scripts/prepare_t2_authentic.py tests/test_t2_data.py`
Expected: `All checks passed!` and no reformat needed. If format check fails, run `uv run ruff format <file>` and re-check.

- [ ] **Step 7: Create the branch and commit**

```bash
git checkout -b feat/t2-lite-training
git add scripts/__init__.py scripts/prepare_t2_authentic.py tests/test_t2_data.py
git commit -m "feat(t2): authentic-frame downloader for one MIDV-500 type

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Synthetic-forgery dataset builder

Turns each real authentic frame into one synthetic forgery, producing a balanced `data/train/forged/` using the existing generator.

**Files:**
- Create: `scripts/build_t2_dataset.py`
- Modify: `tests/test_t2_data.py` (append one test)

**Interfaces:**
- Consumes: real authentic JPGs in `data/train/authentic/` produced by Task 1.
- Produces: `build_forgeries(authentic_dir: Path, forged_dir: Path, seed: int = 42) -> int` — writes one `forged_XXXX.jpg` per authentic image and returns the count. Used by Task 2's script and tested by Task 2's test.

- [ ] **Step 1: Write the failing test for `build_forgeries`**

Append to `tests/test_t2_data.py` (keep the existing imports; add `from pathlib import Path` at the top with the others and the new import below it):

```python
from pathlib import Path  # add alongside existing imports

from scripts.build_t2_dataset import build_forgeries  # add below the prepare import


def test_build_forgeries_is_balanced_and_changes_pixels(tmp_path: Path):
    authentic = tmp_path / "authentic"
    authentic.mkdir()
    base = np.full((224, 224, 3), 200, dtype=np.uint8)
    for i in range(3):
        cv2.imwrite(str(authentic / f"auth_{i:04d}.jpg"), base)

    forged = tmp_path / "forged"
    count = build_forgeries(authentic, forged, seed=1)

    assert count == 3
    assert len(list(forged.glob("*.jpg"))) == 3
    # A forgery must differ from the flat authentic source somewhere
    sample = cv2.imread(str(next(forged.glob("*.jpg"))))
    assert sample is not None
    assert not np.array_equal(sample, cv2.cvtColor(base, cv2.COLOR_RGB2BGR))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_t2_data.py::test_build_forgeries_is_balanced_and_changes_pixels -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.build_t2_dataset'`

- [ ] **Step 3: Implement `scripts/build_t2_dataset.py`**

```python
#!/usr/bin/env python3
"""Generate synthetic forgeries from real authentic frames (T2-lite).

For every authentic frame in data/train/authentic/, applies one synthetic
forgery (cycling the 4 types and 3 intensities of SyntheticForgeryGenerator)
and writes it to data/train/forged/. The result is a balanced dataset whose
authentic class is real and whose forged class is synthetic.

Usage:
    uv run python scripts/build_t2_dataset.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.data.augmentation import ForgeryConfig, ForgeryType, Intensity, SyntheticForgeryGenerator

_INTENSITIES: list[Intensity] = ["mild", "medium", "strong"]


def build_forgeries(authentic_dir: Path, forged_dir: Path, seed: int = 42) -> int:
    """Write one synthetic forgery per authentic image. Returns the count."""
    forged_dir.mkdir(parents=True, exist_ok=True)
    generator = SyntheticForgeryGenerator(seed=seed)
    types = list(ForgeryType)

    paths = sorted(authentic_dir.glob("*.jpg")) + sorted(authentic_dir.glob("*.png"))
    count = 0
    for i, path in enumerate(paths):
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        config = ForgeryConfig(
            forgery_type=types[i % len(types)],
            intensity=_INTENSITIES[i % len(_INTENSITIES)],
            seed=seed + i,
        )
        forged = generator.apply(rgb, config)
        cv2.imwrite(str(forged_dir / f"forged_{i:04d}.jpg"), cv2.cvtColor(forged, cv2.COLOR_RGB2BGR))
        count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authentic", type=Path, default=Path("data/train/authentic"))
    ap.add_argument("--forged", type=Path, default=Path("data/train/forged"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    n = build_forgeries(args.authentic, args.forged, seed=args.seed)
    print(f"Wrote {n} synthetic forgeries to {args.forged}/")


if __name__ == "__main__":
    main()
```

> Note: `Intensity` is the `Literal["mild", "medium", "strong"]` alias already exported from `src/data/augmentation.py`. Verify it is importable; if not, replace the import with `from typing import Literal` and define `Intensity = Literal["mild", "medium", "strong"]` locally.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_t2_data.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Lint**

Run: `uv run ruff check scripts/build_t2_dataset.py tests/test_t2_data.py && uv run ruff format --check scripts/build_t2_dataset.py tests/test_t2_data.py`
Expected: `All checks passed!` (run `uv run ruff format <file>` if format check fails, then re-check)

- [ ] **Step 6: Commit**

```bash
git add scripts/build_t2_dataset.py tests/test_t2_data.py
git commit -m "feat(t2): synthetic-forgery dataset builder over real frames

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Parametrize the trainer entrypoint

Lets `train_demo.py` train on any dataset directory so the same script serves T1 and T2 without duplication.

**Files:**
- Modify: `scripts/train_demo.py`

**Interfaces:**
- Consumes: a directory with `authentic/` and `forged/` subdirs (the T2 dataset from Tasks 1–2).
- Produces: a `--data-dir` CLI argument (default `data/train`) wired into `create_dataloaders`.

- [ ] **Step 1: Add the `--data-dir` argument**

In `scripts/train_demo.py`, replace the top of `main()` (the device line through the `create_dataloaders` call) with:

```python
def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/train"))
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}  |  data: {args.data_dir}")

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=32,
        num_workers=0,  # DocumentPreprocessor holds an unpicklable cv2.CLAHE
    )
```

Leave the rest of `main()` (model build, `TrainerConfig`, `Trainer(...).run()`, evaluation) unchanged.

- [ ] **Step 2: Verify the script still imports and shows the new flag**

Run: `uv run python scripts/train_demo.py --help`
Expected: usage text that includes `--data-dir`.

- [ ] **Step 3: Lint**

Run: `uv run ruff check scripts/train_demo.py && uv run ruff format --check scripts/train_demo.py`
Expected: `All checks passed!`

- [ ] **Step 4: Commit**

```bash
git add scripts/train_demo.py
git commit -m "refactor(t2): add --data-dir to train_demo for reuse across datasets

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Run the T2-lite pipeline end-to-end

Produces the real-data checkpoint and verifies it separates authentic from forged. This is a run/verify task (network + compute); it writes no committed code.

**Files:**
- Produces (gitignored): `data/train/authentic/*.jpg`, `data/train/forged/*.jpg`, `models/saved/efficientnet_b0_best.pt`

**Interfaces:**
- Consumes: the three scripts from Tasks 1–3.
- Produces: a trained `models/saved/efficientnet_b0_best.pt`.

- [ ] **Step 1: Check free disk before downloading**

Run: `df -h /Users/julio/Desktop`
Expected: confirm ≥ ~600 MB free (one zip ≈ 150–250 MB transient). If lower, lower `--n` and proceed — frames are read from the zip and the zip is deleted immediately.

- [ ] **Step 2: Download and prepare the authentic frames**

Run: `uv run python scripts/prepare_t2_authentic.py --type 01_alb_id --n 100 --max-side 384`
Expected: prints `Downloaded NN MB` then `Saved <=100 authentic frames to data/train/authentic/ ; removed 01_alb_id.zip`.
Verify the zip is gone: `ls data/*.zip 2>/dev/null || echo "no zip (good)"`

- [ ] **Step 3: Build the forged class**

Run: `uv run python scripts/build_t2_dataset.py`
Expected: `Wrote N synthetic forgeries to data/train/forged/` where N equals the authentic count.
Verify balance: `echo "auth=$(ls data/train/authentic | wc -l) forged=$(ls data/train/forged | wc -l)"`

- [ ] **Step 4: Train on the real data**

Run: `uv run python scripts/train_demo.py --data-dir data/train`
Expected: two-phase training logs, then a `Test — acc=… f1=… auc_roc=…` line. Note the numbers. AUC should be clearly above 0.5 (the synthetic forgery is learnable); if F1 is low but AUC high, that is the default-threshold artifact, acceptable for the demo.

- [ ] **Step 5: Verify the checkpoint separates the two classes**

Run:
```bash
uv run python -c "
import base64, glob
from api.predictor import DocumentPredictor
p = DocumentPredictor('models/saved/efficientnet_b0_best.pt')
for cls in ('authentic','forged'):
    probs=[]
    for f in sorted(glob.glob(f'data/train/{cls}/*.jpg'))[:8]:
        b=base64.b64encode(open(f,'rb').read()).decode()
        probs.append(p.predict(b)['probability'])
    print(cls, 'mean P(forged)=%.3f'%(sum(probs)/len(probs)))
"
```
Expected: `authentic` mean clearly below 0.5 and `forged` mean clearly above 0.5. If they are not separated, increase `phase_b_epochs` in `scripts/train_demo.py` and retrain.

- [ ] **Step 6: Reclaim disk (optional)**

Run: `du -sh data/train models/saved/efficientnet_b0_best.pt`
The dataset is small downscaled JPGs; keep it for reproducibility or `rm -rf data/train` after the checkpoint is confirmed.

---

### Task 5: Redeploy the checkpoint and open the PR

Ships the real-data checkpoint to the live Space and the scripts/tests to GitHub.

**Files:**
- Modify (Space repo only): `/Users/julio/Desktop/hf-space/models/saved/efficientnet_b0_best.pt`
- Uses: the existing branch `feat/t2-lite-training` from Tasks 1–3.

**Interfaces:**
- Consumes: the trained checkpoint from Task 4.
- Produces: an updated live Space and an open GitHub PR.

- [ ] **Step 1: Copy the new checkpoint into the Space clone**

Run: `cp /Users/julio/Desktop/document-authentication-CV-clone/models/saved/efficientnet_b0_best.pt /Users/julio/Desktop/hf-space/models/saved/efficientnet_b0_best.pt`

- [ ] **Step 2: Push the checkpoint to the Space (LFS)**

```bash
cd /Users/julio/Desktop/hf-space
git add models/saved/efficientnet_b0_best.pt
git commit -m "model: retrain on real MIDV-500 frames (T2-lite)"
git -c credential.helper='!f() { echo username=Pradnas; echo "password=$(hf auth token)"; }; f' push
```
Expected: `Uploading LFS objects: 100%` and a successful push.

- [ ] **Step 3: Verify the Space rebuilt and is running**

Run:
```bash
until s=$(curl -s "https://huggingface.co/api/spaces/Pradnas/document-authentication-dashboard" | python3 -c "import sys,json;print(json.load(sys.stdin).get('runtime',{}).get('stage'))"); [ "$s" = "RUNNING" ] || echo "$s" | grep -q ERROR; do echo "stage=$s"; sleep 20; done; echo "FINAL=$s"
```
Expected: `FINAL=RUNNING`. Then open the Space in a browser, run the **Demo** tab, and confirm verdicts on real frames look separated with credible Grad-CAM heatmaps.

- [ ] **Step 4: Update the README status note (honesty)**

In `/Users/julio/Desktop/document-authentication-CV-clone/README.md`, in the "Estado del proyecto y resultados" blockquote, change the demo description from "dataset sintético pequeño" to: "un **tipo real de MIDV-500** (clase auténtica) con falsificaciones **sintéticas** generadas sobre esas imágenes (la clase forjada sigue siendo sintética)". Keep the rest of the honesty note intact.

- [ ] **Step 5: Push the branch and open the PR**

```bash
cd /Users/julio/Desktop/document-authentication-CV-clone
git add README.md
git commit -m "docs: note T2-lite real-data training in status

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push -u origin feat/t2-lite-training
gh pr create --title "feat: T2-lite — entrenar sobre un tipo real de MIDV-500" \
  --body "Pipeline ligero (<2GB disco) para entrenar la clase auténtica sobre frames reales de un tipo de MIDV-500, con forgeries sintéticas. Incluye scripts + tests CI-safe y nota de honestidad en el README. El checkpoint se sube al Space por LFS (no a git).

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```
Expected: a PR URL is printed.

- [ ] **Step 6: Confirm CI is green, then merge**

Run: `gh pr checks` (wait for lint, test, docker-build to pass), then `gh pr merge --squash --delete-branch`.
Expected: all checks pass; PR merged into `main`.

---

## Self-Review

**1. Spec coverage** (the request: lightest/simplest way to do T2 on <2 GB disk, slow M1, one MIDV type, add value):
- Lightest disk: Task 1 reads frames from the zip, skips the video, deletes the zip immediately, downscales to 384 px → covered. Task 4 Step 1 checks free disk; Step 6 reclaims it.
- Slow M1: ≤250 small 384-px images + `num_workers=0` + reduced epochs (existing `train_demo` config) → covered.
- One MIDV type: `--type` default `01_alb_id`, `_zip_url` resolves a single link → covered.
- Adds value: real authentic substrate + credible Grad-CAM on real documents, redeployed to the live Space (Task 5) → covered.
- Honesty preserved: Global Constraints + Task 5 Step 4 → covered.

**2. Placeholder scan:** no "TBD"/"handle edge cases"/"similar to"/"write tests for the above" — every code step shows full code; every run step shows the command and expected output. The one conditional note (Task 2 Step 3, `Intensity` import) gives the exact fallback code. OK.

**3. Type consistency:** `downscale_rgb(img, max_side)` defined in Task 1, used identically in Task 1's script. `build_forgeries(authentic_dir, forged_dir, seed) -> int` defined and called identically in Task 2. `--data-dir` (Task 3) is consumed by Task 4 Step 4 with the same name. `ForgeryConfig`, `ForgeryType`, `Intensity`, `SyntheticForgeryGenerator` match the exports verified in `src/data/augmentation.py`. `create_dataloaders(data_dir=…, batch_size=…, num_workers=…)` matches `src/data/loader.py`. OK.

---

## Notes / Risks

- **FTP download reliability:** `midv500_links` uses `ftp://smartengines.com/...`. If FTP is blocked on the network, Task 4 Step 2 fails at `urllib.request.urlretrieve`. Fallback: download the single zip manually in a browser into `data/`, then run a variant of Task 1 that reads the local zip (skip the download line). This does not change any committed code.
- **Per-type zip size** is ~150–250 MB transient; with <2 GB free this fits, but Task 4 Step 1 gates on it and the zip is deleted in a `finally` block even on error.
- **Forged class stays synthetic** — this is intentional and documented; do not present T2-lite as real-forgery validation.
