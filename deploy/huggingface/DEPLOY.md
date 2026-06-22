# Deploy the dashboard to Hugging Face Spaces (Docker)

The Space builds the `dashboard` stage of the root `Dockerfile` (the last stage,
which is what `docker build` targets by default) and serves Streamlit on 8501.

## Prerequisites

- A Hugging Face account: https://huggingface.co/join
- `git` and the HF CLI: `uv run pip install huggingface_hub` then `huggingface-cli login`
- `git-lfs` installed (`brew install git-lfs` on macOS) — the checkpoint ships via LFS.

## One-time: verify the build locally (optional but recommended)

```bash
# From the project root — builds the dashboard stage and runs it
docker build --target dashboard -t doc-auth-dashboard .
docker run -p 8501:8501 doc-auth-dashboard
# open http://localhost:8501
```

## Create the Space and push

1. Create a new Space on the website: **New → Space → SDK: Docker → Blank**.
   Name it e.g. `document-authentication-dashboard`. Note the URL:
   `https://huggingface.co/spaces/<your-user>/document-authentication-dashboard`

2. Clone the (empty) Space and copy the project into it:

```bash
git lfs install
git clone https://huggingface.co/spaces/<your-user>/document-authentication-dashboard hf-space
cd hf-space

# Copy the project files the dashboard stage needs (Dockerfile + code + model + samples)
rsync -a --exclude '.git' --exclude '.venv' --exclude 'hf-space' \
  ../document-authentication-CV-clone/ .

# The Space README MUST carry the HF front-matter — use the prepared one:
cp deploy/huggingface/README.md README.md
```

3. The checkpoint is `.gitignore`d in the project but **must** be in the Space.
   It is already tracked by LFS via `.gitattributes`. Force-add it:

```bash
git lfs track "*.pt"            # already in .gitattributes, harmless to repeat
git add -f models/saved/efficientnet_b0_best.pt
git add .
git commit -m "Deploy document-authentication dashboard"
git push
```

4. Watch the build logs on the Space page. First build takes a few minutes
   (installs CPU-only torch). When it goes green, the dashboard is live at the
   Space URL.

## Notes

- **No checkpoint = no model.** If you skip step 3, the dashboard loads but shows
  "Checkpoint no encontrado". The `.pt` is ~17.6 MB — fine for LFS.
- The Dockerfile resolves **CPU-only torch on Linux** (HF builds on Linux), so the
  image stays small and avoids the ~3 GB CUDA stack.
- To update the model later (e.g. after training on MIDV-500), just replace
  `models/saved/efficientnet_b0_best.pt` and `git push` again — the Space rebuilds.
- Free tier: 16 GB RAM / 2 vCPU — comfortable for EfficientNet-B0 CPU inference.

## Alternative without LFS

If you prefer not to use LFS, force-add the file as a normal blob (works because
it is under GitHub/HF's 100 MB limit):

```bash
# remove the *.pt lines from .gitattributes first, then:
git add -f models/saved/efficientnet_b0_best.pt
```
