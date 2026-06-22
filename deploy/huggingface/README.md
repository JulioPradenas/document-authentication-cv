---
title: Document Authentication Dashboard
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
license: mit
---

# Document Authentication — Dashboard

Streamlit dashboard for document forgery detection (EfficientNet-B0 + Grad-CAM).

> Demo checkpoint trained on a small synthetic dataset (T1). Verdicts are real
> but the task is a toy separability task, not production document forensics.
> See the project repo for the full MLOps pipeline and the MIDV-500 training path.

This Space builds the `dashboard` stage of the project's multi-stage Dockerfile
and serves the Streamlit app on port 8501.
