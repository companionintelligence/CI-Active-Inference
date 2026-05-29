# ANTIGRAVITY.md — CI-Active-Inference

## Context

**CI-Active-Inference** — **App/Agent** layer of the Companion Intelligence platform (Lifescope Inc).

Architecture: `CI-Engineering/architecture/architecture.md`
Roadmap: `CI-Engineering/architecture/roadmap.md`

## Purpose

Active inference agent framework for Companion Intelligence. Implements a personal life agent using active inference (Free Energy Principle) over CI-Server data.

## Stack

- Python 3.11+
- ONNX / PyTorch
- FastAPI (if serving)
- PostgreSQL client (reads CI-Server DB)

## Commands

```bash
pip install -r requirements.txt
python main.py
pytest
```

## Rules

- Package manager: do not switch or mix.
- Destructive git operations (`reset --hard`, `clean -f`) require explicit user confirmation.
- Commits reference CI-Engineering issues.
- All content is **private and confidential** — Property of Lifescope Inc.
