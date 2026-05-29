# CLAUDE.md — CI-Active-Inference

> **Private & Confidential — Property of Lifescope Inc. Do not distribute.**

## Platform role

**CI-Active-Inference** sits in the **App/Agent** layer of the Companion Intelligence platform.

Architecture: `CI-Engineering/architecture/architecture.md`
Roadmap: `CI-Engineering/architecture/roadmap.md`
Issues: <https://github.com/companionintelligence/CI-Engineering/issues>

## What this repo does

Active inference agent framework for Companion Intelligence. Implements a personal life agent using active inference (Free Energy Principle) over CI-Server data.

## Stack

- Python 3.11+
- ONNX / PyTorch
- FastAPI (if serving)
- PostgreSQL client (reads CI-Server DB)

## Key commands

```bash
# Install dependencies
pip install -r requirements.txt

# Start development server / service
python main.py

# Run tests
pytest

# Build for production
# no build step — Python
```

## Package manager

**pip** — use exclusively; do not mix with npm/yarn/pnpm/bun unless the repo explicitly requires it.

## Cross-repo connections

- Reads ci.events, ci.content from CI-Server PostgreSQL
- Runs as a sidecar or standalone agent process



## Multi-agent safety

Multiple AI agents work these repos in parallel. Never use `git reset --hard`, `git clean -f`, or any destructive git command without explicit confirmation from the user.

## Confidentiality

Private & Confidential — Property of Lifescope Inc. Do not distribute.
