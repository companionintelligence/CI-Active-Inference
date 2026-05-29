# AGENTS.md — CI-Active-Inference

## Platform context

CI-Active-Inference is the **App/Agent** layer component of the Companion Intelligence platform — a private, local-first digital memory system.

Full architecture: `CI-Engineering/architecture/architecture.md`

## Repo summary

Active inference agent framework for Companion Intelligence. Implements a personal life agent using active inference (Free Energy Principle) over CI-Server data.

## Stack

- Python 3.11+
- ONNX / PyTorch
- FastAPI (if serving)
- PostgreSQL client (reads CI-Server DB)

## Setup & commands

```bash
pip install -r requirements.txt   # install
python main.py   # dev
pytest   # test
# no build step — Python   # build
```



## Git discipline

- Never `git reset --hard` or `git clean -f` without user confirmation.
- Commit messages: imperative mood, ≤ 72 chars.
- Reference CI-Engineering issues in PRs (e.g. `Closes companionintelligence/CI-Engineering#32`).

## Confidentiality

Private & Confidential — Property of Lifescope Inc. Do not distribute.
