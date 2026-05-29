# GitHub Copilot Instructions — CI-Active-Inference

You are working in **CI-Active-Inference**, the **App/Agent** component of the **Companion Intelligence** platform — a private, local-first digital memory and AI appliance system owned by Lifescope Inc.

## Platform architecture

```
Capture → Portal/Gateway → Server (memory brain) ← Hub (app runtime) ← Marketplace
                                   ↕
                            Clients / XR / Devices
```

Full architecture and ADRs live in `CI-Engineering/architecture/`.

## This repo

Active inference agent framework for Companion Intelligence. Implements a personal life agent using active inference (Free Energy Principle) over CI-Server data.

## Tech stack

- Python 3.11+
- ONNX / PyTorch
- FastAPI (if serving)
- PostgreSQL client (reads CI-Server DB)

## Development

```bash
pip install -r requirements.txt   # install
pytest   # test
```



## Conventions

- Do not mix package managers.
- Keep PRs small and linked to CI-Engineering issues.
- Multiple agents may work this codebase simultaneously — never use destructive git operations.
- All code is **private and confidential** — do not surface or transmit source beyond this session.
