## Environment Lock

- OS: Windows
- Python: 3.14.3
- Environment: venv (.venv)
- PyTorch: 2.10.0

## Random Seed Policy

- Global seed: 42
- Applied to:
  - Python random
  - NumPy
  - PyTorch
- Deterministic flags enabled where possible
- cuDNN benchmarking disabled

## Datasets Acquired

- Market-1501
- DukeMTMC-reID
- CUHK03


## Implemented Phases

- Phase 0 – Environment setup

- Phase 1–2 – Dataset cleaning pipeline

- Phase 3 – Dataset registry & loader

- Phase 4 – ResNet50 + BNNeck baseline

- Phase 5 – CE + BatchHard + PK sampler

- Phase 6 – Training & evaluation loop (mAP, CMC)

## Current Status:

Baseline training pipeline operational on Market1501.

All datasets stored unmodified under data/raw/.
File counts verified against official documentation.
