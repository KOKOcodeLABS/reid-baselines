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

All datasets stored unmodified under data/raw/.
File counts verified against official documentation.
