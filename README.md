## CS787: Multimodal Generative Modeling – Project Guide

### Purpose
This repository hosts a learning-and-research implementation of a text-to-image foundational model built largely from scratch in PyTorch, followed by a materials inverse-design model (MatterGPT-style) targeting SLICES crystal notation and property conditioning.

### Tooling and Environments
- **Python management (uv)**: fast package/dependency manager and runner.
- **Lint/format (ruff)**: single tool for linting and formatting.

### Quickstart
1) Install uv:
   - **macOS/Linux**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   - **Windows (PowerShell)**:
   ```powershell
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
   - **Windows (cmd)**:
   ```cmd
   curl -LsSf https://astral.sh/uv/install.bat | cmd
   ```

2) Create/activate a virtualenv and install project in editable mode with dev tools:
```bash
uv venv
uv pip install -e .
```

3) Run the app:
```bash
uv run python main.py
```

4) Format code and run lints:
```bash
uv run ruff format .
```

### Training System Specifications
- **Platform**: Linux 4.15.0-194-generic (x86_64)
- **Python**: 3.11.13
- **CPU**: 16 physical / 32 logical cores @ 813 MHz
- **RAM**: 46.81 GB (43.46 GB available)
- **Storage**: 250.92 GB total, 80.26 GB free
- **GPU 0**: NVIDIA TITAN Xp (12GB VRAM, 0% load)
- **GPU 1**: NVIDIA TITAN Xp (12GB VRAM, 0% load)s

### Data, Security, and Compliance
- No secrets in repo; use environment variables for API keys and compute backends.
- Validate and sanitize prompts and structure strings (SLICES) before training/eval.

### Performance and Observability
- Profile bottlenecks with PyTorch profiler and simple timers.
- Export basic metrics (loss curves, FID, CLIPScore, property MSE) and keep GPU/memory logs for runs.
