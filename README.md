## CS787 Course Project

This repository tracks a from-scratch reimplementation of the MINIM model for text-conditioned medical image generation and its training/evaluation stack. It's a diffusion-based text-to-medical-image generative model capable of synthesizing high-quality, multi-modal medical images conditioned on natural language prompts and modality specifications. The project will create a new codebase, independently replicating and enhancing the core functionalities described in the original MINIM repository to ensure modularity, extensibility, and compliance with best practices in machine learning for medical imaging.

### Tooling and Environments
- **Python management (uv)**: fast package/dependency manager and runner.
- **Lint/format (ruff)**: single tool for linting and formatting.

## Tooling

- **Python management (uv)**: fast package/dependency manager/runner
- **Training launcher**: Accelerate (multi-device, mixed precision)
- **DL libs**: PyTorch, Diffusers, Transformers
- **Eval**: clean-fid/torchvision metrics, scikit-image
- **Lint/format**: ruff

## Install

1. Install uv
   - macOS/Linux:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   - Windows (PowerShell):

   ```powershell
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
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
