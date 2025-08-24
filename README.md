## CS787 Course Project

This is a sophisticated AI system designed to generate high-quality synthetic medical images from text descriptions. 

It's a diffusion-based text-to-medical-image generative model capable of synthesizing high-quality, multi-modal medical images conditioned on natural language prompts and modality specifications.

Supports multiple medical imaging modalities including OCT (ophthalmology), Fundus (retinal), Chest X-ray, Chest CT, Brain MRI, and Breast MRI.

## Data Format

### Dataset Structure
The system expects a single CSV file with three columns: `path`, `Text`, and `modality`.

```csv
path,Text,modality
datasets/oct/retina_healthy.png,OCT: healthy retinal layers with clear foveal depression,OCT
datasets/ct/chest_normal.dcm,Chest CT: normal lung parenchyma without nodules,CT
datasets/mri/brain_tumor.nii.gz,Brain MRI: enhancing mass in right temporal lobe,MRI
datasets/xray/chest_pneumonia.jpg,Chest X-ray: bilateral infiltrates consistent with pneumonia,X-Ray
datasets/fundus/diabetic_retinopathy.png,Fundus: microaneurysms and hard exudates in diabetic patient,Fundus
datasets/mri/breast_lesion.png,Breast MRI: irregular enhancing mass in upper outer quadrant,MRI
```

### Supported Modalities
- **OCT** (Optical Coherence Tomography): Retinal imaging for ophthalmology
- **CT** (Computed Tomography): Cross-sectional imaging (chest, abdomen, brain)
- **X-Ray**: 2D radiographic imaging
- **MRI** (Magnetic Resonance Imaging): Multi-planar imaging (brain, breast, spine)
- **Fundus**: Retinal photography for eye examination

### Image Format Support
- **Standard formats**: PNG, JPG, JPEG
- **Medical formats**: DICOM (.dcm), NIfTI (.nii, .nii.gz)
- **Resolution**: Configurable (default: 512x512)
- **Channels**: Grayscale (L) or RGB

### Dataset Organization
```
datasets/
├── oct/           # Optical Coherence Tomography
│   ├── healthy/
│   ├── diabetic_retinopathy/
│   └── macular_degeneration/
├── ct/            # Computed Tomography
│   ├── chest/
│   ├── abdomen/
│   └── brain/
├── mri/           # Magnetic Resonance Imaging
│   ├── brain/
│   ├── breast/
│   └── spine/
├── xray/          # X-Ray imaging
│   ├── chest/
│   ├── abdomen/
│   └── extremities/
├── fundus/        # Fundus photography
│   ├── normal/
│   ├── diabetic/
│   └── hypertensive/
└── metadata.csv   # Main dataset file
```

### Text Prompt Guidelines
- **Format**: `{Modality}: {detailed description}`
- **Be specific**: Include anatomical location, pathology, image characteristics
- **Use medical terminology**: Leverage standard radiological descriptions
- **Examples**:
  - `OCT: healthy retinal layers with clear foveal depression and normal choroidal vasculature`
  - `CT: chest scan showing bilateral ground glass opacities in lower lobes`
  - `MRI: T1-weighted brain image with enhancing mass in right temporal lobe`

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

# References 

* [Self-improving generative foundation model for synthetic medical image generation and clinical applications](https://www.nature.com/articles/s41591-024-03359-y)
* [MINIM](https://github.com/WithStomach/MINIM)