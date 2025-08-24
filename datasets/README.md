# Datasets Directory

This directory contains the medical image datasets and metadata for training the MINIM model.

## Directory Structure

```
datasets/
├── README.md              # This file
├── metadata.csv           # Main dataset metadata file
├── oct/                   # Optical Coherence Tomography
│   ├── healthy/           # Normal retinal images
│   ├── diabetic/          # Diabetic retinopathy cases
│   ├── macular/           # Macular degeneration
│   └── other_pathologies/ # Additional OCT pathologies
├── ct/                    # Computed Tomography
│   ├── chest/             # Chest CT scans
│   ├── abdomen/           # Abdominal CT scans
│   ├── brain/             # Brain CT scans
│   └── other/             # Other CT examinations
├── mri/                   # Magnetic Resonance Imaging
│   ├── brain/             # Brain MRI scans
│   ├── breast/            # Breast MRI scans
│   ├── spine/             # Spinal MRI scans
│   └── other/             # Other MRI examinations
├── xray/                  # X-Ray imaging
│   ├── chest/             # Chest X-rays
│   ├── abdomen/           # Abdominal X-rays
│   ├── extremities/       # Limb X-rays
│   └── other/             # Other X-ray examinations
└── fundus/                # Fundus photography
    ├── normal/            # Normal fundus images
    ├── diabetic/          # Diabetic retinopathy
    ├── hypertensive/      # Hypertensive retinopathy
    └── other/             # Other fundus pathologies
```

## Dataset Format Requirements

### CSV Metadata File
The main dataset file `metadata.csv` must contain exactly three columns:

| Column | Description | Example |
|--------|-------------|---------|
| `path` | Relative path to image file | `datasets/oct/healthy/retina_001.png` |
| `Text` | Text description/prompt | `OCT: healthy retinal layers with clear foveal depression` |
| `modality` | Imaging modality | `OCT` |

## Modality-Specific Considerations

### OCT (Optical Coherence Tomography)
- **Use case**: Retinal imaging for ophthalmology
- **Key features**: Layer visualization, retinal thickness, choroidal details
- **Common pathologies**: Diabetic retinopathy, macular degeneration, retinal detachment

### CT (Computed Tomography)
- **Use case**: Cross-sectional imaging for various body parts
- **Key features**: Bone detail, soft tissue contrast, 3D reconstruction
- **Common pathologies**: Tumors, fractures, infections, vascular disease

### MRI (Magnetic Resonance Imaging)
- **Use case**: Multi-planar imaging with excellent soft tissue contrast
- **Key features**: T1/T2 weighting, contrast enhancement, diffusion imaging
- **Common pathologies**: Brain tumors, breast lesions, spinal disorders

### X-Ray
- **Use case**: 2D radiographic imaging
- **Key features**: Bone detail, lung fields, cardiac silhouette
- **Common pathologies**: Fractures, pneumonia, heart disease

### Fundus
- **Use case**: Retinal photography for eye examination
- **Key features**: Retinal vessels, optic disc, macula
- **Common pathologies**: Diabetic retinopathy, hypertensive retinopathy, glaucoma
