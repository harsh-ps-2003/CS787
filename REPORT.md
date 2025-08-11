## CS787 Technical Report: Foundational Text-to-Image and MatterGPT-style Inverse Design

### Scope
This report documents the technical plan and architecture for:
1) A foundational text-to-image model implemented largely from scratch in PyTorch, using transformer-based text encoding and U-Net/ResNet-based image generation with diffusion or GAN back-ends and classifier-free guidance.
2) A follow-on MatterGPT-style crystal inverse design model using SLICES notation with a generative transformer decoder, optionally conditioned on properties; validated via DFT pipelines.

### References
- Transformers: [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- CLIP (multi-modal alignment): [CLIP](https://arxiv.org/pdf/2103.00020)
- Word2Vec for element embeddings: [word2vec](https://arxiv.org/pdf/1301.3781)
- MatterGPT (SLICES-based inverse design): [MatterGPT](https://arxiv.org/pdf/2408.07608)

---

### Foundational Text-to-Image

#### Components
- Text tokenizer/encoder:
  - Custom tokenizer supporting BERT-style WordPiece.
  - Transformer encoder (multi-head self-attention) producing a fixed-length embedding.
  - Optional BERT initialization for weights and tokenization pipeline.
- Image encoder/decoder:
  - CLIP-like image encoder (ResNet/ViT) for multimodal contrastive training in early phases.
  - U-Net generator (diffusion) or ResNet generator (GAN) for image synthesis.
- Conditioning and guidance:
  - Classifier-Free Guidance (CFG) for controllable adherence to text prompts.
  - Embedding adapters to fuse text features into the generator (cross-attention or FiLM-like conditioning).

#### Training Phases
1) Contrastive pretraining (CLIP-like): align text/image embeddings using InfoNCE.
2) Generative training:
   - Diffusion: noise scheduling, epsilon-prediction U-Net; CFG via null-conditioning paths.
   - GAN (optional baseline): conditional generator/discriminator; compare FID/IS/CLIPScore.
---

### MatterGPT-style Inverse Design

#### SLICES and Tokenization
- Implement a reversible SLICES tokenizer that maps crystal structures to unambiguous string sequences suitable for transformer decoding.
- Add element embeddings (can evaluate word2vec-style pretraining for element tokens).

#### Generator and Property Heads
- Autoregressive transformer decoder trained with next-token prediction.
- Single and multi-property conditioning via learned adapters; properties include formation energy, band gap, etc.
- Sampling: nucleus/top-k/top-p, temperature, repetition penalties; validate validity/uniqueness/novelty.

#### Verification via DFT
- Integrate ASE-based pipelines and support common DFT backends.
- Cache computations, track provenance, and store relaxed structures and computed properties.

---

### Architecture and Hexagonal Boundaries
- Domain: tokenizers, encoders/decoders, diffusion/GAN primitives, schedulers, property heads.
- Application: trainers, evaluators, sampling pipelines, DFT orchestration.
- Adapters: datasets, logging/metrics, file I/O, CLI/HTTP interfaces.

### Operational Considerations
- Reproducibility: fixed seeds, config files, and run manifests.
- Observability: metrics dashboards (loss, FID, CLIPScore, property RMSE), resource usage.
- Performance: mixed precision, gradient checkpointing, efficient attention variants for long prompts.
- Security: sanitization of prompts and SLICES strings; safe file handling; no secrets in code.

### Next Actions
- Set up project skeleton and tests for tokenizer and text encoder.
- Implement minimal CLIP-like contrastive pretrain to validate embeddings.
- Build diffusion U-Net with CFG path; train on a toy dataset for smoke tests.
- Start SLICES tokenizer and small decoder; design property conditioning heads and stub DFT hooks.

---

### Training System Specifications (CPU-only Plan)
- Platform: Linux 4.15.0-194-generic (x86_64)
- Python: 3.11.13
- CPU: 16 physical / 32 logical cores @ 813 MHz
- RAM: 46.81 GB (43.46 GB available)
- Storage: 250.92 GB total, 80.26 GB free
- GPUs detected: NVIDIA TITAN Xp x2, but we will operate in CPU-only mode for training and verification.

Operational guidance for CPU-only execution:
- Prefer PyTorch MKL/OpenBLAS builds; export OMP_NUM_THREADS/torch.set_num_threads to fully utilize cores.
- Start with small image sizes (e.g., 64x64) and progressively grow (PAG) if feasible.
- Use gradient accumulation to emulate larger effective batch sizes.
- Favor compact model widths/depths, parameter-efficient adapters, and checkpointing to reduce memory.
- Aggressively cache datasets/feature embeddings and DFT intermediates on disk.

---

### Roadmap: Foundational Model to MatterGPT (CPU-only)

Goal A: Foundational text-to-image system that takes a text prompt and generates an image, using classifier-free guidance (CFG). Early stage uses a CLIP-like setup; text encoded by transformers (initially BERT), image handled by U-Net or ResNet. Custom tokenizer will replace BERT tokenizer once stable. Generation via diffusion or GAN.

Phase 1: Foundational Model Development (Months 1–8)

1) Text Encoder and Tokenizer
- Implement a Transformer-based text encoder; initialize with pre-trained BERT to reduce CPU training time.
- Build a custom tokenizer; initially reuse BERT WordPiece, then iterate toward domain-tailored vocabulary. While the custom tokenizer targets SLICES long-term, keep it general enough for generic text prompts in this phase.
- Optional: Train/fine-tune text embeddings to capture material descriptors and crystal-related attributes where applicable.

2) Image/Structure Encoder
- Implement U-Net and ResNet variants for image embeddings and generation backbones.
- Start with lightweight configurations (e.g., base channels 32–64, depth 2–3) to be CPU friendly.
- Optional: add a compact graph encoder prototype to experiment with crystal connectivity features for structure-as-image mappings.

3) Contrastive Multimodal Training (CLIP-like)
- Train a CLIP-style model to align text and image/structure embeddings using InfoNCE.
- Use small curated datasets (e.g., thumbnail images or rendered structure diagrams) to keep CPU runs tractable; iterate with synthetic data if needed.

4) Diffusion or GAN Generator with Classifier-Free Guidance
- Implement a diffusion model with a U-Net epsilon-predictor and unconditional path for CFG; alternatively, prototype a conditional GAN baseline for faster CPU iteration.
- Integrate text conditioning via cross-attention or FiLM-like adapters; expose CFG scale to control prompt adherence.
- Validate samples qualitatively (prompt-image coherence) and quantitatively (CLIPScore proxy, FID on small splits) at 64x64; grow later if feasible.

Phase 2: Transition to MatterGPT

Goal B: Autoregressive transformer generator over SLICES for inverse design with multi-property conditioning; evaluated with CPU-based DFT pipelines.

1) Text Encoder and Tokenizer Refinement (Months 9–10)
- Finalize a SLICES-compatible tokenizer and vocabulary for MatterGPT.
- Fine-tune the text encoder on crystal-specific corpora to improve conditioning semantics.

2) Dataset Preparation and SLICES Optimization (Months 9–11)
- Expand datasets with diverse, validated SLICES strings and associated property labels (formation energy, band gap, elastic moduli).
- Implement strict validators and reversible round-trips for SLICES to ensure training data quality.

3) Implement MatterGPT Transformer Decoder (Months 10–12)
- Develop a GPT-style decoder trained with next-token prediction over SLICES sequences.
- Add property-conditioning tokens/embeddings to enable multi-objective inverse design.
- Optimize for CPU: reduce model width/depth, tie embeddings, share projections, apply gradient checkpointing.

4) Evaluation and Validation Pipeline (Months 11–12)
- Define metrics for validity, uniqueness, novelty, and property accuracy; compute with CPU-only tooling.
- Run tiered verification: fast ML property predictors first; DFT only for top-k promising candidates.
- Integrate ASE-based flows for CPU DFT backends and cache all intermediate artifacts.

5) Optimization and Deployment (Months 12–13)
- Tune schedules/hyperparameters for fidelity given CPU constraints (smaller batches, accumulation, longer but fewer runs).
- Provide CLI/SDK to sample structures with property targets; support batch generation and filtering.
- Prepare minimal UI/notebooks for exploration; document prompts, CFG scales, and sampling recipes.

CPU tactics for Phase 2:
- Prefer nucleus/top-k sampling to reduce decode time; cache logits for repeated prompts.
- Property heads trained with small MLPs; consider distillation from heavier models offline.
- Active learning with small DFT budgets per cycle; prioritize diversity and uncertainty.
