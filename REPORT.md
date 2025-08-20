## CS787 Technical Report

### Scope
Generate synthetic medical images based on textual descriptions and specified imaging modality (e.g., “breast with tumor, MRI”; “lung with pneumonia, chest X-ray”).

Support for image generation across various imaging modalities:

* Optical Coherence Tomography (OCT)
* Fundus
* Chest X-ray
* Chest Computed Tomography (CT)
* Brain MRI

further extensible to new modalities and prompts.

### Possible improvements over existing implementation
- Basic cross-attention with general-domain BERT tokenizer lacks medical vocabulary specificity and spatial reasoning capabilities. Replace with medical-domain pre-trained transformers and enhanced attention mechanisms. Implementing PubMedBERT or ClinicalBERT would provide substantial improvements in medical terminology understanding. These models are pre-trained on millions of medical texts, offering better semantic representations for clinical concepts like "bilateral infiltrates" or "contrast enhancement patterns". The medical vocabulary coverage is significantly higher—PubMedBERT contains specialized tokens for anatomical structures, pathological findings, and imaging characteristics that general BERT models lack.
- Adding transformer blocks within U-Net layers creates spatial-textual feature alignment at multiple resolution scales. Unlike basic cross-attention that operates at fixed resolutions, integrating self-attention within encoder/decoder blocks enables the model to capture long-range spatial dependencies crucial for anatomical coherence. This architecture allows features at 64×64 resolution to attend to both local tissue patterns and global anatomical context simultaneously.
- Standard skip connections can be enhanced with attention-gated mechanisms that selectively emphasize clinically relevant features. For medical images, certain anatomical landmarks (heart borders, organ boundaries) require higher preservation than background regions. Attention gates learn to weight skip connections based on clinical importance, improving anatomical consistency across generated images.
- Medical images contain features at vastly different scales—from cellular structures to entire organ systems. Implementing Feature Pyramid Networks (FPN) within the U-Net allows simultaneous processing of fine pathological details and gross anatomical structures. This is particularly crucial for modalities like histopathology or retinal imaging where microscopic and macroscopic features must be coherently generated.
- Maybe instead of direct "modality + description" input, implement a multi-stage reasoning process - Anatomical Localization: "Identify the chest region and lung fields", Pathology Assessment: "Detect signs of consolidation in the right lower lobe", and then Visual Feature Mapping: "Generate corresponding opacity patterns with air bronchograms". Implement clinical reporting templates that guide generation through standard radiological interpretation frameworks like "History → Technique → Findings → Impression." This structured approach ensures generated images align with established medical imaging protocols.
- Implementing RAG with medical databases (RadLex, SNOMED-CT, medical literature) enables dynamic retrieval of relevant case studies during generation. When generating "pneumonia chest X-ray," the system retrieves similar confirmed cases, anatomical references, and established imaging patterns to guide synthesis. This approach significantly improves clinical plausibility by grounding generation in validated medical knowledge.
- Deploy specialized medical AI agents for generated image verification. Agents can provide structured feedback for automated refinement: "Generated chest X-ray shows heart size inconsistent with described cardiomegaly—increase cardiac silhouette by 15%." This creates a self-improving system that continuously enhances clinical accuracy without human intervention.

This approach mirrors clinical decision-making and produces more diagnostically accurate images. The intermediate steps provide interpretable checkpoints for quality validation and enable targeted improvements during training.

### High Level Architecture

- Parses CSV  and loads paired image-text-modality [path, text, modality] data. Loads and pre-processes images to standard resolution 256×256 with aspect ratio preservation. Normalize to [-1, 1] for diffusion model compatibility. Work on .png and .jpg formats. Maybe some batch processing? Load data in chunks rather than entire dataset in memory to handle large CSV files without memory overflow?
- Tokenizer will convert textual prompts ("modality, finding, organ") into token IDs and attention masks. We can use some medical-domain transformer tokenizer (e.g., PubMedBERT - smaller vocabulary, faster processing) which would handle both modality tokens (MRI, CT, etc.) and rich description tokens supporting prompt templates for data augmentation. Implement prompt augmentation for training robustness.
- Text-condition encoder (Transformer) would embed tokenized prompts to provide text conditioning for the generator (Token IDs → embeddings). Pre-trained transformer encoder (BioBERT, ClinicalBERT, etc) outputs sequence of embeddings (per token) for cross-attension and pooled embedding (whole sequence) for global conditioning
- Modality Encoder (Modality → embedding, fusion with text) explicitly embeds imaging modality for conditioning or element-wise fusion (Hadamard product + linear projection). We would get Transform combined embeddings for U-Net consumption
- We would use a U-Net architecture (encoder-decoder with skip connections) as main generator (image noise reversal), that handles progressive noise addition and removal and supports mixed-precision (fp16 for forward pass and fp32 for gradient computation) for efficiency.
- Next, we would use cross-attension blocks to merge text (and modality) features with image features at each U-Net stage
- Also, we can apply LoRA to cross-attention layers, self-attention in middle blocks, and final output layers
- We would use Classifier-Free Guidance (CFG) for adjustable strength of prompt adherence, Negative Prompting to suppress unwanted features, and Support for unconditional/conditional sampling and dynamic CFG scaling. Now, we would have end-to-end model accepting text+modality input, producing conditioned noise predictions
- We would use Stochastic noise scheduler (linear or cosine) with L2 loss for noise prediction. Might need some gradient clipping for preventing explosion of gradinents. Adds controlled Gaussian noise at each timestep to the image input and Trains U-Net to predict noise or x0 (clean image) per timestep. Generate images every N steps.
- For quantitative evaluation, we can use FID (Frechet Inception Distance) for per modality and overall, IS (Inception Score) to measure image quality and diversity, MS-SSIM for structural similarity assessment, CLIP Score for Text-image alignment measurement, etc
- If time permits, we can also have 2-stage RL pipeline using RAG.
- We will do distributed training on GPUs
---
