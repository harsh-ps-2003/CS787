#!/bin/bash

# Generate diverse fundus images with different medical conditions and variations
# Similar to MINIM project's diverse medical image generation

set -e

echo "=== Generating Diverse Fundus Images ==="
echo "Creating varied fundus images with different medical conditions and anatomical variations"
echo

# Set up paths
MODEL_PATH="./checkpoints/medical-model-biomedbert-256"
OUTPUT_DIR="generated_diverse_fundus"
DEVICE="cuda:1"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Define diverse prompts for different fundus conditions and variations
declare -a PROMPTS=(
    "A fundus image showing diabetic retinopathy with microaneurysms and hemorrhages scattered throughout the retina, indicating early stage diabetic eye disease"
    "Fundus photograph revealing macular degeneration with drusen deposits and geographic atrophy in the central retina"
    "Retinal image displaying hypertensive retinopathy with arteriolar narrowing and cotton wool spots"
    "Fundus view showing glaucoma with increased cup-to-disc ratio and optic nerve head excavation"
    "Retinal photograph demonstrating retinal vein occlusion with extensive hemorrhages and macular edema"
    "Fundus image revealing age-related macular degeneration with large drusen and pigmentary changes"
    "Retinal view showing retinal detachment with elevated retina and visible retinal tears"
    "Fundus photograph displaying optic neuritis with optic disc swelling and peripapillary hemorrhages"
    "Retinal image showing choroidal neovascularization with subretinal fluid and hemorrhage"
    "Fundus view revealing retinal artery occlusion with cherry-red spot and retinal whitening"
    "A healthy fundus image with clear optic disc, normal vessel caliber, and intact macula"
    "Fundus photograph showing myopic degeneration with tilted optic disc and peripheral retinal changes"
    "Retinal image displaying retinitis pigmentosa with bone spicule pigmentation and vessel attenuation"
    "Fundus view revealing central serous chorioretinopathy with serous retinal detachment"
    "Retinal photograph showing vitreous hemorrhage with obscured retinal details"
)

echo "Generating ${#PROMPTS[@]} diverse fundus images..."
echo

# Generate images for each prompt
for i in "${!PROMPTS[@]}"; do
    prompt="${PROMPTS[$i]}"
    image_num=$((i + 1))
    
    echo "[$image_num/${#PROMPTS[@]}] Generating: ${prompt:0:80}..."
    
    # Create a temporary output directory for this specific image
    temp_output_dir="${OUTPUT_DIR}/temp_${image_num}"
    mkdir -p "$temp_output_dir"
    
    # Use the working generate.py script with BioMedBERT support
    UV_NO_SYNC=1 uv run python generate.py \
        --pretrained_model "$MODEL_PATH" \
        --prompt "$prompt" \
        --img_num 1 \
        --device "$DEVICE" \
        --num_inference_steps 50 \
        --precision float32 \
        --height 256 \
        --width 256 \
        --scheduler "euler_a" \
        --output_dir "$temp_output_dir"
    
    # Move the generated image to the final location with proper naming
    if [ -f "$temp_output_dir/generated_1.png" ]; then
        mv "$temp_output_dir/generated_1.png" "$OUTPUT_DIR/diverse_${image_num}.png"
        echo "Generated: diverse_${image_num}.png"
    else
        echo "Warning: No image generated for prompt $image_num"
    fi
    
    # Clean up temp directory
    rm -rf "$temp_output_dir"
    
    echo "Completed: diverse_${image_num}.png"
    echo
done

echo "=== Generation Complete ==="
echo "Generated ${#PROMPTS[@]} diverse fundus images in: $OUTPUT_DIR"
echo
echo "Images include:"
echo "- Diabetic retinopathy"
echo "- Macular degeneration" 
echo "- Hypertensive retinopathy"
echo "- Glaucoma"
echo "- Retinal vein occlusion"
echo "- Retinal detachment"
echo "- Optic neuritis"
echo "- Choroidal neovascularization"
echo "- Retinal artery occlusion"
echo "- Healthy fundus"
echo "- Myopic degeneration"
echo "- Retinitis pigmentosa"
echo "- Central serous chorioretinopathy"
echo "- Vitreous hemorrhage"
echo
echo "These diverse images can be used for:"
echo "- Medical education"
echo "- Clinical training"
echo "- Dataset augmentation"
echo "- Model evaluation"
echo "- Research applications"
