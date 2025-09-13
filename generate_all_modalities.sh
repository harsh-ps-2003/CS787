#!/bin/bash

# Set GPU environment variables
export CUDA_VISIBLE_DEVICES=1
export CUDNN_DISABLE=1
export TRITON_DISABLE=1

# Common parameters
MODEL="runwayml/stable-diffusion-v1-5"
DEVICE="cuda:0"
STEPS=15
PRECISION="float32"
HEIGHT=256
WIDTH=256
SCHEDULER="euler_a"
NUM_IMAGES=3

# Function to generate images
generate_images() {
    local prompt="$1"
    local output_dir="$2"
    
    echo "Generating: $output_dir"
    python generate.py --use_pretrained_only \
        --pretrained_model "$MODEL" \
        --prompt "$prompt" \
        --img_num "$NUM_IMAGES" \
        --device "$DEVICE" \
        --num_inference_steps "$STEPS" \
        --precision "$PRECISION" \
        --height "$HEIGHT" \
        --width "$WIDTH" \
        --scheduler "$SCHEDULER" \
        --output_dir "$output_dir"
}

# Generate all modalities
generate_images "Chest X-ray: normal lung fields without infiltrates" "generated_cxr_normal"
generate_images "Chest X-ray: bilateral pneumonia with consolidation" "generated_cxr_pneumonia"
generate_images "Brain MRI: T1-weighted image showing normal anatomy" "generated_mri_normal"
generate_images "Brain MRI: T1-weighted image with enhancing mass in right temporal lobe" "generated_mri_tumor"
generate_images "Chest CT: normal lung parenchyma without nodules" "generated_ct_normal"
generate_images "Chest CT: lung nodule in right upper lobe" "generated_ct_nodule"
generate_images "Fundus: healthy retina with clear optic disc" "generated_fundus_normal"
generate_images "Fundus: diabetic retinopathy with microaneurysms and hard exudates" "generated_fundus_diabetic"
generate_images "OCT: healthy retinal layers with clear foveal depression" "generated_oct_normal"
generate_images "OCT: macular degeneration with drusen and retinal thinning" "generated_oct_macular"
generate_images "Breast MRI: normal breast tissue without masses" "generated_breast_normal"
generate_images "Breast MRI: irregular enhancing mass in upper outer quadrant" "generated_breast_lesion"

echo "All modalities generated!"
