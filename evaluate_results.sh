#!/bin/bash

# Comprehensive evaluation script for comparing different model outputs
# This script evaluates FID, IS, SSIM, and medical-specific metrics

set -e

echo "=== Medical Image Generation Evaluation ==="
echo "Comparing different model outputs against ground truth fundus images"
echo

# Set up paths
REAL_IMAGES_DIR="datasets/example/fundus"
BIOMEDBERT_256_DIR="generated_biomedbert_256"
FUNDUS_320_DIR="generated_fundus_320_9"

# Create CSV files for metrics evaluation
create_csv_for_metrics() {
    local dir=$1
    local csv_file=$2
    
    echo "path" > "$csv_file"
    for img in "$dir"/*.png; do
        echo "$img" >> "$csv_file"
    done
}

# Create CSV files
echo "Creating CSV files for metrics evaluation..."
create_csv_for_metrics "$REAL_IMAGES_DIR" "real_images.csv"
create_csv_for_metrics "$BIOMEDBERT_256_DIR" "biomedbert_256_images.csv"
create_csv_for_metrics "$FUNDUS_320_DIR" "fundus_320_images.csv"

echo "CSV files created:"
echo "- real_images.csv (ground truth)"
echo "- biomedbert_256_images.csv (BioMedBERT 256px)"
echo "- fundus_320_images.csv (Standard 320px)"

echo
echo "=== Running Standard Metrics (FID, IS, SSIM) ==="

# Run metrics evaluation
echo "Evaluating BioMedBERT 256px vs Ground Truth..."
UV_NO_SYNC=1 uv run python metrics/metrics.py \
    --real_images real_images.csv \
    --generated_images biomedbert_256_images.csv \
    --device cuda:1 > biomedbert_256_metrics.txt 2>&1

echo "Evaluating Standard 320px vs Ground Truth..."
UV_NO_SYNC=1 uv run python metrics/metrics.py \
    --real_images real_images.csv \
    --generated_images fundus_320_images.csv \
    --device cuda:1 > fundus_320_metrics.txt 2>&1

echo
echo "=== Medical-Specific Quality Metrics ==="

# Create a Python script for medical metrics evaluation
cat > evaluate_medical_metrics.py << 'EOF'
import numpy as np
import os
from PIL import Image
from scipy import ndimage
import json

def compute_medical_metrics(image_path):
    """Compute medical-specific quality metrics for fundus images"""
    img = Image.open(image_path).convert('L')
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    metrics = {}
    
    # Signal-to-Noise Ratio (SNR)
    metrics['snr'] = np.mean(img_array) / (np.std(img_array) + 1e-8)
    
    # Contrast
    metrics['contrast'] = np.max(img_array) - np.min(img_array)
    
    # Edge sharpness (using Sobel gradient)
    gradient = ndimage.sobel(img_array)
    metrics['sharpness'] = np.mean(np.abs(gradient))
    
    # Vessel visibility (high-frequency content)
    from scipy.ndimage import gaussian_filter
    low_freq = gaussian_filter(img_array, sigma=2)
    high_freq = img_array - low_freq
    metrics['vessel_visibility'] = np.std(high_freq)
    
    # Brightness uniformity
    center_region = img_array[img_array.shape[0]//4:3*img_array.shape[0]//4,
                             img_array.shape[1]//4:3*img_array.shape[1]//4]
    edge_region = np.concatenate([
        img_array[:img_array.shape[0]//4, :].flatten(),
        img_array[3*img_array.shape[0]//4:, :].flatten(),
        img_array[:, :img_array.shape[1]//4].flatten(),
        img_array[:, 3*img_array.shape[1]//4:].flatten()
    ])
    metrics['uniformity'] = np.mean(center_region) - np.mean(edge_region)
    
    return metrics

def evaluate_directory(dir_path, name):
    """Evaluate all images in a directory"""
    print(f"\n=== {name} Medical Metrics ===")
    
    all_metrics = []
    for img_file in sorted(os.listdir(dir_path)):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(dir_path, img_file)
            metrics = compute_medical_metrics(img_path)
            all_metrics.append(metrics)
            print(f"{img_file}: SNR={metrics['snr']:.3f}, Contrast={metrics['contrast']:.3f}, "
                  f"Sharpness={metrics['sharpness']:.3f}, Vessel={metrics['vessel_visibility']:.3f}")
    
    # Compute averages
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    print(f"\n{name} Averages:")
    for key, value in avg_metrics.items():
        print(f"  {key}: {value:.3f}")
    
    return avg_metrics

if __name__ == "__main__":
    # Evaluate different model outputs
    real_metrics = evaluate_directory("datasets/example/fundus", "Ground Truth Fundus")
    biomedbert_metrics = evaluate_directory("generated_biomedbert_256", "BioMedBERT 256px")
    fundus_320_metrics = evaluate_directory("generated_fundus_320_9", "Standard 320px")
    
    # Save results
    results = {
        "ground_truth": real_metrics,
        "biomedbert_256": biomedbert_metrics,
        "fundus_320": fundus_320_metrics
    }
    
    with open("medical_metrics_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n=== Summary Comparison ===")
    print("Metric\t\tGround Truth\tBioMedBERT\tStandard 320px")
    print("-" * 60)
    for metric in ["snr", "contrast", "sharpness", "vessel_visibility", "uniformity"]:
        print(f"{metric:<15}\t{real_metrics[metric]:.3f}\t\t{biomedbert_metrics[metric]:.3f}\t\t{fundus_320_metrics[metric]:.3f}")
EOF

# Run medical metrics evaluation
echo "Running medical-specific quality metrics..."
UV_NO_SYNC=1 uv run python evaluate_medical_metrics.py

echo
echo "=== Results Summary ==="

# Display standard metrics results
echo "Standard Metrics (FID, IS, SSIM):"
echo "-----------------------------------"
echo "BioMedBERT 256px:"
cat biomedbert_256_metrics.txt
echo
echo "Standard 320px:"
cat fundus_320_metrics.txt

echo
echo "Medical Metrics saved to: medical_metrics_results.json"
echo "Standard metrics saved to: biomedbert_256_metrics.txt, fundus_320_metrics.txt"

# Clean up temporary files
rm -f real_images.csv biomedbert_256_images.csv fundus_320_images.csv evaluate_medical_metrics.py

echo
echo "=== Evaluation Complete ==="
echo "Check the generated files for detailed metrics comparison."
