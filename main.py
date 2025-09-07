import yaml
import torch
from pathlib import Path
from RLHF.main import train as rlhf_train


def detect_gpu_config():
    """Detect and display GPU configuration."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"CUDA available: {gpu_count} GPU(s) detected")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            memory_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({memory_gb:.1f} GB)")
        return gpu_count
    else:
        print("CUDA not available, training will use CPU")
        return 0


def main():
    # Detect GPU configuration
    gpu_count = detect_gpu_config()
    
    # Use root config.yaml
    config_path = Path("config.yaml")
    config = yaml.safe_load(config_path.read_text())
    
    # Update config based on GPU availability
    if gpu_count > 0:
        print(f"Using {gpu_count} GPU(s) for RLHF training")
        # You can modify config here based on GPU count if needed
        # For example, adjust batch sizes, learning rates, etc.
    else:
        print("Using CPU for RLHF training (this will be slow)")
    
    rlhf_train(config)


if __name__ == "__main__":
    main()


