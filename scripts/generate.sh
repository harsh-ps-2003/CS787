# Generate images using trained models

UV_NO_SYNC=1 uv run python ../generate.py --model_used=checkpoint/sd-model-finetuned-on-fundus/checkpoint-100 --prompt="fundus" --device="cuda:0" --img_num=100 --output_dir="generated_img/fundus"
