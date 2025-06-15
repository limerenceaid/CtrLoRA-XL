export MODEL_DIR="/data/jingshirou/diffusers/examples/controlnet/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="sdxl_weight/lora-01/square_masker_brush_border_masker_mix_lineart_lineart_color"
export HF_DATASETS_TRUST_REMOTE_CODE=True

CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 29501 scripts/train_lora.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --lora_condition="square_masker_brush_border_masker_mix_lineart_lineart_color" \
 --train_batch_size=1 \
 --gradient_accumulation_steps=1 \
 --img_root=data_original.train/image \
 --annotation_root=data_original.train/annotation_snowy \
 --annotation_list_path=data_original.train/annotation_snowy.train.jsonl \
 --checkpointing_steps 1 100 500 1000 2000 3000 5000 10000 20000 100000 \
 --resolution=1024 \
 --read_from_LAION2B=/ssd/hezhenliang/Projects/Meta_Projects/CtrLoRA-XL_Data_Processing/data_processed/laion2B-en-aesthetic_1024_plus \
 --controlnet_model_name_or_path=/data/jingshirou/diffusers/examples/controlnet/sdxl_weight/sdxl_weight_multi_lora_1024_finetune/checkpoint-28514-densepose-controlnet \
 --mixed_precision="fp16" \
 --learning_rate=1e-5 \
 --max_train_steps=100000 \
 --seed=42
