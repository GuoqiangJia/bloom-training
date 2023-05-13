#!/bin/bash

model_name_or_path=bigscience/bloomz-7b1-mt
data_path=data/data.json
gpu_ids=0
model_max_length=2048
output_dir=checkpoints/llms_bloom_7b/

accelerate launch --config_file ./peft/accelerate_config.yaml --gpu_ids ${gpu_ids} ./peft/train.py \
  --model_name_or_path ${model_name_or_path} \
  --model_max_length ${model_max_length} \
  --data_path ${data_path} \
  --output_dir ${output_dir} \
  --bf16 True \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 500 \
  --evaluation_strategy "no" \
  --save_total_limit 3 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
  --tf32 True \
  --gradient_checkpointing True \
  --lora True \
  --lora_dim 16 \
  --lora_alpha 16 \
  --lora_droppout 0.05