#!/bin/bash

model_name_or_path=bigscience/bloomz-7b1-mt
data_path=data/data.json
model_max_length=2048
output_dir=checkpoints/llms_bloom_7b/

accelerate launch --config_file ./peft/accelerate_config.yaml ./peft/train.py \
  --model_name_or_path ${model_name_or_path} \
  --model_max_length ${model_max_length} \
  --data_path ${data_path} \
  --output_dir ${output_dir} \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --save_strategy "steps" \
  --save_steps 500 \
  --evaluation_strategy "no" \
  --save_total_limit 3 \
  --learning_rate 3e-3 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --gradient_checkpointing True \
  --lora True \
  --lora_r 8 \
  --lora_alpha 32 \
  --lora_dropout 0.1