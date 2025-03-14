#!/bin/bash

# run "accelerate config" first!
# Set environment variables to optimize memory allocation and increase NCCL timeout
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1200 # Increase NCCL timeout to 1200 seconds
# Specify the GPUs to be used
# export CUDA_VISIBLE_DEVICES=2,3

accelerate launch LaMed/src/train/train-external.py \
    --version v0 \
    --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
    --model_type phi3 \
    --lora_enable True \
    --vision_tower vit3d \
    --pretrain_vision_model ./LaMed/pretrained_model/M3D-CLIP/pretrained_ViT.bin \
    --pretrain_mm_mlp_adapter ./LaMed/output/LaMed-pretrain-0000/mm_projector.bin \
    --segmentation_module segvol \
    --pretrain_seg_module ./LaMed/pretrained_model/SegVol/pytorch_model.bin \
    --bf16 True \
    --output_dir ./LaMed/output/LaMed-finetune-0000 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.04 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True\
    --dataloader_num_workers 8 \
    --report_to tensorboard \
    --cap_data_path /media/Datacenter_storage/Devam/CT/CurateCTdatasets/All_findings_impressionv3.csv