# Trajectory Consistency Distillation
Trajectory Consistency Distillation is a method introduced in [TCD paper](https://arxiv.org/abs/2402.19159).

The method improve [LCM](https://arxiv.org/abs/2310.04378)'s quality by optimize Consistency Model on not only timestep 0 but also timesteps `s` away from 0.
Originally stated:
> We first observed that several errors in the distillation process are related to the time interval
t → s of the consistency function. Thus, we leverage the semi-linear structure with exponential integrators of the PF-ODE
for parameterization, which also supports a shorter interval (i.e., moderately moving the upper limit s away from 0).

## Full model distillation

```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="/path/to/saved/model"

accelerate launch train_tcd_distill_sd_wds.py \
    --pretrained_teacher_model=$MODEL_NAME \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision=fp16 \
    --resolution=512 \
    --learning_rate=4e-6 --loss_type="l2" --ema_decay=0.95 --adam_weight_decay=0.01 \
    --max_train_steps=100000 \
    --max_train_samples=4000000 \
    --dataloader_num_workers=8 \
    --train_shards_path_or_url="pipe:curl -L -s https://huggingface.co/datasets/laion/conceptual-captions-12m-webdataset/resolve/main/data/{00000..01099}.tar?download=true" \
    --validation_steps=200 \
    --checkpointing_steps=200 --checkpoints_total_limit=10 \
    --train_batch_size=12 \
    --gradient_checkpointing --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=1 \
    --use_8bit_adam \
    --resume_from_checkpoint=latest \
    --report_to=tensorboard \
    --seed=453645634 \
```

## LoRA distillation
```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="path/to/saved/model"

accelerate launch train_tcd_distill_lora_sd_wds.py \
    --pretrained_teacher_model=$MODEL_NAME \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision=fp16 \
    --resolution=512 \
    --lora_rank=64 \
    --learning_rate=4.5e-6 --loss_type="l2" --adam_weight_decay=0.01 \
    --max_train_steps=10000 \
    --max_train_samples=4000000 \
    --dataloader_num_workers=8 \
    --train_shards_path_or_url="pipe:curl -L -s https://huggingface.co/datasets/laion/conceptual-captions-12m-webdataset/resolve/main/data/{00000..01099}.tar?download=true" \
    --validation_steps=200 \
    --checkpointing_steps=200 --checkpoints_total_limit=10 \
    --train_batch_size=12 \
    --gradient_checkpointing --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=1 \
    --use_8bit_adam \
    --resume_from_checkpoint=latest \
    --report_to=tensorboard \
    --seed=453645634 \
```
