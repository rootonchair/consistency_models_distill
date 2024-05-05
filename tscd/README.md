# Trajectory Segmented Consistency Distillation
Trajectory Segmented Consistency Distillation is a method introduced in [HyperSD](https://arxiv.org/abs/2404.13686), which divides the ODE trajectory into `k` segments then train Consistency Model on each segment.

The method using different `k` values to distill the model in multiple stages, the authors use `[8, 4, 2, 1]` in the paper. At first stage `k=8`, then in next stage the model is return training with `k=4`, and so on.

**Note:** Currently the script only support training consistency model. As stated in the original paper, they use a hybrid of MSE loss (higher weight in first stages) and (higher weight in latter stages) adversarial loss.

> For the distance
metric d, we employ a hybrid of adversarial loss, as proposed in sdxl-lightning[8], and Mean Squared Error (MSE)
Loss. Empirically, we observe that MSE Loss is more effective when the predictions and target values are proximate
(e.g., for k = 8, 4), whereas adversarial loss proves more
precise as the divergence between predictions and targets
increases (e.g., for k = 2, 1). Accordingly, we dynamically
increase the weight of the adversarial loss and diminish that
of the MSE loss across the training stages

## Full model distillation
```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="/path/to/saved/model"

accelerate launch train_tscd_distill_sd_wds.py \
    --pretrained_teacher_model=$MODEL_NAME \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision=fp16 \
    --resolution=512 \
    --learning_rate=1e-6 --loss_type="l2" --ema_decay=0.95 --adam_weight_decay=0.0 \
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

CUDA_VISIBLE_DEVICES=5 accelerate launch train_tscd_distill_lora_sd_wds.py \
    --pretrained_teacher_model=$MODEL_NAME \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision=fp16 \
    --resolution=512 \
    --lora_rank=64 \
    --learning_rate=1e-6 --loss_type="l2" --adam_weight_decay=0.0 \
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
