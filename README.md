# Consistency Models Distillation
Unofficial training scripts for consistency distillation models. Currently support:
- Trajectory Consistency Distillation
- Trajectory Segmented Consistency Distillation (introduced in Hyper-SD)

## Requirements

```bash
pip install diffusers["torch"] transformers accelerate
```

## Methods
- Start distill with [Trajectory Consistency Distillation-TCD](tcd/README.md)
- Start distill with [Trajectory Segmented Consistency Distillation-TSCD](tscd/README.md)

## Acknowledgments
The training scripts are originally base on [Diffusers's LCM training script](https://github.com/huggingface/diffusers/tree/main/examples/consistency_distillation)
