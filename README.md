# Tinker Cookbook Fireworks Fork

This fork keeps the Tinker Cookbook training abstractions and recipes, with extra support for running them against Firetitan / Fireworks training infrastructure.

The original upstream project is [thinking-machines-lab/tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook).

## What This Fork Adds

- Firetitan service client wiring for SFT, RL, and distillation recipes.
- Full-parameter training and LoRA training.
- Support for more model families, including Qwen, GLM5-class models, Gemma4, and MiniMax M2-class models.
- Long-context training shapes, up to 256k context.

## Basic Client Example

```python
import tinker
from fireworks.training.sdk import FiretitanServiceClient

service_client = FiretitanServiceClient(base_url="https://api.fireworks.ai/training/v1/...")
training_client = service_client.create_lora_training_client(
    base_model="Qwen/Qwen3-4B-Instruct-2507",
    rank=0,
)
training_client.forward_backward(...)
training_client.optim_step(...)
training_client.save_state(...)
training_client.load_state(...)
```

Use `rank=0` for full-parameter fine-tuning, or a positive rank for LoRA fine-tuning.

## Setup

Install the Fireworks training cookbook provisioner in your environment:

```text
"fireworks-training-cookbook @ git+https://github.com/fw-ai/cookbook.git#subdirectory=training ; python_version >= '3.11'",
```

Full provisioning docs for RL/RFT, SFT, and distillation live in
[`tinker_cookbook/fireworks/PROVISIONING.md`](tinker_cookbook/fireworks/PROVISIONING.md).

The short version for RL trainer + rollout deployment:

```bash
export FIREWORKS_API_KEY=...

python -m training.provision.provision \
  --config-name fireworks_rft \
  common.base_model=accounts/fireworks/models/qwen3p5-35b-a3b \
  common.tokenizer_model=Qwen/Qwen3.5-35B-A3B \
  common.lora_rank=128 \
  deployments.rollout.replica_count=1 \
  trainers.policy.training_shape_id=accounts/fireworks/trainingShapes/qwen3p5-35b-a3b-256k-lora \
  trainers.policy.replica_count=1
```

After provisioning, pass the printed trainer endpoint and deployment identifiers
into the recipe you want to run:

```bash
python -m tinker_cookbook.recipes.math_rl.train \
    base_url="https://api.fireworks.ai/training/v1/rlorTrainerJobs/<account>/<job-id>" \
    fireworks_deployment_id=<deployment-id> \
    fireworks_base_model_name=accounts/fireworks/models/<model>
```
