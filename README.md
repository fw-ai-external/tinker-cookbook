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

Install the Fireworks training cookbook in your environment and set your API key:

```text
"fireworks-training-cookbook @ git+https://github.com/fw-ai/cookbook.git#subdirectory=training ; python_version >= '3.11'",
```

```bash
export FIREWORKS_API_KEY=...
```

The RL, on-policy distillation, and SDFT training loops need a trainer and a
rollout deployment. They can create both for the run, or reuse ones you already
have.

### Let the run provision

Leave `base_url` and `fireworks_deployment_id` unset, and set
`fireworks_training_shape_id`. It selects the shape of the new trainer and the
new deployment:

```bash
python -m tinker_cookbook.recipes.math_rl.train \
    fireworks_base_model=accounts/fireworks/models/qwen3p5-9b \
    fireworks_training_shape_id=accounts/fireworks/trainingShapes/qwen3p5-9b-256k-lora
```

The run logs the trainer job ID and deployment ID it uses. When it ends, a
trainer it created is deleted and a deployment it created is scaled to zero.
Set `fireworks_cleanup_on_exit=False` on the training config to keep them.

### Reuse an existing trainer and deployment

Pass the trainer as `base_url` and the deployment as `fireworks_deployment_id`.
You can also set only one of them and let the run create the other; creating
the deployment still needs `fireworks_training_shape_id`. Reused resources are
never deleted or scaled down.

If you have an rlor-trainer-job ID instead of a URL, build the URL as
`https://api.fireworks.ai/training/v1/rlorTrainerJobs/<account>/<job-id>`. The
`fireworks` Claude Code skill in `skills/fireworks/` teaches agents the same
translation.

```bash
python -m tinker_cookbook.recipes.math_rl.train \
    base_url="https://api.fireworks.ai/training/v1/rlorTrainerJobs/<account>/<job-id>" \
    fireworks_deployment_id=<deployment-id> \
    fireworks_base_model=accounts/fireworks/models/<model>
```

Supervised training needs only a trainer and handles it the same way: leave
`base_url` unset to create one (optionally with `fireworks_training_shape_id`),
or pass a trainer URL to reuse it.

Off-policy distillation and distillation teachers on a separate trainer
(`teacher_base_url`) are not provisioned by the run. Pass an existing trainer
as `base_url` for those.
