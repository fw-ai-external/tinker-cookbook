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

Leave `fireworks_trainer_job_id`, `base_url`, and `fireworks_deployment_id`
unset, and set `fireworks_training_shape_id`. It selects the shape of the new trainer and the
new deployment:

```bash
python -m tinker_cookbook.recipes.math_rl.train \
    fireworks_base_model=accounts/fireworks/models/qwen3p5-9b \
    fireworks_training_shape_id=accounts/fireworks/trainingShapes/qwen3p5-9b-256k-lora
```

The run logs the trainer job ID and deployment ID it uses, and leaves both in
place when it ends or fails, so you can reuse them. An idle trainer is cleaned
up by Fireworks; scale down or delete the deployment yourself when you are
done with it.

### Reuse an existing trainer and deployment

Pass the rlor-trainer-job ID as `fireworks_trainer_job_id` and the deployment as
`fireworks_deployment_id`.
You can also set only one of them and let the run create the other; creating
the deployment still needs `fireworks_training_shape_id`.

Reusing a trainer does not reuse its training state: every run starts from the
base model with a fresh optimizer unless it resumes an earlier run (same
`log_path`) or sets `load_checkpoint_path`.

A trainer URL in `base_url`
(`https://api.fireworks.ai/training/v1/rlorTrainerJobs/<account>/<job-id>`)
names a trainer too. Either option alone is enough; if both are set they must
name the same job, or the run stops with a configuration error. The `fireworks`
Claude Code skill in `skills/fireworks/` teaches agents these rules.

```bash
python -m tinker_cookbook.recipes.math_rl.train \
    fireworks_trainer_job_id=<job-id> \
    fireworks_deployment_id=<deployment-id> \
    fireworks_base_model=accounts/fireworks/models/<model>
```

Supervised training and off-policy distillation need only a trainer and handle
it the same way: set `fireworks_trainer_job_id` to reuse one, or leave it unset
to create one (optionally with `fireworks_training_shape_id`).

Distillation teachers on a separate trainer (`teacher_base_url`) are not
provisioned by the run. Pass an existing trainer URL for those.
