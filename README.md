# Tinker Cookbook Fireworks Fork

This fork keeps the Tinker Cookbook training abstractions and recipes, with extra support for running them against Firetitan / Fireworks training infrastructure.

The original upstream project is [thinking-machines-lab/tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook).

## What This Fork Adds

- SFT, RL, and distillation recipes that train and sample on Fireworks.
- LoRA and full-parameter fine-tuning.
- Every model Fireworks can fine-tune, across the DeepSeek, Gemma, GLM, Kimi, Llama, MiniMax, Mistral, Nemotron, and Qwen families. See the [fine-tuning model list](https://docs.fireworks.ai/fine-tuning/models) for the current models and what each one supports.
- Long-context training, up to 524k tokens on the largest training shapes.

## Basic Client Example

Fireworks serverless training needs no trainer or deployment. Point the client
at the serverless endpoint and train a LoRA adapter, here on Qwen3.8-27B:

```python
import tinker
from fireworks.training.sdk import FiretitanServiceClient

service_client = FiretitanServiceClient(
    base_url="https://api.fireworks.ai/training/v1/serverless"
)
training_client = service_client.create_lora_training_client(
    base_model="accounts/fireworks/models/qwen3p8-27b",
    rank=32,
)
training_client.forward_backward(...)
training_client.optim_step(...)
training_client.save_state(...)
training_client.load_state(...)
```

Serverless training is LoRA only, so `rank` must be positive. Full-parameter
fine-tuning (`rank=0`) needs a dedicated trainer; see [Setup](#setup).

## Setup

Install the Fireworks training cookbook in your environment and set your API key:

```text
"fireworks-training-cookbook @ git+https://github.com/fw-ai/cookbook.git#subdirectory=training ; python_version >= '3.11'",
```

```bash
export FIREWORKS_API_KEY=...
```

### Serverless training

Serverless training runs on a shared trainer, so there is nothing to provision.
Pass the serverless endpoint as `base_url`:

```bash
python -m tinker_cookbook.recipes.math_rl.train \
    base_url=https://api.fireworks.ai/training/v1/serverless \
    model_name=Qwen/Qwen3.8-27B \
    fireworks_base_model=accounts/fireworks/models/qwen3p8-27b
```

Serverless training is LoRA only. RL, supervised training, and off-policy
distillation support it; on-policy distillation and SDFT need a dedicated
trainer and deployment. See the
[fine-tuning model list](https://docs.fireworks.ai/fine-tuning/models) for the
models that support serverless training.

### Dedicated training

For a dedicated run, the trainer and deployment instances are provisioned
automatically, and Fireworks cleans them up once they are idle. Leave
`fireworks_trainer_job_id`, `base_url`, and `fireworks_deployment_id` unset, and
set `fireworks_training_shape_id`. It selects the shape of the new trainer and
the new deployment:

```bash
python -m tinker_cookbook.recipes.math_rl.train \
    fireworks_base_model=accounts/fireworks/models/qwen3p5-9b \
    fireworks_training_shape_id=accounts/fireworks/trainingShapes/qwen3p5-9b-256k-lora
```

The run logs the trainer job ID and deployment ID it uses and leaves both in
place when it ends or fails, so you can reuse them in a later run.

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

Distillation teachers on a separate trainer are not provisioned by the run.
Name an existing trainer with `teacher_fireworks_trainer_job_id` (or
`teacher_base_url`).
