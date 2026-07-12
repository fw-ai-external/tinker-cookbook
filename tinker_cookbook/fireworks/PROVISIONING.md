# Fireworks Trainer and Deployment Provisioning

Use the Fireworks provisioning helper when a cookbook recipe needs a live
Fireworks trainer, and for RL/RFT/distillation-style recipes, a hot-loadable
rollout deployment (plus an optional forward-only teacher for distillation).
The helper creates the resources, prints their IDs, keeps them alive while you
work, and deletes newly-created resources when you stop it.

## Prerequisites

- Install the Fireworks training cookbook package in the environment that will
  run provisioning:

```text
"fireworks-training-cookbook @ git+https://github.com/fw-ai/cookbook.git#subdirectory=training ; python_version >= '3.11'",
```

- Set `FIREWORKS_API_KEY` for the account that should own the resources.

```bash
export FIREWORKS_API_KEY=fw-...
```

If you use `uv`, prefix the commands below with `uv run`.

## RL / RFT Trainer + Deployment

This provisions:

- one policy trainer from `trainers.policy`
- one rollout deployment from `deployments.rollout`
- weight sync from the trainer into that rollout deployment

Example for Qwen3.5 35B-A3B LoRA:

```bash
python -m training.provision.provision \
  --config-name fireworks_rft \
  common.base_model=accounts/fireworks/models/qwen3p5-35b-a3b \
  common.tokenizer_model=Qwen/Qwen3.5-35B-A3B \
  common.lora_rank=128 \
  deployments.rollout.replica_count=1 \
  trainers.policy.training_shape_id=accounts/fireworks/trainingShapes/qwen3p5-35b-a3b-256k-lora \
  trainers.policy.replica_count=1
```

The process prints heartbeat lines like:

```text
Fireworks rl infra alive | trainer=<trainer_job_id> | deployment=<deployment_id>
```

Copy those IDs into the training config or script that will connect from
`tinker-cookbook`:

- `base_url`: `https://api.fireworks.ai/training/v1/rlorTrainerJobs/<account>/<trainer_job_id>`
- `fireworks_deployment_id`: the printed deployment ID
- `fireworks_base_model`: usually the same value as `common.base_model`

For example, if the provisioner prints trainer `abc123` and deployment
`qwen3p5-35b-a3b-xyz` in account `fireworks`, pass:

```bash
base_url=https://api.fireworks.ai/training/v1/rlorTrainerJobs/fireworks/abc123
fireworks_deployment_id=qwen3p5-35b-a3b-xyz
fireworks_base_model=accounts/fireworks/models/qwen3p5-35b-a3b
```

Keep the provisioning process running while the training process uses the
resources. Press `Ctrl+C` when you are done; the provisioner will clean up the
resources it created.

## Distillation Student + Teacher

This provisions:

- one student policy trainer from `trainers.policy`
- one student rollout deployment from `deployments.rollout`
- one forward-only teacher trainer from `trainers.teacher_forward_only`

Example for a Qwen3.5 9B LoRA student with a Qwen3.5 9B forward-only teacher:

```bash
python -m training.provision.provision \
  --config-name fireworks_distillation \
  common.base_model=accounts/fireworks/models/qwen3p5-9b \
  common.tokenizer_model=Qwen/Qwen3.5-9B \
  common.lora_rank=128 \
  deployments.rollout.replica_count=1 \
  trainers.policy.training_shape_id=accounts/fireworks/trainingShapes/qwen3p5-9b-256k-lora \
  trainers.policy.base_model=accounts/fireworks/models/qwen3p5-9b \
  trainers.policy.replica_count=1 \
  trainers.teacher_forward_only.training_shape_id=accounts/fireworks/trainingShapes/qwen3p5-9b-256k-lora \
  trainers.teacher_forward_only.base_model=accounts/fireworks/models/qwen3p5-9b \
  trainers.teacher_forward_only.replica_count=1
```

The process prints heartbeat lines like:

```text
Fireworks distillation infra alive | trainer=<student_job_id> | deployment=<deployment_id> | teachers=<teacher_job_id>
```

Copy those IDs into the training config:

- `base_url`: student trainer
  `https://api.fireworks.ai/training/v1/rlorTrainerJobs/<account>/<student_job_id>`
- `fireworks_deployment_id`: the printed student rollout deployment ID
- teacher trainer URL (when the recipe scores against a Fireworks forward-only
  teacher):
  `https://api.fireworks.ai/training/v1/rlorTrainerJobs/<account>/<teacher_job_id>`

Teacher `base_model` / training shape should match the teacher you want to score
against. Student `common.base_model` / `trainers.policy.*` should match the
student being trained. Keep tokenizer IDs aligned between student and teacher.

## Common Overrides

Use Hydra dotlist overrides to change the shipped YAML without editing it:

```bash
common.base_model=accounts/<account>/models/<model>
common.tokenizer_model=<hf-tokenizer-or-model-id>
common.lora_rank=128
trainers.policy.training_shape_id=accounts/<account>/trainingShapes/<shape>
trainers.policy.replica_count=1
deployments.rollout.replica_count=1
# Distillation teacher (fireworks_distillation only):
trainers.teacher_forward_only.base_model=accounts/<account>/models/<teacher>
trainers.teacher_forward_only.training_shape_id=accounts/<account>/trainingShapes/<forward-only-shape>
trainers.teacher_forward_only.replica_count=1
```

For full-parameter training, set:

```bash
common.lora_rank=0
```

Leave `trainers.policy.region=null` unless you explicitly need a region; the
backend should choose placement from the training shape and account capacity.

## Reusing Existing Resources

To reattach instead of creating fresh resources, override the resource IDs:

```bash
python -m training.provision.provision \
  --config-name fireworks_rft \
  trainers.policy.job_id=<existing_trainer_job_id> \
  deployments.rollout.deployment_id=<existing_deployment_id>
```

By default, reattached resources are not deleted on exit. Newly-created
resources are deleted on exit.

## SFT Trainer Only

SFT does not need a rollout deployment:

```bash
python -m training.provision.provision \
  --config-name fireworks_sft \
  common.base_model=accounts/fireworks/models/qwen3p5-35b-a3b \
  common.tokenizer_model=Qwen/Qwen3.5-35B-A3B \
  common.lora_rank=128 \
  trainers.policy.training_shape_id=accounts/fireworks/trainingShapes/qwen3p5-35b-a3b-256k-lora \
  trainers.policy.replica_count=1
```

Use the printed trainer job ID when starting the SFT recipe from this repo.
The Tinker config still expects the trainer as a `base_url`:

```bash
base_url=https://api.fireworks.ai/training/v1/rlorTrainerJobs/<account>/<trainer_job_id>
```

## Notes

- `fireworks_rft` and `rl` use the same provisioning mode.
- `fireworks_distillation` also provisions a forward-only teacher trainer.
- `deployments.rollout.replica_count` controls sampling capacity.
- `trainers.policy.replica_count` controls trainer replicas when the selected
  training shape supports it.
- The provisioner is not the training loop. It only holds infrastructure open so
  a recipe in this repo can connect to it.
