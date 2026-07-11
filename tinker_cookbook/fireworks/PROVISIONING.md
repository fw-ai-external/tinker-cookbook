# Fireworks Trainer and Deployment Provisioning

Use the Fireworks provisioning helper when a cookbook recipe needs a live
Fireworks trainer, and for RL/RFT-style recipes, a hot-loadable rollout
deployment. The helper creates the resources, prints their IDs, keeps them alive
while you work, and deletes newly-created resources when you stop it.

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
- `fireworks_base_model_name`: usually the same value as `common.base_model`

For example, if the provisioner prints trainer `abc123` and deployment
`qwen3p5-35b-a3b-xyz` in account `fireworks`, pass:

```bash
base_url=https://api.fireworks.ai/training/v1/rlorTrainerJobs/fireworks/abc123
fireworks_deployment_id=qwen3p5-35b-a3b-xyz
fireworks_base_model_name=accounts/fireworks/models/qwen3p5-35b-a3b
```

Keep the provisioning process running while the training process uses the
resources. Press `Ctrl+C` when you are done; the provisioner will clean up the
resources it created.

## Common Overrides

Use Hydra dotlist overrides to change the shipped YAML without editing it:

```bash
common.base_model=accounts/<account>/models/<model>
common.tokenizer_model=<hf-tokenizer-or-model-id>
common.lora_rank=128
trainers.policy.training_shape_id=accounts/<account>/trainingShapes/<shape>
trainers.policy.replica_count=1
deployments.rollout.replica_count=1
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
- `deployments.rollout.replica_count` controls sampling capacity.
- `trainers.policy.replica_count` controls trainer replicas when the selected
  training shape supports it.
- The provisioner is not the training loop. It only holds infrastructure open so
  a recipe in this repo can connect to it.
