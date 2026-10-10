---
name: fireworks
description: Run tinker-cookbook training on Fireworks trainers and deployments. Use this skill whenever the user mentions Fireworks, Firetitan, an rlor-trainer-job (or "trainer job") ID, a Fireworks deployment ID, a training shape, or `fireworks_*` config options — and especially when they want to resume, reuse, or reattach to an existing rlor-trainer-job. The cookbook configs take a trainer as a `base_url`, not as a job ID, so a job ID must be translated before it is passed to a config, and a reused trainer must never be used as it is: the run has to load the base checkpoint or resume from a DCP checkpoint first.
---

# Fireworks trainers and deployments

The training configs in this repo (`supervised/train.py`, `rl/train.py`, `distillation/`) do not have a trainer-job-ID field. They identify a Fireworks trainer by `base_url`. Users, the Fireworks console, and `firectl` talk about the same trainer as an **rlor-trainer-job ID**. Translate one into the other before you build a config or a command line.

## Translate an rlor-trainer-job ID into `base_url`

```text
base_url = https://api.fireworks.ai/training/v1/rlorTrainerJobs/<account>/<job-id>
```

- `<job-id>` is the rlor-trainer-job ID the user gave you.
- `<account>` is the Fireworks account that owns the job.

Users give the job in several forms. Map each of them to the same URL:

| What the user gives | `<account>` | `<job-id>` |
| --- | --- | --- |
| `abc123` | ask, or take it from context | `abc123` |
| `accounts/my-team/rlorTrainerJobs/abc123` | `my-team` | `abc123` |
| `https://api.fireworks.ai/training/v1/rlorTrainerJobs/my-team/abc123` | already a `base_url` | pass it through unchanged |

If the user gives only a bare job ID and the account is not clear from the conversation, a resource name they pasted, or their other `fireworks_*` values (for example `accounts/<account>/models/...` on a model they own), ask for it. Do not guess the account, and do not use `fireworks` as a default: that is the account of the public models, not the user's.

Then pass the URL as `base_url`. Never invent a `trainer_job_id`, `job_id`, or `rlor_trainer_job_id` config field; the configs do not have one.

```bash
python -m tinker_cookbook.recipes.math_rl.train \
    base_url="https://api.fireworks.ai/training/v1/rlorTrainerJobs/my-team/abc123" \
    fireworks_deployment_id=<deployment-id> \
    fireworks_base_model=accounts/fireworks/models/<model>
```

```python
config = train.Config(
    base_url="https://api.fireworks.ai/training/v1/rlorTrainerJobs/my-team/abc123",
    fireworks_base_model="accounts/fireworks/models/<model>",
    ...
)
```

The training loops read the job ID back out of the URL with `checkpoint_utils.extract_trainer_job_id`, which takes the path segment after `/rlorTrainerJobs/<account>/`. A URL without the account segment, or without `/rlorTrainerJobs/`, yields no job ID, and the run then creates a new trainer instead of reusing the one the user asked for. After building the URL, check it has both segments.

## Reuse or create

Each resource is reused when you identify it and created by the run when you leave it unset.

| Resource | Reuse | Create |
| --- | --- | --- |
| Trainer | `base_url=<trainer URL>` | leave `base_url` unset |
| Rollout deployment (RL, on-policy distillation, SDFT) | `fireworks_deployment_id=<id>` | leave it unset and set `fireworks_training_shape_id` |

- `fireworks_base_model` is always required, and it must be the model the reused trainer was started with.
- `fireworks_training_shape_id` selects the shape of a new trainer and of a new deployment. It is required whenever a deployment has to be created.
- Supervised training uses a trainer only and never creates a deployment.
- When the run ends, a trainer it created is deleted and a deployment it created is scaled to zero. Set `fireworks_cleanup_on_exit=False` on the training config to keep them. Reused resources are never deleted or scaled down.
- Off-policy distillation and distillation teachers on a separate trainer (`teacher_base_url`, `teacher_config.base_url`) are not created by the run. Pass an existing trainer URL for those, built the same way as above.

The run logs the trainer job ID and deployment ID it uses. If the user may want to come back to a trainer the run creates, set `fireworks_cleanup_on_exit=False` and record those IDs.

## Never reuse a trainer as it is

A reused rlor-trainer-job still holds whatever the last run left in it: trained weights and optimizer state (Adam moments, step count). Passing `base_url` only attaches to the trainer; it does not reset it. Starting a run on top of that state silently continues someone else's training.

So every run on a reused trainer must begin by loading a checkpoint. There are exactly two valid starts:

| Intent | What to load | Optimizer state |
| --- | --- | --- |
| Start a new run | The base checkpoint | Cleared |
| Continue an earlier run | That run's DCP checkpoint (the state saved by `save_state`) | Restored |

Before you launch on a reused trainer, decide which of the two the user wants and confirm the run will do it. If neither applies, do not launch; ask the user, or let the run create a new trainer instead (leave `base_url` unset).

How the loops in this repo cover the two starts:

- **Continue from a DCP checkpoint.** Reuse the `log_path` of the earlier run. The loop finds the last checkpoint recorded there and loads it with its optimizer state (`load_state_with_optimizer`).
- **Load a checkpoint with a cleared optimizer.** On-policy distillation and SDFT accept `load_checkpoint_path` and load it weights-only (`load_state`), which resets the optimizer.
- **RL and supervised training reject `load_checkpoint_path`.** With a new `log_path` they attach to the trainer and train on whatever state it holds. Do not start a new RL or supervised run on a reused trainer unless the user confirms the trainer is already at the base checkpoint; otherwise let the run create its own trainer.

A sampler snapshot from `save_weights_for_sampler` is not a DCP checkpoint and cannot be used to resume.

## Resume an interrupted run

Resuming is the "continue from a DCP checkpoint" start above. It needs two things:

1. The same `log_path` as the interrupted run. The loop finds the last checkpoint there and continues from it.
2. The trainer that run used, passed as `base_url` (translated from its rlor-trainer-job ID as above).

If the original run created its own trainer and cleaned it up on exit, that trainer no longer exists. Tell the user instead of silently starting a new one: a run on a new trainer is not guaranteed to be able to load the old run's checkpoint.
