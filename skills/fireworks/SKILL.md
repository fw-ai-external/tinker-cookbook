---
name: fireworks
description: Run tinker-cookbook training on Fireworks trainers and deployments. Use this skill whenever the user mentions Fireworks, Firetitan, an rlor-trainer-job (or "trainer job") ID, a Fireworks deployment ID, a training shape, or `fireworks_*` config options — and especially when they want to resume, reuse, or reattach to an existing rlor-trainer-job. Pass the job ID as `fireworks_trainer_job_id`; `base_url` is the other way to name the same trainer, and the two must agree when both are set. When neither is provided, the run provisions a new trainer automatically, and likewise a new rollout deployment when `fireworks_deployment_id` is not provided. Also use it to decide what state a run starts from: a new training client always starts from the base model with a fresh optimizer, and a checkpoint is loaded only when the run resumes or sets `load_checkpoint_path`.
---

# Fireworks trainers and deployments

Users, the Fireworks console, and `firectl` name a trainer by its **rlor-trainer-job ID**. The training configs in this repo (`supervised/train.py`, `rl/train.py`, `distillation/`) and the recipes built on them accept that ID directly as `fireworks_trainer_job_id`.

**Provisioning is automatic.** If neither `fireworks_trainer_job_id` nor a trainer `base_url` is provided, the run provisions a new rlor-trainer-job by itself. If `fireworks_deployment_id` is not provided, a loop that samples (RL, on-policy distillation, SDFT) provisions a new rollout deployment too. There is no separate provisioning step to run first, and you never need to ask the user to create a trainer or deployment before launching. Only name an existing resource when the user wants to reuse one.

## Reuse an rlor-trainer-job

Pass the job ID as `fireworks_trainer_job_id`. You do not need a `base_url`, and you do not need the account name:

```bash
python -m tinker_cookbook.recipes.math_rl.train \
    fireworks_trainer_job_id=abc123 \
    fireworks_deployment_id=<deployment-id> \
    fireworks_base_model=accounts/fireworks/models/<model> \
    fireworks_training_shape_id=accounts/fireworks/trainingShapes/<shape>
```

```python
config = train.Config(
    fireworks_trainer_job_id="abc123",
    fireworks_base_model="accounts/fireworks/models/<model>",
    ...
)
```

Pass the bare job ID. If the user gives a resource name such as `accounts/my-team/rlorTrainerJobs/abc123`, pass its last segment, `abc123`.

### `base_url` and `fireworks_trainer_job_id`

A trainer URL in `base_url` names the same thing:

```text
https://api.fireworks.ai/training/v1/rlorTrainerJobs/<account>/<job-id>
```

| `fireworks_trainer_job_id` | `base_url` | Trainer the run uses |
| --- | --- | --- |
| set | unset | the job ID |
| unset | trainer URL | the job ID in the URL |
| set | trainer URL with the same job ID | that job |
| set | anything else | none; the run stops with a `ConfigurationError` |
| unset | unset | a new trainer, created by the run |
| unset | serverless endpoint | the shared serverless trainer |

- Prefer `fireworks_trainer_job_id` when the user gives a job ID. Do not build a URL from it; that needs the account name and adds a way to get it wrong.
- If the user gives a trainer URL, pass it as `base_url` unchanged. Do not also set `fireworks_trainer_job_id`.
- A `base_url` without `/rlorTrainerJobs/<account>/<job-id>` names no trainer. On its own it makes the run create a new one.

## Reuse or create

Each resource is reused when you identify it and provisioned automatically by the run when you leave it unset.

| Resource | Reuse | Create |
| --- | --- | --- |
| Trainer | `fireworks_trainer_job_id=<job-id>` (or a trainer URL in `base_url`) | leave both unset; `fireworks_training_shape_id` is optional and picks the shape |
| Rollout deployment (RL, on-policy distillation, SDFT) | `fireworks_deployment_id=<id>` | leave it unset |

To provision both for the run, pass none of `fireworks_trainer_job_id`, `base_url`, and `fireworks_deployment_id`. The run creates a new rlor-trainer-job and a new rollout deployment from the training shape:

```bash
python -m tinker_cookbook.recipes.math_rl.train \
    model_name=Qwen/Qwen3.5-9B \
    fireworks_base_model=accounts/fireworks/models/qwen3p5-9b \
    fireworks_training_shape_id=accounts/fireworks/trainingShapes/qwen3p5-9b-256k-lora
```

`model_name` is the public model name used for the tokenizer and renderer; `fireworks_base_model` is the Fireworks model the trainer and deployment load.

- `fireworks_base_model` is always required, and it must be the model the reused trainer was started with.
- `fireworks_training_shape_id` selects the shape of a new trainer and of a new deployment. It is required for every dedicated run that uses a rollout deployment, including one that reuses both the trainer and the deployment.
- Supervised training and off-policy distillation use a trainer only. Off-policy distillation uses a deployment, reused or created like the one above, only when the run has evaluators that sample from the model.
- Supervised training never uses a deployment. On a dedicated trainer it skips its infrequent evaluators with a warning; on serverless training they run, and a sampling evaluator samples from a saved snapshot.
- The run never deletes or scales down a trainer or deployment, whether it created them or reused them, and whether it succeeded or failed. An idle trainer is cleaned up by Fireworks on its own; a deployment the run created keeps running until the user scales it down or deletes it.
- A distillation teacher in on-policy or off-policy distillation is handled like the student's trainer. Name an existing trainer by job ID (`fireworks_trainer_job_id` on `TeacherConfig`, `teacher_fireworks_trainer_job_id` on the recipes) or by URL (`base_url`, `teacher_base_url`), with the same consistency rule. With neither set, the run provisions a forward-only trainer for the teacher; `fireworks_training_shape_id` on `TeacherConfig` (`teacher_fireworks_training_shape_id` on the recipes) is optional and picks its shape.
- SDFT's teacher defaults to the frozen base model on the student's trainer, so it needs no trainer of its own. Set `teacher_fireworks_trainer_job_id` or `teacher_base_url` to use a separate, existing one.

The run logs the trainer job ID and deployment ID it uses. Record them: a later run, or a rerun after a failure, can reuse both by passing `fireworks_trainer_job_id` and `fireworks_deployment_id` along with `fireworks_training_shape_id`.

## Serverless training

Serverless training uses a shared trainer, so there is nothing to provision or reuse. Select it with `base_url=https://api.fireworks.ai/training/v1/serverless`:

```bash
python -m tinker_cookbook.recipes.math_rl.train \
    base_url=https://api.fireworks.ai/training/v1/serverless \
    model_name=Qwen/Qwen3.8-27B \
    fireworks_base_model=accounts/fireworks/models/qwen3p8-27b
```

- Every loop supports it: supervised training, RL, on-policy and off-policy distillation, SDFT, and the decisions recipe (`recipes/decisions`, whose demo and forecasting loop default to it). The loops that sample (RL, on-policy distillation, SDFT) save a sampler snapshot after each update and sample from it instead of a rollout deployment.
- `fireworks_deployment_id` is a `ConfigurationError` on serverless. The RL KL penalty (`kl_penalty_coef > 0`) works there: its reference is an untrained training client in a new serverless session.
- A distillation teacher that names no trainer of its own runs on serverless with the student, as a LoRA adapter that is never trained, so it behaves as the base model.
- An account can run a limited number of serverless training runs at once. A `429` that says the maximum concurrent serverless training runs is exceeded means earlier runs still hold their slots; wait for them to finish or expire, then rerun.
- Leave `fireworks_trainer_job_id` unset. A serverless `base_url` together with a job ID is a `ConfigurationError`.
- It is LoRA only: `lora_rank` must be positive.
- `fireworks_training_shape_id` has no effect.

## What state a run starts from

Reusing a trainer does not reuse its training state. `create_training_client()` creates a new model initialized from `fireworks_base_model` with a fresh optimizer, on a new trainer and on a reused rlor-trainer-job alike. Nothing from an earlier run carries over unless the code loads it.

A checkpoint is loaded only by an explicit call:

| Call | Loads | Optimizer state | When the loops make it |
| --- | --- | --- | --- |
| none | nothing; the run starts from the base model | Fresh | New `log_path`, no `load_checkpoint_path` |
| `load_state_with_optimizer(path)` | A DCP checkpoint (the state saved by `save_state`) | Restored | The `log_path` already holds a checkpoint of an earlier run |
| `load_state(path)` | The weights of a DCP checkpoint | Fresh | `load_checkpoint_path` is set |

So:

- To start a new run on a reused trainer, pass `fireworks_trainer_job_id` and a new `log_path`. You do not need to load a base checkpoint or reset anything first.
- To continue an earlier run, reuse its `log_path`. Do not expect a reused trainer to continue on its own: with a new `log_path` the run starts from the base model again, whatever the trainer did before.
- To start from earlier weights with a fresh optimizer, set `load_checkpoint_path`.

A sampler snapshot from `save_weights_for_sampler` is not a DCP checkpoint and cannot be passed to either load call.

## Resume an interrupted run

Resuming needs two things:

1. The same `log_path` as the interrupted run. The loop finds the last checkpoint there and continues from it.
2. A trainer, passed as `fireworks_trainer_job_id`: the one that run used, or on dedicated training another one (see below).

The run does not delete its trainer, so a run that just failed can be resumed on the same job ID.

On a dedicated trainer, each save also records the checkpoint's `cross_job://<trainer-job-id>/step-N` reference in `checkpoints.jsonl` as `state_path`, next to the `trainer_job_id` that saved it, and resuming loads that checkpoint: by its `step-N` name on the trainer that saved it, by the reference on any other trainer. It keeps loading after the trainer has stopped for inactivity and started again, and from a different trainer: to move a run, reuse its `log_path` and pass the other trainer as `fireworks_trainer_job_id`.

- A save waits for the trainer to report the checkpoint, normally a second or two. If it logs `Could not find ... among the trainer's checkpoints` after 90 seconds, no reference was recorded. Resuming then loads the saved name, which works only while the trainer keeps running, and the save itself may be incomplete. Tell the user rather than assuming the checkpoint is usable.
- To start a new run from a checkpoint of another run, set `load_checkpoint_path` to its `cross_job://` reference; this loads the weights with a fresh optimizer.

Checkpoints of a serverless run are named by their full path and need none of this.

## Troubleshooting

Errors seen on live runs, what caused them, and what fixed them.

| Symptom | Cause | Fix |
| --- | --- | --- |
| `APIConnectionError` from `tinker`, or a TLS error such as `invalid peer certificate` | The code created a `tinker.ServiceClient`, which connects to Tinker, not Fireworks | Create the client with the helpers in `fireworks_utils` (or `FiretitanServiceClient`). Nothing in a Fireworks run should connect to Tinker, so fix the code instead of the network or certificates |
| `create_model requested lora_rank=128, but trainer is configured with max lora_rank=32` | The dedicated trainer's shape allows a lower rank than the loop's default (SDFT defaults to 128) | Pass a `lora_rank` the shape allows, such as `lora_rank=32` |
| `RuntimeError: ERROR: Trainer job <id> failed ... Internal error occurred` | The trainer job itself failed | Rerun on a new trainer: leave `fireworks_trainer_job_id` unset, or pass another job |
| Saves log `Could not find ... among the trainer's checkpoints`, and resuming fails with `not a complete DCP training checkpoint` | The trainer did not persist the checkpoint. This happened on every save of a trainer that had failed earlier, while training on it still worked | Treat that trainer's checkpoints as unusable and rerun on a new trainer. Check for the warning before relying on a checkpoint |
| `429` that says the maximum concurrent serverless training runs is exceeded | Earlier serverless runs still hold their slots | Wait for them to finish or expire, then rerun |
| `Could not fetch renderer metadata ... Invalid tinker path` on resume | Fireworks checkpoint paths are not `tinker://` paths, so the renderer check is skipped | Nothing; the run resumes normally |
| `NotImplementedError` for `topk_prompt_logprobs` or `target_prompt_logprobs` | Fireworks samplers do not return prompt logprobs | Score with a training client's `forward_async` instead, as off-policy distillation, SDFT and `sample_decision` do |
| `forward_backward_custom_async` fails with `2D target_tokens ... but loss_fn_inputs.weights has N values` | Tinker sends a custom loss's gradient as sparse weights, which the trainer rejects for multi-token targets | Run `forward_async` first and pass its result as `precomputed_forward=` to `forward_backward_custom_async`, as `recipes/decisions/forecasting.py` does |

When a live check fails, read the first error in the log rather than the last: a failed run is often followed by further errors from cleanup or from a resume that had nothing to load.
