# Recipes on Fireworks

Which recipes have been run on Fireworks, and what worked. Every run was short (one to ten steps) and checked that the loop trains, evaluates, saves checkpoints and, where listed, resumes; none was run long enough to compare final quality with Tinker.

- **Serverless**: `base_url=https://api.fireworks.ai/training/v1/serverless`, Qwen3.8-27B (`accounts/fireworks/models/qwen3p8-27b`), LoRA.
- **Dedicated**: a dedicated trainer and, for loops that sample, a rollout deployment, Qwen3.5-9B (`accounts/fireworks/models/qwen3p5-9b`) on the `qwen3p5-9b-256k-lora` training shape, LoRA rank 32.

See [`skills/fireworks/SKILL.md`](skills/fireworks/SKILL.md) for how to configure runs and for the errors below in more detail.

## Recipes

✅ worked · ⚠️ works with a limitation · ❌ does not work yet · — not run

| Recipe | Serverless | Dedicated | Resume | Notes |
| --- | --- | --- | --- | --- |
| Supervised (`recipes/chat_sl`, `supervised/train.py`) | ✅ | ✅ | ✅ both | Resumed after a killed run, on the same trainer, on a new trainer and across trainers (`cross_job://` references); `load_checkpoint_path` works on both |
| Supervised evaluators | ✅ | ⚠️ | | On dedicated, infrequent evaluators are skipped with a warning and a sampling evaluator is a `ConfigurationError` (work in progress) |
| `recipes/sl_loop.py` | ✅ | — | | `train_mean_bpb` is not logged: Fireworks' `forward_backward` returns no per-token logprobs |
| `recipes/rl_loop.py` | ✅ | ✅ | | |
| Math RL (`recipes/math_rl`), sync, async and streaming | ✅ | ✅ | ✅ both | With evals |
| Math RL with a KL penalty (`kl_penalty_coef > 0`) | ✅ | ✅ | | The reference is always the base model; a reference loaded from a checkpoint is not supported yet |
| DPO (`recipes/preference`) | ✅ | ✅ | ✅ both | `load_checkpoint_path` works on serverless; on dedicated it is a `ConfigurationError` (work in progress), since the reference there is always the base model |
| Multiplayer RL: guess the number | ✅ | ✅ | | |
| Multiplayer RL: twenty questions, text arena | ✅ | ❌ | | Serverless only; a dedicated `base_url` is a `ConfigurationError` (work in progress) |
| On-policy distillation (`recipes/distillation/on_policy_distillation.py`) | ✅ | ✅ | ✅ both | Teacher on its own forward-only trainer (dedicated) or as an untrained adapter (serverless) |
| On-policy multi-teacher (`recipes/distillation/on_policy_multi_teacher.py`) | — | — | | Shares the on-policy loop, but was not run |
| Off-policy distillation with teacher logits (`distillation/train_off_policy.py`) | ✅ | ✅ | ✅ serverless, ❌ dedicated | Teacher top-k from a forward pass. The dedicated resume failed on a trainer whose checkpoints were incomplete (see below), so it is untested on a healthy dedicated trainer |
| Off-policy reasoning SFT (`recipes/distillation/off_policy_reasoning.py`) | — | — | | Supervised training on a dataset; not run |
| SDFT (`recipes/sdft`) | ✅ | ✅ | ✅ both | `teacher_sync_every` is a `ConfigurationError` (work in progress); the teacher stays frozen |
| Decisions: `sample_decision` and the demo (`recipes/decisions`) | ✅ | — | | Labels are scored with a forward pass of a training client |
| Decisions: hard-label training (`decision_choice_datums`) | ✅ | — | | Two steps moved the target label from 0.00004 to 0.78 |
| Decisions: soft-label training and forecasting (`decision_dist_datums`) | ⚠️ | — | ✅ serverless | Needs `precomputed_forward`; see below |

## What did not work, though it was expected to

Each of these failed on a live run. The ones marked *fixed* no longer occur; the others are open and marked `TODO(fireworks)` in the code.

| What failed | Error | Cause | Status |
| --- | --- | --- | --- |
| A recipe that created a `tinker.ServiceClient` | `APIConnectionError` | It connected to Tinker instead of Fireworks | Fixed: the decisions recipe creates Fireworks clients |
| Scoring with `target_prompt_logprobs` or `topk_prompt_logprobs` | `NotImplementedError` from the sampler; type errors for `target_prompt_logprobs` | Fireworks samplers return no prompt logprobs, and the `tinker` version fireworks-ai pins predates `target_prompt_logprobs` | Fixed: off-policy distillation, SDFT and `sample_decision` score with a training client's forward pass |
| `forward_backward_custom_async` with multi-token targets (`decision_dist_datums`) | `2D target_tokens ... but loss_fn_inputs.weights has N values` | Tinker sends the custom loss's gradient as sparse weights, which the trainer rejects | Worked around: run `forward_async` first and pass it as `precomputed_forward=` (`recipes/decisions/forecasting.py`, README examples). Open |
| SDFT on a reused dedicated trainer at its default `lora_rank=128` | `create_model requested lora_rank=128, but trainer is configured with max lora_rank=32` | A trainer's maximum LoRA rank is the rank it was created with | Fixed: a reused trainer's rank is checked first, with a `ConfigurationError` naming both ranks |
| Resuming off-policy distillation on a dedicated trainer | `not a complete DCP training checkpoint` | The trainer had failed earlier (`Internal error occurred`) and none of its later saves was persisted; each save logged `Could not find ... among the trainer's checkpoints` | Not a code issue; rerun on a new trainer. Open on the trainer side |
| Several serverless runs at once | `429` maximum concurrent serverless training runs | Account limit on concurrent serverless runs | Expected; wait and rerun |
| `forward_backward` per-token logprobs | (metrics missing) | Fireworks returns no per-token logprobs | Open: `train_mean_bpb` and per-token NLL are not logged |
| Renderer recorded with a checkpoint | `Could not fetch renderer metadata ... Invalid tinker path` (warning) | Fireworks checkpoint paths are not `tinker://` paths | Harmless; the run resumes. Open |
