"""
Self-Distillation Fine-Tuning (SDFT).

Implements the SDFT algorithm from
`"Self-Distillation Enables Continual Learning" <https://arxiv.org/abs/2601.19897>`_
(Shenfeld et al., 2026). SDFT is an on-policy distillation method that learns new
skills from demonstrations while preserving prior capabilities.

**How it works:** A teacher model (frozen base weights) sees the question **and** a
golden answer as an in-context demonstration. The student sees only the question and
generates completions on-policy. The teacher's top-K token distribution at each
position is recovered via Firetitan's top-K forward API and used as soft
targets for ``cross_entropy`` loss — approximating the paper's full-vocabulary
forward KL divergence.

Two distillation modes are supported (controlled by :class:`Config` ``.topk``):

- **Top-K distillation** (``topk > 0``, default): Recovers the teacher's top-K
  token distribution and trains with ``cross_entropy``. Validated to match
  full-vocabulary KL on the
  `reference implementation <https://github.com/idanshen/Self-Distillation>`_.

- **Per-token importance sampling** (``topk = 0``): Single-sample approximation
  using ``advantage = teacher_lp - student_lp`` with ``importance_sampling`` loss.

Example usage::

    # SDFT with top-K=20 distillation on tool-use data
    python -m tinker_cookbook.recipes.sdft.train \\
        model_name=Qwen/Qwen3.6-35B-A3B \\
        dataset=toolalpaca \\
        toolalpaca_data_path=~/Self-Distillation/data/tooluse_data/train_data \\
        groups_per_batch=128 \\
        learning_rate=5e-4 \\
        topk=20 \\
        lora_rank=64

See the `recipe README <https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/sdft>`_
for full setup instructions and continual learning results. For background on the
loss functions used, see the `Tinker loss docs <https://tinker-docs.thinkingmachines.ai/tinker/losses>`_.
"""

import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import chz
import tinker
import torch
from fireworks.training.sdk import FiretitanTrainingClient
from tinker.types import LossFnType

from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.display import colorize_example
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator, SamplingClientEvaluatorBuilder
from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.fireworks_utils import (
    create_service_client_with_trainer_and_deployment,
    create_service_client_with_trainer_only,
    is_serverless_base_url,
    resolve_trainer_job_id,
)
from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
)
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator, compute_trajectory_metrics
from tinker_cookbook.rl.rollouts import do_group_rollout_and_filter_constant_reward
from tinker_cookbook.rl.train import (
    save_checkpoint_and_get_sampling_client,
    train_step,
)
from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    RLDataset,
    TrajectoryGroup,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log, trace
from tinker_cookbook.utils.git_rev import recipe_user_metadata
from tinker_cookbook.utils.misc_utils import safezip, split_list

logger = logging.getLogger(__name__)

DEFAULT_DEMO_TEMPLATE = (
    "{question}\n\n"
    "This is an example for a response to the question:\n"
    "{golden_answer}\n\n"
    "Now answer with a response of your own, including the thinking process."
)


@runtime_checkable
class SDFTBatchProvider(Protocol):
    """Protocol for SDFT datasets that return builders alongside golden answers."""

    def get_batch(self, index: int) -> tuple[Sequence[EnvGroupBuilder], list[str], list[str]]:
        """Return (env_group_builders, questions, golden_answers) for a batch.

        Each list has the same length (one per problem in the batch).
        """
        ...

    def __len__(self) -> int: ...


class _SDFTEvalDatasetAdapter(RLDataset):
    """Expose only environment builders from an SDFT batch provider."""

    def __init__(self, provider: SDFTBatchProvider):
        self._provider = provider

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        builders, _questions, _golden_answers = self._provider.get_batch(index)
        return builders

    def __len__(self) -> int:
        return len(self._provider)


def build_sdft_teacher_prompt(
    question: str,
    golden_answer: str,
    renderer: renderers.Renderer,
    system_prompt: str | None = None,
    demo_template: str = DEFAULT_DEMO_TEMPLATE,
) -> tinker.ModelInput:
    """Build teacher ModelInput with golden answer as an in-context demonstration.

    The teacher prompt presents the question alongside the golden answer so the
    model can attend to the demonstration when scoring student completions.

    Returns a ModelInput suitable for appending student completion tokens and
    computing logprobs via a FiretitanTrainingClient.
    """
    teacher_content = demo_template.format(question=question, golden_answer=golden_answer)
    messages: list[renderers.Message] = []
    if system_prompt:
        msg: renderers.Message = {"role": "system", "content": system_prompt}  # type: ignore[typeddict-item]
        messages.append(msg)
    user_msg: renderers.Message = {"role": "user", "content": teacher_content}  # type: ignore[typeddict-item]
    messages.append(user_msg)
    return renderer.build_generation_prompt(messages)


def _extract_completion_tokens(
    datum: tinker.Datum,
    teacher_prompt: tinker.ModelInput,
    max_context_length: int,
) -> tuple[list[int], int, int, bool]:
    """Extract student completion tokens and compute teacher prompt length.

    Returns (completion_tokens, teacher_prompt_len, completion_start_in_student, was_truncated).
    completion_tokens may be empty if there are no completion tokens or context overflows.
    """
    mask = datum.loss_fn_inputs["mask"].to_torch()
    completion_mask_indices = torch.where(mask > 0)[0]
    teacher_prompt_len = teacher_prompt.length

    if len(completion_mask_indices) == 0:
        return [], teacher_prompt_len, 0, False

    # Reconstruct full student sequence (model_input is left-shifted, missing last target)
    student_full = datum.model_input.append_int(
        cast(int, datum.loss_fn_inputs["target_tokens"].data[-1])
    )
    student_full_tokens = student_full.to_ints()
    # Completion starts at first mask position + 1 (target is left-shifted)
    completion_start = int(completion_mask_indices[0].item()) + 1
    completion_tokens = student_full_tokens[completion_start:]

    available = max_context_length - teacher_prompt_len
    truncated = False
    if available <= 0:
        return [], teacher_prompt_len, completion_start, True
    if len(completion_tokens) > available:
        completion_tokens = completion_tokens[:available]
        truncated = True

    return completion_tokens, teacher_prompt_len, completion_start, truncated


def _build_teacher_forced_sequence(
    teacher_prompt: tinker.ModelInput,
    completion_tokens: list[int],
) -> tinker.ModelInput:
    """Append completion tokens to teacher prompt to form the teacher-forced sequence."""
    teacher_forced = teacher_prompt
    for token in completion_tokens:
        teacher_forced = teacher_forced.append_int(token)
    return teacher_forced


def _teacher_forward_datum(sequence: tinker.ModelInput) -> tinker.Datum:
    """Shift a full sequence into cross-entropy input and targets for a forward pass."""
    tokens = sequence.to_ints()
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(
                torch.tensor(tokens[1:], dtype=torch.long)
            )
        },
    )


async def _compute_teacher_logprobs(
    teacher_client: FiretitanTrainingClient,
    sequences_D: list[tinker.ModelInput],
    completion_lengths_D: list[int],
) -> list[list[float | None]]:
    """Teacher logprobs per sequence, indexed like ``compute_logprobs`` (None at position 0).

    One batched forward; a sequence with no completion is not sent and gets ``[]``.
    """
    indices = [i for i, n in enumerate(completion_lengths_D) if n > 0]
    logprobs_D: list[list[float | None]] = [[] for _ in sequences_D]
    if not indices:
        return logprobs_D
    future = await teacher_client.forward_async(
        [_teacher_forward_datum(sequences_D[i]) for i in indices], loss_fn="cross_entropy"
    )
    result = await future.result_async()
    for i, output in safezip(indices, result.loss_fn_outputs):
        logprobs_D[i] = [None, *output["logprobs"].to_torch().tolist()]
    return logprobs_D


async def _compute_teacher_topk_prompt_logprobs(
    teacher_client: FiretitanTrainingClient,
    sequences_D: list[tinker.ModelInput],
    completion_lengths_D: list[int],
    topk: int,
) -> list[list[list[tuple[int, float]] | None] | None]:
    """Teacher top-K per sequence, indexed like ``topk_prompt_logprobs`` (None at position 0).

    One batched forward with ``top_k``; a sequence with no completion is not sent and
    gets ``None``.
    """
    indices = [i for i, n in enumerate(completion_lengths_D) if n > 0]
    topk_D: list[list[list[tuple[int, float]] | None] | None] = [None for _ in sequences_D]
    if not indices:
        return topk_D
    future = await teacher_client.forward_async(
        [_teacher_forward_datum(sequences_D[i]) for i in indices],
        loss_fn="cross_entropy",
        loss_fn_config={"top_k": topk},
    )
    result = await future.result_async()
    for i, output in safezip(indices, result.loss_fn_outputs):
        token_ids = output["top_k_indices"].to_torch().tolist()
        logprobs = output["top_k_logprobs"].to_torch().tolist()
        topk_D[i] = [
            None,
            *[list(zip(t, lp, strict=True)) for t, lp in zip(token_ids, logprobs, strict=True)],
        ]
    return topk_D


@trace.scope
async def compute_sdft_advantages(
    data_D: list[tinker.Datum],
    metadata_D: list[dict[str, int]],
    teacher_client: FiretitanTrainingClient,
    teacher_prompts_P: list[tinker.ModelInput],
    max_context_length: int = 32768,
) -> dict[str, float]:
    """Replace advantages with teacher_lp - student_lp (per-token).

    For each datum, builds the full teacher sequence (teacher_prompt + completion
    tokens), computes teacher logprobs, and sets advantages to the per-token
    difference between teacher and student logprobs.

    Modifies data_D in-place (replaces the ``advantages`` field).

    Args:
        data_D: List of datums from rollout. Must have ``logprobs`` and ``mask``
            fields in ``loss_fn_inputs``.
        metadata_D: Per-datum metadata with ``group_idx`` mapping to teacher_prompts_P.
        teacher_client: Firetitan training client for the teacher model.
        teacher_prompts_P: Per-problem teacher prompts (one per group in the batch).
        max_context_length: Maximum context for teacher logprob computation.
            Completion tokens are truncated if teacher_prompt + completion exceeds this.
    """
    teacher_full_sequences_D: list[tinker.ModelInput] = []
    teacher_prompt_lengths_D: list[int] = []
    completion_lengths_D: list[int] = []
    truncated_count = 0

    for i, datum in enumerate(data_D):
        group_idx = metadata_D[i]["group_idx"]
        teacher_prompt = teacher_prompts_P[group_idx]

        completion_tokens, teacher_prompt_len, _, was_truncated = _extract_completion_tokens(
            datum, teacher_prompt, max_context_length
        )
        if was_truncated:
            truncated_count += 1

        if not completion_tokens:
            teacher_full_sequences_D.append(teacher_prompt)
            teacher_prompt_lengths_D.append(teacher_prompt_len)
            completion_lengths_D.append(0)
            continue

        teacher_full = _build_teacher_forced_sequence(teacher_prompt, completion_tokens)
        teacher_full_sequences_D.append(teacher_full)
        teacher_prompt_lengths_D.append(teacher_prompt_len)
        completion_lengths_D.append(len(completion_tokens))

    # Compute teacher logprobs in one batched forward
    teacher_logprobs_D = await _compute_teacher_logprobs(
        teacher_client, teacher_full_sequences_D, completion_lengths_D
    )

    # Replace advantages with teacher_lp - student_lp
    sampled_logprobs_D = [datum.loss_fn_inputs["logprobs"].to_torch() for datum in data_D]
    float_masks_D = [datum.loss_fn_inputs["mask"].to_torch().float() for datum in data_D]

    total_advantage_sum = 0.0
    total_mask_sum = 0.0
    total_teacher_lp_sum = 0.0
    total_student_lp_sum = 0.0

    for i, datum in enumerate(data_D):
        mask = float_masks_D[i]
        student_lp = sampled_logprobs_D[i]
        teacher_prompt_len = teacher_prompt_lengths_D[i]
        completion_len = completion_lengths_D[i]

        if completion_len == 0:
            continue

        raw_teacher_lps = teacher_logprobs_D[i]
        teacher_completion_lps = [
            lp if lp is not None else 0.0
            for lp in raw_teacher_lps[teacher_prompt_len : teacher_prompt_len + completion_len]
        ]
        teacher_lp_tensor = torch.tensor(teacher_completion_lps, dtype=torch.float32)

        new_advantages = torch.zeros_like(mask)
        completion_mask_indices = torch.where(mask > 0)[0]

        num_tokens = min(len(teacher_lp_tensor), len(completion_mask_indices))
        for t in range(num_tokens):
            idx = int(completion_mask_indices[t].item())
            new_advantages[idx] = teacher_lp_tensor[t] - student_lp[idx]

        datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(new_advantages)

        masked_advantages = new_advantages * mask
        total_advantage_sum += masked_advantages.sum().item()
        total_mask_sum += mask.sum().item()
        total_teacher_lp_sum += (teacher_lp_tensor[:num_tokens]).sum().item()
        total_student_lp_sum += sum(
            student_lp[int(completion_mask_indices[t].item())].item() for t in range(num_tokens)
        )

    metrics: dict[str, float] = {}
    if total_mask_sum > 0:
        metrics["sdft/mean_advantage"] = total_advantage_sum / total_mask_sum
        metrics["sdft/mean_teacher_lp"] = total_teacher_lp_sum / total_mask_sum
        metrics["sdft/mean_student_lp"] = total_student_lp_sum / total_mask_sum
    metrics["sdft/teacher_truncated_count"] = float(truncated_count)
    metrics["sdft/num_datums"] = float(len(data_D))

    return metrics


@trace.scope
async def build_topk_distillation_datums(
    data_D: list[tinker.Datum],
    metadata_D: list[dict[str, int]],
    teacher_client: FiretitanTrainingClient,
    teacher_prompts_P: list[tinker.ModelInput],
    topk: int = 20,
    max_context_length: int = 32768,
    vocab_size: int | None = None,
    skip_first_n_tokens: int = 3,
) -> tuple[list[tinker.Datum], dict[str, float]]:
    """Build cross_entropy datums with top-K teacher soft targets.

    Teacher-forces each student completion through the teacher model to recover
    the teacher's top-K token distribution at each position using Firetitan's
    ``loss_fn_config={"top_k": K}`` forward API. Returns new datums with
    ``(N, K)``-shaped ``target_tokens`` and ``weights`` for ``cross_entropy``
    loss.

    This implements forward KL distillation restricted to the top-K vocabulary.
    At each of the T completion token positions, the loss is the cross-entropy
    between the teacher's renormalized top-K distribution and the student::

        L = (1/T) * sum_{t=1}^{T} [ -sum_{k=1}^{K} P_teacher(x_k|t) * log P_student(x_k|t) ]

    This is equivalent to forward KL (up to constant teacher entropy) over the
    top-K tokens that carry most of the probability mass. Validated to match
    full-vocabulary KL on the
    `reference implementation <https://github.com/idanshen/Self-Distillation>`_
    (68.04% vs 68.04% on tooluse with Qwen2.5-7B).

    Args:
        data_D: Datums from rollout (used for model_input and mask alignment).
        metadata_D: Per-datum metadata with ``group_idx`` mapping.
        teacher_client: Firetitan training client for the teacher model.
        teacher_prompts_P: Per-problem teacher prompts (built by
            :func:`build_sdft_teacher_prompt`).
        topk: Number of top tokens to distill (K). K=20 is recommended.
        max_context_length: Maximum teacher context length.
        vocab_size: If set, filter out token IDs >= vocab_size (handles
            special tokens from vLLM that exceed the tokenizer's vocabulary).
        skip_first_n_tokens: Skip the first N completion tokens from the
            loss (default 3, matching the reference implementation).

    Returns:
        (new_datums, metrics) where new_datums have ``cross_entropy``
        loss_fn_inputs with ``target_tokens`` shape ``(N, K)`` and
        ``weights`` shape ``(N, K)``.
    """
    # Step 1: Build teacher-forced sequences
    teacher_forced_sequences_D: list[tinker.ModelInput] = []
    teacher_prompt_lengths_D: list[int] = []
    completion_lengths_D: list[int] = []
    truncated_count = 0

    for i, datum in enumerate(data_D):
        group_idx = metadata_D[i]["group_idx"]
        teacher_prompt = teacher_prompts_P[group_idx]

        completion_tokens, teacher_prompt_len, _, was_truncated = _extract_completion_tokens(
            datum, teacher_prompt, max_context_length
        )
        if was_truncated:
            truncated_count += 1

        if not completion_tokens:
            teacher_forced_sequences_D.append(teacher_prompt)
            teacher_prompt_lengths_D.append(teacher_prompt_len)
            completion_lengths_D.append(0)
            continue

        teacher_forced = _build_teacher_forced_sequence(teacher_prompt, completion_tokens)
        teacher_forced_sequences_D.append(teacher_forced)
        teacher_prompt_lengths_D.append(teacher_prompt_len)
        completion_lengths_D.append(len(completion_tokens))

    # Step 2: Get top-K logprobs from teacher in one batched forward
    topk_D = await _compute_teacher_topk_prompt_logprobs(
        teacher_client, teacher_forced_sequences_D, completion_lengths_D, topk
    )

    # Step 3: Build new datums with (N, K) shaped target_tokens and weights.
    # First pass: collect raw weights and count completion tokens per datum.
    raw_datums: list[tuple[torch.Tensor, torch.Tensor, int]] = []  # (targets, weights, n_comp)
    total_completion_tokens = 0.0
    total_teacher_entropy = 0.0

    for i, datum in enumerate(data_D):
        mask = datum.loss_fn_inputs["mask"].to_torch()
        completion_mask_indices = torch.where(mask > 0)[0]
        N = datum.model_input.length
        completion_len = completion_lengths_D[i]
        teacher_prompt_len = teacher_prompt_lengths_D[i]

        target_tokens_NK = torch.zeros(N, topk, dtype=torch.long)
        weights_NK = torch.zeros(N, topk, dtype=torch.float32)
        n_completion_positions = 0

        if completion_len > 0 and len(completion_mask_indices) > 0:
            topk_all = topk_D[i]

            num_tokens = min(completion_len, len(completion_mask_indices))
            for t in range(num_tokens):
                # Skip first N completion tokens (reference skips 3)
                if t < skip_first_n_tokens:
                    continue

                teacher_pos = teacher_prompt_len + t
                student_pos = int(completion_mask_indices[t].item())

                if topk_all is None or teacher_pos >= len(topk_all):
                    continue
                topk_entries = topk_all[teacher_pos]
                if topk_entries is None:
                    continue

                # Filter out token IDs that exceed the student's vocab size
                # (teacher may return IDs for special/added tokens)
                filtered = [
                    (tok_id, lp)
                    for tok_id, lp in topk_entries[:topk]
                    if vocab_size is None or tok_id < vocab_size
                ]
                if not filtered:
                    continue

                k_actual = len(filtered)
                token_ids = torch.tensor([tok_id for tok_id, _ in filtered], dtype=torch.long)
                logprobs = torch.tensor([lp for _, lp in filtered], dtype=torch.float32)

                # Renormalize over top-K via logsumexp
                logprobs -= torch.logsumexp(logprobs, dim=0)
                probs = logprobs.exp()

                target_tokens_NK[student_pos, :k_actual] = token_ids
                weights_NK[student_pos, :k_actual] = probs
                n_completion_positions += 1

                # Teacher entropy for monitoring (H = -sum p log p)
                total_teacher_entropy += -(probs * logprobs).sum().item()

            total_completion_tokens += num_tokens

        raw_datums.append((target_tokens_NK, weights_NK, n_completion_positions))

    # No weight normalization — Tinker's CE loss uses raw sum, same convention
    # as the SFT loss. Both produce gradients proportional to num_tokens * lr.
    # Use the same LR range for both.
    new_datums: list[tinker.Datum] = []

    for i, datum in enumerate(data_D):
        target_tokens_NK, weights_NK, n_comp = raw_datums[i]

        new_datum = tinker.Datum(
            model_input=datum.model_input,
            loss_fn_inputs={
                "target_tokens": tinker.TensorData.from_torch(target_tokens_NK),
                "weights": tinker.TensorData.from_torch(weights_NK),
            },
        )
        new_datums.append(new_datum)

    metrics: dict[str, float] = {
        "sdft/teacher_truncated_count": float(truncated_count),
        "sdft/num_datums": float(len(data_D)),
        "sdft/topk": float(topk),
    }
    if total_completion_tokens > 0:
        metrics["sdft/total_completion_tokens"] = total_completion_tokens
        metrics["sdft/mean_teacher_entropy"] = total_teacher_entropy / total_completion_tokens

    return new_datums, metrics


@trace.scope
async def build_reverse_kl_datums(
    data_D: list[tinker.Datum],
    metadata_D: list[dict[str, int]],
    teacher_client: FiretitanTrainingClient,
    teacher_prompts_P: list[tinker.ModelInput],
    topk: int = 20,
    max_context_length: int = 32768,
    vocab_size: int | None = None,
    skip_first_n_tokens: int = 3,
) -> tuple[list[tinker.Datum], dict[str, float]]:
    """Build datums for reverse-KL custom-loss training over the teacher's top-K.

    Teacher-forces each student completion through the teacher, extracts the
    teacher's top-K token distribution at every completion position, and
    renormalizes it.

    Tinker's ``forward_backward_custom`` only accepts ``loss_fn_inputs`` keys in
    ``{"target_tokens", "weights"}`` (and only JSON-serializable floats), so we
    pack the renormalized teacher probability into ``weights`` with ``0`` as
    the sentinel for both (a) invalid top-K slots (after vocab filtering) and
    (b) masked positions (prompt, trailing, or skipped). This matches the
    forward-KL CE encoding. :func:`reverse_kl_custom_loss` recovers the mask
    via ``weights > 0`` and reconstructs ``log q_renorm = log(weights)``.

    The loss (REINFORCE form, stop-grad on the advantage):

        L = sum_t sum_{k in S_t} p_renorm(x_k|t) * sg[log p_renorm - log q_renorm]
    """
    teacher_forced_sequences_D: list[tinker.ModelInput] = []
    teacher_prompt_lengths_D: list[int] = []
    completion_lengths_D: list[int] = []
    truncated_count = 0

    for i, datum in enumerate(data_D):
        group_idx = metadata_D[i]["group_idx"]
        teacher_prompt = teacher_prompts_P[group_idx]

        completion_tokens, teacher_prompt_len, _, was_truncated = _extract_completion_tokens(
            datum, teacher_prompt, max_context_length
        )
        if was_truncated:
            truncated_count += 1

        if not completion_tokens:
            teacher_forced_sequences_D.append(teacher_prompt)
            teacher_prompt_lengths_D.append(teacher_prompt_len)
            completion_lengths_D.append(0)
            continue

        teacher_forced = _build_teacher_forced_sequence(teacher_prompt, completion_tokens)
        teacher_forced_sequences_D.append(teacher_forced)
        teacher_prompt_lengths_D.append(teacher_prompt_len)
        completion_lengths_D.append(len(completion_tokens))

    topk_D = await _compute_teacher_topk_prompt_logprobs(
        teacher_client, teacher_forced_sequences_D, completion_lengths_D, topk
    )

    new_datums: list[tinker.Datum] = []
    total_positions = 0.0
    total_teacher_entropy = 0.0

    for i, datum in enumerate(data_D):
        mask = datum.loss_fn_inputs["mask"].to_torch()
        completion_mask_indices = torch.where(mask > 0)[0]
        N = datum.model_input.length
        completion_len = completion_lengths_D[i]
        teacher_prompt_len = teacher_prompt_lengths_D[i]

        target_tokens_NK = torch.zeros(N, topk, dtype=torch.long)
        # All slots start at 0 (masked). Valid slots get the renormalized
        # teacher probability; masked positions and unused slots keep 0.
        weights_NK = torch.zeros(N, topk, dtype=torch.float32)

        if completion_len > 0 and len(completion_mask_indices) > 0:
            topk_all = topk_D[i]

            num_tokens = min(completion_len, len(completion_mask_indices))
            for t in range(num_tokens):
                if t < skip_first_n_tokens:
                    continue

                teacher_pos = teacher_prompt_len + t
                student_pos = int(completion_mask_indices[t].item())

                if topk_all is None or teacher_pos >= len(topk_all):
                    continue
                topk_entries = topk_all[teacher_pos]
                if topk_entries is None:
                    continue

                filtered = [
                    (tok_id, lp)
                    for tok_id, lp in topk_entries[:topk]
                    if vocab_size is None or tok_id < vocab_size
                ]
                if not filtered:
                    continue

                k_actual = len(filtered)
                token_ids = torch.tensor([tok_id for tok_id, _ in filtered], dtype=torch.long)
                raw_lps = torch.tensor([lp for _, lp in filtered], dtype=torch.float32)

                teacher_log_renorm = raw_lps - torch.logsumexp(raw_lps, dim=0)
                teacher_probs = teacher_log_renorm.exp()

                target_tokens_NK[student_pos, :k_actual] = token_ids
                weights_NK[student_pos, :k_actual] = teacher_probs

                total_teacher_entropy += -(teacher_probs * teacher_log_renorm).sum().item()
                total_positions += 1

        new_datums.append(
            tinker.Datum(
                model_input=datum.model_input,
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData.from_torch(target_tokens_NK),
                    "weights": tinker.TensorData.from_torch(weights_NK),
                },
            )
        )

    metrics: dict[str, float] = {
        "sdft/teacher_truncated_count": float(truncated_count),
        "sdft/num_datums": float(len(data_D)),
        "sdft/topk": float(topk),
    }
    if total_positions > 0:
        metrics["sdft/total_completion_positions"] = total_positions
        metrics["sdft/mean_teacher_entropy"] = total_teacher_entropy / total_positions

    return new_datums, metrics


def reverse_kl_custom_loss(
    data: list[tinker.Datum],
    logprobs_list: list[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Analytical reverse-KL loss over the teacher's top-K (REINFORCE form).

    Consumes datums built by :func:`build_reverse_kl_datums`. The server
    returns student logprobs at the teacher-top-K ``target_tokens`` (shape
    ``(N, K)``); we renormalize student over those K slots, stop-grad the
    ``[log p_renorm - log q_renorm]`` bracket, and take the mass-weighted
    sum. Gradient flows only through the outer ``p_renorm`` weight, matching
    Tinker's ``importance_sampling`` convention.

    The per-slot ``weights`` carry renormalized teacher probability (``q_renorm``)
    at valid slots and ``0`` at padding slots or masked positions. We recover
    the per-slot validity mask via ``weights > 0`` and reconstruct
    ``log q_renorm = log(weights)`` for the KL computation.
    """
    device = logprobs_list[0].device if logprobs_list else torch.device("cpu")
    total_loss = torch.zeros((), device=device)
    sum_kl = 0.0
    sum_student_entropy = 0.0
    sum_positions = 0.0

    for i, datum in enumerate(data):
        student_logp_NK = logprobs_list[i]
        weights_NK = datum.loss_fn_inputs["weights"].to_torch()  # q_renorm or 0

        slot_mask_NK = weights_NK > 0  # (N, K)
        position_mask_N = slot_mask_NK.any(dim=-1).float()  # (N,)

        # Reconstruct log q_renorm safely; clamp away from 0 for the log op.
        safe_weights = weights_NK.clamp(min=1e-30)
        teacher_log_renorm_NK = torch.where(
            slot_mask_NK, torch.log(safe_weights), torch.zeros_like(weights_NK)
        )

        neg_inf = torch.full_like(student_logp_NK, float("-inf"))
        masked_logp = torch.where(slot_mask_NK, student_logp_NK, neg_inf)
        log_p_renorm = torch.log_softmax(masked_logp, dim=-1)
        # Invalid slots land at 0 here (→ p_renorm=1), but are zeroed below via slot_mask_NK.
        log_p_renorm = torch.nan_to_num(log_p_renorm, nan=0.0, neginf=0.0)
        p_renorm = log_p_renorm.exp()

        adv_NK = (log_p_renorm - teacher_log_renorm_NK).detach()
        per_pos_NK = p_renorm * adv_NK * slot_mask_NK.float()
        per_pos_N = per_pos_NK.sum(dim=-1)
        loss_d = (per_pos_N * position_mask_N).sum()
        total_loss = total_loss + loss_d

        with torch.no_grad():
            kl_NK = p_renorm * (log_p_renorm - teacher_log_renorm_NK) * slot_mask_NK.float()
            per_pos_kl = kl_NK.sum(dim=-1)
            sum_kl += (per_pos_kl * position_mask_N).sum().item()
            ent_NK = -(p_renorm * log_p_renorm * slot_mask_NK.float())
            per_pos_ent = ent_NK.sum(dim=-1)
            sum_student_entropy += (per_pos_ent * position_mask_N).sum().item()
            sum_positions += position_mask_N.sum().item()

    metrics: dict[str, float] = {"sdft/reverse_kl_loss": total_loss.item()}
    if sum_positions > 0:
        metrics["sdft/reverse_kl_mean"] = sum_kl / sum_positions
        metrics["sdft/student_entropy_mean"] = sum_student_entropy / sum_positions
    metrics["sdft/reverse_kl_positions"] = sum_positions

    return total_loss, metrics


@trace.scope
async def _train_step_reverse_kl(
    data_D: list[tinker.Datum],
    training_client: FiretitanTrainingClient,
    learning_rate: float,
    num_substeps: int,
    metrics: dict[str, Any] | None = None,
) -> None:
    """Substep-pipelined custom forward/backward for the reverse-KL loss.

    Mirrors :func:`tinker_cookbook.rl.train.train_step` but uses
    ``forward_backward_custom_async`` with :func:`reverse_kl_custom_loss`.
    The datums produced by :func:`build_reverse_kl_datums` carry only
    ``target_tokens`` and ``weights`` (with ``0`` in ``weights`` acting as
    the mask sentinel), so no ``mask`` stripping is needed.
    """
    batches = split_list(data_D, min(num_substeps, len(data_D)))
    if not batches:
        return

    adam_params = tinker.AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
    fwd_bwd_future = await training_client.forward_backward_custom_async(
        batches[0], reverse_kl_custom_loss
    )
    optim_future = await training_client.optim_step_async(adam_params)
    loss_metrics: dict[str, float] = {}
    optim_result: tinker.OptimStepResponse | None = None

    for i in range(len(batches)):
        if i + 1 < len(batches):
            next_fwd_bwd_future = await training_client.forward_backward_custom_async(
                batches[i + 1], reverse_kl_custom_loss
            )
            next_optim_future = await training_client.optim_step_async(adam_params)
        else:
            next_fwd_bwd_future = None
            next_optim_future = None

        fwd_bwd_result = await fwd_bwd_future.result_async()
        if fwd_bwd_result.metrics:
            loss_metrics.update(fwd_bwd_result.metrics)
        optim_result = await optim_future.result_async()

        if next_fwd_bwd_future is not None and next_optim_future is not None:
            fwd_bwd_future = next_fwd_bwd_future
            optim_future = next_optim_future

    if metrics is not None:
        metrics.update(loss_metrics)
        if optim_result is not None and optim_result.metrics:
            metrics.update(optim_result.metrics)


@chz.chz
class Config:
    """Configuration for SDFT training.

    Key parameters:

    - ``topk``: Number of top tokens for distillation (default 20). Set to 0
      for the importance-sampling fallback. K=20 matches full-vocabulary KL
      in practice.
    - ``learning_rate``: For LoRA, use 5e-4 to 1e-3. The top-K CE loss
      produces larger gradients than SFT at the same LR due to more
      completion tokens per step (on-policy generation), so use the lower
      end of the range.
    - ``teacher_sync_every``: Optional periodic hard-sync of student weights
      into the teacher (approximating EMA). ``None`` = static frozen teacher,
      which works comparably to EMA in our experiments.

    See :func:`main` for the training loop.
    """

    # Model
    model_name: str
    recipe_name: str
    renderer_name: str | None = None
    lora_rank: int = 128
    base_url: str | None = None
    fireworks_base_model: str | None = None
    # Trainer job to reuse. None uses the trainer in base_url, or creates one.
    fireworks_trainer_job_id: str | None = None
    # Rollout deployment to reuse. None creates one.
    fireworks_deployment_id: str | None = None
    # Training shape, required on a dedicated trainer. It selects the shape of a
    # new trainer and a new deployment.
    fireworks_training_shape_id: str | None = None
    fireworks_hot_load_timeout: int = 600
    # A separate trainer for the teacher. None uses the student's trainer (the
    # base model) on a dedicated trainer, and the student's session on serverless.
    teacher_base_url: str | None = None
    teacher_fireworks_trainer_job_id: str | None = None
    teacher_fireworks_base_model: str | None = None

    # Training
    learning_rate: float = 2e-5
    max_tokens: int = 2048
    temperature: float = 1.0
    loss_fn: LossFnType = "cross_entropy"

    # SDFT-specific
    topk: int = 20
    reverse: bool = False
    demo_template: str = DEFAULT_DEMO_TEMPLATE
    system_prompt: str | None = None
    teacher_sync_every: int | None = None
    max_context_length: int = 32768

    # Evaluation
    evaluator_builders: list[SamplingClientEvaluatorBuilder] = chz.field(default_factory=list)
    eval_every: int = 20
    save_every: int = 20

    # Standard infra
    num_substeps: int = 1
    log_path: str = chz.field(munger=lambda _, s: str(Path(s).expanduser()))
    wandb_project: str | None = None
    wandb_name: str | None = None
    load_checkpoint_path: str | None = None
    max_steps: int | None = None

    enable_trace: bool = False
    span_chart_every: int = 0


@trace.scope
async def main(
    cfg: Config,
    sdft_dataset: SDFTBatchProvider,
    test_dataset: SDFTBatchProvider | None = None,
) -> None:
    """Main training loop for SDFT.

    Runs on-policy self-distillation: at each step, the student generates
    completions, the teacher scores them (conditioned on the golden answer
    demonstration), and the student is trained to match the teacher's
    distribution.

    When ``cfg.topk > 0``, uses top-K distillation via Firetitan's
    ``cross_entropy`` loss with ``(N, K)``-shaped soft targets. When
    ``cfg.topk == 0``, falls back to importance sampling with per-token
    advantages.

    Args:
        cfg: Training configuration. See :class:`Config`.
        sdft_dataset: Dataset providing (builders, questions, golden_answers)
            batches. Use :class:`~tinker_cookbook.recipes.sdft.datasets.SDFTDataset`.
        test_dataset: Optional test dataset for periodic evaluation.
    """
    if cfg.reverse and cfg.topk == 0:
        raise ValueError(
            "reverse=True requires topk>0: the analytical reverse KL runs over "
            "the teacher's top-K set, which doesn't exist when topk=0."
        )
    # TODO(fireworks): sync the student's weights into the teacher.
    if cfg.teacher_sync_every is not None:
        raise ConfigurationError(
            "teacher_sync_every is not supported on Fireworks yet (work in progress); "
            "the teacher is static"
        )
    if cfg.fireworks_base_model is None:
        raise ConfigurationError(
            "fireworks_base_model must be specified when creating a Fireworks training client."
        )

    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name,
    )
    store = ml_logger.store
    if cfg.enable_trace:
        current_task = asyncio.current_task()
        if current_task is not None:
            current_task.set_name("main")
        trace_events_path = str(Path(cfg.log_path) / "trace_events.jsonl")
        logger.info(f"Tracing enabled. Events saved to {trace_events_path}")
        trace.trace_init(output_file=trace_events_path)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)

    # Resume handling
    resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    start_batch = (resume_info.batch or 0) if resume_info else 0

    # Service and training client setup. Serverless training uses a shared
    # trainer and samples from saved snapshots, so it has no trainer job or
    # rollout deployment.
    serverless = is_serverless_base_url(cfg.base_url)
    service_client = create_service_client_with_trainer_and_deployment(
        trainer_job_id=resolve_trainer_job_id(
            base_url=cfg.base_url,
            fireworks_trainer_job_id=cfg.fireworks_trainer_job_id,
        ),
        base_model=cfg.fireworks_base_model,
        lora_rank=cfg.lora_rank,
        deployment_id=cfg.fireworks_deployment_id,
        training_shape_id=cfg.fireworks_training_shape_id,
        hot_load_timeout=cfg.fireworks_hot_load_timeout,
        base_url=cfg.base_url,
        user_metadata=recipe_user_metadata(cfg.recipe_name),
    )
    user_metadata: dict[str, str] = {}
    if wandb_link := ml_logger.get_logger_url():
        user_metadata["wandb_link"] = wandb_link
    checkpoint_utils.add_renderer_name_to_user_metadata(user_metadata, cfg.renderer_name)
    model_info.warn_if_renderer_not_recommended(cfg.model_name, cfg.renderer_name)

    training_client = service_client.create_training_client(
        base_model=cfg.fireworks_base_model,
        lora_rank=cfg.lora_rank,
        user_metadata=user_metadata,
    )
    if resume_info:
        await checkpoint_utils.check_renderer_name_for_checkpoint_async(
            service_client, resume_info.state_path, cfg.renderer_name
        )
        load_path = await checkpoint_utils.load_resume_state_async(
            training_client, service_client, resume_info
        )
        logger.info(f"Resumed training from {load_path}")
    elif cfg.load_checkpoint_path:
        await checkpoint_utils.check_renderer_name_for_checkpoint_async(
            service_client, cfg.load_checkpoint_path, cfg.renderer_name
        )
        await training_client.load_state(cfg.load_checkpoint_path).result_async()
        logger.info(f"Loaded weights from {cfg.load_checkpoint_path}")

    tokenizer = get_tokenizer(cfg.model_name)
    assert cfg.renderer_name is not None, "renderer_name must be set (resolve before calling main)"
    renderer = renderers.get_renderer(cfg.renderer_name, tokenizer=tokenizer)

    num_batches = len(sdft_dataset)
    if cfg.max_steps is not None:
        num_batches = min(cfg.max_steps, num_batches)
    logger.info(f"Will train on {num_batches} batches")

    # Evaluators
    evaluators: list[SamplingClientEvaluator] = [e() for e in cfg.evaluator_builders]
    if test_dataset is not None:
        evaluators.append(
            RLTestSetEvaluator(_SDFTEvalDatasetAdapter(test_dataset), max_tokens=cfg.max_tokens)
        )

    # Teacher client (static base-model weights)
    teacher_base_model = cfg.teacher_fireworks_base_model or cfg.fireworks_base_model
    teacher_base_url = cfg.teacher_base_url or (cfg.base_url if serverless else None)
    teacher_trainer_job_id = resolve_trainer_job_id(
        base_url=teacher_base_url,
        fireworks_trainer_job_id=cfg.teacher_fireworks_trainer_job_id,
    )
    if teacher_base_url is None and teacher_trainer_job_id is None:
        # The student's trainer serves the frozen base model as the teacher.
        teacher_client = service_client.create_reference_client(
            teacher_base_model, policy_client=training_client
        )
    else:
        # Serverless training is LoRA only, so a serverless teacher is an adapter
        # that is never trained: it leaves the base model's outputs unchanged.
        teacher_lora_rank = cfg.lora_rank if is_serverless_base_url(teacher_base_url) else 0
        teacher_service_client = create_service_client_with_trainer_only(
            trainer_job_id=teacher_trainer_job_id,
            base_model=teacher_base_model,
            lora_rank=teacher_lora_rank,
            training_shape_id=None,
            base_url=teacher_base_url,
            forward_only=True,
        )
        teacher_client = await asyncio.to_thread(
            teacher_service_client.create_training_client,
            base_model=teacher_base_model,
            lora_rank=teacher_lora_rank,
        )
    logger.info(f"Created static teacher client for {teacher_base_model}")

    checkpoint_mgr = checkpoint_utils.CheckpointManager(
        training_client=training_client,
        service_client=service_client,
        log_path=cfg.log_path,
        save_every=cfg.save_every,
        store=store,
    )

    # Initial sampling client for student
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client, checkpoint_mgr, service_client, tokenizer, start_batch, start_batch
    )

    log_path = Path(cfg.log_path)

    for i_batch in range(start_batch, num_batches):
        metrics: dict[str, Any] = {
            "progress/batch": i_batch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }

        with trace.trace_iteration(step=i_batch) as window:
            # Evaluation
            if cfg.eval_every > 0 and i_batch % cfg.eval_every == 0:
                async with trace.scope_span("run_evals"):
                    for evaluator in evaluators:
                        eval_metrics = await evaluator(sampling_client)
                        metrics.update({f"test/{k}": v for k, v in eval_metrics.items()})

            # Get batch: builders + questions + golden answers
            builders_P, questions_P, golden_answers_P = sdft_dataset.get_batch(i_batch)

            # Rollout: student generates completions on-policy.
            # Uses the RL rollout infrastructure (do_group_rollout) rather than
            # raw sampling so that group_size > 1 and multi-turn environments
            # work out of the box. With the default group_size=1, this is
            # equivalent to a single sample_async call per problem.
            async with trace.scope_span("sample"):
                trajectory_groups_raw = await asyncio.gather(
                    *[
                        asyncio.create_task(
                            do_group_rollout_and_filter_constant_reward(
                                sampling_client,
                                builder,
                                temperature=cfg.temperature,
                                max_tokens=cfg.max_tokens,
                                do_remove_constant_reward_groups=False,
                            ),
                            name=f"sample_task_{i}",
                        )
                        for i, builder in enumerate(builders_P)
                    ],
                )
            trajectory_groups_P: list[TrajectoryGroup] = [
                tg for tg in trajectory_groups_raw if tg is not None
            ]

            # Compute trajectory metrics
            taglist_P = [b.logging_tags() for b in builders_P]
            metrics.update(compute_trajectory_metrics(trajectory_groups_P, taglist_P))

            # Assemble training data (advantages start as 0 since rewards are all 0)
            async with trace.scope_span("assemble_training_data"):
                advantages_P = compute_advantages(trajectory_groups_P)
                data_D, metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)

            # Log one example
            if data_D:
                logger.info(colorize_example(data_D[0], tokenizer, key="mask"))

            # Build teacher prompts (one per problem)
            teacher_prompts_P = [
                build_sdft_teacher_prompt(
                    question=question,
                    golden_answer=golden_answer,
                    renderer=renderer,
                    system_prompt=cfg.system_prompt,
                    demo_template=cfg.demo_template,
                )
                for question, golden_answer in zip(questions_P, golden_answers_P)
            ]

            if cfg.topk > 0 and cfg.reverse:
                # Analytical reverse KL over teacher top-K via custom loss
                async with trace.scope_span("build_reverse_kl_datums"):
                    rev_datums, rev_metrics = await build_reverse_kl_datums(
                        data_D,
                        metadata_D,
                        teacher_client,
                        teacher_prompts_P,
                        topk=cfg.topk,
                        max_context_length=cfg.max_context_length,
                        vocab_size=len(tokenizer),
                    )
                metrics.update(rev_metrics)

                async with trace.scope_span("train"):
                    await _train_step_reverse_kl(
                        data_D=rev_datums,
                        training_client=training_client,
                        learning_rate=cfg.learning_rate,
                        num_substeps=cfg.num_substeps,
                        metrics=metrics,
                    )
            elif cfg.topk > 0:
                # Top-K CE distillation (forward KL)
                async with trace.scope_span("build_topk_distillation_datums"):
                    topk_datums, topk_metrics = await build_topk_distillation_datums(
                        data_D,
                        metadata_D,
                        teacher_client,
                        teacher_prompts_P,
                        topk=cfg.topk,
                        max_context_length=cfg.max_context_length,
                        vocab_size=len(tokenizer),
                    )
                metrics.update(topk_metrics)

                async with trace.scope_span("train"):
                    # Top-K CE only (no IS, no student renormalization)
                    await train_step(
                        data_D=topk_datums,
                        training_client=training_client,
                        learning_rate=cfg.learning_rate,
                        num_substeps=cfg.num_substeps,
                        loss_fn="cross_entropy",
                        metrics=metrics,
                    )
            else:
                # DEPRECATED: single-sample importance-sampling approximation of
                # reverse KL. Superseded by `reverse=True` (analytical top-K
                # reverse KL over the teacher distribution). Kept only so that
                # historical configs with `topk=0` continue to run; new work
                # should set `topk>0` and toggle `reverse` for the forward/reverse
                # choice.
                async with trace.scope_span("compute_sdft_advantages"):
                    is_metrics = await compute_sdft_advantages(
                        data_D,
                        metadata_D,
                        teacher_client,
                        teacher_prompts_P,
                        max_context_length=cfg.max_context_length,
                    )
                metrics.update(is_metrics)

                async with trace.scope_span("train"):
                    await train_step(
                        data_D=data_D,
                        training_client=training_client,
                        learning_rate=cfg.learning_rate,
                        num_substeps=cfg.num_substeps,
                        loss_fn=cfg.loss_fn,
                        metrics=metrics,
                    )

            # Refresh sampling client
            sampling_client, _ = await save_checkpoint_and_get_sampling_client(
                training_client,
                checkpoint_mgr,
                service_client,
                tokenizer,
                i_batch + 1,
                start_batch,
            )

        # Log timing
        metrics.update(window.get_timing_metrics())
        window.save_timing(i_batch, store=store)
        if cfg.span_chart_every > 0 and i_batch % cfg.span_chart_every == 0:
            trace.save_gantt_chart_html(
                window, i_batch, log_path / f"timing_gantt_{i_batch:06d}.html"
            )
        ml_logger.log_metrics(metrics, step=i_batch)

    # Final checkpoint
    if start_batch < num_batches:
        await checkpoint_mgr.save_final_async(loop_state={"batch": num_batches})

    ml_logger.close()
    logger.info("SDFT training completed successfully")
