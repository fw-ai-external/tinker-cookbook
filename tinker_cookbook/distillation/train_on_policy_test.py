"""Tests for out-of-vocab token sanitization in on-policy distillation.

On-policy sampling can emit "phantom" token ids that live in a model's padded
embedding matrix but outside the tokenizer's real vocabulary (e.g. Qwen3 has
151,669 real tokens but a larger ``lm_head``, so ids in ``[151669, embedding_rows)``
are samplable yet rejected by the forward pass). These tests verify that
``sanitize_data_vocab`` rewrites such ids before any API call and excludes the
affected positions from the loss/KL.
"""

import pytest
import tinker
import torch

# train_on_policy imports the Fireworks SDK at module load; skip if unavailable.
pytest.importorskip("fireworks.training.sdk")

from tinker_cookbook.distillation.train_on_policy import (  # noqa: E402
    _teacher_forward_datum,
    sanitize_data_vocab,
)

VOCAB = 151669  # Qwen3 real vocab: valid range is [0, 151668]
PHANTOM = 151807  # the id from the original RequestFailedError
EOS = 151643  # a valid replacement token


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_datum(full_sequence: list[int]) -> tinker.Datum:
    """Build a datum the way rl.data_processing does: model_input is
    ``full_sequence[:-1]`` and target_tokens is ``full_sequence[1:]``."""
    n_targets = len(full_sequence) - 1
    return tinker.Datum(
        model_input=tinker.ModelInput(
            chunks=[tinker.EncodedTextChunk(tokens=full_sequence[:-1])]
        ),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(
                torch.tensor(full_sequence[1:], dtype=torch.long)
            ),
            "logprobs": tinker.TensorData.from_torch(
                torch.tensor([-1.0] * n_targets, dtype=torch.float32)
            ),
            "advantages": tinker.TensorData.from_torch(
                torch.tensor([5.0] * n_targets, dtype=torch.float32)
            ),
            "mask": tinker.TensorData.from_torch(
                torch.tensor([1.0] * n_targets, dtype=torch.float32)
            ),
        },
    )


def _ints(datum: tinker.Datum) -> list[int]:
    return datum.model_input.to_ints()


def _targets(datum: tinker.Datum) -> list[int]:
    return datum.loss_fn_inputs["target_tokens"].to_torch().tolist()


def _mask(datum: tinker.Datum) -> list[float]:
    return datum.loss_fn_inputs["mask"].to_torch().tolist()


def _advantages(datum: tinker.Datum) -> list[float]:
    return datum.loss_fn_inputs["advantages"].to_torch().tolist()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_phantom_token_replaced_in_input_and_targets():
    # Phantom id sits at absolute index 2 of the full sequence.
    data_D = [_make_datum([100, 200, PHANTOM, 300, 400])]

    n = sanitize_data_vocab(data_D, VOCAB, replacement_token_id=EOS)

    d = data_D[0]
    # The phantom appears in both model_input (idx 2) and target_tokens (idx 1).
    assert n == 2
    assert _ints(d) == [100, 200, EOS, 300]
    assert _targets(d) == [200, EOS, 300, 400]
    # Every id sent to the server is now within the valid range.
    assert all(0 <= t < VOCAB for t in _ints(d))
    assert all(0 <= t < VOCAB for t in _targets(d))


def test_phantom_target_position_is_masked_out():
    data_D = [_make_datum([100, 200, PHANTOM, 300, 400])]
    sanitize_data_vocab(data_D, VOCAB, replacement_token_id=EOS)
    d = data_D[0]

    # Phantom is target index 1 -> masked out of loss and KL; neighbors untouched.
    assert _mask(d) == [1.0, 0.0, 1.0, 1.0]
    assert _advantages(d) == [5.0, 0.0, 5.0, 5.0]


def test_logprobs_are_left_untouched():
    data_D = [_make_datum([100, 200, PHANTOM, 300, 400])]
    sanitize_data_vocab(data_D, VOCAB, replacement_token_id=EOS)
    logprobs = data_D[0].loss_fn_inputs["logprobs"].to_torch().tolist()
    assert logprobs == [-1.0, -1.0, -1.0, -1.0]


def test_teacher_forward_path_is_clean_after_sanitization():
    data_D = [_make_datum([100, 200, PHANTOM, 300, 400])]
    sanitize_data_vocab(data_D, VOCAB, replacement_token_id=EOS)
    d = data_D[0]

    # incorporate_kl_penalty appends the last target token, then builds the
    # teacher datum via _teacher_forward_datum. Both must be in-range.
    full_seq_input = d.model_input.append_int(int(_targets(d)[-1]))
    teacher_datum = _teacher_forward_datum(full_seq_input)
    assert all(0 <= t < VOCAB for t in teacher_datum.model_input.to_ints())
    assert all(
        0 <= t < VOCAB
        for t in teacher_datum.loss_fn_inputs["target_tokens"].to_torch().tolist()
    )


def test_clean_datum_is_returned_unchanged():
    original = _make_datum([100, 200, 300, 400])
    data_D = [original]

    n = sanitize_data_vocab(data_D, VOCAB, replacement_token_id=EOS)

    assert n == 0
    # No rewrite -> same object identity (no needless tensor copies).
    assert data_D[0] is original


def test_boundary_id_is_kept_threshold_id_is_replaced():
    # vocab_size is exclusive: vocab_size-1 is valid, vocab_size is not.
    data_D = [_make_datum([VOCAB - 1, VOCAB, 100])]

    n = sanitize_data_vocab(data_D, VOCAB, replacement_token_id=EOS)

    d = data_D[0]
    assert n == 2  # VOCAB appears in both model_input (idx 1) and targets (idx 0)
    assert _ints(d) == [VOCAB - 1, EOS]
    assert _targets(d) == [EOS, 100]


def test_negative_token_id_is_replaced():
    data_D = [_make_datum([100, -1, 200])]

    n = sanitize_data_vocab(data_D, VOCAB, replacement_token_id=EOS)

    d = data_D[0]
    assert n == 2
    assert all(t >= 0 for t in _ints(d))
    assert all(t >= 0 for t in _targets(d))


def test_phantom_as_last_token_only_hits_targets():
    # The final token of the sequence appears only in target_tokens, not in
    # model_input (which is full_sequence[:-1]).
    data_D = [_make_datum([100, 200, 300, PHANTOM])]

    n = sanitize_data_vocab(data_D, VOCAB, replacement_token_id=EOS)

    d = data_D[0]
    assert n == 1
    assert _ints(d) == [100, 200, 300]  # unchanged
    assert _targets(d) == [200, 300, EOS]
    assert _mask(d) == [1.0, 1.0, 0.0]


def test_batch_aggregates_count_and_only_touches_offenders():
    clean = _make_datum([10, 20, 30])
    dirty = _make_datum([10, PHANTOM, 30])
    data_D = [clean, dirty]

    n = sanitize_data_vocab(data_D, VOCAB, replacement_token_id=EOS)

    assert n == 2  # only the phantom in `dirty`, counted in input + targets
    assert data_D[0] is clean  # untouched datum keeps its identity
    assert all(0 <= t < VOCAB for t in _ints(data_D[1]))
    assert all(0 <= t < VOCAB for t in _targets(data_D[1]))
