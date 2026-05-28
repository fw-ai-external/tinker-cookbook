"""Verify that a custom cross_entropy implementation matches Tinker's built-in
`cross_entropy` loss when target_tokens / weights have shape (N, K).

Run with: python scratch/custom_ce_nk.py
"""

import asyncio
import os

import tinker
import torch
from tinker import TensorData


def custom_cross_entropy(
    data: list[tinker.Datum],
    logprobs_list: list[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Reverse-engineered cross_entropy.

    For (N,) shaped target_tokens/weights this is just `-(lp * w).sum()`.
    For (N, K) the server returns lp with shape (N, K) — one logprob per
    (position, k-th target token) — so the same elementwise expression
    works without any reshape.
    """
    total_loss = torch.zeros((), device=logprobs_list[0].device)
    for datum, lp in zip(data, logprobs_list):
        weights = datum.loss_fn_inputs["weights"].to_torch().to(lp.device).to(lp.dtype)
        total_loss = total_loss + (-lp * weights).sum()
    return total_loss, {"custom_ce": total_loss.item()}


def make_datum_nk(
    tokenizer, prompt: str, K: int, seed: int
) -> tuple[tinker.Datum, int]:
    """Build a Datum whose target_tokens/weights have shape (N, K).

    Per-position weights are random non-negative numbers summing to 1 over K
    (a soft target). Prompt positions are zeroed out so they don't contribute.
    """
    g = torch.Generator().manual_seed(seed)
    ids = tokenizer.encode(prompt)
    N = len(ids) - 1
    model_input = tinker.ModelInput.from_ints(ids[:-1])

    # Random target tokens (deduplicated per row to avoid double-counting the
    # same student logprob).
    vocab_size = tokenizer.vocab_size
    target_tokens = torch.zeros(N, K, dtype=torch.long)
    for n in range(N):
        # Sample K distinct token ids per position.
        perm = torch.randperm(vocab_size, generator=g)[:K]
        target_tokens[n] = perm

    # Random soft weights per active position, normalised to sum to 1.
    raw = torch.rand(N, K, generator=g)
    weights = raw / raw.sum(dim=-1, keepdim=True)

    # Mask out the first half of positions (treat as "prompt", weight=0).
    prompt_len = N // 2
    weights[:prompt_len] = 0.0

    return (
        tinker.Datum(
            model_input=model_input,
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(target_tokens),
                "weights": TensorData.from_torch(weights),
            },
        ),
        N,
    )


async def main():
    assert "TINKER_API_KEY" in os.environ, "set TINKER_API_KEY"

    MODEL = "Qwen/Qwen3-4B-Instruct-2507"
    K = 5
    prompts = [
        "The capital of France is Paris, which is also famous for the Eiffel Tower.",
        "Photosynthesis converts sunlight into chemical energy stored in glucose.",
    ]

    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=MODEL, rank=8
    )
    tokenizer = training_client.get_tokenizer()

    data = []
    lengths = []
    for i, p in enumerate(prompts):
        d, N = make_datum_nk(tokenizer, p, K=K, seed=i)
        data.append(d)
        lengths.append(N)
    print(f"Built {len(data)} datums with K={K}, N={lengths}")

    # 1) Built-in cross_entropy
    builtin_fut = await training_client.forward_backward_async(data, loss_fn="cross_entropy")
    builtin_res = await builtin_fut.result_async()
    builtin_loss = builtin_res.metrics["loss:sum"]

    # 2) Custom loss via forward_backward_custom
    custom_fut = await training_client.forward_backward_custom_async(
        data, custom_cross_entropy
    )
    custom_res = await custom_fut.result_async()
    custom_loss = custom_res.metrics["custom_ce"]

    print(f"\nbuilt-in cross_entropy loss:sum = {builtin_loss:.6f}")
    print(f"custom   cross_entropy loss     = {custom_loss:.6f}")
    print(f"abs diff                        = {abs(builtin_loss - custom_loss):.2e}")
    print(f"rel diff                        = "
          f"{abs(builtin_loss - custom_loss) / max(abs(builtin_loss), 1e-12):.2e}")

    # Also sanity-check per-datum logprobs match between the two paths.
    builtin_lps = builtin_res.loss_fn_outputs[0]["logprobs"]
    custom_lps = custom_res.loss_fn_outputs[0]["logprobs"]
    builtin_t = (
        builtin_lps.to_torch() if hasattr(builtin_lps, "to_torch") else torch.as_tensor(builtin_lps)
    )
    custom_t = (
        custom_lps.to_torch() if hasattr(custom_lps, "to_torch") else torch.as_tensor(custom_lps)
    )
    print(f"\nlogprobs[0] shapes: builtin={tuple(builtin_t.shape)} custom={tuple(custom_t.shape)}")
    print(f"max |Δlogprobs[0]|              = {(builtin_t - custom_t).abs().max().item():.2e}")


if __name__ == "__main__":
    asyncio.run(main())
