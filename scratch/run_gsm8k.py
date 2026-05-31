import asyncio
import inspect
import os
import re
from pathlib import Path
from typing import Any, cast

import tinker
from fireworks.training.sdk import FiretitanSamplingClient  # type: ignore[import-not-found]
from tinker_cookbook import model_info, renderers
from tinker_cookbook.eval.benchmarks import BenchmarkConfig, run_benchmark
from tinker_cookbook.tokenizer_utils import get_tokenizer


BASE_MODEL = "Qwen/Qwen3-8B-Base"
# MODEL_PATH = "accounts/pyroworks/deployments/otoy65mg"
MODEL_PATH = "accounts/pyroworks/deployments/br7kn1tp"
# SAVE_DIR = str(Path(__file__).resolve().parent / "evals/qwen3-8b-base-promote-step-58-gsm8k-boxed-stop-v2")
# SAVE_DIR = str(Path(__file__).resolve().parent / "evals/qwen3-8b-base")
SAVE_DIR = str(Path(__file__).resolve().parent / "evals/openthoughts-off-policy-qwen-8b-base")


def make_sampled_sequence(tokens: list[int]) -> tinker.SampledSequence:
    logprobs = [0.0] * len(tokens)
    sampled_sequence = cast(Any, tinker.SampledSequence)
    if "tokens" in inspect.signature(tinker.SampledSequence).parameters:
        return sampled_sequence(
            stop_reason="stop",
            tokens=tokens,
            logprobs=logprobs,
        )
    return sampled_sequence(
        stop_reason="stop",
        _tokens_list=tokens,
        _logprobs_list=logprobs,
    )


def make_sample_response(
    sequences: list[tinker.SampledSequence],
    prompt_logprobs: list[float | None] | None,
) -> tinker.SampleResponse:
    sample_response = cast(Any, tinker.SampleResponse)
    if "prompt_logprobs" in inspect.signature(tinker.SampleResponse).parameters:
        return sample_response(
            sequences=sequences,
            prompt_logprobs=prompt_logprobs,
        )
    return sample_response(
        sequences=sequences,
        _prompt_logprobs_list=prompt_logprobs,
    )


class StopAfterBoxedClient:
    """Treat the first boxed answer as a clean RoleColon stop for this eval."""

    def __init__(self, client: FiretitanSamplingClient, tokenizer: Any):
        self.client = client
        self.tokenizer = tokenizer

    async def sample_async(
        self,
        prompt: tinker.ModelInput,
        num_samples: int,
        sampling_params: tinker.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ) -> tinker.SampleResponse:
        response = await self.client.sample_async(
            prompt,
            num_samples,
            sampling_params,
            include_prompt_logprobs,
            topk_prompt_logprobs,
        )

        sequences = []
        for sequence in response.sequences:
            text = self.tokenizer.decode(sequence.tokens)
            match = re.search(r"\\boxed\{[^{}]*\}", text)
            if match is None:
                sequences.append(sequence)
                continue

            trimmed_text = text[: match.end()] + "\n\nUser:"
            trimmed_tokens = self.tokenizer.encode(trimmed_text, add_special_tokens=False)
            sequences.append(make_sampled_sequence(trimmed_tokens))

        return make_sample_response(sequences, response.prompt_logprobs)


async def main() -> None:
    tokenizer = get_tokenizer(BASE_MODEL)
    renderer_name = model_info.get_recommended_renderer_name(BASE_MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    sampling_client = FiretitanSamplingClient.create(
        inference_url=os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai"),
        model=MODEL_PATH,
        api_key=os.environ["FIREWORKS_API_KEY"],
        tokenizer=tokenizer,
    )
    stopping_client = StopAfterBoxedClient(sampling_client, tokenizer)
    try:
        result = await run_benchmark(
            "gsm8k",
            cast(Any, stopping_client),
            renderer,
            BenchmarkConfig(
                max_examples=100,
                max_tokens=2048,
                save_dir=SAVE_DIR,
                system_prompt=(
                    "Solve the problem, then write the final numeric answer in \\boxed{}."
                ),
            ),
        )
        print(f"GSM8K: {result.score:.1%} ({result.num_correct}/{result.num_examples})")
    finally:
        sampling_client.close()


if __name__ == "__main__":
    asyncio.run(main())
