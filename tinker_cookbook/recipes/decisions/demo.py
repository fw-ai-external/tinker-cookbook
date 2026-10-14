"""Interactive demo: route a question about Tinker to the team that can answer it.

Prints the probability of each team as JSON; an empty line or Ctrl-D quits.

    uv run -m tinker_cookbook.recipes.decisions.demo
    echo 'How do I pick a learning rate for LoRA?' | uv run -m tinker_cookbook.recipes.decisions.demo

    # A fine-tuned training state saved with save_state
    uv run -m tinker_cookbook.recipes.decisions.demo --model_path=<account>/<run>/final-state

Runs on Fireworks serverless training; requires ``FIREWORKS_API_KEY``.
"""

from __future__ import annotations

import asyncio
import json

import chz

from tinker_cookbook import model_info, renderers
from tinker_cookbook.fireworks_utils import (
    FIREWORKS_SERVERLESS_TRAINING_BASE_URL,
    create_service_client_with_trainer_only,
)
from tinker_cookbook.recipes.decisions import Decision, DecisionRenderer, sample_decision
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils.git_rev import recipe_user_metadata

TEAMS = [
    ("billing", "Charges, invoices, credits, payment methods and pricing"),
    ("sdk", "The tinker Python package: installation, API errors, training and sampling calls"),
    ("console", "The web console: signing in, API keys, viewing runs and checkpoints"),
    ("cookbook", "The Tinker Cookbook recipes: training loops, renderers, evaluations and docs"),
]


@chz.chz
class Config:
    model_name: str = "Qwen/Qwen3.8-27B"
    """Base model to use as the decision model."""

    fireworks_base_model: str = "accounts/fireworks/models/qwen3p8-27b"
    """The Fireworks model of ``model_name``."""

    lora_rank: int = 32

    model_path: str | None = None
    """Optional path to a fine-tuned training state for ``model_name``."""

    base_url: str | None = FIREWORKS_SERVERLESS_TRAINING_BASE_URL
    fireworks_trainer_job_id: str | None = None
    fireworks_training_shape_id: str | None = None


async def main(config: Config) -> None:
    service_client = create_service_client_with_trainer_only(
        trainer_job_id=config.fireworks_trainer_job_id,
        base_model=config.fireworks_base_model,
        lora_rank=config.lora_rank,
        training_shape_id=config.fireworks_training_shape_id,
        base_url=config.base_url,
        user_metadata=recipe_user_metadata("recipe_decisions_demo"),
    )
    training_client = service_client.create_training_client(
        base_model=config.fireworks_base_model, lora_rank=config.lora_rank
    )
    if config.model_path:
        await (await training_client.load_state_async(config.model_path)).result_async()
    renderer = renderers.get_renderer(
        model_info.get_recommended_renderer_name(config.model_name),
        get_tokenizer(config.model_name),
    )
    decision_renderer = DecisionRenderer(renderer)

    while True:
        try:
            question = input("Ask a question about Tinker (empty to quit): ").strip()
        except EOFError:
            break
        if not question:
            break
        decision = Decision(
            f"A user asked the following question about Tinker:\n\n{question}\n\n"
            "Which team should answer it?",
            TEAMS,
        )
        probabilities = await sample_decision(training_client, decision_renderer.render(decision))
        print(json.dumps({team: round(p, 4) for team, p in probabilities.items()}, indent=2))


if __name__ == "__main__":
    asyncio.run(main(chz.entrypoint(Config, allow_hyphens=True)))
