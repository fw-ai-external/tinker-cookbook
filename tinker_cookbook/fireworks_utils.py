"""Helpers for training loops that sample from a Fireworks hot-load deployment."""

import asyncio
import time
from typing import Any

import tinker
from fireworks.training.sdk import FiretitanServiceClient, FiretitanTrainingClient

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.tokenizer_utils import Tokenizer


def create_service_client_with_deployment(
    *,
    base_url: str | None,
    base_model: str,
    lora_rank: int,
    deployment_id: str | None,
    hot_load_timeout: int,
    user_metadata: dict[str, str] | None = None,
) -> FiretitanServiceClient:
    """Create a service client bound to an existing trainer and rollout deployment.

    The SDK attaches the deployment to the trainer, so sampler weights saved with
    ``save_weights_for_sampler`` can be served through
    ``create_sampling_client(model_path=...)``.

    Args:
        base_url: Trainer URL of the form
            ``https://api.fireworks.ai/training/v1/rlorTrainerJobs/{account}/{job_id}``.
        base_model: Fireworks model ID the trainer was created with.
        lora_rank: LoRA rank of the policy (0 for full-parameter training).
        deployment_id: Rollout deployment that serves the policy.
        hot_load_timeout: Seconds to wait for each hot-load.
        user_metadata: Optional run metadata.

    Raises:
        ConfigurationError: If ``base_url`` is not a trainer URL or
            ``deployment_id`` is not set.
    """
    trainer_job_id = checkpoint_utils.extract_trainer_job_id(base_url)
    if base_url is None or trainer_job_id is None:
        raise ConfigurationError(
            "base_url must be a Fireworks trainer URL of the form "
            "https://api.fireworks.ai/training/v1/rlorTrainerJobs/<account>/<trainer_job_id>, "
            f"got {base_url!r}"
        )
    if deployment_id is None:
        raise ConfigurationError(
            "fireworks_deployment_id must be set to sample from the policy being trained"
        )
    return FiretitanServiceClient.from_firetitan_config(
        base_url=base_url[: base_url.index("/rlorTrainerJobs/")].removesuffix("/training/v1"),
        base_model=base_model,
        lora_rank=lora_rank,
        trainer_job_id=trainer_job_id,
        deployment_id=deployment_id,
        hotload_timeout_s=hot_load_timeout,
        user_metadata=user_metadata,
    )


async def save_weights_and_get_sampling_client(
    training_client: FiretitanTrainingClient,
    service_client: FiretitanServiceClient,
    tokenizer: Tokenizer,
    name: str,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    """Save sampler weights, hot-load them, and return a sampling client for them.

    Args:
        training_client: Training client whose current weights are saved.
        service_client: Service client created by
            :func:`create_service_client_with_deployment`.
        tokenizer: Tokenizer used by the returned sampling client.
        name: Name of the sampler snapshot.

    Returns:
        A ``(tinker.SamplingClient, metrics)`` pair. The metrics hold the save
        and hot-load durations in seconds.
    """
    t0 = time.time()
    save_future = await training_client.save_weights_for_sampler_async(name)
    path = (await save_future.result_async()).path
    if not path:
        raise RuntimeError(f"save_weights_for_sampler({name!r}) returned no path")
    t1 = time.time()
    # Creating the sampling client hot-loads the snapshot, which blocks until
    # the deployment serves it.
    sampling_client = await asyncio.to_thread(
        service_client.create_sampling_client, model_path=path, tokenizer=tokenizer
    )
    metrics = {
        "weight_sync/save_time_s": t1 - t0,
        "weight_sync/hotload_time_s": time.time() - t1,
    }
    return sampling_client, metrics
