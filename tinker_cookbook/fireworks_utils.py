"""Helpers for training loops that sample from a Fireworks hot-load deployment."""

from collections.abc import Callable
from typing import Any

import tinker
from fireworks.training.sdk import FiretitanServiceClient, FiretitanTrainingClient

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.tokenizer_utils import Tokenizer

WeightSync = Callable[..., tinker.SamplingClient]
"""Publishes the current policy weights and returns a sampling client for them."""


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

    The SDK attaches the deployment to the trainer, so weights published with
    :func:`make_weight_sync` are served by the deployment.

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


def make_weight_sync(
    training_client: FiretitanTrainingClient,
    service_client: FiretitanServiceClient,
    tokenizer: Tokenizer,
) -> WeightSync:
    """Select how policy weights reach the rollout deployment.

    The returned function publishes the training client's current weights and
    returns a sampling client for the deployment that serves them. It blocks
    until the deployment serves the new weights, so call it with
    ``asyncio.to_thread`` from async code.

    Args:
        training_client: Training client whose weights are published.
        service_client: Service client created by
            :func:`create_service_client_with_deployment`.
        tokenizer: Tokenizer used by the returned sampling clients.

    Returns:
        ``publish(name, **save_kwargs)``. ``name`` and ``save_kwargs`` (for
        example ``checkpoint_type="base"``) apply to the sampler snapshot that
        is saved when the weights are not synced over RDMA.
    """
    if training_client.supports_rdma_weight_sync:

        def publish(name: str, **save_kwargs: Any) -> tinker.SamplingClient:
            training_client.weight_sync().result()
            return service_client.create_sampling_client(tokenizer=tokenizer)

        return publish

    def publish(name: str, **save_kwargs: Any) -> tinker.SamplingClient:
        saved = training_client.save_weights_for_sampler(name, **save_kwargs).result()
        service_client.hotload_sampler_snapshot(saved.path)
        return service_client.create_sampling_client(tokenizer=tokenizer)

    return publish
