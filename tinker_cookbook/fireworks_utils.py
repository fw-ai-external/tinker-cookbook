"""Helpers for training loops that sample from a Fireworks hot-load deployment."""

from collections.abc import Callable
from typing import Any

import tinker
import training.utils as fireworks_cookbook
from fireworks.training.sdk import FiretitanServiceClient, FiretitanTrainingClient
from training.utils import DeployConfig, ReconnectableClient

from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.tokenizer_utils import Tokenizer

WeightSync = Callable[..., tinker.SamplingClient]
"""Publishes the current policy weights and returns a sampling client for them."""


def create_service_client_with_deployment(
    *,
    trainer_job_id: str | None,
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
        trainer_job_id: ID of the trainer job to train on.
        base_model: Fireworks model ID the trainer was created with.
        lora_rank: LoRA rank of the policy (0 for full-parameter training).
        deployment_id: Rollout deployment that serves the policy.
        hot_load_timeout: Seconds to wait for each hot-load.
        user_metadata: Optional run metadata.

    Raises:
        ConfigurationError: If ``trainer_job_id`` or ``deployment_id`` is not set.
    """
    if trainer_job_id is None:
        raise ConfigurationError(
            "A trainer job ID is required: set base_url to a Fireworks trainer URL of the form "
            "https://api.fireworks.ai/training/v1/rlorTrainerJobs/<account>/<trainer_job_id>"
        )
    if deployment_id is None:
        raise ConfigurationError(
            "fireworks_deployment_id must be set to sample from the policy being trained"
        )
    return FiretitanServiceClient.from_firetitan_config(
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
    *,
    base_model: str,
    lora_rank: int,
) -> WeightSync:
    """Select how policy weights reach the rollout deployment.

    Wraps ``training.utils.make_weight_sync`` from the Fireworks training
    cookbook. The returned function publishes the training client's current
    weights and returns a sampling client for the deployment that serves them.
    It blocks until the deployment serves the new weights, so call it with
    ``asyncio.to_thread`` from async code.

    Args:
        training_client: Training client whose weights are published.
        service_client: Service client created by
            :func:`create_service_client_with_deployment`.
        tokenizer: Tokenizer used by the returned sampling clients.
        base_model: Fireworks model ID of the policy.
        lora_rank: LoRA rank of the policy (0 for full-parameter training).

    Returns:
        ``publish(name, **save_kwargs)``. ``name`` and ``save_kwargs`` (for
        example ``checkpoint_type="base"``) apply to the sampler snapshot that
        is saved when the weights are not synced over RDMA.
    """
    policy = ReconnectableClient.from_training_client(
        training_client,
        base_model=base_model,
        lora_rank=lora_rank,
        job_id=service_client.trainer_job_id,
        service=service_client,
    )
    publish_weights = fireworks_cookbook.make_weight_sync(
        policy, service_client, DeployConfig(deployment_id=service_client.deployment_id)
    )

    def publish(name: str, **save_kwargs: Any) -> tinker.SamplingClient:
        publish_weights(name, **save_kwargs)
        return service_client.create_sampling_client(tokenizer=tokenizer)

    return publish
