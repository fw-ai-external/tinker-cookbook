"""Helpers for training loops that sample from a Fireworks hot-load deployment."""

from collections.abc import Callable
from typing import Any

from fireworks.training.sdk import FiretitanServiceClient

from tinker_cookbook.exceptions import ConfigurationError

WeightSync = Callable[..., Any]
"""Publishes the current policy weights: ``training.utils.service.make_weight_sync``'s result."""


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
    ``training.utils.service.make_weight_sync`` are served by the deployment.

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
