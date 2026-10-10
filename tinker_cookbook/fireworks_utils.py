"""Helpers for training loops that run on Fireworks trainers and deployments."""

from collections.abc import Callable
from typing import Any

from fireworks.training.sdk import (
    CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO,
    FiretitanServiceClient,
)

from tinker_cookbook.exceptions import ConfigurationError

WeightSync = Callable[..., Any]
"""Publishes the current policy weights: ``training.utils.service.make_weight_sync``'s result."""


def create_service_client_with_trainer_only(
    *,
    trainer_job_id: str | None,
    base_model: str,
    lora_rank: int,
    training_shape_id: str | None,
    cleanup_on_exit: bool = True,
    user_metadata: dict[str, str] | None = None,
) -> FiretitanServiceClient:
    """Create a service client bound to a trainer, without a rollout deployment.

    Use this for loops that only train (supervised learning). An existing
    trainer is reused when its ID is given; otherwise the SDK creates one when
    the first client is created.

    Args:
        trainer_job_id: Trainer job to reuse. ``None`` creates one.
        base_model: Fireworks model ID to train.
        lora_rank: LoRA rank of the policy (0 for full-parameter training).
        training_shape_id: Training shape for a new trainer. ``None`` lets the
            backend choose it.
        cleanup_on_exit: When the service client is closed, delete the trainer
            if this run created it. A reused trainer is never touched.
        user_metadata: Optional run metadata.
    """
    return FiretitanServiceClient.from_firetitan_config(
        base_model=base_model,
        lora_rank=lora_rank,
        training_shape_id=training_shape_id,
        trainer_job_id=trainer_job_id,
        create_deployment=False,
        cleanup_trainer_on_close=cleanup_on_exit,
        user_metadata=user_metadata,
    )


def create_service_client_with_trainer_and_deployment(
    *,
    trainer_job_id: str | None,
    base_model: str,
    lora_rank: int,
    deployment_id: str | None,
    training_shape_id: str | None,
    hot_load_timeout: int,
    cleanup_on_exit: bool = True,
    reference_required: bool = False,
    user_metadata: dict[str, str] | None = None,
) -> FiretitanServiceClient:
    """Create a service client bound to a trainer and a rollout deployment.

    An existing trainer or deployment is reused when its ID is given; otherwise
    the SDK creates one. The SDK attaches the deployment to the trainer, so
    weights published with ``training.utils.service.make_weight_sync`` are
    served by the deployment. Resources are provisioned when the first client
    is created.

    Args:
        trainer_job_id: Trainer job to reuse. ``None`` creates one.
        base_model: Fireworks model ID to train.
        lora_rank: LoRA rank of the policy (0 for full-parameter training).
        deployment_id: Rollout deployment to reuse. ``None`` creates one.
        training_shape_id: Training shape for a new trainer; it also selects
            the shape of a new deployment. ``None`` lets the backend choose the
            trainer shape.
        hot_load_timeout: Seconds to wait for each hot-load.
        cleanup_on_exit: When the service client is closed, delete the trainer
            and scale the deployment to zero if this run created them. Reused
            resources are never touched.
        reference_required: Whether the run needs a frozen reference model
            (see ``FiretitanServiceClient.create_reference_client``).
        user_metadata: Optional run metadata.

    Raises:
        ConfigurationError: If a deployment has to be created and
            ``training_shape_id`` is not set.
    """
    if deployment_id is None and training_shape_id is None:
        raise ConfigurationError(
            "fireworks_training_shape_id must be set to create a rollout deployment; "
            "set fireworks_deployment_id instead to reuse an existing one"
        )
    return FiretitanServiceClient.from_firetitan_config(
        base_model=base_model,
        lora_rank=lora_rank,
        training_shape_id=training_shape_id,
        trainer_job_id=trainer_job_id,
        deployment_id=deployment_id,
        hotload_timeout_s=hot_load_timeout,
        reference_required=reference_required,
        cleanup_trainer_on_close=cleanup_on_exit,
        cleanup_deployment_on_close=(
            CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO if cleanup_on_exit else None
        ),
        user_metadata=user_metadata,
    )
