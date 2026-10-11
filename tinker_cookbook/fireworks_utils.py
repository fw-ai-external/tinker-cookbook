"""Helpers for training loops that run on Fireworks trainers and deployments."""

import os
from collections.abc import Callable
from typing import Any

import tinker
from fireworks.training.sdk import (
    FiretitanServiceClient,
    FiretitanTrainingClient,
    TrainerJobManager,
)

from tinker_cookbook.checkpoint_utils import extract_trainer_job_id
from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.tokenizer_utils import Tokenizer

WeightSync = Callable[..., Any]
"""Publishes the current policy weights: ``training.utils.service.make_weight_sync``'s result."""


def is_serverless_base_url(base_url: str | None) -> bool:
    """Whether ``base_url`` is the Fireworks serverless training endpoint."""
    return bool(base_url) and base_url.rstrip("/").endswith("/training/v1/serverless")


def resolve_trainer_job_id(
    *, base_url: str | None, fireworks_trainer_job_id: str | None
) -> str | None:
    """Return the trainer job a run should reuse, or ``None`` to create one.

    A trainer can be named by ``fireworks_trainer_job_id`` or by a trainer
    ``base_url`` (``.../rlorTrainerJobs/{account}/{job_id}``). Either one alone
    is enough. When both are set they must name the same job.

    Raises:
        ConfigurationError: If both are set and ``base_url`` does not name
            ``fireworks_trainer_job_id``.
    """
    job_id_from_url = extract_trainer_job_id(base_url)
    if not fireworks_trainer_job_id:
        return job_id_from_url
    if base_url and job_id_from_url != fireworks_trainer_job_id:
        raise ConfigurationError(
            f"base_url={base_url!r} and fireworks_trainer_job_id="
            f"{fireworks_trainer_job_id!r} name different trainers; set only one of "
            "them, or make them agree"
        )
    return fireworks_trainer_job_id


def resolve_teacher_base_url(
    *, base_url: str | None, fireworks_trainer_job_id: str | None
) -> str | None:
    """Return the URL of a teacher's trainer, or ``None`` when none is named.

    A teacher connects to its trainer by URL. ``base_url`` is returned as it is.
    A ``fireworks_trainer_job_id`` without a ``base_url`` is looked up on
    Fireworks, waiting for the trainer to be ready. The two must name the same
    job when both are set (see :func:`resolve_trainer_job_id`).

    Raises:
        ConfigurationError: If the two name different trainers, or if the job
            has to be looked up and ``FIREWORKS_API_KEY`` is not set.
    """
    trainer_job_id = resolve_trainer_job_id(
        base_url=base_url, fireworks_trainer_job_id=fireworks_trainer_job_id
    )
    if base_url or trainer_job_id is None:
        return base_url
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise ConfigurationError(
            "FIREWORKS_API_KEY must be set to look up the trainer for "
            f"fireworks_trainer_job_id={trainer_job_id!r}"
        )
    api_url = os.environ.get("FIREWORKS_BASE_URL")
    manager = (
        TrainerJobManager(api_key=api_key, base_url=api_url)
        if api_url
        else TrainerJobManager(api_key=api_key)
    )
    return manager.wait_for_existing(trainer_job_id).base_url


def create_service_client_with_trainer_only(
    *,
    trainer_job_id: str | None,
    base_model: str,
    lora_rank: int,
    training_shape_id: str | None,
    base_url: str | None = None,
    user_metadata: dict[str, str] | None = None,
) -> FiretitanServiceClient:
    """Create a service client bound to a trainer, without a rollout deployment.

    Use this for loops that only train (supervised learning). An existing
    trainer is reused when its ID is given; otherwise the SDK creates one when
    the first client is created. With the serverless ``base_url`` the client
    attaches to the shared serverless trainer and nothing is created.

    Args:
        trainer_job_id: Trainer job to reuse. ``None`` creates one.
        base_url: The run's ``base_url``. Only the serverless endpoint
            (``.../training/v1/serverless``) changes what this returns.
        base_model: Fireworks model ID to train.
        lora_rank: LoRA rank of the policy (0 for full-parameter training).
        training_shape_id: Training shape for a new trainer. ``None`` lets the
            backend choose it.
        user_metadata: Optional run metadata.
    """
    if is_serverless_base_url(base_url):
        return FiretitanServiceClient(base_url=base_url, user_metadata=user_metadata)
    return FiretitanServiceClient.from_firetitan_config(
        base_model=base_model,
        lora_rank=lora_rank,
        training_shape_id=training_shape_id,
        trainer_job_id=trainer_job_id,
        create_deployment=False,
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
        user_metadata=user_metadata,
    )


class ServerlessWeightSync:
    """Publishes the current policy weights on serverless training.

    Serverless training has no rollout deployment to hot-load. Publishing saves
    a sampler snapshot instead, and a sampling client is then bound to that
    snapshot (see :func:`create_policy_sampling_client`).
    """

    def __init__(self, training_client: FiretitanTrainingClient) -> None:
        self._training_client = training_client
        self.sampler_path: str | None = None

    def __call__(self, name: str, **kwargs: Any) -> str:
        sampler_path = self._training_client.save_weights_for_sampler(name).result().path
        self.sampler_path = sampler_path
        return sampler_path


def create_policy_sampling_client(
    service_client: FiretitanServiceClient,
    publish_weights: WeightSync,
    tokenizer: Tokenizer,
) -> tinker.SamplingClient:
    """Create a sampling client for the weights ``publish_weights`` last published.

    On a dedicated trainer the client samples from the rollout deployment,
    which serves the published weights. On serverless training it samples from
    the snapshot that :class:`ServerlessWeightSync` saved.
    """
    if isinstance(publish_weights, ServerlessWeightSync):
        if publish_weights.sampler_path is None:
            raise RuntimeError("publish_weights must be called before sampling")
        return service_client.create_sampling_client(
            model_path=publish_weights.sampler_path, tokenizer=tokenizer
        )
    return service_client.create_sampling_client(tokenizer=tokenizer)
