"""Helpers for training loops that run on Fireworks trainers and deployments."""

import os
import re
from collections.abc import Callable
from typing import Any

import tinker
from fireworks.training.sdk import (
    FiretitanServiceClient,
    FiretitanTrainingClient,
    FireworksClient,
)

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
    """Return the trainer job a config names, or ``None`` when it names none.

    A trainer is named by ``fireworks_trainer_job_id`` or by a trainer
    ``base_url`` (``.../rlorTrainerJobs/{account}/{job_id}``). When both are
    set they must name the same job.
    """
    match = re.search(r"/rlorTrainerJobs/[^/]+/([^/]+)", base_url or "")
    job_id_from_url = match.group(1) if match else None
    if not fireworks_trainer_job_id:
        return job_id_from_url
    if base_url and job_id_from_url != fireworks_trainer_job_id:
        raise ConfigurationError(
            f"base_url={base_url!r} and fireworks_trainer_job_id="
            f"{fireworks_trainer_job_id!r} name different trainers; set only one of "
            "them, or make them agree"
        )
    return fireworks_trainer_job_id


def resolve_trainer_base_url(
    *, base_url: str | None, fireworks_trainer_job_id: str | None
) -> str | None:
    """Return the ``base_url`` a config names, or ``None`` when it names no trainer.

    For clients that connect by URL. A ``fireworks_trainer_job_id`` without a
    ``base_url`` becomes the trainer's URL, which contains the account of
    ``FIREWORKS_API_KEY``.
    """
    trainer_job_id = resolve_trainer_job_id(
        base_url=base_url, fireworks_trainer_job_id=fireworks_trainer_job_id
    )
    if base_url or trainer_job_id is None:
        return base_url
    api_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai").rstrip("/")
    account_id = FireworksClient(
        api_key=os.environ["FIREWORKS_API_KEY"], base_url=api_url
    ).account_id
    return f"{api_url}/training/v1/rlorTrainerJobs/{account_id}/{trainer_job_id}"


def create_service_client_with_trainer_only(
    *,
    trainer_job_id: str | None,
    base_model: str,
    lora_rank: int,
    training_shape_id: str | None,
    base_url: str | None = None,
    forward_only: bool = False,
    user_metadata: dict[str, str] | None = None,
) -> FiretitanServiceClient:
    """Create a service client for a loop that only trains.

    The trainer ``trainer_job_id`` is reused; ``None`` creates one, with
    ``training_shape_id`` if given. ``forward_only`` creates a trainer that
    only runs forward passes, for a frozen model. A serverless ``base_url``
    attaches to the shared serverless trainer instead and creates nothing.
    """
    if is_serverless_base_url(base_url):
        return FiretitanServiceClient(base_url=base_url, user_metadata=user_metadata)
    return FiretitanServiceClient.from_firetitan_config(
        base_model=base_model,
        lora_rank=lora_rank,
        training_shape_id=training_shape_id,
        trainer_job_id=trainer_job_id,
        create_deployment=False,
        forward_only=forward_only,
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
    base_url: str | None = None,
    reference_required: bool = False,
    user_metadata: dict[str, str] | None = None,
) -> FiretitanServiceClient:
    """Create a service client for a loop that trains and samples.

    The trainer ``trainer_job_id`` and the rollout deployment ``deployment_id``
    are reused; ``None`` creates one. ``training_shape_id`` selects the shape
    of a new trainer and deployment and is required to create a deployment.
    The deployment serves the weights published with ``make_weight_sync``.

    A serverless ``base_url`` attaches to the shared serverless trainer instead
    and creates nothing; the loop then publishes with
    :class:`ServerlessWeightSync`.
    """
    if is_serverless_base_url(base_url):
        if deployment_id is not None:
            raise ConfigurationError(
                "fireworks_deployment_id cannot be used with serverless training"
            )
        if lora_rank <= 0:
            raise ConfigurationError("serverless training requires lora_rank > 0")
        return FiretitanServiceClient(base_url=base_url, user_metadata=user_metadata)
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


def create_teacher_client(
    *,
    base_url: str | None,
    fireworks_trainer_job_id: str | None,
    base_model: str,
    training_shape_id: str | None = None,
) -> FiretitanTrainingClient:
    """Create a frozen client for a distillation teacher's forward passes.

    A teacher named by ``fireworks_trainer_job_id`` or ``base_url`` (a trainer
    URL or the serverless endpoint) runs there. Otherwise a forward-only
    trainer is created for it, with ``training_shape_id`` if given.
    """
    teacher_base_url = resolve_trainer_base_url(
        base_url=base_url, fireworks_trainer_job_id=fireworks_trainer_job_id
    )
    if teacher_base_url is not None:
        return FiretitanServiceClient(base_url=teacher_base_url).create_base_training_client(
            base_model=base_model
        )
    return create_service_client_with_trainer_only(
        trainer_job_id=None,
        base_model=base_model,
        lora_rank=0,
        training_shape_id=training_shape_id,
        forward_only=True,
    ).create_training_client(base_model=base_model, lora_rank=0)


class ServerlessWeightSync:
    """Publishes policy weights on serverless training by saving a sampler snapshot.

    Serverless training has no rollout deployment to hot-load.
    """

    def __init__(self, training_client: FiretitanTrainingClient) -> None:
        self._training_client = training_client
        self.sampler_path: str | None = None

    def __call__(self, name: str, **kwargs: Any) -> None:
        self.sampler_path = self._training_client.save_weights_for_sampler(name).result().path


def create_policy_sampling_client(
    service_client: FiretitanServiceClient,
    publish_weights: WeightSync,
    tokenizer: Tokenizer,
) -> tinker.SamplingClient:
    """Create a sampling client for the weights ``publish_weights`` last published.

    It samples from the rollout deployment, or on serverless training from the
    snapshot that :class:`ServerlessWeightSync` saved.
    """
    if isinstance(publish_weights, ServerlessWeightSync):
        return service_client.create_sampling_client(
            model_path=publish_weights.sampler_path, tokenizer=tokenizer
        )
    return service_client.create_sampling_client(tokenizer=tokenizer)
