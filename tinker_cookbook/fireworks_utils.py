"""Helpers for training loops that run on Fireworks trainers and deployments."""

import asyncio
import os
import re

import tinker
from fireworks.training.sdk import (
    FiretitanServiceClient,
    FiretitanTrainingClient,
    FireworksClient,
)
from training.utils.client import ReconnectableClient
from training.utils.config import DeployConfig
from training.utils.service import make_weight_sync

from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.tokenizer_utils import Tokenizer

FIREWORKS_SERVERLESS_TRAINING_BASE_URL = "https://api.fireworks.ai/training/v1/serverless"


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
    reference_required: bool = False,
    user_metadata: dict[str, str] | None = None,
) -> FiretitanServiceClient:
    """Create a service client for a loop that only trains.

    The trainer ``trainer_job_id`` is reused; ``None`` creates one, with
    ``training_shape_id`` if given. ``forward_only`` creates a trainer that
    only runs forward passes, for a frozen model. ``reference_required`` is
    for a loop that also scores with a frozen reference model. A serverless
    ``base_url`` attaches to the shared serverless trainer instead and creates
    nothing.
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
        reference_required=reference_required,
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
    are reused; ``None`` creates one. ``training_shape_id`` is required and
    selects the shape of a new trainer and deployment.
    The deployment serves the weights published with
    :func:`weight_sync_and_get_sampling_client`.

    A serverless ``base_url`` attaches to the shared serverless trainer instead
    and creates nothing.
    """
    if is_serverless_base_url(base_url):
        if deployment_id is not None:
            raise ConfigurationError(
                "fireworks_deployment_id cannot be used with serverless training"
            )
        if lora_rank <= 0:
            raise ConfigurationError("serverless training requires lora_rank > 0")
        return FiretitanServiceClient(base_url=base_url, user_metadata=user_metadata)
    if training_shape_id is None:
        raise ConfigurationError(
            "fireworks_training_shape_id must be set for a dedicated trainer and its "
            "rollout deployment"
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


def create_base_model_sampling_client(
    base_url: str | None, base_model: str, tokenizer: Tokenizer, lora_rank: int = 32
) -> tinker.SamplingClient:
    """Create a sampling client for ``base_model`` on Fireworks serverless training.

    For a fixed model that is not being trained, such as an opponent or a
    judge. The client samples from the snapshot of an adapter that is never
    trained, in its own serverless session, so it samples from the base model.
    """
    if not is_serverless_base_url(base_url):
        # TODO(fireworks): sample from a fixed base model on a dedicated trainer too.
        raise ConfigurationError(
            f"Fireworks samples from the fixed base model {base_model!r} on serverless "
            "training only (work in progress); set base_url to the serverless training "
            "endpoint"
        )
    service_client = FiretitanServiceClient(base_url=base_url)
    training_client = service_client.create_training_client(
        base_model=base_model, lora_rank=lora_rank
    )
    sampler_path = training_client.save_weights_for_sampler("base-model").result().path
    return service_client.create_sampling_client(model_path=sampler_path, tokenizer=tokenizer)


def weight_sync_and_get_sampling_client(
    service_client: FiretitanServiceClient,
    training_client: FiretitanTrainingClient,
    name: str,
    tokenizer: Tokenizer,
    checkpoint_type: str | None = None,
) -> tinker.SamplingClient:
    """Publish the current weights as ``name`` and return a sampling client for them.

    A dedicated trainer syncs the weights to the rollout deployment, which the
    sampling client samples from. Serverless training saves a sampler snapshot,
    which the sampling client samples from directly.
    """
    # Only a serverless service client owns a training session.
    if service_client.training_session_id is not None:
        sampler_path = training_client.save_weights_for_sampler(name).result().path
        return service_client.create_sampling_client(model_path=sampler_path, tokenizer=tokenizer)
    policy = ReconnectableClient.from_training_client(
        training_client,
        base_model="",  # required, but never read by a wrapper built from a training client
        job_id=service_client.trainer_job_id,
        service=service_client,
    )
    weight_sync = make_weight_sync(
        policy, service_client, DeployConfig(deployment_id=service_client.deployment_id)
    )
    weight_sync(name, checkpoint_type=checkpoint_type)
    return service_client.create_sampling_client(tokenizer=tokenizer)


async def weight_sync_and_get_sampling_client_async(
    service_client: FiretitanServiceClient,
    training_client: FiretitanTrainingClient,
    name: str,
    tokenizer: Tokenizer,
    checkpoint_type: str | None = None,
) -> tinker.SamplingClient:
    """Async version of :func:`weight_sync_and_get_sampling_client`."""
    return await asyncio.to_thread(
        weight_sync_and_get_sampling_client,
        service_client,
        training_client,
        name,
        tokenizer,
        checkpoint_type,
    )
