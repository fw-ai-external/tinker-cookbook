"""Helpers for training loops that run on Fireworks trainers and deployments."""

import os
import re

from fireworks.training.sdk import FiretitanServiceClient, FireworksClient

from tinker_cookbook.exceptions import ConfigurationError


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
