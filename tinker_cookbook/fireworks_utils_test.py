import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.fireworks_utils import (
    create_service_client_with_deployment,
    save_weights_and_get_sampling_client,
)

TRAINER_URL = "https://api.fireworks.ai/training/v1/rlorTrainerJobs/my-account/job-123"


def _create(**overrides):
    kwargs = {
        "base_url": TRAINER_URL,
        "base_model": "accounts/fireworks/models/qwen3-8b",
        "lora_rank": 32,
        "deployment_id": "my-deployment",
        "hot_load_timeout": 600,
    }
    kwargs.update(overrides)
    return create_service_client_with_deployment(**kwargs)


def test_create_service_client_binds_trainer_and_deployment():
    with patch("tinker_cookbook.fireworks_utils.FiretitanServiceClient") as service_cls:
        service_client = _create(user_metadata={"k": "v"})

    assert service_client is service_cls.from_firetitan_config.return_value
    service_cls.from_firetitan_config.assert_called_once_with(
        base_url="https://api.fireworks.ai",
        base_model="accounts/fireworks/models/qwen3-8b",
        lora_rank=32,
        trainer_job_id="job-123",
        deployment_id="my-deployment",
        hotload_timeout_s=600,
        user_metadata={"k": "v"},
    )


@pytest.mark.parametrize("base_url", [None, "https://api.fireworks.ai"])
def test_create_service_client_requires_trainer_url(base_url):
    with pytest.raises(ConfigurationError, match="base_url must be a Fireworks trainer URL"):
        _create(base_url=base_url)


def test_create_service_client_requires_deployment_id():
    with pytest.raises(ConfigurationError, match="fireworks_deployment_id must be set"):
        _create(deployment_id=None)


def test_save_weights_and_get_sampling_client_hot_loads_saved_snapshot():
    save_future = MagicMock()
    save_future.result_async = AsyncMock(return_value=MagicMock(path="snapshot-path"))
    training_client = MagicMock()
    training_client.save_weights_for_sampler_async = AsyncMock(return_value=save_future)
    service_client = MagicMock()
    tokenizer = MagicMock()

    sampling_client, metrics = asyncio.run(
        save_weights_and_get_sampling_client(training_client, service_client, tokenizer, "step-3")
    )

    training_client.save_weights_for_sampler_async.assert_awaited_once_with("step-3")
    service_client.create_sampling_client.assert_called_once_with(
        model_path="snapshot-path", tokenizer=tokenizer
    )
    assert sampling_client is service_client.create_sampling_client.return_value
    assert set(metrics) == {"weight_sync/save_time_s", "weight_sync/hotload_time_s"}


def test_save_weights_and_get_sampling_client_rejects_missing_path():
    save_future = MagicMock()
    save_future.result_async = AsyncMock(return_value=MagicMock(path=None))
    training_client = MagicMock()
    training_client.save_weights_for_sampler_async = AsyncMock(return_value=save_future)
    service_client = MagicMock()

    with pytest.raises(RuntimeError, match="returned no path"):
        asyncio.run(
            save_weights_and_get_sampling_client(
                training_client, service_client, MagicMock(), "step-3"
            )
        )
    service_client.create_sampling_client.assert_not_called()
