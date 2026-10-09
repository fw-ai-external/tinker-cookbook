from unittest.mock import patch

import pytest

from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.fireworks_utils import create_service_client_with_deployment


def _create(**overrides):
    kwargs = {
        "trainer_job_id": "job-123",
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
        base_model="accounts/fireworks/models/qwen3-8b",
        lora_rank=32,
        trainer_job_id="job-123",
        deployment_id="my-deployment",
        hotload_timeout_s=600,
        user_metadata={"k": "v"},
    )


def test_create_service_client_requires_trainer_job_id():
    with pytest.raises(ConfigurationError, match="A trainer job ID is required"):
        _create(trainer_job_id=None)


def test_create_service_client_requires_deployment_id():
    with pytest.raises(ConfigurationError, match="fireworks_deployment_id must be set"):
        _create(deployment_id=None)
