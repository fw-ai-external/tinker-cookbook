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
        "training_shape_id": None,
        "hot_load_timeout": 600,
    }
    kwargs.update(overrides)
    with patch("tinker_cookbook.fireworks_utils.FiretitanServiceClient") as service_cls:
        service_client = create_service_client_with_deployment(**kwargs)
    assert service_client is service_cls.from_firetitan_config.return_value
    return service_cls.from_firetitan_config.call_args.kwargs


def test_create_service_client_reuses_trainer_and_deployment():
    assert _create(user_metadata={"k": "v"}) == {
        "base_model": "accounts/fireworks/models/qwen3-8b",
        "lora_rank": 32,
        "training_shape_id": None,
        "trainer_job_id": "job-123",
        "deployment_id": "my-deployment",
        "hotload_timeout_s": 600,
        "reference_required": False,
        "cleanup_trainer_on_close": True,
        "cleanup_deployment_on_close": "scale_to_zero",
        "user_metadata": {"k": "v"},
    }


def test_create_service_client_creates_trainer_and_deployment():
    shape = "accounts/fireworks/trainingShapes/qwen3-8b-lora"
    config = _create(trainer_job_id=None, deployment_id=None, training_shape_id=shape)

    assert config["trainer_job_id"] is None
    assert config["deployment_id"] is None
    assert config["training_shape_id"] == shape


def test_create_service_client_can_keep_created_resources():
    config = _create(cleanup_on_exit=False)

    assert config["cleanup_trainer_on_close"] is False
    assert config["cleanup_deployment_on_close"] is None


def test_create_service_client_needs_shape_to_create_deployment():
    with pytest.raises(ConfigurationError, match="fireworks_training_shape_id must be set"):
        _create(deployment_id=None)
