from unittest.mock import MagicMock, patch

import pytest

from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.fireworks_utils import (
    create_service_client_with_deployment,
    make_weight_sync,
)


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


def test_weight_sync_saves_and_hot_loads_snapshot():
    training_client = MagicMock(supports_rdma_weight_sync=False)
    training_client.save_weights_for_sampler.return_value.result.return_value.path = "snapshot"
    service_client = MagicMock()
    tokenizer = MagicMock()

    publish_weights = make_weight_sync(training_client, service_client, tokenizer)
    sampling_client = publish_weights("step-3", checkpoint_type="base")

    training_client.save_weights_for_sampler.assert_called_once_with(
        "step-3", checkpoint_type="base"
    )
    service_client.hotload_sampler_snapshot.assert_called_once_with("snapshot")
    service_client.create_sampling_client.assert_called_once_with(tokenizer=tokenizer)
    assert sampling_client is service_client.create_sampling_client.return_value


def test_weight_sync_uses_rdma_when_supported():
    training_client = MagicMock(supports_rdma_weight_sync=True)
    service_client = MagicMock()
    tokenizer = MagicMock()

    publish_weights = make_weight_sync(training_client, service_client, tokenizer)
    sampling_client = publish_weights("step-3", checkpoint_type="base")

    training_client.weight_sync.return_value.result.assert_called_once_with()
    training_client.save_weights_for_sampler.assert_not_called()
    service_client.hotload_sampler_snapshot.assert_not_called()
    assert sampling_client is service_client.create_sampling_client.return_value
