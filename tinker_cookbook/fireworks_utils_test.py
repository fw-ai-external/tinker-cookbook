import asyncio
from unittest.mock import MagicMock, patch

import pytest

from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.fireworks_utils import (
    create_base_model_sampling_client,
    create_service_client_with_trainer_and_deployment,
    create_service_client_with_trainer_only,
    resolve_trainer_base_url,
    resolve_trainer_job_id,
    weight_sync_and_get_sampling_client,
    weight_sync_and_get_sampling_client_async,
)

_SHAPE = "accounts/fireworks/trainingShapes/qwen3-8b-lora"


def _create(**overrides):
    kwargs = {
        "trainer_job_id": "job-123",
        "base_model": "accounts/fireworks/models/qwen3-8b",
        "lora_rank": 32,
        "deployment_id": "my-deployment",
        "training_shape_id": _SHAPE,
        "hot_load_timeout": 600,
    }
    kwargs.update(overrides)
    with patch("tinker_cookbook.fireworks_utils.FiretitanServiceClient") as service_cls:
        service_client = create_service_client_with_trainer_and_deployment(**kwargs)
    assert service_client is service_cls.from_firetitan_config.return_value
    return service_cls.from_firetitan_config.call_args.kwargs


def test_create_service_client_reuses_trainer_and_deployment():
    assert _create(user_metadata={"k": "v"}) == {
        "base_model": "accounts/fireworks/models/qwen3-8b",
        "lora_rank": 32,
        "training_shape_id": _SHAPE,
        "trainer_job_id": "job-123",
        "deployment_id": "my-deployment",
        "hotload_timeout_s": 600,
        "reference_required": False,
        "user_metadata": {"k": "v"},
    }


def test_create_service_client_creates_trainer_and_deployment():
    shape = "accounts/fireworks/trainingShapes/qwen3-8b-lora"
    config = _create(trainer_job_id=None, deployment_id=None, training_shape_id=shape)

    assert config["trainer_job_id"] is None
    assert config["deployment_id"] is None
    assert config["training_shape_id"] == shape


def test_create_service_client_needs_a_training_shape():
    with pytest.raises(ConfigurationError, match="fireworks_training_shape_id must be set"):
        _create(training_shape_id=None)


def test_create_service_client_with_trainer_only_skips_the_deployment():
    with patch("tinker_cookbook.fireworks_utils.FiretitanServiceClient") as service_cls:
        service_client = create_service_client_with_trainer_only(
            trainer_job_id=None,
            base_model="accounts/fireworks/models/qwen3-8b",
            lora_rank=32,
            training_shape_id="accounts/fireworks/trainingShapes/qwen3-8b-lora",
        )

    assert service_client is service_cls.from_firetitan_config.return_value
    assert service_cls.from_firetitan_config.call_args.kwargs == {
        "base_model": "accounts/fireworks/models/qwen3-8b",
        "lora_rank": 32,
        "training_shape_id": "accounts/fireworks/trainingShapes/qwen3-8b-lora",
        "trainer_job_id": None,
        "create_deployment": False,
        "forward_only": False,
        "reference_required": False,
        "user_metadata": None,
    }


_TRAINER_URL = "https://api.fireworks.ai/training/v1/rlorTrainerJobs/my-team/job-123"


@pytest.mark.parametrize(
    ("base_url", "job_id", "expected"),
    [
        (None, None, None),
        (None, "job-123", "job-123"),
        (_TRAINER_URL, None, "job-123"),
        (_TRAINER_URL, "job-123", "job-123"),
        ("", "job-123", "job-123"),
        (_TRAINER_URL, "", "job-123"),
    ],
)
def test_resolve_trainer_job_id(base_url, job_id, expected):
    assert resolve_trainer_job_id(base_url=base_url, fireworks_trainer_job_id=job_id) == expected


@pytest.mark.parametrize("base_url", [_TRAINER_URL, "https://api.fireworks.ai/training/v1"])
def test_resolve_trainer_job_id_rejects_a_base_url_for_another_trainer(base_url):
    with pytest.raises(ConfigurationError, match="name different trainers"):
        resolve_trainer_job_id(base_url=base_url, fireworks_trainer_job_id="job-456")


@pytest.mark.parametrize(
    "base_url",
    [
        _TRAINER_URL + "/",
        "https://api.fireworks.ai/training/v1/rlorTrainerJobs/my-team/job-123/api/v1",
    ],
)
def test_resolve_trainer_job_id_reads_the_job_id_from_a_trainer_url(base_url):
    assert resolve_trainer_job_id(base_url=base_url, fireworks_trainer_job_id=None) == "job-123"


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.fireworks.ai/training/v1/rlorTrainerJobs/job-123",
        "https://api.fireworks.ai/training/v1/serverless",
    ],
)
def test_resolve_trainer_job_id_is_none_for_a_url_without_a_trainer_job(base_url):
    assert resolve_trainer_job_id(base_url=base_url, fireworks_trainer_job_id=None) is None


def test_resolve_base_url_keeps_a_base_url():
    with patch("tinker_cookbook.fireworks_utils.FireworksClient") as client_cls:
        url = resolve_trainer_base_url(base_url=_TRAINER_URL, fireworks_trainer_job_id="job-123")

    assert url == _TRAINER_URL
    client_cls.assert_not_called()


def test_resolve_base_url_is_none_without_a_trainer():
    assert resolve_trainer_base_url(base_url=None, fireworks_trainer_job_id=None) is None


def test_resolve_base_url_builds_the_url_for_a_job_id(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "key")
    monkeypatch.delenv("FIREWORKS_BASE_URL", raising=False)
    with patch("tinker_cookbook.fireworks_utils.FireworksClient") as client_cls:
        client_cls.return_value.account_id = "my-team"
        url = resolve_trainer_base_url(base_url=None, fireworks_trainer_job_id="job-123")

    assert url == _TRAINER_URL
    client_cls.assert_called_once_with(api_key="key", base_url="https://api.fireworks.ai")


_SERVERLESS_URL = "https://api.fireworks.ai/training/v1/serverless"


def test_create_service_client_with_trainer_only_attaches_to_serverless():
    with patch("tinker_cookbook.fireworks_utils.FiretitanServiceClient") as service_cls:
        service_client = create_service_client_with_trainer_only(
            trainer_job_id=None,
            base_model="accounts/fireworks/models/qwen3p8-27b",
            lora_rank=32,
            training_shape_id=None,
            base_url=_SERVERLESS_URL + "/",
            user_metadata={"k": "v"},
        )

    assert service_client is service_cls.return_value
    service_cls.assert_called_once_with(base_url=_SERVERLESS_URL + "/", user_metadata={"k": "v"})
    service_cls.from_firetitan_config.assert_not_called()


def test_serverless_base_url_cannot_be_combined_with_a_trainer_job_id():
    with pytest.raises(ConfigurationError, match="name different trainers"):
        resolve_trainer_job_id(base_url=_SERVERLESS_URL, fireworks_trainer_job_id="job-123")


def test_serverless_weight_sync_samples_from_the_saved_snapshot():
    training_client = MagicMock()
    training_client.save_weights_for_sampler.return_value.result.return_value.path = "snap-1"
    service_client = MagicMock(training_session_id="ts-abc123")
    tokenizer = MagicMock()

    sampling_client = weight_sync_and_get_sampling_client(
        service_client, training_client, "step-0", tokenizer, checkpoint_type="base"
    )

    training_client.save_weights_for_sampler.assert_called_once_with("step-0")
    service_client.hotload_sampler_snapshot.assert_not_called()
    service_client.create_sampling_client.assert_called_once_with(
        model_path="snap-1", tokenizer=tokenizer
    )
    assert sampling_client is service_client.create_sampling_client.return_value


def test_dedicated_weight_sync_hot_loads_and_samples_from_the_deployment():
    training_client = MagicMock(supports_rdma_weight_sync=False)
    training_client.save_weights_for_sampler.return_value.result.return_value.path = "snap-1"
    service_client = MagicMock(training_session_id=None)
    tokenizer = MagicMock()

    weight_sync_and_get_sampling_client(
        service_client, training_client, "step-0", tokenizer, checkpoint_type="base"
    )

    training_client.save_weights_for_sampler.assert_called_once_with(
        "step-0", ttl_seconds=None, checkpoint_type="base"
    )
    service_client.hotload_sampler_snapshot.assert_called_once_with("snap-1")
    service_client.create_sampling_client.assert_called_once_with(tokenizer=tokenizer)


def test_dedicated_weight_sync_uses_the_trainer_sync_when_supported():
    training_client = MagicMock(supports_rdma_weight_sync=True)
    service_client = MagicMock(training_session_id=None)
    tokenizer = MagicMock()

    weight_sync_and_get_sampling_client(service_client, training_client, "step-3", tokenizer)

    training_client.weight_sync.return_value.result.assert_called_once()
    training_client.save_weights_for_sampler.assert_not_called()
    service_client.hotload_sampler_snapshot.assert_not_called()
    service_client.create_sampling_client.assert_called_once_with(tokenizer=tokenizer)


def test_async_weight_sync_matches_the_sync_one():
    training_client = MagicMock(supports_rdma_weight_sync=False)
    training_client.save_weights_for_sampler.return_value.result.return_value.path = "snap-2"
    service_client = MagicMock(training_session_id=None)
    tokenizer = MagicMock()

    sampling_client = asyncio.run(
        weight_sync_and_get_sampling_client_async(
            service_client, training_client, "step-1", tokenizer
        )
    )

    training_client.save_weights_for_sampler.assert_called_once_with(
        "step-1", ttl_seconds=None, checkpoint_type=None
    )
    service_client.hotload_sampler_snapshot.assert_called_once_with("snap-2")
    assert sampling_client is service_client.create_sampling_client.return_value


def test_create_service_client_with_trainer_and_deployment_attaches_to_serverless():
    with patch("tinker_cookbook.fireworks_utils.FiretitanServiceClient") as service_cls:
        service_client = create_service_client_with_trainer_and_deployment(
            trainer_job_id=None,
            base_model="accounts/fireworks/models/qwen3p8-27b",
            lora_rank=32,
            deployment_id=None,
            training_shape_id=None,
            hot_load_timeout=600,
            base_url=_SERVERLESS_URL,
        )

    assert service_client is service_cls.return_value
    service_cls.assert_called_once_with(base_url=_SERVERLESS_URL, user_metadata=None)
    service_cls.from_firetitan_config.assert_not_called()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"deployment_id": "my-deployment"}, "fireworks_deployment_id cannot be used"),
        ({"lora_rank": 0}, "requires lora_rank > 0"),
    ],
)
def test_serverless_rejects_a_deployment_and_full_parameter_training(overrides, message):
    with pytest.raises(ConfigurationError, match=message):
        _create(
            **{
                "trainer_job_id": None,
                "deployment_id": None,
                "base_url": _SERVERLESS_URL,
                **overrides,
            }
        )


def test_base_model_sampling_client_samples_an_untrained_adapter_on_serverless():
    tokenizer = MagicMock()
    with patch("tinker_cookbook.fireworks_utils.FiretitanServiceClient") as service_cls:
        service_client = service_cls.return_value
        training_client = service_client.create_training_client.return_value
        training_client.save_weights_for_sampler.return_value.result.return_value.path = "snap-0"
        sampling_client = create_base_model_sampling_client(
            _SERVERLESS_URL, "accounts/fireworks/models/qwen3p8-27b", tokenizer
        )

    service_cls.assert_called_once_with(base_url=_SERVERLESS_URL)
    service_client.create_training_client.assert_called_once_with(
        base_model="accounts/fireworks/models/qwen3p8-27b", lora_rank=32
    )
    training_client.optim_step.assert_not_called()
    service_client.create_sampling_client.assert_called_once_with(
        model_path="snap-0", tokenizer=tokenizer
    )
    assert sampling_client is service_client.create_sampling_client.return_value


@pytest.mark.parametrize(
    "base_url", [None, "https://api.fireworks.ai/training/v1/rlorTrainerJobs/a/j"]
)
def test_base_model_sampling_client_needs_serverless_training(base_url):
    with (
        patch("tinker_cookbook.fireworks_utils.FiretitanServiceClient") as service_cls,
        pytest.raises(ConfigurationError, match="serverless training only"),
    ):
        create_base_model_sampling_client(base_url, "accounts/fireworks/models/m", MagicMock())
    service_cls.assert_not_called()
