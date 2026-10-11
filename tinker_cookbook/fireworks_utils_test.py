from unittest.mock import patch

import pytest

from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.fireworks_utils import (
    create_service_client_with_trainer_and_deployment,
    create_service_client_with_trainer_only,
    resolve_teacher_base_url,
    resolve_trainer_job_id,
)


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
        service_client = create_service_client_with_trainer_and_deployment(**kwargs)
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
        "user_metadata": {"k": "v"},
    }


def test_create_service_client_creates_trainer_and_deployment():
    shape = "accounts/fireworks/trainingShapes/qwen3-8b-lora"
    config = _create(trainer_job_id=None, deployment_id=None, training_shape_id=shape)

    assert config["trainer_job_id"] is None
    assert config["deployment_id"] is None
    assert config["training_shape_id"] == shape


def test_create_service_client_needs_shape_to_create_deployment():
    with pytest.raises(ConfigurationError, match="fireworks_training_shape_id must be set"):
        _create(deployment_id=None)


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


def test_resolve_teacher_base_url_keeps_a_base_url():
    with patch("tinker_cookbook.fireworks_utils.TrainerJobManager") as manager_cls:
        url = resolve_teacher_base_url(base_url=_TRAINER_URL, fireworks_trainer_job_id="job-123")

    assert url == _TRAINER_URL
    manager_cls.assert_not_called()


def test_resolve_teacher_base_url_is_none_without_a_trainer():
    assert resolve_teacher_base_url(base_url=None, fireworks_trainer_job_id=None) is None


def test_resolve_teacher_base_url_looks_up_a_job_id(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "key")
    monkeypatch.delenv("FIREWORKS_BASE_URL", raising=False)
    with patch("tinker_cookbook.fireworks_utils.TrainerJobManager") as manager_cls:
        manager_cls.return_value.wait_for_existing.return_value.base_url = _TRAINER_URL
        url = resolve_teacher_base_url(base_url=None, fireworks_trainer_job_id="job-123")

    assert url == _TRAINER_URL
    manager_cls.assert_called_once_with(api_key="key")
    manager_cls.return_value.wait_for_existing.assert_called_once_with("job-123")


def test_resolve_teacher_base_url_needs_an_api_key_to_look_up_a_job_id(monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="FIREWORKS_API_KEY must be set"):
        resolve_teacher_base_url(base_url=None, fireworks_trainer_job_id="job-123")


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
