from unittest.mock import patch

import pytest

from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.fireworks_utils import (
    create_service_client_with_trainer_only,
    resolve_trainer_base_url,
    resolve_trainer_job_id,
)


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
