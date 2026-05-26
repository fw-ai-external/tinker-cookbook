import logging
import os
from concurrent.futures import ThreadPoolExecutor

import hydra
from omegaconf import DictConfig

from fireworks.training.sdk import (
    TrainerJobManager,
    TrainerServiceEndpoint,
)
from tinker_cookbook.fireworks.setup_config import to_infra_config
from tinker_cookbook.fireworks.utils import ReconnectableClient, create_trainer_job

logger = logging.getLogger(__name__)


def init_fireworks_infra(cfg: DictConfig) -> TrainerServiceEndpoint:
    """Create Fireworks TrainerJobManager, DeploymentManager,
    ReconnectableClient, WeightSyncer, and sampling client.

    Expects a fully-resolved ``DictConfig`` matching the schema in
    ``fireworks_sft.yaml``.  Typically called from a ``@hydra.main`` entry point.
    """
    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = cfg.get("fireworks_base_url", "https://api.fireworks.ai")

    rlor_mgr = TrainerJobManager(api_key=api_key, base_url=base_url)

    infra = to_infra_config(cfg.training_infra)

    # Resolve training shape profile and auto-derive config values
    profile = None
    if infra.training_shape_id:
        profile = rlor_mgr.resolve_training_profile(infra.training_shape_id)
        if profile.max_supported_context_length and not cfg.training.get("max_length"):
            cfg.training.max_length = profile.max_supported_context_length
            logger.info("Auto-derived max_length from training shape: %d", cfg.training.max_length)


    with ThreadPoolExecutor(max_workers=2) as pool:
        pol_fut = pool.submit(
            create_trainer_job,
            rlor_mgr,
            base_model=cfg.model.name,
            infra=infra,
            profile=profile,
            lora_rank=cfg.model.get("lora_rank", 0),
            max_seq_len=cfg.training.max_length,
            learning_rate=cfg.training.learning_rate,
            display_name=cfg.get("train_display_name", "policy"),
            job_id=cfg.training_infra.training_job_id,
        )
        policy_ep = pol_fut.result()

    policy_rc = ReconnectableClient(
        rlor_mgr, policy_ep.job_id, cfg.model.name,
        lora_rank=cfg.model.get("lora_rank", 0),
    )

    return policy_ep


@hydra.main(config_path=".", config_name="fireworks_sft", version_base=None)
def main(cfg: DictConfig) -> None:
    policy_ep = init_fireworks_infra(cfg)
    logger.info("Fireworks policy endpoint ready (policy=%s)", policy_ep.base_url)


if __name__ == "__main__":
    main()
