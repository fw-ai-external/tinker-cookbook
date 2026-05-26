import logging
import os
from concurrent.futures import ThreadPoolExecutor

import hydra
import tinker
from omegaconf import DictConfig

from fireworks.training.sdk import (
    DeploymentManager,
    TrainerJobManager,
    TrainerServiceEndpoint,
    WeightSyncer,
)
from tinker_cookbook.fireworks.setup_config import to_deploy_config, to_infra_config
from tinker_cookbook.fireworks.utils import ReconnectableClient, create_trainer_job, setup_deployment

logger = logging.getLogger(__name__)


def init_fireworks_infra(
    cfg: DictConfig,
) -> tuple[TrainerServiceEndpoint, TrainerServiceEndpoint | None, tinker.SamplingClient, WeightSyncer]:
    """Create Fireworks TrainerJobManager, DeploymentManager,
    ReconnectableClient, WeightSyncer, and sampling client.

    Expects a fully-resolved ``DictConfig`` matching the schema in
    ``fireworks_rl.yaml``.  Typically called from a ``@hydra.main`` entry point.
    """
    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = cfg.get("fireworks_base_url", "https://api.fireworks.ai")

    rlor_mgr = TrainerJobManager(api_key=api_key, base_url=base_url)
    deploy_mgr = DeploymentManager(api_key=api_key, base_url=base_url)

    infra = to_infra_config(cfg.training_infra)
    reference_infra_cfg = cfg.get("reference_training_infra") or cfg.training_infra
    reference_infra = to_infra_config(reference_infra_cfg)
    deploy = to_deploy_config(cfg.deployment)

    # Resolve training shape profile and auto-derive config values
    profile = None
    if infra.training_shape_id:
        profile = rlor_mgr.resolve_training_profile(infra.training_shape_id)
        dep_shape = getattr(profile, "deployment_shape", None) or getattr(profile, "deployment_shape_version", None)
        if dep_shape and not deploy.deployment_shape:
            deploy.deployment_shape = dep_shape
            logger.info("Auto-derived deployment_shape from training shape: %s", dep_shape)
        if profile.max_supported_context_length and not cfg.training.get("max_length"):
            cfg.training.max_length = profile.max_supported_context_length
            logger.info("Auto-derived max_length from training shape: %d", cfg.training.max_length)

    dep_info = setup_deployment(deploy_mgr, deploy, cfg.model.name, infra)
    deployment_id = dep_info.deployment_id
    use_reference = cfg.algorithm.get("kl_beta", 0.0) > 0

    ref_profile = None
    if use_reference:
        if reference_infra.training_shape_id:
            ref_profile = rlor_mgr.resolve_training_profile(reference_infra.training_shape_id)
        elif infra.ref_training_shape_id:
            ref_profile = rlor_mgr.resolve_training_profile(infra.ref_training_shape_id)
        elif profile is not None:
            ref_profile = profile

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
            hot_load_deployment_id=deployment_id,
        )
        ref_fut = None
        if use_reference:
            ref_fut = pool.submit(
                create_trainer_job,
                rlor_mgr,
                base_model=cfg.model.name,
                infra=reference_infra,
                profile=ref_profile,
                lora_rank=cfg.model.get("lora_rank", 0),
                max_seq_len=cfg.training.max_length,
                learning_rate=cfg.training.learning_rate,
                display_name=cfg.get("ref_display_name", "reference"),
                job_id=reference_infra_cfg.get("training_job_id"),
                forward_only=True,
            )
        policy_ep = pol_fut.result()
        reference_ep = ref_fut.result() if ref_fut is not None else None

    # policy_job_id = policy_ep.job_id
    # reference_job_id = reference_ep.job_id if reference_ep else None

    policy_rc = ReconnectableClient(
        rlor_mgr, policy_ep.job_id, cfg.model.name,
        lora_rank=cfg.model.get("lora_rank", 0),
    )
    reference_rc = (
        ReconnectableClient(
            rlor_mgr, reference_ep.job_id, cfg.model.name,
            lora_rank=cfg.model.get("lora_rank", 0),
        )
        if reference_ep else None
    )

    weight_syncer = WeightSyncer(
        policy_client=policy_rc.inner,
        deploy_mgr=deploy_mgr,
        deployment_id=deployment_id,
        base_model=cfg.model.name,
        hotload_timeout=cfg.hotload.hot_load_timeout,
    )
    sampling_client = weight_syncer.get_sampling_client()

    return policy_ep, reference_ep, sampling_client, weight_syncer


@hydra.main(config_path=".", config_name="fireworks_rl", version_base=None)
def main(cfg: DictConfig) -> None:
    policy_ep, reference_ep, sampling_client, weight_syncer = init_fireworks_infra(cfg)
    logger.info("Fireworks policy endpoint ready (policy=%s)", policy_ep.base_url)

    logger.info("Fireworks reference endpoint ready (reference=%s)", reference_ep.base_url if reference_ep else None)
    sampling_model = getattr(getattr(sampling_client, "deployment_sampler", None), "model", None)
    logger.info("Fireworks sampling client ready (sampling_client=%s)", sampling_model)
    # logger.info("Fireworks weight syncer ready (weight_syncer=%s)", weight_syncer.base_url)


if __name__ == "__main__":
    main()
