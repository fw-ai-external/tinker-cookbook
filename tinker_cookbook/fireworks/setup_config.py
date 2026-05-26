from omegaconf import DictConfig

from tinker_cookbook.fireworks.utils.config import DeployConfig, InfraConfig


def to_infra_config(cfg_section: DictConfig) -> InfraConfig:
    """Convert an OmegaConf infra section to an ``InfraConfig`` dataclass."""
    return InfraConfig(
        training_shape_id=cfg_section.get("training_shape_id"),
        ref_training_shape_id=cfg_section.get("ref_training_shape_id"),
        region=cfg_section.get("region"),
        custom_image_tag=cfg_section.get("custom_image_tag"),
        accelerator_type=cfg_section.get("accelerator_type"),
        accelerator_count=cfg_section.get("accelerator_count"),
        node_count=cfg_section.get("node_count", 1),
        skip_validations=cfg_section.get("skip_validations", False),
        extra_args=list(cfg_section.get("extra_args") or []),
    )


def to_deploy_config(cfg_section: DictConfig) -> DeployConfig:
    """Convert an OmegaConf ``deployment`` section to a ``DeployConfig`` dataclass."""
    return DeployConfig(
        deployment_id=cfg_section.get("deployment_id"),
        deployment_shape=cfg_section.get("deployment_shape"),
        deployment_region=cfg_section.get("deployment_region"),
        replica_count=cfg_section.get("replica_count"),
        deployment_accelerator_type=cfg_section.get("deployment_accelerator_type"),
        hot_load_bucket_type=cfg_section.get("hot_load_bucket_type", "FW_HOSTED"),
        deployment_timeout_s=cfg_section.get("deployment_timeout_s", 5400),
        deployment_extra_args=list(cfg_section.get("deployment_extra_args") or []) or None,
        tokenizer_model=cfg_section.get("tokenizer_model"),
        sample_timeout=cfg_section.get("sample_timeout", 600),
        disable_speculative_decoding=cfg_section.get("disable_speculative_decoding", True),
        extra_values=dict(cfg_section.get("extra_values") or {}) or None,
    )
