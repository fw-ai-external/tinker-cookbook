import logging
import os
from concurrent.futures import ThreadPoolExecutor

import hydra
from fireworks.training.sdk import (
    DeploymentManager,
    TrainerJobManager,
    TrainerServiceEndpoint,
)
from omegaconf import DictConfig

from tinker_cookbook.fireworks.setup_config import to_deploy_config, to_infra_config
from tinker_cookbook.fireworks.utils import create_trainer_job, setup_deployment

logger = logging.getLogger(__name__)


def _get_teacher_model(cfg: DictConfig) -> str:
    """Resolve the teacher Fireworks model from optional distillation config."""
    teacher_cfg = cfg.get("teacher")
    if teacher_cfg is not None and teacher_cfg.get("name"):
        return teacher_cfg.get("name")
    teacher_model = cfg.get("teacher_model") or cfg.model.name
    if teacher_model == cfg.model.name:
        logger.warning(
            "No teacher.name or teacher_model configured; using the student model as teacher."
        )
    return teacher_model


def init_fireworks_infra(
    cfg: DictConfig,
) -> tuple[TrainerServiceEndpoint, TrainerServiceEndpoint, str, str, str]:
    """Create student trainer, teacher trainer, and student sampling deployment.

    The returned endpoints map directly to
    ``tinker_cookbook.recipes.distillation.on_policy_distillation`` CLI fields:
    ``base_url`` for the student and ``teacher_base_url`` for the teacher.
    """
    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = cfg.get("fireworks_base_url", "https://api.fireworks.ai")

    trainer_mgr = TrainerJobManager(api_key=api_key, base_url=base_url)
    deploy_mgr = DeploymentManager(api_key=api_key, base_url=base_url)

    student_model = cfg.model.name
    student_lora_rank = cfg.model.get("lora_rank", 0)
    teacher_model = _get_teacher_model(cfg)

    student_infra = to_infra_config(cfg.training_infra)
    teacher_infra_cfg = (
        cfg.get("teacher_training_infra")
        or cfg.get("reference_training_infra")
        or cfg.training_infra
    )
    teacher_infra = to_infra_config(teacher_infra_cfg)
    deploy = to_deploy_config(cfg.deployment)

    student_profile = None
    if student_infra.training_shape_id:
        student_profile = trainer_mgr.resolve_training_profile(student_infra.training_shape_id)
        dep_shape = getattr(student_profile, "deployment_shape", None) or getattr(
            student_profile, "deployment_shape_version", None
        )
        if dep_shape and not deploy.deployment_shape:
            deploy.deployment_shape = dep_shape
            logger.info("Auto-derived deployment_shape from student training shape: %s", dep_shape)
        if student_profile.max_supported_context_length and not cfg.training.get("max_length"):
            cfg.training.max_length = student_profile.max_supported_context_length
            logger.info(
                "Auto-derived max_length from student training shape: %d", cfg.training.max_length
            )

    teacher_profile = None
    if teacher_infra.training_shape_id:
        teacher_profile = trainer_mgr.resolve_training_profile(teacher_infra.training_shape_id)

    dep_info = setup_deployment(deploy_mgr, deploy, student_model, student_infra)
    deployment_id = dep_info.deployment_id

    student_display_name = cfg.get("train_display_name") or cfg.get("display_name", "student")
    teacher_display_name = cfg.get("teacher_display_name") or cfg.get("ref_display_name", "teacher")

    with ThreadPoolExecutor(max_workers=2) as pool:
        student_fut = pool.submit(
            create_trainer_job,
            trainer_mgr,
            base_model=student_model,
            infra=student_infra,
            profile=student_profile,
            lora_rank=student_lora_rank,
            max_seq_len=cfg.training.max_length,
            learning_rate=cfg.training.learning_rate,
            display_name=student_display_name,
            job_id=cfg.training_infra.training_job_id,
            hot_load_deployment_id=deployment_id,
        )
        teacher_fut = pool.submit(
            create_trainer_job,
            trainer_mgr,
            base_model=teacher_model,
            infra=teacher_infra,
            profile=teacher_profile,
            lora_rank=0,
            max_seq_len=cfg.training.max_length,
            learning_rate=cfg.training.learning_rate,
            display_name=teacher_display_name,
            job_id=teacher_infra_cfg.get("training_job_id"),
            forward_only=True,
        )
        student_ep = student_fut.result()
        teacher_ep = teacher_fut.result()

    return student_ep, teacher_ep, deployment_id, student_model, teacher_model


@hydra.main(config_path=".", config_name="fireworks_distillation", version_base=None)
def main(cfg: DictConfig) -> None:
    student_ep, teacher_ep, deployment_id, student_model, teacher_model = init_fireworks_infra(cfg)
    logger.info("Fireworks student endpoint ready (base_url=%s)", student_ep.base_url)
    logger.info("Fireworks teacher endpoint ready (teacher_base_url=%s)", teacher_ep.base_url)
    logger.info("Fireworks student deployment ready (fireworks_deployment_id=%s)", deployment_id)
    logger.info("Use these on_policy_distillation overrides:")
    logger.info("  base_url=%s", student_ep.base_url)
    logger.info("  lora_rank=%s", cfg.model.get("lora_rank", 0))
    logger.info("  fireworks_base_model=%s", student_model)
    logger.info("  fireworks_deployment_id=%s", deployment_id)
    logger.info("  teacher_base_url=%s", teacher_ep.base_url)
    logger.info("  teacher_fireworks_base_model=%s", teacher_model)


if __name__ == "__main__":
    main()
