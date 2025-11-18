from __future__ import annotations
from typing import TYPE_CHECKING, TypedDict

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class ObjectTerminationStage(TypedDict):
    step: int
    pos_threshold: float
    ori_threshold: float


def object_termination_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    stages: list[ObjectTerminationStage],
) -> dict[str, float]:
    del env_ids  # unused

    # Same TerminationTermCfg objects that were passed in the env cfg
    term_pos = env.termination_manager.cfg["object_pos"]
    term_ori = env.termination_manager.cfg["object_ori"]

    for stage in reversed(stages):
        if env.common_step_counter > stage["step"]:
            term_pos.params["threshold"] = stage["pos_threshold"]
            term_ori.params["threshold"] = stage["ori_threshold"]
            break

    return {
        "pos_threshold": term_pos.params["threshold"],
        "ori_threshold": term_ori.params["threshold"],
    }