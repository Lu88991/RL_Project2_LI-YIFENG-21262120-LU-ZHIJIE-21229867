"""训练对手配置：与 main / train 解耦，便于单独选择对手与参数。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class TrainOpponentConfig:
    """在 main.py 里只改这个结构体即可切换训练对手（无需手写一长串 CLI）。"""

    # random | heuristic | mixed | self_snapshot | curriculum
    mode: str = "curriculum"
    # --- curriculum：先打随机若干局，再切换到 after_curriculum_opponent ---
    curriculum_switch_episodes: int = 300
    after_curriculum_opponent: str = "mixed"  # random | heuristic | mixed | self_snapshot
    # --- mixed / curriculum 后半段若为 mixed 时生效 ---
    heuristic_prob: float = 0.7
    # --- self_snapshot：对手为「上一快照」的自己的策略（定期从当前 policy 同步）---
    self_play_sync_iters: int = 10
    # 快照对手是否使用必胜/必堵战术层；False=纯 Q 贪心（与默认 greedy_action_masked 方案 B 一致）
    self_play_opponent_tactical: bool = False


class SelfPlayOpponentBridge:
    """可交给 Gym 的 opponent_policy：在对手视角 obs 上调用冻结策略。"""

    def __init__(self, policy_module, device, *, use_tactical: bool) -> None:
        self.policy_module = policy_module
        self.device = device
        self.use_tactical = use_tactical

    def __call__(self, obs_opp: object, mask: object) -> int:
        from model import greedy_action_masked

        return greedy_action_masked(
            self.policy_module,
            obs_opp,
            mask,
            self.device,
            use_tactical=self.use_tactical,
        )


def build_train_opponent_argv(cfg: TrainOpponentConfig) -> List[str]:
    """生成传给 train.py 的 argparse 片段（放在其它 TRAIN_EXTRA_ARGS 之前，后者可覆盖同名项）。"""
    m = cfg.mode.strip().lower()
    if m not in ("random", "heuristic", "mixed", "self_snapshot", "curriculum"):
        raise ValueError(
            f"未知 TrainOpponentConfig.mode={cfg.mode!r}；"
            "可选：random / heuristic / mixed / self_snapshot / curriculum"
        )

    if m == "curriculum":
        ac = cfg.after_curriculum_opponent.strip().lower()
        if ac not in ("random", "heuristic", "mixed", "self_snapshot"):
            raise ValueError(
                f"after_curriculum_opponent={cfg.after_curriculum_opponent!r} 非法；"
                "可选：random / heuristic / mixed / self_snapshot"
            )
        if cfg.curriculum_switch_episodes <= 0:
            raise ValueError("curriculum 模式下 curriculum_switch_episodes 必须 > 0")
        out = [
            "--curriculum-switch-episodes",
            str(int(cfg.curriculum_switch_episodes)),
            "--opponent",
            ac,
            "--opponent-heuristic-prob",
            str(float(cfg.heuristic_prob)),
        ]
        if ac == "self_snapshot":
            out += [
                "--self-play-sync-iters",
                str(int(cfg.self_play_sync_iters)),
            ]
            if not cfg.self_play_opponent_tactical:
                out.append("--self-play-opponent-q-only")
        return out

    # 非课程：全程单一对手
    out = [
        "--curriculum-switch-episodes",
        "0",
        "--opponent",
        m,
        "--opponent-heuristic-prob",
        str(float(cfg.heuristic_prob)),
    ]
    if m == "self_snapshot":
        out += [
            "--self-play-sync-iters",
            str(int(cfg.self_play_sync_iters)),
        ]
        if not cfg.self_play_opponent_tactical:
            out.append("--self-play-opponent-q-only")
    return out
