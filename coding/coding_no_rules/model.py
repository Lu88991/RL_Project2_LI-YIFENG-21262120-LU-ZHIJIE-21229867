"""TorchRL model stack: rewritten CNN dueling Q-network."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch import nn

try:
    from tensordict import TensorDict
    from tensordict.nn import TensorDictModule as Mod
    from tensordict.nn import TensorDictSequential as Seq
except ImportError as _e:
    raise ImportError(
        "未安装 tensordict（TorchRL 依赖）。请在项目目录执行：\n"
        "  pip install tensordict torchrl\n"
        "或一次性安装全部依赖：\n"
        "  pip install -r requirements.txt\n"
    ) from _e

from environment import AGENT, OPPONENT, _immediate_winning_actions, register_super_tic_tac_toe

OBS_DIM = 96
N_ACTIONS = 96
PROJ_H = 12
PROJ_W = 12

# Keep consistent with environment.py projection.
_BOARD_ORIGIN = {
    0: (0, 4),
    1: (4, 2),
    2: (4, 6),
    3: (8, 0),
    4: (8, 4),
    5: (8, 8),
}


def _build_proj_index() -> np.ndarray:
    idx = []
    for g in range(N_ACTIONS):
        b = g // 16
        loc = g % 16
        r, c = loc // 4, loc % 4
        br, bc = _BOARD_ORIGIN[b]
        idx.append((br + r) * PROJ_W + (bc + c))
    return np.asarray(idx, dtype=np.int64)


class ObsToBoardPlanes(nn.Module):
    """(B,96) -> (B,3,12,12): agent plane, opponent plane, valid-cell plane."""

    def __init__(self):
        super().__init__()
        proj_idx = _build_proj_index()
        self.register_buffer("proj_idx", torch.as_tensor(proj_idx, dtype=torch.long))
        valid = torch.zeros(PROJ_H * PROJ_W, dtype=torch.float32)
        valid[self.proj_idx] = 1.0
        self.register_buffer("valid_flat", valid)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        bsz = observation.shape[0]
        flat_agent = observation.new_zeros((bsz, PROJ_H * PROJ_W))
        flat_opp = observation.new_zeros((bsz, PROJ_H * PROJ_W))
        flat_agent.index_copy_(1, self.proj_idx, (observation > 0.5).to(observation.dtype))
        flat_opp.index_copy_(1, self.proj_idx, (observation < -0.5).to(observation.dtype))
        valid = self.valid_flat.unsqueeze(0).expand(bsz, -1)
        board = torch.stack([flat_agent, flat_opp, valid], dim=1)
        return board.view(bsz, 3, PROJ_H, PROJ_W)


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class DuelingConvQNet(nn.Module):
    """CNN backbone + dueling heads for Q-values(96)."""

    def __init__(self):
        super().__init__()
        self.encoder = ObsToBoardPlanes()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64),
            ResBlock(64),
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        feat_dim = 96 * PROJ_H * PROJ_W
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, N_ACTIONS),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x = self.encoder(observation)
        x = self.backbone(x)
        v = self.value_head(x)
        a = self.adv_head(x)
        return v + (a - a.mean(dim=-1, keepdim=True))


class LegalActionMaskFromObs(nn.Module):
    """Fast action mask for training: empty cell == legal action."""

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        return (observation.abs() < 1e-6).to(dtype=torch.bool)


class ApplyActionMask(nn.Module):
    """Set illegal action-values to a large negative number."""

    def forward(self, action_value: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        if action_mask.ndim == 1 and action_value.ndim == 2:
            action_mask = action_mask.unsqueeze(0).expand_as(action_value)
        masked = action_value.clone()
        masked[~action_mask] = -1e9
        return masked


def make_rl_env(
    device: Optional[torch.device] = None,
    opponent: str = "random",
    opponent_heuristic_prob: float = 0.7,
    opponent_policy=None,
):
    """Gymnasium 注册 + TorchRL GymEnv + StepCounter。

    opponent_policy 若提供，则环境内对手步完全由该可调用决定（签名 (obs_opp_96, mask_96)->int），
    此时 opponent 字符串仅作占位（仍传给 Gym 构造函数以满足注册签名）。
    """
    register_super_tic_tac_toe()
    from torchrl.envs import GymEnv, StepCounter, TransformedEnv

    kw: dict[str, Any] = dict(
        opponent=opponent,
        opponent_heuristic_prob=opponent_heuristic_prob,
    )
    if opponent_policy is not None:
        kw["opponent_policy"] = opponent_policy
    return TransformedEnv(
        GymEnv(
            "SuperTicTacToe-v0",
            device=device,
            **kw,
        ),
        StepCounter(),
    )


def build_policy_pair(
    env,
    eps_init: float,
    eps_end: float,
    annealing_num_steps: int,
) -> Tuple[Seq, Seq, Any]:
    from torchrl.modules import EGreedyModule, QValueModule

    value_net = Mod(DuelingConvQNet(), in_keys=["observation"], out_keys=["action_value"])
    mask_net = Mod(LegalActionMaskFromObs(), in_keys=["observation"], out_keys=["action_mask"])
    mask_q_net = Mod(ApplyActionMask(), in_keys=["action_value", "action_mask"], out_keys=["action_value"])
    policy = Seq(value_net, mask_net, mask_q_net, QValueModule(spec=env.action_spec))
    exploration = EGreedyModule(
        env.action_spec,
        eps_init=eps_init,
        eps_end=eps_end,
        annealing_num_steps=annealing_num_steps,
        action_mask_key="action_mask",
    )
    policy_explore = Seq(policy, exploration)
    return policy, policy_explore, exploration


def greedy_action_masked(
    policy: Seq,
    observation: np.ndarray,
    legal_mask: np.ndarray,
    device: torch.device,
    *,
    use_tactical: bool = False,
) -> int:
    """合法掩码上的贪心：默认仅 Q-argmax（方案 B，靠 RL+奖励学杀棋/堵棋）。

    use_tactical=True 时恢复旧行为：必胜步 > 必堵步 > Q-argmax。
    """
    obs = np.asarray(observation).reshape(-1)
    legal = np.asarray(legal_mask).reshape(-1) > 0.5
    if not use_tactical:
        with torch.inference_mode():
            td = TensorDict(
                {"observation": torch.as_tensor(obs, dtype=torch.float32, device=device)},
                batch_size=torch.Size([]),
            )
            out = policy(td)
            q = out["action_value"].detach().cpu().numpy().reshape(-1)
        q = q.copy()
        q[~legal] = -1e9
        return int(np.argmax(q))
    board = np.zeros(N_ACTIONS, dtype=np.int8)
    board[obs > 0.5] = AGENT
    board[obs < -0.5] = OPPONENT
    wins = _immediate_winning_actions(board, AGENT)
    if wins:
        valid = [a for a in wins if legal[a]]
        if valid:
            return int(valid[0])
    blocks = _immediate_winning_actions(board, OPPONENT)
    if blocks:
        valid = [a for a in blocks if legal[a]]
        if valid:
            return int(valid[0])

    with torch.inference_mode():
        td = TensorDict(
            {"observation": torch.as_tensor(obs, dtype=torch.float32, device=device)},
            batch_size=torch.Size([]),
        )
        out = policy(td)
        q = out["action_value"].detach().cpu().numpy().reshape(-1)
    q = q.copy()
    q[~legal] = -1e9
    return int(np.argmax(q))
