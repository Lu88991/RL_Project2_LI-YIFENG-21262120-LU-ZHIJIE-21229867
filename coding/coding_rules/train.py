#!/usr/bin/env python3
"""使用 TorchRL 训练 DQN：DQNLoss、ReplayBuffer、env.rollout（非手写 PyTorch 循环）。"""

from __future__ import annotations

import argparse
import os
import random
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam

from environment import SuperTicTacToeEnv
from model import build_policy_pair, greedy_action_masked, make_rl_env
from training_opponent import SelfPlayOpponentBridge


def _episode_returns_from_rollout(roll) -> list[float]:
    """从一次 rollout 的 tensordict 中拆出各局回报。"""
    rewards = roll["next", "reward"].squeeze(-1)
    dones = roll["next", "done"].squeeze(-1)
    out: list[float] = []
    ep = 0.0
    for t in range(rewards.shape[0]):
        ep += float(rewards[t].item())
        if bool(dones[t].item()):
            out.append(ep)
            ep = 0.0
    return out


def _safe_torch_save(obj, path: str) -> None:
    """
    Atomic save: write temp file in target directory then replace.
    This avoids partial/corrupted checkpoints on interruption and provides
    clearer error context if filesystem issues occur.
    """
    abs_path = os.path.abspath(path)
    parent = os.path.dirname(abs_path) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_ckpt_", suffix=".pt", dir=parent)
    os.close(fd)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, abs_path)
    except Exception as e:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise RuntimeError(
            "模型保存失败。\n"
            f"  target={abs_path}\n"
            f"  cwd={os.getcwd()}\n"
            f"  error={e!r}"
        ) from e


def _normalize_td_action_shapes(td):
    """TorchRL Gym wrapper may keep an extra env-dim (..,1,96) for action-like keys."""
    for key in ("action", "action_value", "chosen_action_value"):
        try:
            v = td.get(key)
        except Exception:
            continue
        if hasattr(v, "ndim") and v.ndim >= 2 and v.shape[-2] == 1:
            td.set(key, v.squeeze(-2))
    return td


def _evaluate_win_rate(
    policy,
    device: torch.device,
    episodes: int,
    opponent: str,
    opponent_heuristic_prob: float = 0.7,
) -> tuple[float, float]:
    policy.eval()
    wins = draws = 0
    for ep in range(episodes):
        env = SuperTicTacToeEnv(
            seed=ep,
            opponent=opponent,
            opponent_heuristic_prob=opponent_heuristic_prob,
        )
        obs, _ = env.reset(seed=ep)
        while True:
            mask = env.legal_actions_mask()
            a = greedy_action_masked(policy, obs, mask, device)
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                if info.get("winner") == "agent" or r > 0.5:
                    wins += 1
                elif info.get("draw") or abs(r) <= 0.2:
                    draws += 1
                break
    policy.train()
    return wins / max(episodes, 1), draws / max(episodes, 1)


def train_dqn(args: argparse.Namespace) -> None:
    try:
        from torchrl.data import LazyTensorStorage, ReplayBuffer
        from torchrl.objectives import DQNLoss, SoftUpdate
        from torchrl.objectives.utils import ValueEstimators
    except ImportError as e:
        raise ImportError(
            "请安装 TorchRL：pip install torchrl tensordict\n" + str(e)
        ) from e

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    # 初始环境：课程学习前半段固定 random；全程 self_snapshot 时先用 random 占位以构建 policy，再在下方换入带 bridge 的环境。
    start_opp = (
        "random"
        if args.curriculum_switch_episodes > 0
        else ("random" if args.opponent == "self_snapshot" else args.opponent)
    )
    env = make_rl_env(
        device=device,
        opponent=start_opp,
        opponent_heuristic_prob=args.opponent_heuristic_prob,
    )
    env.set_seed(args.seed)

    policy, policy_explore, exploration = build_policy_pair(
        env,
        eps_init=args.eps_start,
        eps_end=args.eps_end,
        annealing_num_steps=args.eps_anneal_steps,
    )
    policy.to(device)
    policy_explore.to(device)

    snapshot_policy = None
    if args.curriculum_switch_episodes <= 0 and args.opponent == "self_snapshot":
        snapshot_policy, _, _ = build_policy_pair(
            env,
            eps_init=1.0,
            eps_end=1.0,
            annealing_num_steps=1,
        )
        snapshot_policy.to(device)
        snapshot_policy.load_state_dict(policy.state_dict())
        snapshot_policy.eval()
        tact = not bool(getattr(args, "self_play_opponent_q_only", False))
        bridge = SelfPlayOpponentBridge(snapshot_policy, device, use_tactical=tact)
        env = make_rl_env(
            device=device,
            opponent="random",
            opponent_heuristic_prob=args.opponent_heuristic_prob,
            opponent_policy=bridge,
        )
        env.set_seed(args.seed)
        print(
            "[self-play] 全程对手=self_snapshot（定期把当前 policy 同步到快照作为对手）",
            flush=True,
        )

    loss_module = DQNLoss(
        value_network=policy,
        action_space=env.action_spec,
        delay_value=True,
        double_dqn=True,
        loss_function="smooth_l1",
    ).to(device)
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=args.gamma)
    optim = Adam(loss_module.parameters(), lr=args.lr)
    updater = SoftUpdate(loss_module, tau=args.tau)

    rb = ReplayBuffer(storage=LazyTensorStorage(args.buffer))

    returns: list[float] = []
    total_episodes = 0
    best_mean = -1e9
    it = 0
    switched_to_target_opponent = args.curriculum_switch_episodes <= 0
    train_updates = 0

    while total_episodes < args.episodes:
        it += 1
        env.set_seed(args.seed + it)
        roll = env.rollout(
            max_steps=args.rollout_steps,
            policy=policy_explore,
            auto_reset=True,
        )
        roll = _normalize_td_action_shapes(roll)
        rb.extend(roll)
        ep_rets = _episode_returns_from_rollout(roll)
        returns.extend(ep_rets)
        total_episodes += len(ep_rets)

        if (
            not switched_to_target_opponent
            and total_episodes >= args.curriculum_switch_episodes
        ):
            # Curriculum：前半 random；后半切到 args.opponent（可为 mixed / heuristic / self_snapshot 等）
            if args.opponent == "self_snapshot":
                snapshot_policy, _, _ = build_policy_pair(
                    env,
                    eps_init=1.0,
                    eps_end=1.0,
                    annealing_num_steps=1,
                )
                snapshot_policy.to(device)
                snapshot_policy.load_state_dict(policy.state_dict())
                snapshot_policy.eval()
                tact = not bool(getattr(args, "self_play_opponent_q_only", False))
                bridge = SelfPlayOpponentBridge(snapshot_policy, device, use_tactical=tact)
                env = make_rl_env(
                    device=device,
                    opponent="random",
                    opponent_heuristic_prob=args.opponent_heuristic_prob,
                    opponent_policy=bridge,
                )
            else:
                env = make_rl_env(
                    device=device,
                    opponent=args.opponent,
                    opponent_heuristic_prob=args.opponent_heuristic_prob,
                )
            env.set_seed(args.seed + it)
            switched_to_target_opponent = True
            print(
                f"[curriculum] switch opponent -> {args.opponent} at episodes {total_episodes}",
                flush=True,
            )

        if len(rb) >= args.min_buffer:
            for _ in range(args.optim_steps):
                sample = rb.sample(args.batch)
                sample = _normalize_td_action_shapes(sample)
                if device.type == "cuda":
                    sample = sample.to(device)
                loss_vals = loss_module(sample)
                loss_vals["loss"].backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 10.0)
                optim.step()
                optim.zero_grad()
                train_updates += 1
            exploration.step(int(roll.numel()))
            updater.step()
            if snapshot_policy is not None and train_updates > 0:
                se = int(args.self_play_sync_iters)
                # se<=0：每次发生更新后都同步快照（最强「上一时刻自己」）
                sync_now = (se <= 0) or (it % se == 0)
                if sync_now:
                    snapshot_policy.load_state_dict(policy.state_dict())
                    snapshot_policy.eval()
                    if it <= 3 or it % max(args.log_every, 1) == 0:
                        print(f"[self-play] snapshot <- policy (iter {it})", flush=True)

        if it % args.log_every == 0 and returns:
            w = min(50, len(returns))
            m = float(np.mean(returns[-w:]))
            eps = float(exploration.eps)
            print(
                f"iter {it}  episodes_total {total_episodes}  "
                f"mean_return_{w} {m:.3f}  eps {eps:.3f}  buffer {len(rb)}"
            )
            if args.eval_every > 0 and it % args.eval_every == 0:
                wr_r, dr_r = _evaluate_win_rate(policy, device, args.eval_episodes, opponent="random")
                wr_h, dr_h = _evaluate_win_rate(
                    policy,
                    device,
                    max(10, args.eval_episodes // 2),
                    opponent="heuristic",
                )
                print(
                    f"[eval] win_rate_random={wr_r:.3f} draw_random={dr_r:.3f} "
                    f"win_rate_heuristic={wr_h:.3f} draw_heuristic={dr_h:.3f}"
                )
            if m > best_mean and total_episodes > args.save_after:
                best_mean = m
                path = os.path.join(args.out_dir, "best_model.pt")
                _safe_torch_save(
                    {
                        "policy": policy.state_dict(),
                        "torchrl": True,
                        "obs_dim": 96,
                        "n_actions": 96,
                    },
                    path,
                )

    os.makedirs(args.out_dir, exist_ok=True)
    plt.figure(figsize=(9, 4))
    n = len(returns)
    x = np.arange(1, n + 1, dtype=np.float64)
    plt.plot(x, returns, alpha=0.35, label="return")
    w = min(50, len(returns))
    if w >= 2:
        ma = [np.mean(returns[max(0, i - w + 1) : i + 1]) for i in range(len(returns))]
        plt.plot(x, ma, label=f"MA{w}")
    plt.xlabel("Episode (1-based)")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    curve_path = os.path.join(args.out_dir, "training_curve.png")
    plt.savefig(curve_path, dpi=150)
    print("saved", curve_path)

    final_path = os.path.join(args.out_dir, "final_model.pt")
    _safe_torch_save(
        {
            "policy": policy.state_dict(),
            "torchrl": True,
            "obs_dim": 96,
            "n_actions": 96,
        },
        final_path,
    )
    print("saved", final_path)
    if train_updates == 0:
        print(
            "[warn] 本次训练没有发生任何参数更新（train_updates=0）。"
            "请提高 episodes 或降低 min-buffer。",
            flush=True,
        )
    else:
        print(f"[info] train_updates={train_updates}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=8000, help="目标对局数（达到后停止）")
    p.add_argument("--rollout-steps", type=int, default=128, help="每次 rollout 最大步数")
    p.add_argument("--lr", type=float, default=7e-5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--buffer", type=int, default=100_000)
    p.add_argument("--min-buffer", type=int, default=2000, help="开始训练前最少缓冲条数")
    p.add_argument("--optim-steps", type=int, default=4, help="每轮 rollout 后优化步数")
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--eps-start", type=float, default=0.5)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument(
        "--eps-anneal-steps",
        type=int,
        default=3_000,
        help="EGreedy 从 eps_start 线性降到 eps_end 的步数",
    )
    p.add_argument(
        "--opponent",
        type=str,
        default="mixed",
        choices=["random", "heuristic", "mixed", "self_snapshot"],
        help="训练对手：random / heuristic / mixed / self_snapshot（与「上一快照」的自己对弈）。",
    )
    p.add_argument(
        "--opponent-heuristic-prob",
        type=float,
        default=0.8,
        help="当 opponent=mixed 时，使用 heuristic 的概率。",
    )
    p.add_argument(
        "--curriculum-switch-episodes",
        type=int,
        default=0,
        help="先用随机对手训练多少局后切换到 --opponent；设 0 表示全程使用 --opponent。",
    )
    p.add_argument(
        "--self-play-sync-iters",
        type=int,
        default=10,
        help="self_snapshot：每隔多少个外层 train iter 把当前 policy 权重复制到快照对手；<=0 表示每次有更新后都同步。",
    )
    p.add_argument(
        "--self-play-opponent-q-only",
        action="store_true",
        help="self_snapshot：快照对手不使用必胜/必堵战术层，仅按 Q 做合法贪心。",
    )
    p.add_argument("--eval-every", type=int, default=100, help="每隔多少个 iter 做一次胜率评估；0 关闭")
    p.add_argument("--eval-episodes", type=int, default=20, help="每次评估的对局数")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-after", type=int, default=100)
    p.add_argument("--out-dir", type=str, default="artifacts")
    p.add_argument("--cpu", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_dqn(args)


if __name__ == "__main__":
    main()
