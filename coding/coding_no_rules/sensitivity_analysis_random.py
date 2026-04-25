#!/usr/bin/env python3
"""
单参数敏感性分析（OAT），random 对手等价 main 中 mode=random。
依赖本文件 + train/model/environment；奖励通过 patch environment 常量。
make_rl_env_sensitivity + _patch_torchrl_gym_read_action 兼容 TorchRL 0.11。

用法见 --help；分参数入口见 sensitivity_params/<name>.py。
各轴输出目录 oat_random_sensitivity/by_param/<轴名>/figures/，收敛图为 MA50。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment,misc]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam


def _iter_param_values_with_progress(axis_name: str, values: Sequence[Any]) -> Iterable[Any]:
    if tqdm is None:
        yield from values
        return
    yield from tqdm(values, desc=f"param:{axis_name}", unit="run", leave=True, file=sys.stderr)


def _strip_jupyter_kernel_argv(argv: list[str]) -> list[str]:
    if len(argv) < 1:
        return argv
    out = [argv[0]]
    i = 1
    while i < len(argv):
        a = argv[i]
        if a.startswith("--f="):
            i += 1
            continue
        if a == "-f":
            i += 2
            continue
        out.append(a)
        i += 1
    return out


def make_rl_env_sensitivity(
    device: Optional[torch.device] = None,
    opponent: str = "random",
    opponent_heuristic_prob: float = 0.7,
    opponent_policy: Any = None,
):
    from environment import register_super_tic_tac_toe
    from torchrl.envs import GymEnv, StepCounter, TransformedEnv

    register_super_tic_tac_toe()
    kw: dict[str, Any] = dict(opponent=opponent, opponent_heuristic_prob=opponent_heuristic_prob)
    if opponent_policy is not None:
        kw["opponent_policy"] = opponent_policy
    return TransformedEnv(
        GymEnv(
            "SuperTicTacToe-v0",
            device=device,
            categorical_action_encoding=True,
            **kw,
        ),
        StepCounter(),
    )


_READ_ACTION_PATCHED = False


def _patch_torchrl_gym_read_action() -> None:
    global _READ_ACTION_PATCHED
    if _READ_ACTION_PATCHED:
        return
    from torchrl.data import Categorical, OneHot
    from torchrl.envs.gym_like import GymLikeEnv
    from torchrl.envs.libs.gym import GymWrapper

    def read_action(self, action):  # type: ignore[no-untyped-def]
        out = GymLikeEnv.read_action(self, action)
        spec = self.action_spec
        if isinstance(spec, Categorical):
            arr = np.asarray(out)
            if arr.ndim == 0:
                return int(arr)
            if arr.size == 1:
                return int(arr.reshape(-1)[0])
            return int(arr.argmax())
        if isinstance(spec, OneHot):
            arr = np.asarray(out)
            if arr.ndim == 0:
                return int(arr)
            if arr.size == 1:
                return int(arr.reshape(-1)[0])
            return int(arr.argmax())
        return out

    GymWrapper.read_action = read_action  # type: ignore[method-assign]
    _READ_ACTION_PATCHED = True


MAIN_BASELINE: dict[str, Any] = {
    "episodes": 500,
    "rollout_steps": 64,
    "lr": 7e-5,
    "gamma": 0.99,
    "batch": 64,
    "buffer": 100_000,
    "min_buffer": 100,
    "optim_steps": 1,
    "tau": 0.005,
    "eps_start": 0.6,
    "eps_end": 0.05,
    "eps_anneal_steps": int(round(900 * 500 / 750)),
    "opponent": "random",
    "curriculum_switch_episodes": 0,
    "opponent_heuristic_prob": 0.7,
    "self_play_sync_iters": 10,
    "self_play_opponent_q_only": False,
    "eval_every": 0,
    "eval_episodes": 20,
    "seed": 0,
    "log_every": 200,
    "save_after": 0,
    "cpu": False,
    "tactical_inference": False,
}

ENV_REWARD_DEFAULTS: dict[str, float] = {
    "WIN_REWARD": 3.0,
    "LOSE_REWARD": -3.0,
    "ILLEGAL_REWARD": -0.2,
    "FORFEIT_REWARD": -0.1,
    "THREAT_REDUCE_BONUS": 0.1,
    "THREAT_EXIST_PENALTY": 0.08,
    "WIN_INTENT_BONUS": 1.15,
    "MISS_WIN_PENALTY": 1.35,
    "BLOCK_THREAT_BONUS": 0.85,
    "MISS_BLOCK_PENALTY": 1.15,
    "CREATE_WIN_THREAT_BONUS": 0.12,
    "SHAPING_LAMBDA": 0.25,
    "SHAPING_GAMMA": 0.99,
}

_ENV_BACKUP: dict[str, float] = {}


def _backup_env_constants() -> None:
    import environment as env

    for k in ENV_REWARD_DEFAULTS:
        _ENV_BACKUP[k] = float(getattr(env, k))


def _restore_env_constants() -> None:
    import environment as env

    for k, v in _ENV_BACKUP.items():
        setattr(env, k, v)


def _apply_env_overrides(overrides: Mapping[str, float] | None) -> None:
    import environment as env

    if not overrides:
        return
    for k, v in overrides.items():
        if not hasattr(env, k):
            raise KeyError(f"environment 无属性 {k!r}")
        setattr(env, k, float(v))


def _five_numeric_values(lo: float, hi: float, baseline: float, *, kind: str) -> list[float]:
    _ = baseline
    if hi < lo:
        lo, hi = hi, lo
    xs = np.geomspace(lo, hi, num=5) if (kind == "log" and lo > 0 and hi > 0) else np.linspace(lo, hi, num=5)
    out = [float(x) for x in xs]
    uniq: list[float] = []
    for x in out:
        if not any(abs(x - u) <= 1e-12 * max(1.0, abs(u)) for u in uniq):
            uniq.append(x)
    while len(uniq) < 5:
        uniq.append(uniq[-1] + 1e-9)
    return uniq[:5]


def _five_int_values(lo: int, hi: int, baseline: int) -> list[int]:
    _ = baseline
    if hi < lo:
        lo, hi = hi, lo
    out: list[int] = []
    for x in np.linspace(lo, hi, num=50):
        iv = int(round(float(x)))
        iv = max(lo, min(hi, iv))
        if iv not in out:
            out.append(iv)
        if len(out) >= 5:
            return out[:5]
    while len(out) < 5 and out:
        out.append(out[-1])
    return out[:5]


def _five_from_discrete(choices: Sequence[int], baseline: int) -> list[int]:
    _ = baseline
    ch = sorted(set(int(x) for x in choices))
    if not ch:
        return []
    idxs = np.linspace(0, len(ch) - 1, num=5)
    return [ch[int(round(float(i)))] for i in idxs]


def _format_value(v: Any) -> str:
    if isinstance(v, float):
        if abs(v) < 1e-3 or abs(v) >= 1e5:
            return f"{v:.6g}"
        return f"{v:.6f}".rstrip("0").rstrip(".")
    return str(v)


from environment import SuperTicTacToeEnv  # noqa: E402
from model import build_policy_pair, greedy_action_masked  # noqa: E402
from training_opponent import SelfPlayOpponentBridge  # noqa: E402
from train import (  # noqa: E402
    _episode_returns_from_rollout,
    _evaluate_win_rate,
    _normalize_td_action_shapes,
    _safe_torch_save,
)


def _train_dqn_collect_returns(args: argparse.Namespace) -> tuple[list[float], int]:
    _patch_torchrl_gym_read_action()
    try:
        from torchrl.data import LazyTensorStorage, ReplayBuffer
        from torchrl.objectives import DQNLoss, SoftUpdate
        from torchrl.objectives.utils import ValueEstimators
    except ImportError as e:
        raise ImportError("请安装 TorchRL：pip install torchrl tensordict\n" + str(e)) from e

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    start_opp = (
        "random"
        if args.curriculum_switch_episodes > 0
        else ("random" if args.opponent == "self_snapshot" else args.opponent)
    )
    env = make_rl_env_sensitivity(
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
            env, eps_init=1.0, eps_end=1.0, annealing_num_steps=1
        )
        snapshot_policy.to(device)
        snapshot_policy.load_state_dict(policy.state_dict())
        snapshot_policy.eval()
        tact = not bool(getattr(args, "self_play_opponent_q_only", False))
        bridge = SelfPlayOpponentBridge(snapshot_policy, device, use_tactical=tact)
        env = make_rl_env_sensitivity(
            device=device,
            opponent="random",
            opponent_heuristic_prob=args.opponent_heuristic_prob,
            opponent_policy=bridge,
        )
        env.set_seed(args.seed)

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
        roll = env.rollout(max_steps=args.rollout_steps, policy=policy_explore, auto_reset=True)
        roll = _normalize_td_action_shapes(roll)
        rb.extend(roll)
        ep_rets = _episode_returns_from_rollout(roll)
        returns.extend(ep_rets)
        total_episodes += len(ep_rets)

        if not switched_to_target_opponent and total_episodes >= args.curriculum_switch_episodes:
            if args.opponent == "self_snapshot":
                snapshot_policy, _, _ = build_policy_pair(
                    env, eps_init=1.0, eps_end=1.0, annealing_num_steps=1
                )
                snapshot_policy.to(device)
                snapshot_policy.load_state_dict(policy.state_dict())
                snapshot_policy.eval()
                tact = not bool(getattr(args, "self_play_opponent_q_only", False))
                bridge = SelfPlayOpponentBridge(snapshot_policy, device, use_tactical=tact)
                env = make_rl_env_sensitivity(
                    device=device,
                    opponent="random",
                    opponent_heuristic_prob=args.opponent_heuristic_prob,
                    opponent_policy=bridge,
                )
            else:
                env = make_rl_env_sensitivity(
                    device=device,
                    opponent=args.opponent,
                    opponent_heuristic_prob=args.opponent_heuristic_prob,
                )
            env.set_seed(args.seed + it)
            switched_to_target_opponent = True
            print(f"[curriculum] switch -> {args.opponent} @ ep {total_episodes}", flush=True)

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
                sync_now = (se <= 0) or (it % se == 0)
                if sync_now:
                    snapshot_policy.load_state_dict(policy.state_dict())
                    snapshot_policy.eval()

        if it % max(args.log_every, 1) == 0 and returns:
            w = min(50, len(returns))
            m = float(np.mean(returns[-w:]))
            eps = float(exploration.eps)
            print(
                f"[sens] iter {it} ep {total_episodes} mean_{w} {m:.3f} eps {eps:.3f} buf {len(rb)}",
                flush=True,
            )
            if args.eval_every > 0 and it % args.eval_every == 0:
                wr_r, dr_r = _evaluate_win_rate(
                    policy,
                    device,
                    args.eval_episodes,
                    opponent="random",
                    use_tactical_inference=args.tactical_inference,
                )
                print(f"[sens][eval] wr_r={wr_r:.3f} dr_r={dr_r:.3f}", flush=True)
            if m > best_mean and total_episodes > args.save_after:
                best_mean = m
                _safe_torch_save(
                    {"policy": policy.state_dict(), "torchrl": True, "obs_dim": 96, "n_actions": 96},
                    os.path.join(args.out_dir, "best_model.pt"),
                )

    os.makedirs(args.out_dir, exist_ok=True)
    plt.figure(figsize=(9, 4))
    x = np.arange(1, len(returns) + 1, dtype=np.float64)
    plt.plot(x, returns, alpha=0.35, label="return")
    w = min(50, len(returns))
    if w >= 2:
        ma = [np.mean(returns[max(0, i - w + 1) : i + 1]) for i in range(len(returns))]
        plt.plot(x, ma, label=f"MA{w}")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    cp = os.path.join(args.out_dir, "training_curve.png")
    plt.savefig(cp, dpi=150)
    plt.close()
    print("saved", cp, flush=True)
    _safe_torch_save(
        {"policy": policy.state_dict(), "torchrl": True, "obs_dim": 96, "n_actions": 96},
        os.path.join(args.out_dir, "final_model.pt"),
    )
    print("saved", os.path.join(args.out_dir, "final_model.pt"), flush=True)
    if train_updates == 0:
        print("[warn] train_updates=0", flush=True)
    else:
        print(f"[info] train_updates={train_updates}", flush=True)
    return returns, train_updates


_MAX_STEPS_PER_EPISODE = 512


def _load_policy_torchrl(path: str, device: torch.device):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if not ckpt.get("torchrl"):
        raise ValueError(f"非 TorchRL checkpoint: {path}")
    env = make_rl_env_sensitivity(device=torch.device("cpu"), opponent="random")
    policy, _, _ = build_policy_pair(env, eps_init=1.0, eps_end=0.05, annealing_num_steps=1)
    policy.load_state_dict(ckpt["policy"])
    policy.to(device)
    policy.eval()
    return policy


def eval_agent_vs_policy(
    path_agent: str,
    opponent_policy: Callable[[np.ndarray, np.ndarray], int],
    device: torch.device,
    episodes: int,
    *,
    use_tactical: bool,
    seed_offset: int,
) -> float:
    pol_a = _load_policy_torchrl(path_agent, device)
    wins = draws = losses = 0
    for ep in range(episodes):
        env = SuperTicTacToeEnv(seed=seed_offset + ep, opponent_policy=opponent_policy)
        obs, _ = env.reset(seed=seed_offset + ep)
        steps = 0
        while True:
            steps += 1
            if steps > _MAX_STEPS_PER_EPISODE:
                draws += 1
                break
            m = env.legal_actions_mask()
            a = greedy_action_masked(pol_a, obs, m, device, use_tactical=use_tactical)
            obs, _r, term, trunc, info = env.step(a)
            if term or trunc:
                if info.get("winner") == "agent":
                    wins += 1
                elif info.get("winner") == "opponent":
                    losses += 1
                elif info.get("draw"):
                    draws += 1
                else:
                    draws += 1
                break
    return wins / max(1, wins + losses + draws)


def eval_agent_vs_base_checkpoint(
    path_agent: str,
    path_base: str,
    device: torch.device,
    episodes: int,
    *,
    use_tactical: bool,
    seed_offset: int,
) -> float:
    pol_o = _load_policy_torchrl(path_base, device)

    def opp_policy(obs_opp: np.ndarray, mask: np.ndarray) -> int:
        return greedy_action_masked(pol_o, obs_opp, mask, device, use_tactical=use_tactical)

    return eval_agent_vs_policy(
        path_agent, opp_policy, device, episodes, use_tactical=use_tactical, seed_offset=seed_offset
    )


def return_variance(returns: Sequence[float]) -> float:
    arr = np.asarray(returns, dtype=np.float64)
    return 0.0 if arr.size < 2 else float(np.var(arr))


def sample_efficiency_slope(returns: Sequence[float]) -> float:
    y = np.asarray(returns, dtype=np.float64)
    n = y.size
    if n < 2:
        return 0.0
    x = np.arange(1, n + 1, dtype=np.float64)
    xm, ym = float(x.mean()), float(y.mean())
    d = float(np.sum((x - xm) ** 2))
    return 0.0 if d <= 0 else float(np.sum((x - xm) * (y - ym)) / d)


@dataclass
class SweepAxis:
    name: str
    values: list[Any]
    env_keys: tuple[str, ...] = field(default_factory=tuple)
    apply_to_args: Callable[[argparse.Namespace, Any], None] | None = None


def _set_attr(ns: argparse.Namespace, key: str, val: Any) -> None:
    setattr(ns, key, val)


def build_sweep_axes(baseline: dict[str, Any]) -> list[SweepAxis]:
    b = baseline
    axes: list[SweepAxis] = []

    def add(name: str, vals: list[Any], key: str) -> None:
        axes.append(
            SweepAxis(name=name, values=vals, apply_to_args=lambda ns, v, k=key: _set_attr(ns, k, v))
        )

    add("lr", _five_numeric_values(1e-5, 1e-3, float(b["lr"]), kind="log"), "lr")
    add("optim_steps", _five_int_values(1, 10, int(b["optim_steps"])), "optim_steps")
    est = max(1, int(b["episodes"]) * int(b["rollout_steps"]))
    lo_a, hi_a = int(0.30 * est), max(int(0.30 * est) + 1, int(0.60 * est))
    add("eps_anneal_steps", [int(max(1, round(x))) for x in np.linspace(lo_a, hi_a, 5)], "eps_anneal_steps")
    add("batch", _five_from_discrete((32, 64, 128, 256), int(b["batch"])), "batch")
    add("rollout_steps", _five_int_values(16, 128, int(b["rollout_steps"])), "rollout_steps")
    add("min_buffer", _five_int_values(20, 400, int(b["min_buffer"])), "min_buffer")
    add("eps_start", _five_numeric_values(0.3, 1.0, float(b["eps_start"]), kind="linear"), "eps_start")
    add("eps_end", _five_numeric_values(0.01, 0.2, float(b["eps_end"]), kind="linear"), "eps_end")
    add("tau", _five_numeric_values(0.001, 0.02, float(b["tau"]), kind="linear"), "tau")
    add("gamma", _five_numeric_values(0.90, 0.999, float(b["gamma"]), kind="linear"), "gamma")
    add(
        "buffer",
        [int(round(x)) for x in _five_numeric_values(100_000, 10_000_000, float(b["buffer"]), kind="log")],
        "buffer",
    )

    def env_axis(attr: str, lo: float, hi: float) -> SweepAxis:
        base = float(ENV_REWARD_DEFAULTS[attr])
        return SweepAxis(
            name=attr.lower(),
            values=_five_numeric_values(lo, hi, base, kind="linear"),
            env_keys=(attr,),
            apply_to_args=None,
        )

    axes += [
        env_axis("WIN_REWARD", 1.0, 10.0),
        env_axis("LOSE_REWARD", -10.0, -1.0),
        env_axis("FORFEIT_REWARD", -0.5, -0.05),
        env_axis("WIN_INTENT_BONUS", 0.5, 2.0),
        env_axis("ILLEGAL_REWARD", -5.0, 0.0),
        env_axis("THREAT_REDUCE_BONUS", 0.0, 1.0),
        env_axis("THREAT_EXIST_PENALTY", 0.0, 1.0),
        env_axis("MISS_WIN_PENALTY", 0.0, 3.0),
        env_axis("BLOCK_THREAT_BONUS", 0.0, 3.0),
        env_axis("MISS_BLOCK_PENALTY", 0.0, 3.0),
        env_axis("CREATE_WIN_THREAT_BONUS", 0.0, 1.0),
        env_axis("SHAPING_LAMBDA", 0.0, 1.0),
        env_axis("SHAPING_GAMMA", 0.0, 1.0),
    ]
    return axes


def list_param_axis_names(baseline: dict[str, Any] | None = None) -> tuple[str, ...]:
    b = deepcopy(MAIN_BASELINE) if baseline is None else deepcopy(baseline)
    return tuple(a.name for a in build_sweep_axes(b))


def namespace_from_baseline(baseline: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(**deepcopy(baseline))


def run_single_experiment(
    base_ns: argparse.Namespace,
    axis: SweepAxis,
    value: Any,
    out_dir: str,
    base_model: str,
    eval_episodes: int,
    device: torch.device,
    *,
    use_tactical_eval: bool,
) -> dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    done_path = os.path.join(out_dir, "result.json")
    if os.path.isfile(done_path):
        with open(done_path, "r", encoding="utf-8") as f:
            return json.load(f)

    ns = deepcopy(base_ns)
    ns.out_dir = out_dir
    env_override: dict[str, float] = {}
    if axis.env_keys:
        env_override[axis.env_keys[0]] = float(value)
    elif axis.apply_to_args is not None:
        axis.apply_to_args(ns, value)
    else:
        raise RuntimeError(axis)

    _restore_env_constants()
    _apply_env_overrides(env_override)
    random.seed(ns.seed)
    np.random.seed(ns.seed)
    torch.manual_seed(ns.seed)

    returns, train_updates = _train_dqn_collect_returns(ns)
    np.save(os.path.join(out_dir, "returns.npy"), np.asarray(returns, dtype=np.float64))
    final_ckpt = os.path.join(out_dir, "final_model.pt")
    win_rate = eval_agent_vs_base_checkpoint(
        final_ckpt,
        base_model,
        device,
        eval_episodes,
        use_tactical=use_tactical_eval,
        seed_offset=hash((axis.name, str(value))) % 10_000,
    )
    metrics = {
        "param": axis.name,
        "value": value,
        "win_rate_vs_base": win_rate,
        "return_variance": return_variance(returns),
        "sample_efficiency_slope": sample_efficiency_slope(returns),
        "n_episodes": len(returns),
        "train_updates": int(train_updates),
        "out_dir": os.path.abspath(out_dir),
    }
    with open(done_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    _restore_env_constants()
    return metrics


def moving_average_n(y: np.ndarray, window: int) -> np.ndarray:
    """Episode 序列的严格窗口移动平均；前 window-1 个点为 nan（窗口未满）。"""
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or window < 1:
        return out
    w = min(window, n)
    c = np.cumsum(np.insert(y, 0, 0.0))
    out[w - 1 :] = (c[w:] - c[:-w]) / w
    return out


def plot_param_figures(axis: SweepAxis, rows: list[dict[str, Any]], fig_dir: str) -> None:
    os.makedirs(fig_dir, exist_ok=True)
    for r in rows:
        npy = os.path.join(str(r["out_dir"]), "returns.npy")
        if not os.path.isfile(npy):
            raise FileNotFoundError(npy)

    def sk(r: dict[str, Any]) -> Any:
        try:
            return float(r["value"])
        except Exception:
            return str(r["value"])

    rows = sorted(rows, key=sk)
    values = [r["value"] for r in rows]
    wr = [float(r["win_rate_vs_base"]) for r in rows]
    va = [float(r["return_variance"]) for r in rows]
    sl = [float(r["sample_efficiency_slope"]) for r in rows]
    labels = [_format_value(v) for v in values]
    idx = np.arange(len(labels))
    written: list[str] = []

    # 所有参数轴：收敛图统一为回报 MA50（前 49 个 episode 为 nan，图中不连线）
    convergence_ma = 50

    try:
        plt.figure(figsize=(10, 5))
        for r in rows:
            y = np.load(os.path.join(str(r["out_dir"]), "returns.npy"))
            y = np.asarray(y, dtype=np.float64).ravel()
            y = moving_average_n(y, convergence_ma)
            plt.plot(np.arange(1, y.size + 1), y, alpha=0.75, lw=1.3, label=f"{axis.name}={_format_value(r['value'])}")
        plt.xlabel("Episode")
        plt.ylabel("Return (MA50)")
        plt.title(f"Convergence — {axis.name} (MA{convergence_ma})")
        plt.legend(fontsize=8)
        plt.tight_layout()
        p1 = os.path.join(fig_dir, f"{axis.name}_convergence.png")
        plt.savefig(p1, dpi=150, format="png", bbox_inches="tight")
        written.append(os.path.basename(p1))
    finally:
        plt.close("all")

    try:
        plt.figure(figsize=(8, 4))
        plt.bar(idx, wr, color="#3b82f6", edgecolor="#1e293b")
        plt.xticks(idx, labels, rotation=35, ha="right")
        plt.ylim(0, 1.05)
        plt.ylabel("Win rate vs base")
        plt.title(f"Win rate — {axis.name}")
        plt.tight_layout()
        p2 = os.path.join(fig_dir, f"{axis.name}_winrate.png")
        plt.savefig(p2, dpi=150, format="png", bbox_inches="tight")
        written.append(os.path.basename(p2))
    finally:
        plt.close("all")

    try:
        plt.figure(figsize=(8, 4))
        plt.bar(idx, va, color="#f97316", edgecolor="#1e293b")
        plt.xticks(idx, labels, rotation=35, ha="right")
        plt.ylabel("Var(return)")
        plt.title(f"Stability — {axis.name}")
        plt.tight_layout()
        p3 = os.path.join(fig_dir, f"{axis.name}_stability_variance.png")
        plt.savefig(p3, dpi=150, format="png", bbox_inches="tight")
        written.append(os.path.basename(p3))
    finally:
        plt.close("all")

    try:
        plt.figure(figsize=(8, 4))
        plt.bar(idx, sl, color="#22c55e", edgecolor="#1e293b")
        plt.xticks(idx, labels, rotation=35, ha="right")
        plt.ylabel("OLS slope")
        plt.title(f"Sample efficiency — {axis.name}")
        plt.tight_layout()
        p4 = os.path.join(fig_dir, f"{axis.name}_sample_efficiency_slope.png")
        plt.savefig(p4, dpi=150, format="png", bbox_inches="tight")
        written.append(os.path.basename(p4))
    finally:
        plt.close("all")

    with open(os.path.join(fig_dir, "figures_manifest.txt"), "w", encoding="utf-8") as mf:
        mf.write("\n".join(written) + "\n")


def replot_all_figures_from_disk(out_root: str, *, only_param: str = "") -> None:
    bp = os.path.join(out_root, "by_param")
    if not os.path.isdir(bp):
        print(f"[replot] 无目录 {bp}", file=sys.stderr)
        return
    op = only_param.strip()
    for name in sorted(os.listdir(bp)):
        if op and name != op:
            continue
        pd = os.path.join(bp, name)
        if not os.path.isdir(pd):
            continue
        rows: list[dict[str, Any]] = []
        for fn in sorted(os.listdir(pd)):
            if not fn.startswith("value_"):
                continue
            vd = os.path.join(pd, fn)
            jp, npy = os.path.join(vd, "result.json"), os.path.join(vd, "returns.npy")
            if os.path.isfile(jp) and os.path.isfile(npy):
                with open(jp, encoding="utf-8") as f:
                    rows.append(json.load(f))
        if not rows:
            print(f"[replot] skip {name}", flush=True)
            continue
        fd = os.path.join(pd, "figures")
        plot_param_figures(SweepAxis(name=name, values=[r["value"] for r in rows]), rows, fd)
        print(f"[replot] {name} -> {fd}", flush=True)


def append_csv(path: str, fieldnames: list[str], row: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    sys.argv[:] = _strip_jupyter_kernel_argv(sys.argv)
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="oat_random_sensitivity")
    p.add_argument("--base-model", type=str, default="base_model.pt")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--tactical-eval", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--replot-only", action="store_true")
    p.add_argument("--only-param", type=str, default="")
    p.add_argument("--list-params", action="store_true")
    args_cli = p.parse_args()

    baseline = deepcopy(MAIN_BASELINE)
    baseline["episodes"] = int(args_cli.episodes)
    baseline["cpu"] = bool(args_cli.cpu)
    baseline["eps_anneal_steps"] = int(round(900 * baseline["episodes"] / 750))

    if args_cli.list_params:
        for a in build_sweep_axes(baseline):
            print(a.name, flush=True)
        return

    base_model = os.path.abspath(args_cli.base_model)
    if not args_cli.dry_run and not args_cli.replot_only and not os.path.isfile(base_model):
        print(f"找不到 base 模型: {base_model}", file=sys.stderr)
        sys.exit(1)

    _backup_env_constants()
    _restore_env_constants()
    out_root = os.path.abspath(args_cli.out_dir)
    os.makedirs(out_root, exist_ok=True)

    base_ns = namespace_from_baseline(baseline)
    all_axes = build_sweep_axes(baseline)
    axes = all_axes
    if args_cli.only_param.strip():
        axes = [a for a in all_axes if a.name == args_cli.only_param.strip()]
        if not axes:
            print(f"--only-param 无匹配: {args_cli.only_param!r}", file=sys.stderr)
            sys.exit(1)

    plan_path = os.path.join(out_root, "experiment_plan.json")
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline": baseline,
                "axes": [{"name": a.name, "values": a.values, "env_keys": list(a.env_keys)} for a in axes],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"已写入: {plan_path}")

    if args_cli.dry_run:
        for ax in axes:
            print(f"\n[{ax.name}] {ax.values}")
        return

    if args_cli.replot_only:
        replot_all_figures_from_disk(out_root, only_param=args_cli.only_param)
        print("[replot] done", out_root, flush=True)
        return

    if tqdm is None:
        print("pip install tqdm 显示进度条", file=sys.stderr, flush=True)

    device = torch.device("cpu" if args_cli.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    summary_csv = os.path.join(out_root, "all_runs.csv")
    fields = [
        "param",
        "value",
        "win_rate_vs_base",
        "return_variance",
        "sample_efficiency_slope",
        "n_episodes",
        "train_updates",
        "out_dir",
    ]

    for axis in axes:
        param_dir = os.path.join(out_root, "by_param", axis.name)
        fig_dir = os.path.join(param_dir, "figures")
        os.makedirs(param_dir, exist_ok=True)
        rows: list[dict[str, Any]] = []
        try:
            for v in _iter_param_values_with_progress(axis.name, axis.values):
                vdir = os.path.join(param_dir, f"value_{_format_value(v)}")
                print(f"\n=== {axis.name} = {v!r} -> {vdir}", flush=True)
                m = run_single_experiment(
                    base_ns,
                    axis,
                    v,
                    vdir,
                    base_model,
                    int(args_cli.eval_episodes),
                    device,
                    use_tactical_eval=bool(args_cli.tactical_eval),
                )
                rows.append(m)
                append_csv(summary_csv, fields, {k: m.get(k, "") for k in fields})
            mcsv = os.path.join(param_dir, "metrics.csv")
            with open(mcsv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k, "") for k in fields})
            plot_param_figures(axis, rows, fig_dir)
            print(f"完成 {axis.name} -> {fig_dir}", flush=True)
        except Exception as e:
            ep = os.path.join(param_dir, "AXIS_ERROR.txt")
            with open(ep, "w", encoding="utf-8") as ef:
                ef.write(f"{type(e).__name__}: {e}\n")
            print(f"[错误] {axis.name}: {e}", file=sys.stderr, flush=True)
            continue

    print(f"\n完成。总表: {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
