#!/usr/bin/env python3
"""
model/ 目录下各 checkpoint 两两对弈：对每个有序对 (A,B), A≠B，A 作 agent、B 作 opponent，共 episodes 局。
输出 CSV、胜率热力图、各模型作先手时的平均胜率柱状图、总表图。

用法:
  python eval_model_pairwise.py
  python eval_model_pairwise.py --model-dir model --episodes 100 --out-dir artifacts/model_pairwise
  python eval_model_pairwise.py --plots-from-csv artifacts/model_pairwise/pairwise_results.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from environment import SuperTicTacToeEnv
from model import build_policy_pair, greedy_action_masked, make_rl_env

_MAX_STEPS_PER_EPISODE = 512


def _strip_jupyter_kernel_argv(argv: list) -> list:
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


def _load_policy(path: str, device: torch.device):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if not ckpt.get("torchrl"):
        raise ValueError(f"非 TorchRL checkpoint（缺少 torchrl=True）: {path}")
    env = make_rl_env(device=torch.device("cpu"), opponent="random")
    policy, _, _ = build_policy_pair(env, eps_init=1.0, eps_end=0.05, annealing_num_steps=1)
    policy.load_state_dict(ckpt["policy"])
    policy.to(device)
    policy.eval()
    return policy


def _discover_checkpoints(model_dir: str) -> List[Tuple[str, str]]:
    if not os.path.isdir(model_dir):
        return []
    out = []
    for fn in os.listdir(model_dir):
        if fn.endswith(".pt"):
            out.append((fn.replace(".pt", ""), os.path.join(model_dir, fn)))

    def _sort_key(item: Tuple[str, str]) -> tuple:
        stem, _ = item
        mid = _plot_model_id_from_stem(stem)
        return (mid is None, mid if mid is not None else 10**9, stem)

    out.sort(key=_sort_key)
    return out


def _eval_ordered_pair(
    path_agent: str,
    path_opp: str,
    policy_cache: Dict[str, object],
    device: torch.device,
    episodes: int,
    *,
    use_tactical: bool,
    seed_offset: int,
) -> Tuple[int, int, int]:
    if path_agent not in policy_cache:
        policy_cache[path_agent] = _load_policy(path_agent, device)
    if path_opp not in policy_cache:
        policy_cache[path_opp] = _load_policy(path_opp, device)
    pol_a = policy_cache[path_agent]
    pol_o = policy_cache[path_opp]

    def opp_policy(obs_opp: np.ndarray, mask: np.ndarray) -> int:
        return greedy_action_masked(
            pol_o, obs_opp, mask, device, use_tactical=use_tactical
        )

    wins = losses = draws = 0
    for ep in range(episodes):
        env = SuperTicTacToeEnv(seed=seed_offset + ep, opponent_policy=opp_policy)
        obs, _ = env.reset(seed=seed_offset + ep)
        steps = 0
        while True:
            steps += 1
            if steps > _MAX_STEPS_PER_EPISODE:
                draws += 1
                break
            m = env.legal_actions_mask()
            a = greedy_action_masked(
                pol_a, obs, m, device, use_tactical=use_tactical
            )
            obs, r, term, trunc, info = env.step(a)
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
    return wins, losses, draws


def _save_csv(rows: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "agent",
                "opponent",
                "wins",
                "losses",
                "draws",
                "games",
                "agent_win_rate",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _rebuild_win_matrix_from_csv(
    path: str,
) -> Tuple[List[str], np.ndarray, int]:
    """从 pairwise_results.csv 恢复 names 顺序与胜率矩阵，供仅重画图使用。"""
    rows: List[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    agents = {r["agent"] for r in rows}
    opps = {r["opponent"] for r in rows}
    if agents != opps:
        raise ValueError(
            "CSV agent set != opponent set; cannot build a square win matrix."
        )
    names = sorted(agents)
    n = len(names)
    if n < 2:
        raise ValueError(f"need at least 2 models in CSV, got {n}")
    expected = n * (n - 1)
    if len(rows) != expected:
        raise ValueError(
            f"CSV row count {len(rows)} != n*(n-1)={expected} for n={n}; "
            "refuse to plot incomplete matrix."
        )
    win_mat = np.zeros((n, n), dtype=float)
    idx = {name: i for i, name in enumerate(names)}
    games_list: List[int] = []
    for r in rows:
        i = idx[r["agent"]]
        j = idx[r["opponent"]]
        win_mat[i, j] = float(r["agent_win_rate"])
        games_list.append(int(r["games"]))
    if len(set(games_list)) != 1:
        raise ValueError(
            f"inconsistent 'games' across rows: {sorted(set(games_list))}; "
            "cannot pick a single episodes value for plot titles."
        )
    episodes = games_list[0]
    return names, win_mat, episodes


_PLOT_MODEL_LABELS: dict[int, str] = {
    1: "Random",
    2: "Heuristic",
    3: "Full-Mix",
    4: "Full-Self",
    5: "Cur+Mix",
    6: "Cur+Self+Heu",
    7: "Cur+Heu",
    8: "Cur+Self",
    9: "NoRule_Random",
    10: "NoRule_Cur+Mix",
    11: "NoRule_Cur+Heu",
}

_PLOT_STEM_TO_ID: dict[str, int] = {v: k for k, v in _PLOT_MODEL_LABELS.items()}


def _plot_model_id_from_stem(stem: str) -> int | None:
    s = stem.strip()
    if s in _PLOT_STEM_TO_ID:
        return _PLOT_STEM_TO_ID[s]
    m = re.search(r"(\d+)$", s)
    if m:
        return int(m.group(1))
    parts = re.findall(r"\d+", s)
    if parts:
        return int(parts[-1])
    return None


def _plot_tick_label(stem: str) -> str:
    mid = _plot_model_id_from_stem(stem)
    if mid is not None and mid in _PLOT_MODEL_LABELS:
        return _PLOT_MODEL_LABELS[mid]
    return (stem[:14] if stem else "?") or "?"


def _plot_heatmap(
    names: List[str],
    win_mat: np.ndarray,
    out_path: str,
    episodes: int,
) -> None:
    labels = [_plot_tick_label(nm) for nm in names]
    n = len(names)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.55), max(7, n * 0.55)))
    disp = win_mat.astype(float).copy()
    np.fill_diagonal(disp, np.nan)
    im = ax.imshow(disp, vmin=0, vmax=1, cmap="RdYlGn", aspect="equal")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Opponent (columns)")
    ax.set_ylabel("Agent as first player (rows)")
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", color="gray", fontsize=9)
            else:
                v = win_mat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Win rate")
    ax.set_title(f"Pairwise win rate matrix ({episodes} games per cell; row vs column)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_row_mean(names: List[str], win_mat: np.ndarray, out_path: str, episodes: int) -> None:
    labels = [_plot_tick_label(nm) for nm in names]
    n = len(names)
    row_mean = np.array(
        [np.nanmean(np.delete(win_mat[i], i)) for i in range(n)],
        dtype=float,
    )
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(max(9, n * 0.45), 4.2))
    ax.bar(x, row_mean, color=plt.cm.viridis(np.linspace(0.2, 0.85, n)), edgecolor="#334155")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean win rate vs other models")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Mean win rate as first player ({episodes} games per opponent)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_table_figure(names: List[str], win_mat: np.ndarray, out_path: str, episodes: int) -> None:
    labels = [_plot_tick_label(nm) for nm in names]
    n = len(names)
    cell_text = [[""] + list(labels)]
    for i in range(n):
        row = [labels[i]] + [f"{win_mat[i, j]:.2f}" if i != j else "—" for j in range(n)]
        cell_text.append(row)
    fig, ax = plt.subplots(figsize=(min(18, 1.4 + n * 0.72), min(14, 1.0 + n * 0.55)))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.0, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0 or c == 0:
            cell.set_facecolor("#1e293b")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif r == c:
            cell.set_facecolor("#e2e8f0")
        else:
            cell.set_facecolor("#f8fafc" if (r + c) % 2 else "#e2e8f0")
    ax.set_title(
        f"Win rate table (row=agent, col=opponent; {episodes} games per cell)",
        fontsize=10,
        pad=10,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sys.argv[:] = _strip_jupyter_kernel_argv(sys.argv)
    p = argparse.ArgumentParser(description="model/ 下模型两两对弈评估")
    p.add_argument("--model-dir", type=str, default="model")
    p.add_argument("--episodes", type=int, default=100, help="每个有序对 (A vs B) 的局数")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="输出目录；完整评估默认 artifacts/model_pairwise；"
        "与 --plots-from-csv 联用时未指定则写到 CSV 同目录。",
    )
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--tactical-inference", action="store_true")
    p.add_argument(
        "--plots-from-csv",
        type=str,
        default="",
        metavar="PATH",
        help="仅根据已有 pairwise_results.csv 重画三张图，不加载模型、不跑对局。",
    )
    args = p.parse_args()

    if args.plots_from_csv:
        csv_in = os.path.abspath(args.plots_from_csv)
        if not os.path.isfile(csv_in):
            print(f"CSV 不存在: {csv_in}", file=sys.stderr)
            sys.exit(1)
        out_dir = (
            os.path.abspath(args.out_dir)
            if args.out_dir is not None
            else (os.path.dirname(csv_in) or ".")
        )
        os.makedirs(out_dir, exist_ok=True)
        try:
            names, win_mat, episodes = _rebuild_win_matrix_from_csv(csv_in)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)
        h1 = os.path.join(out_dir, "pairwise_winrate_heatmap.png")
        h2 = os.path.join(out_dir, "pairwise_row_mean_bar.png")
        h3 = os.path.join(out_dir, "pairwise_table.png")
        _plot_heatmap(names, win_mat, h1, episodes)
        _plot_row_mean(names, win_mat, h2, episodes)
        _plot_table_figure(names, win_mat, h3, episodes)
        print("已从 CSV 重画图（未重新评估）。")
        print(f"CSV: {csv_in}")
        print(f"热力图: {h1}")
        print(f"平均胜率柱图: {h2}")
        print(f"表格图: {h3}")
        return

    out_dir = (
        os.path.abspath(args.out_dir)
        if args.out_dir is not None
        else os.path.abspath("artifacts/model_pairwise")
    )

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    models = _discover_checkpoints(args.model_dir)
    if len(models) < 2:
        print(
            f"需要至少 2 个 .pt，当前目录 {os.path.abspath(args.model_dir)!r} 中只有 {len(models)} 个。",
            file=sys.stderr,
        )
        sys.exit(1)

    names = [m[0] for m in models]
    paths = [m[1] for m in models]
    n = len(names)
    os.makedirs(out_dir, exist_ok=True)
    tact = bool(args.tactical_inference)

    print(
        f"模型数={n}，有序对数={n * (n - 1)}，每对 {args.episodes} 局，"
        f"总对局数≈{n * (n - 1) * args.episodes}；设备={device}；输出={out_dir}",
        flush=True,
    )

    policy_cache: Dict[str, object] = {}
    win_mat = np.zeros((n, n), dtype=float)
    csv_rows: List[dict] = []
    pair_idx = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pair_idx += 1
            seed_off = pair_idx * 10_000
            print(
                f"[{pair_idx}/{n * (n - 1)}] {names[i]} (agent) vs {names[j]} (opp) …",
                flush=True,
            )
            w, l, d = _eval_ordered_pair(
                paths[i],
                paths[j],
                policy_cache,
                device,
                args.episodes,
                use_tactical=tact,
                seed_offset=seed_off,
            )
            wr = w / (w + l + d) if (w + l + d) else 0.0
            win_mat[i, j] = wr
            csv_rows.append(
                {
                    "agent": names[i],
                    "opponent": names[j],
                    "wins": w,
                    "losses": l,
                    "draws": d,
                    "games": w + l + d,
                    "agent_win_rate": f"{wr:.6f}",
                }
            )
            print(f"    -> W {w} L {l} D {d}  win%={100 * wr:.1f}", flush=True)

    csv_path = os.path.join(out_dir, "pairwise_results.csv")
    _save_csv(csv_rows, csv_path)
    h1 = os.path.join(out_dir, "pairwise_winrate_heatmap.png")
    h2 = os.path.join(out_dir, "pairwise_row_mean_bar.png")
    h3 = os.path.join(out_dir, "pairwise_table.png")
    _plot_heatmap(names, win_mat, h1, args.episodes)
    _plot_row_mean(names, win_mat, h2, args.episodes)
    _plot_table_figure(names, win_mat, h3, args.episodes)

    print("\n完成。")
    print(f"CSV: {csv_path}")
    print(f"热力图: {h1}")
    print(f"平均胜率柱图: {h2}")
    print(f"表格图: {h3}")


if __name__ == "__main__":
    main()
