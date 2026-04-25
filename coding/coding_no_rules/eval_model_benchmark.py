#!/usr/bin/env python3
"""
批量评估 model/ 目录下的 checkpoint：每个模型对局若干盘，输出 CSV + 表格图。

用法:
  python eval_model_benchmark.py
  python eval_model_benchmark.py --model-dir model --episodes 100 --out-dir artifacts/model_benchmark
  python eval_model_benchmark.py --plot-only --out-dir artifacts/model_benchmark
  # plot-only 需与生成 CSV 时相同的 --opponent；局数从 CSV 每行 games 读取
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 轴标签含中文时避免 DejaVu 缺字（macOS 常见字体优先）
plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Heiti TC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import torch

from environment import SuperTicTacToeEnv
from model import build_policy_pair, greedy_action_masked, make_rl_env

# 单局最多环境步数（防止异常局面下 while 死循环；正常一局远小于此值）
_MAX_STEPS_PER_EPISODE = 512

# 与 eval_model_pairwise._PLOT_MODEL_LABELS 一致：stem 为简称或与 best_model_N 相同的末尾编号时映射图表简称
_BENCHMARK_PLOT_LABELS: dict[int, str] = {
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

_BENCHMARK_STEM_TO_ID: dict[str, int] = {v: k for k, v in _BENCHMARK_PLOT_LABELS.items()}


def _benchmark_model_id_from_stem(stem: str) -> int | None:
    """与 eval_model_pairwise._plot_model_id_from_stem 相同逻辑。"""
    s = stem.strip()
    if s in _BENCHMARK_STEM_TO_ID:
        return _BENCHMARK_STEM_TO_ID[s]
    m = re.search(r"(\d+)$", s)
    if m:
        return int(m.group(1))
    parts = re.findall(r"\d+", s)
    if parts:
        return int(parts[-1])
    return None


def _strip_jupyter_kernel_argv(argv: list) -> list:
    """Jupyter/IPython 会把 `-f` / `--f=...` 塞进 argv，解析前先去掉。"""
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


@dataclass
class EvalRow:
    name: str
    path: str
    wins: int
    losses: int
    draws: int

    @property
    def n(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n else 0.0


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


def _eval_one_model(
    ckpt_path: str,
    device: torch.device,
    episodes: int,
    opponent: str,
    opponent_heuristic_prob: float,
    *,
    use_tactical: bool,
) -> Tuple[int, int, int]:
    policy = _load_policy(ckpt_path, device)
    wins = losses = draws = 0
    for ep in range(episodes):
        env = SuperTicTacToeEnv(
            seed=ep,
            opponent=opponent,
            opponent_heuristic_prob=opponent_heuristic_prob,
        )
        obs, _ = env.reset(seed=ep)
        steps = 0
        while True:
            steps += 1
            if steps > _MAX_STEPS_PER_EPISODE:
                draws += 1
                break
            m = env.legal_actions_mask()
            a = greedy_action_masked(
                policy, obs, m, device, use_tactical=use_tactical
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


def _discover_checkpoints(model_dir: str) -> List[str]:
    if not os.path.isdir(model_dir):
        return []
    files = []
    for fn in os.listdir(model_dir):
        if fn.endswith(".pt"):
            files.append(os.path.join(model_dir, fn))

    def _sort_key(p: str) -> tuple:
        stem = os.path.basename(p).replace(".pt", "")
        mid = _benchmark_model_id_from_stem(stem)
        return (mid is None, mid if mid is not None else 10**9, stem)

    return sorted(files, key=_sort_key)


def _display_label(stored_name: str) -> str:
    mid = _benchmark_model_id_from_stem(stored_name)
    if mid is not None and mid in _BENCHMARK_PLOT_LABELS:
        return _BENCHMARK_PLOT_LABELS[mid]
    return (stored_name[:14] if stored_name else "?") or "?"


def _load_csv(path: str) -> List[EvalRow]:
    rows: List[EvalRow] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            name = (row.get("model") or "").strip()
            p = (row.get("path") or "").strip()
            w = int(row["wins"])
            l = int(row["losses"])
            d = int(row["draws"])
            rows.append(EvalRow(name=name, path=p, wins=w, losses=l, draws=d))
    return rows


def _rows_for_plot(rows: List[EvalRow]) -> List[EvalRow]:
    """复制行并把 name 换成图表用显示名（不改 CSV 磁盘内容）。"""
    out: List[EvalRow] = []
    for r in rows:
        label = _display_label(r.name)
        out.append(
            EvalRow(name=label, path=r.path, wins=r.wins, losses=r.losses, draws=r.draws)
        )
    return out


def _save_csv(rows: List[EvalRow], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "path", "wins", "losses", "draws", "games", "win_rate"])
        for r in rows:
            w.writerow(
                [
                    r.name,
                    r.path,
                    r.wins,
                    r.losses,
                    r.draws,
                    r.n,
                    f"{r.win_rate:.4f}",
                ]
            )


def _plot_charts(
    rows: List[EvalRow], out_dir: str, opponent: str, episodes: int, *, use_display_names: bool
) -> None:
    if use_display_names:
        rows = _rows_for_plot(rows)
    names = [r.name for r in rows]
    x = np.arange(len(names))
    wins = np.array([r.wins for r in rows], dtype=float)
    losses = np.array([r.losses for r in rows], dtype=float)
    draws = np.array([r.draws for r in rows], dtype=float)
    win_rates = np.array([r.win_rate for r in rows])

    # 1) Win rate bar
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.9), 4.5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(names)))
    ax.bar(x, win_rates, color=colors, edgecolor="#334155", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Win rate")
    ax.set_ylim(0, 1.05)
    total = wins.sum() + losses.sum() + draws.sum()
    if total > 0:
        ax.axhline(
            y=wins.sum() / total,
            color="gray",
            ls="--",
            lw=0.8,
            label="overall win rate",
        )
    ax.set_title(f"Win rate per model (n={episodes} vs {opponent})")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    p1 = os.path.join(out_dir, "benchmark_win_rate.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # 2) Stacked W / L / D
    fig2, ax2 = plt.subplots(figsize=(max(10, len(names) * 0.9), 5))
    ax2.bar(x, wins, label="Wins", color="#22c55e", edgecolor="#14532d", linewidth=0.5)
    ax2.bar(x, losses, bottom=wins, label="Losses", color="#ef4444", edgecolor="#7f1d1d", linewidth=0.5)
    ax2.bar(x, draws, bottom=wins + losses, label="Draws", color="#94a3b8", edgecolor="#334155", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax2.set_ylabel(f"Games (out of {episodes})")
    ax2.set_title(f"Wins / Losses / Draws per model vs {opponent}")
    ax2.legend(loc="upper right")
    ax2.set_ylim(0, episodes + 0.5)
    fig2.tight_layout()
    p2 = os.path.join(out_dir, "benchmark_wld_stacked.png")
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)

    # 3) Table as figure
    fig3, ax3 = plt.subplots(figsize=(max(8, len(names) * 0.35), 0.5 + len(names) * 0.35))
    ax3.axis("off")
    table_data = [
        ["model", "W", "L", "D", "win%"],
    ]
    for r in rows:
        table_data.append(
            [r.name, str(r.wins), str(r.losses), str(r.draws), f"{100*r.win_rate:.1f}"]
        )
    tbl = ax3.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colWidths=[0.42, 0.1, 0.1, 0.1, 0.12],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.8)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1e293b")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#f8fafc" if row % 2 else "#e2e8f0")
    ax3.set_title(f"Summary (vs {opponent}, {episodes} games each)", fontsize=11, pad=12)
    p3 = os.path.join(out_dir, "benchmark_table.png")
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig3)

    return p1, p2, p3


def _print_table(rows: List[EvalRow]) -> None:
    col_w = max(8, max(len(r.name) for r in rows) + 2, 12)
    hdr = f"{'model':<{col_w}} {'W':>5} {'L':>5} {'D':>5} {'win%':>8}"
    print("\n" + "=" * (col_w + 28))
    print(hdr)
    print("-" * (col_w + 28))
    for r in rows:
        print(
            f"{r.name:<{col_w}} {r.wins:5d} {r.losses:5d} {r.draws:5d} {100*r.win_rate:8.2f}"
        )
    print("=" * (col_w + 28) + "\n")


def main() -> None:
    sys.argv[:] = _strip_jupyter_kernel_argv(sys.argv)
    p = argparse.ArgumentParser(description="批量评估 model/ 下各 .pt 对同一对手")
    p.add_argument("--model-dir", type=str, default="model", help="存放 .pt 的目录")
    p.add_argument("--episodes", type=int, default=100, help="每个模型评估局数")
    p.add_argument(
        "--opponent",
        type=str,
        default="random",
        choices=["random", "heuristic", "mixed"],
        help="评估对手类型",
    )
    p.add_argument("--opponent-heuristic-prob", type=float, default=0.7)
    p.add_argument("--out-dir", type=str, default="artifacts/model_benchmark", help="CSV 与图输出目录")
    p.add_argument(
        "--csv",
        type=str,
        default="",
        help="plot-only 时读取的 CSV；默认 <out-dir>/benchmark_results.csv",
    )
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="不跑评估，仅从已有 CSV 重画三张图（需与生成 CSV 时相同的 --opponent / --episodes）",
    )
    p.add_argument(
        "--no-display-names",
        action="store_true",
        help="图表仍用 CSV 中的 model 列，不套用与 pairwise 相同的编号→简称",
    )
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--tactical-inference",
        action="store_true",
        help="评估时启用必胜/必堵（与 play --tactical-inference 一致）",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    use_labels = not bool(args.no_display_names)

    if args.plot_only:
        csv_path = args.csv.strip() or os.path.join(args.out_dir, "benchmark_results.csv")
        if not os.path.isfile(csv_path):
            print(f"未找到 CSV: {os.path.abspath(csv_path)}", file=sys.stderr)
            sys.exit(1)
        rows = _load_csv(csv_path)
        if not rows:
            print(f"CSV 无数据行: {csv_path}", file=sys.stderr)
            sys.exit(1)
        # 标题中的局数以 CSV 为准（各模型 games 应一致）
        ep = rows[0].n if rows[0].n > 0 else int(args.episodes)
        p1, p2, p3 = _plot_charts(
            rows, args.out_dir, args.opponent, ep, use_display_names=use_labels
        )
        _print_table(_rows_for_plot(rows) if use_labels else rows)
        print(f"（plot-only）自 CSV: {os.path.abspath(csv_path)}")
        print(f"图1: {p1}")
        print(f"图2: {p2}")
        print(f"图3: {p3}")
        return

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    paths = _discover_checkpoints(args.model_dir)
    if not paths:
        print(
            f"未在 {os.path.abspath(args.model_dir)!r} 找到任何 .pt 文件。\n"
            "请将 11 个训练好的模型放入该目录后再运行。",
            file=sys.stderr,
        )
        sys.exit(1)

    rows: List[EvalRow] = []
    print(
        f"发现 {len(paths)} 个 checkpoint；设备={device}；对手={args.opponent}；"
        f"每模型 {args.episodes} 局；输出目录={os.path.abspath(args.out_dir)}",
        flush=True,
    )

    for i, ckpt in enumerate(paths, 1):
        name = os.path.basename(ckpt).replace(".pt", "")
        print(f"[{i}/{len(paths)}] 评估 {name} …", flush=True)
        w, l, d = _eval_one_model(
            ckpt,
            device,
            args.episodes,
            args.opponent,
            args.opponent_heuristic_prob,
            use_tactical=bool(args.tactical_inference),
        )
        rows.append(EvalRow(name=name, path=os.path.abspath(ckpt), wins=w, losses=l, draws=d))
        print(f"    -> W {w} L {l} D {d}  (win% {100*w/(w+l+d):.1f})", flush=True)

    csv_path = os.path.join(args.out_dir, "benchmark_results.csv")
    _save_csv(rows, csv_path)
    p1, p2, p3 = _plot_charts(
        rows, args.out_dir, args.opponent, args.episodes, use_display_names=use_labels
    )

    _print_table(_rows_for_plot(rows) if use_labels else rows)
    print(f"CSV: {csv_path}")
    print(f"图1: {p1}")
    print(f"图2: {p2}")
    print(f"图3: {p3}")


if __name__ == "__main__":
    main()
