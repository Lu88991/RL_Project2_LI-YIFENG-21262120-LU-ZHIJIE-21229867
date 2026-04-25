#!/usr/bin/env python3
"""加载 TorchRL 训练保存的 policy，评估、命令行人机、图形界面人机。"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
import sys
import tkinter as tk
from tkinter import messagebox
from typing import Optional

import numpy as np
import torch

from environment import SuperTicTacToeEnv
from model import build_policy_pair, greedy_action_masked, make_rl_env

# Global projection helpers (same layout as environment.py)
_BOARD_ORIGIN = {
    0: (0, 4),
    1: (4, 2),
    2: (4, 6),
    3: (8, 0),
    4: (8, 4),
    5: (8, 8),
}


def _idx_to_global_rc(idx: int) -> tuple[int, int]:
    b = idx // 16
    loc = idx % 16
    r, c = loc // 4, loc % 4
    br, bc = _BOARD_ORIGIN[b]
    return br + r, bc + c


def _global_rc_to_idx(gr: int, gc: int) -> Optional[int]:
    if not (0 <= gr < 12 and 0 <= gc < 12):
        return None
    for b, (br, bc) in _BOARD_ORIGIN.items():
        if br <= gr < br + 4 and bc <= gc < bc + 4:
            r = gr - br
            c = gc - bc
            return b * 16 + r * 4 + c
    return None


def _running_in_ipython() -> bool:
    try:
        get_ipython()  # type: ignore[name-defined]
        return True
    except NameError:
        return False


def _load_policy(path: str, device: torch.device):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if not ckpt.get("torchrl"):
        print(
            "该 checkpoint 不是 TorchRL 格式（缺少 torchrl=True）。请用当前 train.py 重新训练。",
            file=sys.stderr,
        )
        sys.exit(1)
    env = make_rl_env(device=torch.device("cpu"), opponent="random")
    policy, _, _ = build_policy_pair(env, eps_init=1.0, eps_end=0.05, annealing_num_steps=1)
    policy.load_state_dict(ckpt["policy"])
    policy.to(device)
    policy.eval()
    return policy


def _default_play_result_path() -> str:
    """固定写到「当前工作目录」下的 artifacts/play_result.txt，方便在工程里找。"""
    return os.path.abspath(os.path.join(os.getcwd(), "artifacts", "play_result.txt"))


def _result_path_for_model(model_path: str, explicit: Optional[str]) -> Optional[str]:
    if explicit == "":
        return None
    if explicit is not None:
        return os.path.abspath(explicit)
    return _default_play_result_path()


def _save_and_print_result(result_path: Optional[str], line: str) -> None:
    sep = "=" * 62
    banner = f"\n{sep}\n  PLAY 评估结果\n  {line}\n{sep}\n"
    print(banner, flush=True)
    if result_path:
        abs_path = os.path.abspath(result_path)
        parent = os.path.dirname(abs_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        block = f"[{stamp}]\n{line}\n"
        with open(abs_path, "a", encoding="utf-8") as f:
            f.write(block)
        print(
            f">>> 结果已保存到文件（用编辑器打开即可）：\n    {abs_path}\n",
            flush=True,
        )


def run_eval_vs_random(
    path: str,
    episodes: int,
    device: torch.device,
    result_path: Optional[str],
    *,
    use_tactical_inference: bool = False,
) -> None:
    print(f"\n>>> 开始评估：共 {episodes} 局，对手=随机合法落子 …\n", flush=True)
    policy = _load_policy(path, device)
    wins = losses = draws = 0
    for ep in range(episodes):
        env = SuperTicTacToeEnv(seed=ep)
        obs, _ = env.reset(seed=ep)
        while True:
            m = env.legal_actions_mask()
            a = greedy_action_masked(
                policy, obs, m, device, use_tactical=use_tactical_inference
            )
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                if info.get("winner") == "agent":
                    wins += 1
                elif info.get("winner") == "opponent":
                    losses += 1
                else:
                    draws += 1
                break
    line = f"eval {episodes} episodes vs random: W {wins} L {losses} D {draws}"
    _save_and_print_result(result_path, line)


def run_ai_vs_ai(
    path: str,
    episodes: int,
    device: torch.device,
    result_path: Optional[str],
    *,
    use_tactical_inference: bool = False,
) -> None:
    print(f"\n>>> 开始评估：共 {episodes} 局，AI vs AI（同一权重）…\n", flush=True)
    policy = _load_policy(path, device)

    def opp_policy(obs_opp: np.ndarray, mask: np.ndarray) -> int:
        return greedy_action_masked(
            policy, obs_opp, mask, device, use_tactical=use_tactical_inference
        )

    wins = losses = draws = 0
    for ep in range(episodes):
        env = SuperTicTacToeEnv(seed=ep, opponent_policy=opp_policy)
        obs, _ = env.reset(seed=ep)
        while True:
            m = env.legal_actions_mask()
            a = greedy_action_masked(
                policy, obs, m, device, use_tactical=use_tactical_inference
            )
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                if info.get("winner") == "agent":
                    wins += 1
                elif info.get("winner") == "opponent":
                    losses += 1
                else:
                    draws += 1
                break
    line = f"AI vs AI (same weights, flipped obs for P2): W {wins} L {losses} D {draws}"
    _save_and_print_result(result_path, line)


def run_human_cli(
    path: str, device: torch.device, *, use_tactical_inference: bool = False
) -> None:
    print(
        "\n（命令行人机模式不生成 play_result.txt；胜负在下方对局中可见。）\n",
        flush=True,
    )
    policy = _load_policy(path, device)

    def opp_policy(obs_opp: np.ndarray, mask: np.ndarray) -> int:
        return greedy_action_masked(
            policy, obs_opp, mask, device, use_tactical=use_tactical_inference
        )

    env = SuperTicTacToeEnv(seed=0, opponent_policy=opp_policy)
    obs, _ = env.reset(seed=0)
    print("You are O (agent). Enter global index 0-95, or 'b r c' with board 0-5 and r,c 0-3.")
    print(env.render())
    while True:
        m = env.legal_actions_mask()
        if not np.any(m > 0):
            print("Game over (draw).")
            break
        line = input("move> ").strip()
        if line.lower() in ("q", "quit", "exit"):
            break
        if line == "":
            continue
        parts = line.split()
        if len(parts) == 1:
            a = int(parts[0])
        elif len(parts) == 3:
            b, r, c = int(parts[0]), int(parts[1]), int(parts[2])
            a = b * 16 + r * 4 + c
        else:
            print("Expected: <int> or <board> <r> <c>")
            continue
        if a < 0 or a >= 96 or m[a] < 0.5:
            print("Illegal move.")
            continue
        obs, r, term, trunc, info = env.step(a)
        print(f"reward={r:.2f} info={info}")
        print(env.render())
        if term or trunc:
            break


class HumanVsAIGUI:
    """整体棋盘图形界面：12x12 投影（无子棋盘空隙）。"""

    def __init__(self, env: SuperTicTacToeEnv):
        self.env = env
        self.cell = 44
        self.grid_n = 12
        self.margin = 20
        self.board_px = self.cell * self.grid_n
        self.canvas_w = self.margin * 2 + self.board_px
        self.canvas_h = self.margin * 2 + self.board_px + 54

        self.root = tk.Tk()
        self.root.title("Super Tic-Tac-Toe 人机对弈（你=O，AI=X）")
        self.canvas = tk.Canvas(self.root, width=self.canvas_w, height=self.canvas_h, bg="#f6f7fb")
        self.canvas.pack()
        self.status_var = tk.StringVar(value="你的回合：点击任意空格落子")
        self.last_agent_move = None
        self.last_opponent_move = None
        status = tk.Label(self.root, textvariable=self.status_var, anchor="w", justify="left")
        status.pack(fill="x", padx=8, pady=6)
        self.canvas.bind("<Button-1>", self.on_click)
        self.draw()

    def _pixel_to_action(self, x: int, y: int):
        bx, by = self.margin, self.margin
        if not (bx <= x < bx + self.board_px and by <= y < by + self.board_px):
            return None
        gc = (x - bx) // self.cell
        gr = (y - by) // self.cell
        return _global_rc_to_idx(int(gr), int(gc))

    def draw(self):
        self.canvas.delete("all")
        bx, by = self.margin, self.margin
        valid = {_idx_to_global_rc(i) for i in range(96)}

        # Draw full 12x12 board as a single whole grid.
        self.canvas.create_rectangle(bx, by, bx + self.board_px, by + self.board_px, outline="#334155", width=2)
        for i in range(1, self.grid_n):
            self.canvas.create_line(bx + i * self.cell, by, bx + i * self.cell, by + self.board_px, fill="#cbd5e1")
            self.canvas.create_line(bx, by + i * self.cell, bx + self.board_px, by + i * self.cell, fill="#cbd5e1")

        # Shade invalid projection cells.
        for gr in range(self.grid_n):
            for gc in range(self.grid_n):
                if (gr, gc) in valid:
                    continue
                x0 = bx + gc * self.cell
                y0 = by + gr * self.cell
                x1 = x0 + self.cell
                y1 = y0 + self.cell
                self.canvas.create_rectangle(x0, y0, x1, y1, fill="#e5e7eb", outline="")

        # Draw visible cell borders for every valid cell (strong internal grid).
        for gr in range(self.grid_n):
            for gc in range(self.grid_n):
                if (gr, gc) not in valid:
                    continue
                x0 = bx + gc * self.cell
                y0 = by + gr * self.cell
                x1 = x0 + self.cell
                y1 = y0 + self.cell
                self.canvas.create_rectangle(x0, y0, x1, y1, outline="#64748b", width=1)

        # Draw stones on valid cells.
        for idx in range(96):
            v = int(self.env._board[idx])
            if v == 0:
                continue
            gr, gc = _idx_to_global_rc(idx)
            cx = bx + gc * self.cell + self.cell / 2
            cy = by + gr * self.cell + self.cell / 2
            if v == 1:
                self.canvas.create_oval(
                    cx - 12,
                    cy - 12,
                    cx + 12,
                    cy + 12,
                    outline="#2563eb",
                    width=3,
                )
            elif v == 2:
                self.canvas.create_line(cx - 12, cy - 12, cx + 12, cy + 12, fill="#ef4444", width=3)
                self.canvas.create_line(cx - 12, cy + 12, cx + 12, cy - 12, fill="#ef4444", width=3)

        # Visualize intended vs final landing point for the last human move.
        if self.last_agent_move:
            im = self.last_agent_move
            ic = im.get("intended_coord")
            fc = im.get("final_coord")
            if ic is not None:
                gr, gc = int(ic[0]), int(ic[1])
                cx = bx + gc * self.cell + self.cell / 2
                cy = by + gr * self.cell + self.cell / 2
                # yellow dashed box: intended target
                self.canvas.create_rectangle(
                    cx - 16,
                    cy - 16,
                    cx + 16,
                    cy + 16,
                    outline="#f59e0b",
                    width=2,
                    dash=(4, 3),
                )
            if fc is not None:
                gr, gc = int(fc[0]), int(fc[1])
                cx = bx + gc * self.cell + self.cell / 2
                cy = by + gr * self.cell + self.cell / 2
                # green ring: final placed location
                self.canvas.create_oval(
                    cx - 18,
                    cy - 18,
                    cx + 18,
                    cy + 18,
                    outline="#16a34a",
                    width=2,
                )
            if bool(im.get("out_of_board", False)):
                # explicit marker outside board
                tx = bx + self.board_px + 10
                ty = by + 16
                self.canvas.create_text(
                    tx,
                    ty,
                    text="↘ 落在棋盘外(作废)",
                    fill="#b91c1c",
                    anchor="w",
                    font=("Arial", 11, "bold"),
                )

        # Visualize AI(model) intended vs final move.
        if self.last_opponent_move:
            om = self.last_opponent_move
            ic = om.get("intended_coord")
            fc = om.get("final_coord")
            if ic is not None:
                gr, gc = int(ic[0]), int(ic[1])
                cx = bx + gc * self.cell + self.cell / 2
                cy = by + gr * self.cell + self.cell / 2
                # cyan dashed box: AI intended target
                self.canvas.create_rectangle(
                    cx - 12,
                    cy - 12,
                    cx + 12,
                    cy + 12,
                    outline="#06b6d4",
                    width=2,
                    dash=(2, 2),
                )
            if fc is not None:
                gr, gc = int(fc[0]), int(fc[1])
                cx = bx + gc * self.cell + self.cell / 2
                cy = by + gr * self.cell + self.cell / 2
                # purple ring: AI final placed location
                self.canvas.create_oval(
                    cx - 14,
                    cy - 14,
                    cx + 14,
                    cy + 14,
                    outline="#7c3aed",
                    width=2,
                )

        # Legend
        lx = bx + self.board_px + 10
        ly = by + 48
        self.canvas.create_text(lx, ly - 24, text="图例", fill="#111827", anchor="w", font=("Arial", 11, "bold"))
        legend_items = [
            ("○ 蓝色", "你的棋子"),
            ("× 红色", "AI 棋子"),
            ("黄虚线框", "你的计划落子点"),
            ("绿圆环", "你的最终落子点"),
            ("青虚线框", "AI 计划落子点"),
            ("紫圆环", "AI 最终落子点"),
            ("灰色格", "无效区域（不可落子）"),
        ]
        for i, (k, v) in enumerate(legend_items):
            self.canvas.create_text(
                lx,
                ly + i * 18,
                text=f"{k}: {v}",
                fill="#334155",
                anchor="w",
                font=("Arial", 10),
            )
        self.root.update_idletasks()

    def on_click(self, event):
        action = self._pixel_to_action(event.x, event.y)
        if action is None:
            return
        mask = self.env.legal_actions_mask()
        if mask[action] < 0.5:
            self.status_var.set("该格已占用或不可下，请点空格。")
            return
        obs, reward, term, trunc, info = self.env.step(action)
        self.last_agent_move = info.get("agent_move")
        self.last_opponent_move = info.get("opponent_move")
        self.draw()
        if self.last_agent_move:
            im = self.last_agent_move
            msg = (
                f"目标={im.get('intended_coord')} "
                f"最终={im.get('final_coord')} "
                f"{'棋盘外作废' if im.get('out_of_board') else ''}"
            )
            if self.last_opponent_move:
                om = self.last_opponent_move
                msg += (
                    f" | AI计划={om.get('intended_coord')} "
                    f"AI实际={om.get('final_coord')} "
                    f"{'AI棋盘外作废' if om.get('out_of_board') else ''}"
                )
            self.status_var.set(f"{msg} | reward={reward:.2f}")
        else:
            self.status_var.set(f"本步 reward={reward:.2f}，info={info}")
        if term or trunc:
            if info.get("winner") == "agent":
                msg = "你赢了！"
            elif info.get("winner") == "opponent":
                msg = "AI 赢了。"
            else:
                msg = "平局。"
            messagebox.showinfo("对局结束", f"{msg}\nreward={reward:.2f}\n{info}")
            self.canvas.unbind("<Button-1>")

    def run(self):
        self.root.mainloop()


def run_human_gui(
    path: str, device: torch.device, *, use_tactical_inference: bool = False
) -> None:
    if _running_in_ipython():
        print("检测到 Notebook 内核：已切换为 Jupyter 点击棋盘界面（ipywidgets）。", flush=True)
        run_human_widget(path, device, use_tactical_inference=use_tactical_inference)
        return

    policy = _load_policy(path, device)

    def opp_policy(obs_opp: np.ndarray, mask: np.ndarray) -> int:
        return greedy_action_masked(
            policy, obs_opp, mask, device, use_tactical=use_tactical_inference
        )

    env = SuperTicTacToeEnv(seed=0, opponent_policy=opp_policy)
    env.reset(seed=0)
    HumanVsAIGUI(env).run()


def run_human_widget(
    path: str, device: torch.device, *, use_tactical_inference: bool = False
) -> None:
    """Notebook 内的点击棋盘界面（避免 Tk 导致 kernel 崩溃）。"""
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except Exception:
        print(
            "未安装 ipywidgets，无法在 Notebook 显示点击界面。请安装后重试：\n"
            "  pip install ipywidgets",
            file=sys.stderr,
        )
        return

    policy = _load_policy(path, device)

    def opp_policy(obs_opp: np.ndarray, mask: np.ndarray) -> int:
        return greedy_action_masked(
            policy, obs_opp, mask, device, use_tactical=use_tactical_inference
        )

    env = SuperTicTacToeEnv(seed=0, opponent_policy=opp_policy)
    obs, _ = env.reset(seed=0)
    finished = {"done": False}

    status = widgets.HTML("<b>你的回合：点击任意空格（你=O，AI=X）</b>")
    legend = widgets.HTML(
        "<div style='font-size:12px;line-height:1.5'>"
        "<b>图例：</b>"
        "<span style='color:#2563eb'>○ 蓝色=你的棋子</span>；"
        "<span style='color:#ef4444'>× 红色=AI棋子</span>；"
        "<span style='color:#f59e0b'>黄框=你的计划落子点</span>；"
        "<span style='color:#16a34a'>绿框=你的最终落子点</span>；"
        "<span style='color:#0891b2'>青框=AI计划落子点</span>；"
        "<span style='color:#7c3aed'>紫框=AI最终落子点</span>；"
        "灰格=无效区域。"
        "</div>"
    )
    buttons = []
    btn_map = {}
    marker = {
        "intended_idx": None,
        "final_idx": None,
        "out_of_board": False,
        "opp_intended_idx": None,
        "opp_final_idx": None,
        "opp_out_of_board": False,
    }

    cells = []

    def cell_text(v: int) -> str:
        return "." if v == 0 else ("O" if v == 1 else "X")

    def rebuild():
        b = env._board  # noqa: SLF001
        for idx, btn in buttons:
            btn.description = cell_text(int(b[idx]))
            btn.disabled = finished["done"] or int(b[idx]) != 0
            btn.style.button_color = "#ffffff"
            btn.layout.border = "1px solid #94a3b8"

        ii = marker.get("intended_idx")
        fi = marker.get("final_idx")
        if ii is not None and ii in btn_map:
            btn_map[ii].style.button_color = "#fde68a"
            btn_map[ii].layout.border = "3px solid #f59e0b"
        if fi is not None and fi in btn_map:
            btn_map[fi].style.button_color = "#bbf7d0"
            btn_map[fi].layout.border = "3px solid #16a34a"
        if ii is not None and fi is not None and ii == fi and ii in btn_map:
            btn_map[ii].style.button_color = "#ddd6fe"
            btn_map[ii].layout.border = "3px solid #7c3aed"
        oi = marker.get("opp_intended_idx")
        of = marker.get("opp_final_idx")
        if oi is not None and oi in btn_map:
            btn_map[oi].style.button_color = "#a5f3fc"
            btn_map[oi].layout.border = "3px solid #0891b2"
        if of is not None and of in btn_map:
            btn_map[of].style.button_color = "#e9d5ff"
            btn_map[of].layout.border = "3px solid #7c3aed"

    def on_click(idx: int):
        nonlocal obs
        if finished["done"]:
            return
        m = env.legal_actions_mask()
        if idx < 0 or idx >= 96 or m[idx] < 0.5:
            status.value = "<b style='color:#b91c1c'>该位置不可下，请选择空格。</b>"
            return
        obs, reward, term, trunc, info = env.step(idx)
        am = info.get("agent_move", {})
        marker["intended_idx"] = am.get("intended_idx")
        marker["final_idx"] = am.get("final_idx")
        marker["out_of_board"] = bool(am.get("out_of_board", False))
        om = info.get("opponent_move", {})
        marker["opp_intended_idx"] = om.get("intended_idx")
        marker["opp_final_idx"] = om.get("final_idx")
        marker["opp_out_of_board"] = bool(om.get("out_of_board", False))
        rebuild()
        if am is not None:
            status.value = (
                "<b>"
                f"目标={am.get('intended_coord')}，最终={am.get('final_coord')}，"
                f"{'棋盘外作废，' if am.get('out_of_board') else ''}"
                f"AI计划={om.get('intended_coord')}，AI实际={om.get('final_coord')}，"
                f"{'AI棋盘外作废，' if om.get('out_of_board') else ''}"
                f"reward={reward:.2f}"
                "</b>"
            )
        else:
            status.value = f"<b>本步 reward={reward:.2f}，info={info}</b>"
        if term or trunc:
            finished["done"] = True
            if info.get("winner") == "agent":
                msg = "你赢了！"
            elif info.get("winner") == "opponent":
                msg = "AI 赢了。"
            else:
                msg = "平局。"
            status.value = (
                f"<b style='color:#065f46'>对局结束：{msg} "
                f"(reward={reward:.2f}, info={info})</b>"
            )
            rebuild()

    valid = {_idx_to_global_rc(i): i for i in range(96)}
    for gr in range(12):
        for gc in range(12):
            idx = valid.get((gr, gc))
            btn = widgets.Button(
                description="." if idx is not None else " ",
                layout=widgets.Layout(width="34px", height="32px"),
                button_style="",
                disabled=idx is None,
            )
            if idx is not None:
                btn.on_click(lambda _btn, i=idx: on_click(i))
                buttons.append((idx, btn))
                btn_map[idx] = btn
                btn.style.button_color = "#ffffff"
            else:
                btn.style.button_color = "#e5e7eb"
            btn.layout.border = "1px solid #94a3b8"
            cells.append(btn)

    rebuild()
    grid = widgets.GridBox(
        cells,
        layout=widgets.Layout(
            grid_template_columns="repeat(12, 34px)",
            grid_template_rows="repeat(12, 32px)",
            gap="0px",
            justify_content="center",
            margin="6px 0",
        ),
    )
    ui = widgets.VBox([status, legend, grid])
    display(ui)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="artifacts/best_model.pt")
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--mode",
        choices=["eval_random", "ai_vs_ai", "human", "human_cli", "human_gui"],
        default="eval_random",
    )
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument(
        "--result-file",
        type=str,
        default=None,
        metavar="PATH",
        help="评估摘要追加写入的路径；默认为 当前目录/artifacts/play_result.txt；设为 \"\" 则不写文件",
    )
    p.add_argument(
        "--tactical-inference",
        action="store_true",
        help="AI 走子启用必胜/必堵硬规则；默认关闭（纯 Q+合法掩码）。",
    )
    args = p.parse_args()

    if not os.path.isfile(args.model):
        print(f"Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    result_path = _result_path_for_model(args.model, args.result_file)
    if args.mode in ("eval_random", "ai_vs_ai") and result_path:
        print(
            f">>> 本次评估结束后，摘要会追加到：\n    {result_path}\n",
            flush=True,
        )

    tact = bool(args.tactical_inference)
    if args.mode == "eval_random":
        run_eval_vs_random(
            args.model, args.episodes, device, result_path, use_tactical_inference=tact
        )
    elif args.mode == "ai_vs_ai":
        run_ai_vs_ai(
            args.model, args.episodes, device, result_path, use_tactical_inference=tact
        )
    elif args.mode == "human_cli":
        run_human_cli(args.model, device, use_tactical_inference=tact)
    elif args.mode == "human":
        run_human_gui(args.model, device, use_tactical_inference=tact)
    else:
        run_human_gui(args.model, device, use_tactical_inference=tact)


if __name__ == "__main__":
    main()
