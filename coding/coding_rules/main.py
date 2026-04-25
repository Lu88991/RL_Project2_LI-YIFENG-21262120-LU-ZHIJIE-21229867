from __future__ import annotations

import sys

from training_opponent import TrainOpponentConfig, build_train_opponent_argv

# ============================ 配置区（只改这里）============================




# 无子命令运行 main 时：只改这一行
#   RUN_MODE = 1 -> train
#   RUN_MODE = 2 -> play



RUN_MODE = 2




# --- 训练（RUN_MODE == "train" 时生效）---
TRAIN_EPISODES = 750          # 训练总对局数（想训几盘就写几）
TRAIN_OUT_DIR = "artifacts"    # 模型与 training_curve.png 保存目录

# 训练对手（独立配置；会生成 --opponent / --curriculum-* / --self-play-* 等参数）
# mode 可选：
#   random | heuristic | mixed | self_snapshot | curriculum
# curriculum：先随机 curriculum_switch_episodes 局，再切到 after_curriculum_opponent
# self_snapshot：对手为「上一快照」的自己（由 train.py 定期从当前 policy 复制权重）
TRAIN_OPPONENT = TrainOpponentConfig(
    mode="random",
    curriculum_switch_episodes=250,
    after_curriculum_opponent="mixed",
    heuristic_prob=0.7,
    self_play_sync_iters=10,
    self_play_opponent_tactical=True,
)

# 其它训练超参（若与上一段同名，以这里为准——写在列表更靠后的位置）
TRAIN_EXTRA_ARGS: list[str] = [
    "--min-buffer",
    "500",
    "--rollout-steps",
    "64",
    "--batch",
    "64",
    "--optim-steps",
    "1",
    "--eps-start",
    "0.6",
    "--eps-anneal-steps",
    "900",
    "--eval-every",
    "0",
]






# --- 测试 / 人机（RUN_MODE == "play" 时生效）---
PLAY_MODEL = "artifacts/best_model.pt"
# PLAY_MODE = "eval_random"     # eval_random | ai_vs_ai | human
PLAY_MODE = "human"
PLAY_EPISODES = 50            # 连续评估多少盘（与训练局数无关）
PLAY_EXTRA_ARGS: list[str] = []





# ===========================================================================







def _running_in_ipython() -> bool:
    try:
        get_ipython()  # type: ignore[name-defined]
        return True
    except NameError:
        return False


def _exit_or_return(code: int = 1) -> None:
    """在 Jupyter 里不要用 sys.exit，否则会触发 SystemExit 红字异常。"""
    if _running_in_ipython():
        return
    raise SystemExit(code)


def _strip_jupyter_kernel_argv(argv: list) -> list:
    """Jupyter/IPython 会把 `-f` / `--f=...`（内核连接文件）塞进 argv，这里先去掉再解析。"""
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


def _print_usage() -> None:
    print(
        "用法:\n"
        "  python main.py              # 默认 = train\n"
        "  python main.py train [...]  # 训练（参数见 train.py）\n"
        "  python main.py play [...]   # 评估 / 人机（参数见 play.py）\n"
        "  python main.py help         # 显示本说明\n"
        "\n"
        "示例:\n"
        "  python main.py train --episodes 2000 --out-dir artifacts\n"
        "  python main.py play --model artifacts/best_model.pt --mode human\n",
        file=sys.stderr,
    )


def main() -> None:
    sys.argv[:] = _strip_jupyter_kernel_argv(sys.argv)

    if len(sys.argv) < 2:
        script = sys.argv[0]
        if RUN_MODE == 2:
            sys.argv = [
                script,
                "play",
                "--model",
                PLAY_MODEL,
                "--mode",
                PLAY_MODE,
                "--episodes",
                str(PLAY_EPISODES),
                *PLAY_EXTRA_ARGS,
            ]
        else:
            sys.argv = [
                script,
                "train",
                "--episodes",
                str(TRAIN_EPISODES),
                "--out-dir",
                TRAIN_OUT_DIR,
                *build_train_opponent_argv(TRAIN_OPPONENT),
                *TRAIN_EXTRA_ARGS,
            ]
        print(
            "（未传子命令：已使用 RUN_MODE；1=train，2=play）\n",
            file=sys.stderr,
        )

    cmd = sys.argv.pop(1)
    if cmd in ("help", "-h", "--help"):
        _print_usage()
        return
    if cmd == "train":
        from train import main as train_main

        train_main()
    elif cmd == "play":
        from play import main as play_main

        play_main()
    else:
        print(f"未知子命令: {cmd!r}，请使用 train 或 play。", file=sys.stderr)
        _exit_or_return(1)


if __name__ == "__main__":
    main()
