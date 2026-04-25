"""
Super Tic-Tac-Toe (6×4×4 pyramid) — Gymnasium environment.

Rules (assignment):
- 6 boards of 4×4, global indices 0..95 (board = idx // 16, local = idx % 16).
- Win: (1) horizontal 4 within a 4×4 board; (2) vertical 4 on global grid with
  cells not all in the same 4-row band (cross-level); (3) diagonal 5 on global grid.
- Stochastic move: 50% chosen cell; else uniform among 8 neighbors within the same
  4×4; out-of-board or occupied → move forfeited for that player.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Callable, List, Optional, Tuple

# Optional: opponent_policy(obs_flipped_96, legal_mask_96) -> action int
OpponentPolicy = Optional[Callable[[np.ndarray, np.ndarray], int]]

EMPTY = 0
AGENT = 1
OPPONENT = 2

# Map board index + local (r,c) → global 12×12 pyramid layout (top view)
def _board_to_global(board: int, r: int, c: int) -> Tuple[int, int]:
    # Boards are shifted horizontally by level so that upper levels overlap lower
    # levels in projection:
    # - level 1: board 0 at col 4
    # - level 2: boards 1/2 at col 2/6
    # - level 3: boards 3/4/5 at col 0/4/8
    # This makes cross-level vertical / diagonal lines reachable.
    if board == 0:
        return r, c + 4
    if board == 1:
        return r + 4, c + 2
    if board == 2:
        return r + 4, c + 6
    if board == 3:
        return r + 8, c
    if board == 4:
        return r + 8, c + 4
    if board == 5:
        return r + 8, c + 8
    raise ValueError(f"invalid board {board}")


def global_to_board_index(g: int) -> Tuple[int, int, int]:
    """Global linear index 0..95 → (board, r, c)."""
    board = g // 16
    loc = g % 16
    r, c = loc // 4, loc % 4
    return board, r, c


def _build_global_coord_map() -> Tuple[np.ndarray, np.ndarray]:
    """Linear index → (gr, gc) on 12×12; inverse sparse map."""
    gr = np.zeros(96, dtype=np.int32)
    gc = np.zeros(96, dtype=np.int32)
    for g in range(96):
        b, r, c = global_to_board_index(g)
        gr[g], gc[g] = _board_to_global(b, r, c)
    return gr, gc


_GR, _GC = _build_global_coord_map()
_INV_COORD = {(int(_GR[g]), int(_GC[g])): g for g in range(96)}


def _build_winning_lines() -> List[List[int]]:
    inv: dict = {}
    for g in range(96):
        inv[(_GR[g], _GC[g])] = g

    lines: List[List[int]] = []

    # (1) Horizontal on global projection: 4 consecutive cols in same row.
    # This supports both within-board and cross-board horizontal wins.
    for row in range(12):
        for start in range(9):  # start + 3 <= 11
            cols = [start + k for k in range(4)]
            cells: List[int] = []
            ok = True
            for c in cols:
                key = (row, c)
                if key not in inv:
                    ok = False
                    break
                cells.append(inv[key])
            if ok:
                lines.append(cells)

    # (2) Vertical on global grid: 4 consecutive rows, same column; must span ≥2 row bands
    for col in range(12):
        for start in range(9):
            rows = [start + k for k in range(4)]
            if len({r // 4 for r in rows}) < 2:
                continue
            cells: List[int] = []
            ok = True
            for r in rows:
                key = (r, col)
                if key not in inv:
                    ok = False
                    break
                cells.append(inv[key])
            if ok:
                lines.append(cells)

    # (3) Diagonal: 5 in a row on global grid
    for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        for gr in range(12):
            for gcc in range(12):
                pts = [(gr + k * dr, gcc + k * dc) for k in range(5)]
                if any(p[0] < 0 or p[0] >= 12 or p[1] < 0 or p[1] >= 12 for p in pts):
                    continue
                if all(p in inv for p in pts):
                    lines.append([inv[p] for p in pts])

    seen = set()
    uniq: List[List[int]] = []
    for ln in lines:
        t = tuple(ln)
        if t not in seen:
            seen.add(t)
            uniq.append(ln)
    return uniq


WIN_LINES: List[List[int]] = _build_winning_lines()


# 8 directions (including diagonals); assignment: each direction has unconditional prob 1/16,
# chosen cell has 1/2; total 1/2 + 8/16 = 1.
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _stochastic_cell(
    chosen: int, board_state: np.ndarray, rng: np.random.Generator
) -> Tuple[Optional[int], dict]:
    """
    Return (actual_cell_or_None, meta).
    meta contains:
      - intended_idx / intended_coord
      - final_idx / final_coord (if placed)
      - out_of_board / occupied_target / used_neighbor
    """
    meta = {
        "intended_idx": int(chosen),
        "intended_coord": (int(_GR[chosen]), int(_GC[chosen])),
        "used_neighbor": False,
        "out_of_board": False,
        "occupied_target": False,
        "final_idx": None,
        "final_coord": None,
    }
    if rng.random() < 0.5:
        if board_state[chosen] != EMPTY:
            meta["occupied_target"] = True
            return None, meta
        meta["final_idx"] = int(chosen)
        meta["final_coord"] = (int(_GR[chosen]), int(_GC[chosen]))
        return chosen, meta
    gr = int(_GR[chosen])
    gc = int(_GC[chosen])
    meta["used_neighbor"] = True
    dr, dc = _DIRS8[int(rng.integers(0, 8))]
    ncoord = (gr + dr, gc + dc)
    # Boundary rule on the WHOLE pyramid projection:
    # if chosen adjacent coordinate is not a valid cell, move is forfeited.
    if ncoord not in _INV_COORD:
        meta["out_of_board"] = True
        return None, meta
    actual = _INV_COORD[ncoord]
    if board_state[actual] != EMPTY:
        meta["occupied_target"] = True
        return None, meta
    meta["final_idx"] = int(actual)
    meta["final_coord"] = (int(_GR[actual]), int(_GC[actual]))
    return actual, meta


def _check_winner(board: np.ndarray, player: int) -> bool:
    for line in WIN_LINES:
        if all(board[i] == player for i in line):
            return True
    return False


def _strategic_line_score(board: np.ndarray, player: int) -> float:
    """Potential-based score: reward making own open lines, penalize weak positions."""
    other = OPPONENT if player == AGENT else AGENT
    # Stronger emphasis on offensive continuation (3-in-line and 4-in-line)
    # so the agent prefers converting advantages into wins.
    weights = {1: 0.02, 2: 0.10, 3: 0.45, 4: 1.60, 5: 2.20}
    s = 0.0
    for line in WIN_LINES:
        vals = board[line]
        if np.any(vals == other):
            continue
        cnt = int(np.sum(vals == player))
        if cnt > 0:
            s += weights.get(cnt, 0.0)
    return s


def _immediate_winning_actions(board: np.ndarray, player: int) -> List[int]:
    """Deterministic one-step winning actions (ignores stochastic landing deviation)."""
    out: List[int] = []
    legal = np.where(board == EMPTY)[0]
    for a in legal:
        b2 = board.copy()
        b2[a] = player
        if _check_winner(b2, player):
            out.append(int(a))
    return out


# Reward shaping constants
WIN_REWARD = 3.0
LOSE_REWARD = -3.0
ILLEGAL_REWARD = -0.2
FORFEIT_REWARD = -0.1
THREAT_REDUCE_BONUS = 0.1
THREAT_EXIST_PENALTY = 0.08
# If there is an immediate winning move, choosing it should be strongly preferred.
# （方案 B）略加强信号：推理侧不再硬编码必胜/必堵，靠 shaping 引导 Q 学会。
WIN_INTENT_BONUS = 1.15
MISS_WIN_PENALTY = 1.35
# If opponent has an immediate winning move, choosing a blocking action matters a lot.
BLOCK_THREAT_BONUS = 0.85
MISS_BLOCK_PENALTY = 1.15
# Encourage creating immediate winning threats for next turn.
CREATE_WIN_THREAT_BONUS = 0.12
SHAPING_LAMBDA = 0.25
SHAPING_GAMMA = 0.99


class SuperTicTacToeEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        opponent: str = "random",
        seed: Optional[int] = None,
        opponent_policy: OpponentPolicy = None,
        opponent_heuristic_prob: float = 0.7,
    ):
        super().__init__()
        self.opponent = opponent
        self.opponent_policy = opponent_policy
        self.opponent_heuristic_prob = float(opponent_heuristic_prob)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(96,), dtype=np.float32)
        self.action_space = spaces.Discrete(96)
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._board = np.zeros(96, dtype=np.int8)
        self._done = False
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        o = np.zeros(96, dtype=np.float32)
        o[self._board == AGENT] = 1.0
        o[self._board == OPPONENT] = -1.0
        return o

    def _random_opponent_action(self) -> int:
        empty = np.where(self._board == EMPTY)[0]
        if len(empty) == 0:
            return 0
        return int(self._rng.choice(empty))

    def _heuristic_opponent_action(self) -> int:
        # 1) win now if possible
        wins = _immediate_winning_actions(self._board, OPPONENT)
        if wins:
            return int(self._rng.choice(wins))
        # 2) block agent immediate win
        blocks = _immediate_winning_actions(self._board, AGENT)
        if blocks:
            return int(self._rng.choice(blocks))
        # 3) otherwise random legal
        return self._random_opponent_action()

    def _opponent_action(self) -> int:
        if self.opponent_policy is None:
            if self.opponent == "heuristic":
                return self._heuristic_opponent_action()
            if self.opponent == "mixed":
                if self._rng.random() < self.opponent_heuristic_prob:
                    return self._heuristic_opponent_action()
                return self._random_opponent_action()
            return self._random_opponent_action()
        obs = self._get_obs()
        obs_opp = -obs
        mask = self.legal_actions_mask()
        return int(self.opponent_policy(obs_opp, mask))

    def step(self, action: int):
        if self._done:
            return self._get_obs(), 0.0, True, False, {"reason": "already_done"}

        reward = 0.0
        info: dict = {}

        # Potential reference before agent action (for potential-based shaping).
        phi_before = _strategic_line_score(self._board, AGENT) - _strategic_line_score(self._board, OPPONENT)
        opp_threat_before = len(_immediate_winning_actions(self._board, OPPONENT))
        own_win_actions_before = _immediate_winning_actions(self._board, AGENT)
        own_threat_before = len(own_win_actions_before)

        # --- Agent (AGENT) move ---
        can_place = (0 <= action < 96) and (self._board[action] == EMPTY)
        if not can_place:
            # Illegal action is treated as a forfeited move (no immediate termination),
            # otherwise the agent is dominated by one-step short episodes.
            reward += ILLEGAL_REWARD
            info["illegal"] = True
        else:
            # If a deterministic immediate winning action exists, explicitly shape intent.
            if own_win_actions_before:
                if int(action) in own_win_actions_before:
                    reward += WIN_INTENT_BONUS
                else:
                    reward -= MISS_WIN_PENALTY
            elif opp_threat_before > 0:
                # No immediate own win; then blocking opponent immediate win is critical.
                opp_win_actions = _immediate_winning_actions(self._board, OPPONENT)
                if int(action) in opp_win_actions:
                    reward += BLOCK_THREAT_BONUS
                else:
                    reward -= MISS_BLOCK_PENALTY
            actual, move_meta = _stochastic_cell(action, self._board, self._rng)
            info["agent_move"] = move_meta
            if actual is None:
                reward += FORFEIT_REWARD
                info["forfeit"] = "agent"
            else:
                self._board[actual] = AGENT
                if _check_winner(self._board, AGENT):
                    self._done = True
                    return self._get_obs(), WIN_REWARD + reward, True, False, {"winner": "agent"}

        # Potential-based shaping should be credited to the agent action only
        # (before opponent moves).
        phi_after_agent = _strategic_line_score(self._board, AGENT) - _strategic_line_score(self._board, OPPONENT)
        reward += SHAPING_LAMBDA * (SHAPING_GAMMA * phi_after_agent - phi_before)
        opp_threat_after_agent = len(_immediate_winning_actions(self._board, OPPONENT))
        if opp_threat_after_agent < opp_threat_before:
            reward += THREAT_REDUCE_BONUS * float(opp_threat_before - opp_threat_after_agent)
        own_threat_after_agent = len(_immediate_winning_actions(self._board, AGENT))
        if own_threat_after_agent > own_threat_before:
            reward += CREATE_WIN_THREAT_BONUS * float(own_threat_after_agent - own_threat_before)
        # Penalize leaving immediate opponent winning threats unresolved.
        if opp_threat_after_agent > 0:
            reward -= THREAT_EXIST_PENALTY * float(opp_threat_after_agent)

        if np.all(self._board != EMPTY):
            self._done = True
            return self._get_obs(), reward, True, False, {"draw": True}

        # --- Opponent move ---
        opp_act = self._opponent_action()
        if self._board[opp_act] == EMPTY:
            o_actual, o_meta = _stochastic_cell(opp_act, self._board, self._rng)
            info["opponent_move"] = o_meta
            if o_actual is None:
                info["forfeit_opponent"] = True
            else:
                self._board[o_actual] = OPPONENT
                if _check_winner(self._board, OPPONENT):
                    self._done = True
                    return self._get_obs(), reward + LOSE_REWARD, True, False, {"winner": "opponent"}

        if np.all(self._board != EMPTY):
            self._done = True
            return self._get_obs(), reward, True, False, {"draw": True}

        return self._get_obs(), reward, False, False, info

    def render(self):
        lines = []
        lines.append("Super Tic-Tac-Toe (96 cells). Legend: . empty, O agent, X opp")
        for b in range(6):
            br, bc = b // 3, b % 3
            lines.append(f"  board {b}")
            for r in range(4):
                row = []
                for c in range(4):
                    g = b * 16 + r * 4 + c
                    v = self._board[g]
                    row.append("." if v == EMPTY else ("O" if v == AGENT else "X"))
                lines.append("    " + " ".join(row))
        return "\n".join(lines)

    def legal_actions_mask(self) -> np.ndarray:
        m = (self._board == EMPTY).astype(np.float32)
        return m


def register_super_tic_tac_toe() -> None:
    """注册到 Gymnasium，供 TorchRL 的 GymEnv(\"SuperTicTacToe-v0\") 使用。"""
    try:
        gym.spec("SuperTicTacToe-v0")
    except gym.error.Error:
        gym.register(id="SuperTicTacToe-v0", entry_point="environment:SuperTicTacToeEnv")


if __name__ == "__main__":
    import numpy as np

    e = SuperTicTacToeEnv(seed=0)
    obs, _ = e.reset()
    print("win lines count", len(WIN_LINES))
    for _ in range(3):
        a = int(np.random.choice(np.where(e._board == EMPTY)[0]))
        obs, r, term, trunc, inf = e.step(a)
        print(r, term, inf)
