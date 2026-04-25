#!/usr/bin/env python3
"""
Generate a standalone English HTML report (intro, model, training, experiments with pictures/, …).

Usage:
  python generate_report.py
  python generate_report.py --out report.html

Edit TITLE / ORGANIZATION at the top of this file if you prefer not to use CLI. Author names are fixed in code.

---------------------------------------------------------------------------
后续扩展正文时请保持以下 HTML 结构一致（样式由 CSS 自动套用）
---------------------------------------------------------------------------
章节：每个大节 <section id="sec-..."> 内第一层标题用 <h2 class="sec">…</h2>；
小节 <h3>…</h3>，小节下再分用 <h4>…</h4>。

连续叙述：用普通 <p>…</p>（不要加额外 class）。会得到「顶条 + 全圆角」正文框并两端对齐。

多条例 / 并列规则（每条较长、需与正文同等待遇）：用
  <ul class="body-list"><li>…</li></ul>
会得到「虚线框 + 左侧点线」列表框并两端对齐。不要用无 class 的 <ul> 写长条规则。

占位、未完成：用 <p class="todo">…</p>（无正文框）。

代码：先可选 <p class="snippet-caption">…</p>，再接
  <pre class="code-snippet"><code>…</code></pre>
公式 / 单行等宽、不要两端对齐：用 <p class="mono">…</p>。

图注：用 <p class="figure-ref">…</p>。行内路径 / 标识符：<code class="mono">…</code>。

新增大节时：在 <nav class="toc"> 的 <ol class="toc-list"> 里按同样结构增加 <a class="toc-link">…</a>，
并同步更新 .toc-meta 里的节数；正文 id 与 href 保持一致。

不要直接改 report.html；改本文件后重新运行 generate_report.py 生成。
"""

from __future__ import annotations

import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Defaults — edit here or override via CLI
# ---------------------------------------------------------------------------
TITLE = (
    "Deep Reinforcement Learning for Stochastic Super Tic-Tac-Toe: "
    "A Dueling Double DQN Agent"
)
ORGANIZATION = ""  # optional; set to "" to hide affiliation line

# Author line (rendered as boxed name + student ID chips in the HTML header).
AUTHOR_TEAM: tuple[tuple[str, str], ...] = (
    ("Yifeng Li", "21262120"),
    ("Zhijie Lu", "21229867"),
)

# 写新 HTML 片段时参见文件顶部 docstring「后续扩展正文」一节。

def _escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _author_chips_markup() -> str:
    """Two boxed chips: each chip splits display name and student ID in nested cells."""
    parts: list[str] = []
    for name, sid in AUTHOR_TEAM:
        label = _escape(f"{name} {sid}")
        parts.append(
            f'<span class="author-chip" role="group" aria-label="{label}">'
            f'<span class="author-chip-cell author-chip-name">{_escape(name)}</span>'
            f'<span class="author-chip-cell author-chip-id">{_escape(sid)}</span>'
            f"</span>"
        )
    return "".join(parts)


# Abstract (placed above the table of contents in the generated HTML).
ABSTRACT_SECTION_HTML = """
    <section id="sec-abstract">
      <h2 class="sec">Abstract</h2>
      <div class="abstract-prose-box">
        <p>
          We document a <strong>Dueling Double DQN</strong> agent built with <strong>TorchRL</strong> for a stochastic
          <em>Super Tic-Tac-Toe</em> assignment: six 4×4 boards, non-standard global win lines, and noisy move execution.
          The report walks through the Gymnasium environment, the value network and action masking pipeline, the training
          loop (replay, soft target updates, opponent scheduling), and two parallel code paths&mdash;one with explicit
          tactical shaping and hard win/block inference (<code class="mono">coding_rules</code>), and one trained only on
          the plain reward (<code class="mono">coding_no_rules</code>).
        </p>
        <p>
          Empirically we compare <strong>eleven opponent curricula</strong> (training curves, pooled return charts, a
          random benchmark, and an all-pairs matrix at 100 games per pairing), run <strong>one-at-a-time sensitivity</strong>
          sweeps on learning rate, optimiser steps per rollout, epsilon anneal length, batch size, and discount factor
          (500 training episodes and 100 evaluation games per point), and close with <strong>qualitative board captures</strong>
          that contrast mandatory defensive moves against a free-form policy that still seeks its own winning geometry.
        </p>
      </div>
    </section>
"""


# Section 1 (Introduction): not an f-string — code snippets contain literal `{` / `}`.
INTRODUCTION_SECTION_HTML = """
    <section id="sec-intro">
      <h2 class="sec">1. Introduction and Problem Background</h2>

      <p>
        This project studies <strong>deep reinforcement learning</strong> for a course-defined variant of
        <em>Super Tic-Tac-Toe</em>: six independent 4×4 boards (96 cells), non-classical win conditions on a
        global 12×12 projection of a three-level pyramid, and <strong>stochastic move execution</strong> that
        can displace the agent&rsquo;s intended cell to a neighboring square or void the move entirely.
        The environment is implemented as a Gymnasium <code class="mono">Env</code> in
        <code class="mono">coding/coding_no_rules/environment.py</code>, which makes the assignment rules
        executable and provides the interface used by TorchRL training code.
      </p>

      <h3>1.1 Problem statement (assignment rules)</h3>
      <p>The specification encoded in the module docstring can be summarized as follows.</p>
      <ul class="body-list">
        <li><strong>Board layout.</strong> Six 4×4 boards; each global cell index 0&ndash;95 decomposes as
          <code class="mono">board = idx // 16</code>, <code class="mono">local = idx % 16</code>.</li>
        <li><strong>Winning patterns.</strong> A player wins if they complete any of:
          (1) four in a row <em>horizontally</em> within the projection (including within a single 4×4);
          (2) four in a row <em>vertically</em> on the global grid such that the four cells are <em>not</em>
          all in the same 4-row band (forcing cross-level vertical lines);
          (3) <em>five</em> in a row along a diagonal on the global grid.</li>
        <li><strong>Stochastic dynamics.</strong> When a player chooses a legal empty cell, with probability
          1/2 the stone is placed on that cell. Otherwise the environment picks uniformly one of eight
          neighbor directions (including diagonals) on the <em>same</em> 4×4 board; each direction has
          unconditional probability 1/16. If the resulting target is outside the valid pyramid footprint or
          already occupied, that player&rsquo;s move is <strong>forfeited</strong> (no stone placed).</li>
      </ul>
      <p>
        From an RL perspective this yields a sparse-reward, stochastic transition model with a discrete action
        space of 96 move indices. The rest of this section maps each rule cluster to the corresponding
        implementation fragment (file <code class="mono">environment.py</code> unless noted).
      </p>

      <h3>1.2 Mapping rules to code</h3>

      <h4>1.2.1 Global indexing: six boards, 96 cells</h4>
      <p>
        The assignment fixes a linear index over all cells. The code recovers board id and local coordinates
        with integer division and modulo, matching the problem statement.
      </p>
      <p class="snippet-caption">File: <code class="mono">environment.py</code> — index decomposition</p>
      <pre class="code-snippet"><code>def global_to_board_index(g: int) -&gt; Tuple[int, int, int]:
    \"\"\"Global linear index 0..95 → (board, r, c).\"\"\"
    board = g // 16
    loc = g % 16
    r, c = loc // 4, loc % 4
    return board, r, c</code></pre>

      <h4>1.2.2 Pyramid geometry and the 12×12 projection</h4>
      <p>
        Cross-board vertical and diagonal wins are defined on a shared global grid. The implementation maps
        each board-local coordinate to a unique <code class="mono">(row, col)</code> on a 12×12 lattice so that
        win-line enumeration can be expressed uniformly in global coordinates.
      </p>
      <p class="snippet-caption">File: <code class="mono">environment.py</code> — board placement on the projection</p>
      <pre class="code-snippet"><code>def _board_to_global(board: int, r: int, c: int) -&gt; Tuple[int, int]:
    if board == 0:
        return r, c + 4
    if board == 1:
        return r + 4, c + 2
    # ... boards 2–5 offset similarly ...
    raise ValueError(f"invalid board {board}")</code></pre>
      <p>
        At import time, <code class="mono">_build_global_coord_map</code> fills <code class="mono">_GR</code> and
        <code class="mono">_GC</code> for every linear index, and <code class="mono">_INV_COORD</code> reverses
        global coordinates back to a cell index for neighbor lookup during stochastic moves.
      </p>

      <h4>1.2.3 Win detection: horizontal four on the projection</h4>
      <p>
        Horizontal wins (including purely within-board lines) are generated by scanning each projection row
        for four consecutive columns that all correspond to real cells in <code class="mono">_INV_COORD</code>.
      </p>
      <p class="snippet-caption">File: <code class="mono">environment.py</code> — horizontal win lines</p>
      <pre class="code-snippet"><code># (1) Horizontal on global projection: 4 consecutive cols in same row.
for row in range(12):
    for start in range(9):
        cols = [start + k for k in range(4)]
        cells: List[int] = []
        # ... collect inv[(row, c)] for each c if present ...</code></pre>

      <h4>1.2.4 Win detection: vertical four with cross-level constraint</h4>
      <p>
        The assignment requires vertical lines of four to span at least two distinct 4-row bands
        (<code class="mono">row // 4</code>). Lines that stay within a single band are skipped, implementing the
        &ldquo;not all in the same 4-row band&rdquo; rule.
      </p>
      <p class="snippet-caption">File: <code class="mono">environment.py</code> — vertical win lines</p>
      <pre class="code-snippet"><code># (2) Vertical ... must span ≥2 row bands
for col in range(12):
    for start in range(9):
        rows = [start + k for k in range(4)]
        if len({r // 4 for r in rows}) &lt; 2:
            continue
        # ... build cells if all coordinates exist in inv ...</code></pre>

      <h4>1.2.5 Win detection: diagonal five</h4>
      <p>
        Diagonal wins use five aligned cells in any of the four diagonal directions; only segments whose
        endpoints lie inside the 12×12 grid and whose every cell exists in the pyramid footprint are kept.
      </p>
      <p class="snippet-caption">File: <code class="mono">environment.py</code> — diagonal-five lines</p>
      <pre class="code-snippet"><code># (3) Diagonal: 5 in a row on global grid
for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
    for gr in range(12):
        for gcc in range(12):
            pts = [(gr + k * dr, gcc + k * dc) for k in range(5)]
            # ... bounds check; require all(p in inv for p in pts) ...</code></pre>
      <p>
        All generated lines are deduplicated into <code class="mono">WIN_LINES</code>. At play time,
        <code class="mono">_check_winner</code> tests whether any line is monochromatic for the given player.
      </p>

      <h4>1.2.6 Stochastic placement: 1/2 on target, else 1/16 per neighbor direction</h4>
      <p>
        The docstring documents the probability split: 1/2 for the chosen cell, plus eight directions each with
        probability 1/16. The code uses Bernoulli <code class="mono">rng.random() &lt; 0.5</code> for the first
        branch and a discrete uniform draw over <code class="mono">_DIRS8</code> for the second.
      </p>
      <p class="snippet-caption">File: <code class="mono">environment.py</code> — direction set and random branch</p>
      <pre class="code-snippet"><code># 8 directions ... chosen cell has 1/2; total 1/2 + 8/16 = 1.
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

if rng.random() &lt; 0.5:
    # place on chosen cell if empty, else forfeit
    ...
else:
    dr, dc = _DIRS8[int(rng.integers(0, 8))]
    ncoord = (gr + dr, gc + dc)</code></pre>

      <h4>1.2.7 Forfeited moves: invalid neighbor or occupied target</h4>
      <p>
        If the stochastic neighbor step leaves the valid cell set (<code class="mono">ncoord not in _INV_COORD</code>)
        or lands on an occupied index, the function returns <code class="mono">None</code> and metadata flags
        <code class="mono">out_of_board</code> or <code class="mono">occupied_target</code>. The environment step
        then applies a small forfeit penalty for the agent instead of mutating the board for that ply.
      </p>
      <p class="snippet-caption">File: <code class="mono">environment.py</code> — forfeiture conditions</p>
      <pre class="code-snippet"><code>if ncoord not in _INV_COORD:
    meta["out_of_board"] = True
    return None, meta
actual = _INV_COORD[ncoord]
if board_state[actual] != EMPTY:
    meta["occupied_target"] = True
    return None, meta</code></pre>

      <h4>1.2.8 Gymnasium interface: observations and discrete actions</h4>
      <p>
        The RL agent consumes a length-96 vector: +1 for agent stones, −1 for opponent, 0 for empty. The
        action space is <code class="mono">Discrete(96)</code>; legality is exposed as a float mask over empty
        cells, which downstream code uses for masked Q-values.
      </p>
      <p class="snippet-caption">File: <code class="mono">environment.py</code> — spaces and observation</p>
      <pre class="code-snippet"><code>self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(96,), dtype=np.float32)
self.action_space = spaces.Discrete(96)

def _get_obs(self) -&gt; np.ndarray:
    o = np.zeros(96, dtype=np.float32)
    o[self._board == AGENT] = 1.0
    o[self._board == OPPONENT] = -1.0
    return o</code></pre>

      <p>
        Together, these pieces turn the prose rules into a reproducible simulator: global geometry defines
        which tactical lines exist, <code class="mono">WIN_LINES</code> materializes win tests, and
        <code class="mono">_stochastic_cell</code> injects the assignment&rsquo;s move noise before terminal
        checks and opponent replies in <code class="mono">step</code>.
      </p>
    </section>
"""


# Section 2 (Model): narrative order + formulas; not an f-string (code contains `{` / `}`).
MODEL_SECTION_HTML = """
    <section id="sec-model">
      <h2 class="sec">2. Model Architecture and Learning Objective</h2>

      <h3>2.1 TorchRL as the integration layer</h3>
      <p>
        The agent is trained in a <strong>TorchRL</strong> pipeline: the environment is wrapped as a
        <code class="mono">GymEnv</code>, transitions are stored in a replay buffer, and the policy is built
        from composable <strong>TensorDictModule</strong> blocks that read and write keys on a shared
        <code class="mono">TensorDict</code> (e.g. <code class="mono">observation</code>, <code class="mono">action_value</code>,
        <code class="mono">action_mask</code>). The actual function approximator is still a standard
        <strong>PyTorch</strong> <code class="mono">nn.Module</code>; TorchRL supplies the glue for exploration,
        discrete Q-value parsing, and vectorized rollouts.
      </p>
      <p class="snippet-caption">File: <code class="mono">coding/coding_no_rules/model.py</code> — policy stack wired for TorchRL</p>
      <pre class="code-snippet"><code>value_net = Mod(DuelingConvQNet(), in_keys=["observation"], out_keys=["action_value"])
mask_net = Mod(LegalActionMaskFromObs(), in_keys=["observation"], out_keys=["action_mask"])
mask_q_net = Mod(ApplyActionMask(), in_keys=["action_value", "action_mask"], out_keys=["action_value"])
policy = Seq(value_net, mask_net, mask_q_net, QValueModule(spec=env.action_spec))
exploration = EGreedyModule(
    env.action_spec, eps_init=..., eps_end=..., annealing_num_steps=..., action_mask_key="action_mask"
)
policy_explore = Seq(policy, exploration)</code></pre>

      <h3>2.2 Deep RL formulation: Q-learning, dueling value decomposition, exploration</h3>
      <p>
        We follow the <strong>deep Q-network (DQN)</strong> family: a parametric scorer
        <span class="mono">Q<sub>θ</sub>(s,a)</span> is fitted to Bellman targets built from stored transitions.
        Under the usual tabular optimality idea, the fixed-point Bellman backup for the optimal action-value is
      </p>
      <p class="mono" style="margin:0.45rem 0 0.65rem;">Q*(s,a) = E[ r + γ · max_{a′} Q*(s′,a′) ]</p>
      <p>
        where <span class="mono">r</span> is the immediate reward, <span class="mono">γ</span> the discount factor,
        and the expectation is over stochastic transitions (here, including the game&rsquo;s noisy placement rule).
        In practice, <span class="mono">Q<sub>θ</sub></span> is trained with a Huber TD loss, <strong>Double DQN</strong>
        bootstrapping, and a <strong>target network</strong> updated by soft Polyak steps in
        <code class="mono">train.py</code> (Section&nbsp;3); this section focuses on how
        <span class="mono">Q<sub>θ</sub></span> is shaped inside <code class="mono">model.py</code>.
      </p>
      <p>
        Instead of a single monolithic head for all 96 logits, we use a <strong>dueling</strong> parameterisation
        that splits state value and action advantages:
      </p>
      <p class="mono" style="margin:0.45rem 0 0.65rem;">Q<sub>θ</sub>(s,a) = V<sub>θ</sub>(s) + ( A<sub>θ</sub>(s,a) − (1/|A|) Σ<sub>b</sub> A<sub>θ</sub>(s,b) )</p>
      <p>
        Subtracting the mean advantage identically in <span class="mono">a</span> keeps the representation
        identifiable while letting <span class="mono">V<sub>θ</sub></span> absorb global shift and
        <span class="mono">A<sub>θ</sub></span> encode relative preferences among moves.
      </p>
      <p>
        <strong>Exploration</strong> is ε-greedy on <em>legal</em> actions: with probability <span class="mono">1−ε</span>
        the greedy action maximises masked <span class="mono">Q</span>; with probability <span class="mono">ε</span>
        a legal action is chosen uniformly. The schedule is implemented by TorchRL&rsquo;s
        <code class="mono">EGreedyModule</code> on top of the masked policy.
      </p>
      <p class="snippet-caption">File: <code class="mono">model.py</code> — dueling recombination in the network</p>
      <pre class="code-snippet"><code>v = self.value_head(x)
a = self.adv_head(x)
return v + (a - a.mean(dim=-1, keepdim=True))</code></pre>

      <h3>2.3 Input semantics and convolutional state encoding</h3>
      <p>
        The policy consumes the same 96-dimensional observation as the environment: positive entries mark the
        learning agent, negative entries the opponent, zeros empty cells. Before any convolution, an encoder
        maps <span class="mono">(B,96)</span> to <span class="mono">(B,3,12,12)</span>: two occupancy planes on the
        global pyramid layout plus a binary &ldquo;valid cell&rdquo; plane. This aligns the network&rsquo;s spatial
        receptive field with cross-board win lines from the rules.
      </p>
      <p class="snippet-caption">File: <code class="mono">model.py</code> — projection indices and plane stacking</p>
      <pre class="code-snippet"><code>def _build_proj_index() -&gt; np.ndarray:
    idx = []
    for g in range(N_ACTIONS):
        b = g // 16
        loc = g % 16
        r, c = loc // 4, loc % 4
        br, bc = _BOARD_ORIGIN[b]
        idx.append((br + r) * PROJ_W + (bc + c))
    return np.asarray(idx, dtype=np.int64)

# forward (conceptual): scatter agent / opponent into flat 12×12, stack with valid mask → (B,3,H,W)</code></pre>
      <pre class="code-snippet"><code>flat_agent.index_copy_(1, self.proj_idx, (observation &gt; 0.5).to(observation.dtype))
flat_opp.index_copy_(1, self.proj_idx, (observation &lt; -0.5).to(observation.dtype))
board = torch.stack([flat_agent, flat_opp, valid], dim=1)
return board.view(bsz, 3, PROJ_H, PROJ_W)</code></pre>

      <h3>2.4 Convolutional backbone with residual blocks</h3>
      <p>
        The tensor passes through a small CNN: <span class="mono">3→64</span> convolution with batch norm and ReLU,
        two identical <strong>residual</strong> blocks (each two <span class="mono">3×3</span> convolutions, BN, skip
        connection, ReLU), then a <span class="mono">64→96</span> conv-BN-ReLU widening. The flattened feature has
        dimension <span class="mono">96×12×12</span> and feeds both MLP heads.
      </p>
      <p class="snippet-caption">File: <code class="mono">model.py</code> — residual block and backbone excerpt</p>
      <pre class="code-snippet"><code>class ResBlock(nn.Module):
    def forward(self, x: torch.Tensor) -&gt; torch.Tensor:
        return self.act(x + self.net(x))

self.backbone = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    ResBlock(64),
    ResBlock(64),
    nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(96),
    nn.ReLU(inplace=True),
)</code></pre>

      <h3>2.5 Value and advantage heads</h3>
      <p>
        The value stream maps flattened features to a scalar <span class="mono">V<sub>θ</sub>(s)</span>; the
        advantage stream maps to a 96-dimensional vector of raw advantages
        <span class="mono">(A<sub>θ</sub>(s,·))</span>. Their widths (<span class="mono">256</span> and
        <span class="mono">512</span> hidden units) follow the usual pattern of a narrower value trunk and a
        slightly wider advantage trunk before the final linear projections.
      </p>
      <p class="snippet-caption">File: <code class="mono">model.py</code> — MLP heads on flattened CNN features</p>
      <pre class="code-snippet"><code>feat_dim = 96 * PROJ_H * PROJ_W
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
)</code></pre>

      <h3>2.6 Legal-action masking inside the TensorDict pipeline</h3>
      <p>
        Legality is read from the observation: any coordinate with near-zero entry is empty and therefore
        legal. Illegal logits are overwritten with a large negative constant so that downstream
        <code class="mono">QValueModule</code> never assigns positive mass to forbidden moves, and ε-greedy
        sampling respects the same mask key.
      </p>
      <p class="snippet-caption">File: <code class="mono">model.py</code> — mask derivation and application</p>
      <pre class="code-snippet"><code>return (observation.abs() &lt; 1e-6).to(dtype=torch.bool)

masked = action_value.clone()
masked[~action_mask] = -1e9
return masked</code></pre>

      <h3>2.7 Gymnasium bridge for TorchRL rollouts</h3>
      <p>
        The registered Gymnasium environment <code class="mono">SuperTicTacToe-v0</code> is wrapped as a TorchRL
        <code class="mono">GymEnv</code> with optional custom opponent policies (e.g. self-play bridges from
        <code class="mono">train.py</code>) and a <code class="mono">StepCounter</code> transform for horizon bookkeeping.
      </p>
      <p class="snippet-caption">File: <code class="mono">model.py</code> — environment factory</p>
      <pre class="code-snippet"><code>return TransformedEnv(
    GymEnv("SuperTicTacToe-v0", device=device, **kw),
    StepCounter(),
)</code></pre>

      <h3>2.8 Greedy action selection at evaluation time</h3>
      <p>
        For deployment or evaluation, the code runs the same TensorDict policy under <code class="mono">torch.inference_mode()</code>,
        reads <code class="mono">action_value</code>, applies the supplied legal mask, and returns an argmax.
        An optional <code class="mono">use_tactical</code> flag layers hand-crafted win/block priorities on top of
        the network; the default path relies purely on learned Q-values plus shaping from the environment.
      </p>
      <p class="snippet-caption">File: <code class="mono">model.py</code> — masked argmax (default branch)</p>
      <pre class="code-snippet"><code>out = policy(td)
q = out["action_value"].detach().cpu().numpy().reshape(-1)
q = q.copy()
q[~legal] = -1e9
return int(np.argmax(q))</code></pre>

      <h3>2.9 Concept-to-implementation map</h3>
      <p>For cross-reference while reading the source, the narrative blocks above map to symbols in <code class="mono">model.py</code> as follows.</p>
      <ul>
        <li><strong>TorchRL composition / ε-greedy</strong> — <code class="mono">build_policy_pair</code></li>
        <li><strong>Grid encoder</strong> — <code class="mono">ObsToBoardPlanes</code>, <code class="mono">_build_proj_index</code></li>
        <li><strong>CNN + residuals + dueling heads</strong> — <code class="mono">DuelingConvQNet</code>, <code class="mono">ResBlock</code></li>
        <li><strong>Masking</strong> — <code class="mono">LegalActionMaskFromObs</code>, <code class="mono">ApplyActionMask</code></li>
        <li><strong>Env factory</strong> — <code class="mono">make_rl_env</code>; <strong>greedy play</strong> — <code class="mono">greedy_action_masked</code></li>
      </ul>
    </section>
"""


# Section 3 (Training): train.py + coding_no_rules reward logic + hyperparameter tables.
TRAINING_SECTION_HTML = """
    <section id="sec-training">
      <h2 class="sec">3. Training Procedure and Hyperparameters</h2>

      <h3>3.1 TorchRL DQN training framework (<code class="mono">train.py</code>)</h3>
      <p>
        Training is implemented without a hand-written back-prop loop over raw environment steps. Instead,
        <code class="mono">train_dqn</code> composes <strong>TorchRL</strong> primitives: a <strong>replay buffer</strong>
        (<code class="mono">ReplayBuffer</code> with <code class="mono">LazyTensorStorage</code>), a
        <strong>Double DQN</strong> objective (<code class="mono">DQNLoss</code> with delayed value network and
        Huber loss), <strong>Adam</strong> on the loss module&rsquo;s parameters, and <strong>Polyak soft updates</strong>
        (<code class="mono">SoftUpdate</code> with coefficient <span class="mono">τ</span>). Each outer iteration
        collects transitions by calling <code class="mono">env.rollout(...)</code> with the exploratory policy
        (<code class="mono">policy_explore</code>), appends them to the buffer, and—once enough data exist—samples
        mini-batches for several gradient steps before annealing exploration and stepping the target-network tracker.
      </p>
      <p class="snippet-caption">File: <code class="mono">coding/coding_no_rules/train.py</code> — loss, replay, optimiser, soft update</p>
      <pre class="code-snippet"><code>loss_module = DQNLoss(
    value_network=policy,
    action_space=env.action_spec,
    delay_value=True,
    double_dqn=True,
    loss_function="smooth_l1",
).to(device)
loss_module.make_value_estimator(ValueEstimators.TD0, gamma=args.gamma)
optim = Adam(loss_module.parameters(), lr=args.lr)
updater = SoftUpdate(loss_module, tau=args.tau)
rb = ReplayBuffer(storage=LazyTensorStorage(args.buffer))</code></pre>
      <p class="snippet-caption">File: <code class="mono">train.py</code> — rollout, buffer extension, optimisation micro-loop</p>
      <pre class="code-snippet"><code>roll = env.rollout(max_steps=args.rollout_steps, policy=policy_explore, auto_reset=True)
roll = _normalize_td_action_shapes(roll)
rb.extend(roll)
# ...
if len(rb) &gt;= args.min_buffer:
    for _ in range(args.optim_steps):
        sample = rb.sample(args.batch)
        loss_vals = loss_module(sample)
        loss_vals["loss"].backward()
        torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 10.0)
        optim.step()
        optim.zero_grad()
    exploration.step(int(roll.numel()))
    updater.step()</code></pre>

      <h3>3.2 How the <code class="mono">coding_rules</code> track enforces mandatory win-and-block rules</h3>
      <p>
        The assignment dynamics (six boards, stochastic placement, global win lines) are shared across packages.
        The clearest contrast with <code class="mono">coding_no_rules</code> is therefore <strong>compulsory behaviour at
        decision time</strong>, not merely different scalar rewards: in <strong><code class="mono">coding/coding_rules/</code></strong>
        the agent is not allowed to ignore one-ply kill shots whenever the board admits them. Learning signals still echo
        that priority through <strong>reward shaping</strong> in <code class="mono">environment.py</code>
        &mdash; immediate-win moves earn <code class="mono">WIN_INTENT_BONUS</code> while missing them incurs
        <code class="mono">MISS_WIN_PENALTY</code>, and answering an opponent instant threat with a block earns
        <code class="mono">BLOCK_THREAT_BONUS</code> while ignoring it triggers <code class="mono">MISS_BLOCK_PENALTY</code>
        &mdash; but the headline difference is the paired <strong>hard tactical layer</strong> that rewrites greedy actions
        regardless of the Q-head&rsquo;s preference.
      </p>
      <p>
        That layer lives in <code class="mono">model.py</code> as <code class="mono">greedy_action_masked(..., use_tactical=True)</code>,
        which is the default for GUI and scripted play in this tree. After reconstructing occupancy from the observation,
        the routine applies two <strong>sequential mandatory checks</strong> before any masked Q-argmax:
        <strong>Rule&nbsp;1</strong> &mdash; if <code class="mono">_immediate_winning_actions(board, AGENT)</code> is non-empty,
        return the first legal action drawn from that set (play an immediate winning move whenever one exists).
        <strong>Rule&nbsp;2</strong> &mdash; else if <code class="mono">_immediate_winning_actions(board, OPPONENT)</code> is non-empty,
        return the first legal action drawn from that blocking set (deny the opponent&rsquo;s immediate win).
        <strong>Otherwise</strong> mask illegal logits and take the argmax over Q. The companion
        <code class="mono">coding_no_rules</code> tree may reuse similar shaping hooks inside <code class="mono">step</code>,
        but it typically keeps <code class="mono">use_tactical=False</code> at greedy time so evaluation stresses pure learned
        values unless a flag explicitly re-enables the tactical ordering.
      </p>
      <p class="snippet-caption">File: <code class="mono">coding/coding_rules/environment.py</code> — tactical shaping constants (rules branch)</p>
      <pre class="code-snippet"><code>WIN_INTENT_BONUS = 0.90
MISS_WIN_PENALTY = 1.10
BLOCK_THREAT_BONUS = 0.60
MISS_BLOCK_PENALTY = 0.90
SHAPING_LAMBDA = 0.25
SHAPING_GAMMA = 0.99</code></pre>
      <p class="snippet-caption">File: <code class="mono">coding/coding_rules/environment.py</code> — when the agent must address instant win / loss threats</p>
      <pre class="code-snippet"><code>own_win_actions_before = _immediate_winning_actions(self._board, AGENT)
opp_threat_before = len(_immediate_winning_actions(self._board, OPPONENT))
# ...
if own_win_actions_before:
    if int(action) in own_win_actions_before:
        reward += WIN_INTENT_BONUS
    else:
        reward -= MISS_WIN_PENALTY
elif opp_threat_before &gt; 0:
    opp_win_actions = _immediate_winning_actions(self._board, OPPONENT)
    if int(action) in opp_win_actions:
        reward += BLOCK_THREAT_BONUS
    else:
        reward -= MISS_BLOCK_PENALTY</code></pre>
      <p class="snippet-caption">File: <code class="mono">coding/coding_rules/model.py</code> — hard tactical ordering at greedy play (default on)</p>
      <pre class="code-snippet"><code>def greedy_action_masked(..., use_tactical: bool = True) -&gt; int:
    # ...
    if not use_tactical:
        ...
        return int(np.argmax(q))
    board = np.zeros(N_ACTIONS, dtype=np.int8)
    board[obs &gt; 0.5] = AGENT
    board[obs &lt; -0.5] = OPPONENT
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
    # ... masked Q-argmax ...</code></pre>
      <p>
        Illegal moves, forfeited stochastic placements, and potential-based terms built from
        <code class="mono">_strategic_line_score</code> are handled in the same <code class="mono">step</code> implementation;
        see <code class="mono">coding/coding_rules/environment.py</code> for the full reward ledger.
      </p>

      <h3>3.3 Opponent modes from <code class="mono">main.py</code></h3>
      <p>
        Training never hard-codes a single opponent: both <code class="mono">coding_rules/main.py</code> and
        <code class="mono">coding_no_rules/main.py</code> funnel configuration through a shared dataclass
        <code class="mono">TrainOpponentConfig</code> plus <code class="mono">build_train_opponent_argv</code>
        (the two trees carry parallel copies of <code class="mono">training_opponent.py</code>). You pick a
        <code class="mono">mode</code> &mdash; <strong>random</strong> (uniform legal moves),
        <strong>heuristic</strong> (one-ply win then block then random),
        <strong>mixed</strong> (random vs heuristic with probability <code class="mono">heuristic_prob</code>),
        <strong>self_snapshot</strong> (opponent is a periodically refreshed copy of the learner), or
        <strong>curriculum</strong> (first <code class="mono">curriculum_switch_episodes</code> games against random noise,
        then switch to whichever opponent string you set in <code class="mono">after_curriculum_opponent</code>).
        When the post-curriculum opponent is <code class="mono">self_snapshot</code>, extra flags control how often the
        frozen policy is synced and whether that snapshot uses the tactical greedy layer.
      </p>
      <p>
        When you launch <code class="mono">python main.py</code> without sub-commands, the script injects
        <code class="mono">train</code> plus <code class="mono">--episodes</code>, <code class="mono">--out-dir</code>, the
        opponent argv fragment, and any <code class="mono">TRAIN_EXTRA_ARGS</code> overrides, so swapping modes is a
        one-line edit to <code class="mono">TRAIN_OPPONENT</code> rather than memorising long CLI strings.
      </p>
      <p class="snippet-caption">File: <code class="mono">coding/coding_rules/training_opponent.py</code> (same pattern under <code class="mono">coding_no_rules/</code>) — dataclass + argv builder</p>
      <pre class="code-snippet"><code>@dataclass
class TrainOpponentConfig:
    mode: str = "curriculum"
    curriculum_switch_episodes: int = 300
    after_curriculum_opponent: str = "mixed"
    heuristic_prob: float = 0.7
    self_play_sync_iters: int = 10
    self_play_opponent_tactical: bool = False

def build_train_opponent_argv(cfg: TrainOpponentConfig) -&gt; List[str]:
    m = cfg.mode.strip().lower()
    if m == "curriculum":
        ac = cfg.after_curriculum_opponent.strip().lower()
        out = [
            "--curriculum-switch-episodes", str(int(cfg.curriculum_switch_episodes)),
            "--opponent", ac,
            "--opponent-heuristic-prob", str(float(cfg.heuristic_prob)),
        ]
        # ... append self_snapshot flags when after_curriculum_opponent == "self_snapshot"
        return out
    # ... non-curriculum: emit --curriculum-switch-episodes 0 and --opponent == mode ...</code></pre>
      <p class="snippet-caption">File: <code class="mono">coding/coding_no_rules/main.py</code> — wiring training argv (representative snippet)</p>
      <pre class="code-snippet"><code>TRAIN_OPPONENT = TrainOpponentConfig(
    mode="curriculum",
    curriculum_switch_episodes=250,
    after_curriculum_opponent="heuristic",
    heuristic_prob=0.7,
    self_play_sync_iters=10,
    self_play_opponent_tactical=True,
)
# ...
sys.argv = [
    script, "train",
    "--episodes", str(TRAIN_EPISODES),
    "--out-dir", TRAIN_OUT_DIR,
    *build_train_opponent_argv(TRAIN_OPPONENT),
    *TRAIN_EXTRA_ARGS,
]</code></pre>
      <p>
        The <code class="mono">coding_rules/main.py</code> checkout in this workspace shows the same structure with a
        different illustrative <code class="mono">mode</code> (e.g. pure <code class="mono">random</code> with curriculum fields
        kept for quick experiments); functionally both entry points expose the identical opponent vocabulary.
      </p>

      <h3>3.4 Hyperparameter catalogue (grouped)</h3>
      <p>
        The tables consolidate every CLI flag exposed by <code class="mono">train.py</code> plus the reward constants
        from <code class="mono">environment.py</code>. Where <code class="mono">main.py</code> overrides a default
        through <code class="mono">TRAIN_EXTRA_ARGS</code> or <code class="mono">TrainOpponentConfig</code>, the
        <strong>Setting</strong> column reflects the <code class="mono">coding_no_rules/main.py</code> entry point;
        remaining entries fall back to <code class="mono">train.py</code> argparse defaults. Section&nbsp;F lists the
        <strong><code class="mono">coding_rules</code></strong> shaping magnitudes; the <code class="mono">coding_no_rules</code>
        clone uses slightly larger bonuses/penalties for the same logical terms.
      </p>

      <div class="param-wrap" role="region" aria-label="Optimization and TD parameters">
        <table class="param-table">
          <caption>A. Optimisation &amp; temporal-difference</caption>
          <thead><tr><th scope="col">Flag / symbol</th><th scope="col">Role</th><th scope="col">Setting (project / default)</th></tr></thead>
          <tbody>
            <tr><td class="mono-cell">--lr</td><td>Adam learning rate on <code class="mono">DQNLoss</code> parameters.</td><td class="mono-cell">7e-5 (train default)</td></tr>
            <tr><td class="mono-cell">--gamma</td><td>Discount factor in the TD0 estimator.</td><td class="mono-cell">0.99</td></tr>
            <tr><td class="mono-cell">--tau</td><td>Polyak coefficient for soft target updates.</td><td class="mono-cell">0.005</td></tr>
            <tr><td class="mono-cell">--optim-steps</td><td>Gradient steps per rollout once the buffer is warm.</td><td class="mono-cell">1 (via <code class="mono">main.py</code>)</td></tr>
          </tbody>
        </table>
      </div>

      <div class="param-wrap" role="region" aria-label="Replay and rollout">
        <table class="param-table">
          <caption>B. Replay buffer &amp; interaction horizon</caption>
          <thead><tr><th scope="col">Flag</th><th scope="col">Role</th><th scope="col">Setting (project / default)</th></tr></thead>
          <tbody>
            <tr><td class="mono-cell">--buffer</td><td>Maximum transitions stored in <code class="mono">LazyTensorStorage</code>.</td><td class="mono-cell">100000</td></tr>
            <tr><td class="mono-cell">--min-buffer</td><td>Minimum transitions before optimisation begins.</td><td class="mono-cell">500 (<code class="mono">main.py</code>)</td></tr>
            <tr><td class="mono-cell">--batch</td><td>Mini-batch size for replay sampling.</td><td class="mono-cell">64 (<code class="mono">main.py</code>)</td></tr>
            <tr><td class="mono-cell">--rollout-steps</td><td>Max environment steps per outer <code class="mono">rollout</code> call.</td><td class="mono-cell">64 (<code class="mono">main.py</code>)</td></tr>
            <tr><td class="mono-cell">--episodes</td><td>Target completed episodes before stopping.</td><td class="mono-cell">750 (<code class="mono">main.py</code> <code class="mono">TRAIN_EPISODES</code>)</td></tr>
          </tbody>
        </table>
      </div>

      <div class="param-wrap" role="region" aria-label="Exploration">
        <table class="param-table">
          <caption>C. ε-greedy exploration (<code class="mono">EGreedyModule</code>)</caption>
          <thead><tr><th scope="col">Flag</th><th scope="col">Role</th><th scope="col">Setting (project / default)</th></tr></thead>
          <tbody>
            <tr><td class="mono-cell">--eps-start</td><td>Initial exploration rate.</td><td class="mono-cell">0.6 (<code class="mono">main.py</code>)</td></tr>
            <tr><td class="mono-cell">--eps-end</td><td>Floor after annealing.</td><td class="mono-cell">0.05</td></tr>
            <tr><td class="mono-cell">--eps-anneal-steps</td><td>Environment steps over which ε decays linearly.</td><td class="mono-cell">900 (<code class="mono">main.py</code>)</td></tr>
          </tbody>
        </table>
      </div>

      <div class="param-wrap" role="region" aria-label="Logging and checkpoints">
        <table class="param-table">
          <caption>D. Logging, evaluation, checkpoints, hardware</caption>
          <thead><tr><th scope="col">Flag</th><th scope="col">Role</th><th scope="col">Setting (project / default)</th></tr></thead>
          <tbody>
            <tr><td class="mono-cell">--seed</td><td>RNG seed for Python, NumPy, Torch, and per-iter env seeds.</td><td class="mono-cell">0</td></tr>
            <tr><td class="mono-cell">--log-every</td><td>Print rolling return / diagnostics every N outer iterations.</td><td class="mono-cell">50</td></tr>
            <tr><td class="mono-cell">--save-after</td><td>Minimum episodes before <code class="mono">best_model.pt</code> snapshots trigger.</td><td class="mono-cell">100</td></tr>
            <tr><td class="mono-cell">--out-dir</td><td>Directory for checkpoints and <code class="mono">training_curve.png</code>.</td><td class="mono-cell">artifacts (<code class="mono">main.py</code>)</td></tr>
            <tr><td class="mono-cell">--eval-every</td><td>Frequency of optional win-rate sweeps; 0 disables.</td><td class="mono-cell">0 (<code class="mono">main.py</code>)</td></tr>
            <tr><td class="mono-cell">--eval-episodes</td><td>Episodes per evaluation split when enabled.</td><td class="mono-cell">20</td></tr>
            <tr><td class="mono-cell">--cpu</td><td>Force CPU even if CUDA is available.</td><td class="mono-cell">false (omit flag)</td></tr>
            <tr><td class="mono-cell">--tactical-inference</td><td>Enable hard win/block priorities during <code class="mono">_evaluate_win_rate</code>.</td><td class="mono-cell">false (omit flag)</td></tr>
          </tbody>
        </table>
      </div>

      <div class="param-wrap" role="region" aria-label="Opponent curriculum">
        <table class="param-table">
          <caption>E. Opponents, curriculum, and self-play bridges</caption>
          <thead><tr><th scope="col">Flag / config</th><th scope="col">Role</th><th scope="col">Setting (project / default)</th></tr></thead>
          <tbody>
            <tr><td class="mono-cell">TrainOpponentConfig.mode</td><td>Selects scripted opponent mix or curriculum wrapper.</td><td class="mono-cell">curriculum (<code class="mono">main.py</code>)</td></tr>
            <tr><td class="mono-cell">--curriculum-switch-episodes</td><td>Episodes against random play before switching <code class="mono">--opponent</code>.</td><td class="mono-cell">250</td></tr>
            <tr><td class="mono-cell">--opponent</td><td>Post-curriculum opponent type (also standalone mode when curriculum off).</td><td class="mono-cell">heuristic (after switch)</td></tr>
            <tr><td class="mono-cell">--opponent-heuristic-prob</td><td>Probability of heuristic branch when <code class="mono">opponent=mixed</code>.</td><td class="mono-cell">0.7</td></tr>
            <tr><td class="mono-cell">--self-play-sync-iters</td><td>Outer-iter cadence for refreshing the frozen self-play snapshot.</td><td class="mono-cell">10 (active for <code class="mono">self_snapshot</code>)</td></tr>
            <tr><td class="mono-cell">--self-play-opponent-q-only</td><td>Snapshot opponent ignores tactical win/block ordering.</td><td class="mono-cell">omitted (tactical snapshot enabled in <code class="mono">main.py</code> when applicable)</td></tr>
          </tbody>
        </table>
      </div>

      <div class="param-wrap" role="region" aria-label="Reward shaping constants">
        <table class="param-table">
          <caption>F. Reward &amp; shaping constants (<code class="mono">environment.py</code>)</caption>
          <thead><tr><th scope="col">Symbol</th><th scope="col">Role</th><th scope="col">Numeric value</th></tr></thead>
          <tbody>
            <tr><td class="mono-cell">WIN_REWARD</td><td>Terminal bonus when the agent completes a winning line.</td><td class="mono-cell">3.0</td></tr>
            <tr><td class="mono-cell">LOSE_REWARD</td><td>Terminal penalty when the opponent wins.</td><td class="mono-cell">-3.0</td></tr>
            <tr><td class="mono-cell">ILLEGAL_REWARD</td><td>Penalty for illegal / occupied picks (treated as forfeit).</td><td class="mono-cell">-0.2</td></tr>
            <tr><td class="mono-cell">FORFEIT_REWARD</td><td>Penalty when stochastic placement voids the ply.</td><td class="mono-cell">-0.1</td></tr>
            <tr><td class="mono-cell">WIN_INTENT_BONUS / MISS_WIN_PENALTY</td><td>Incentivise taking an immediate winning action.</td><td class="mono-cell">+0.90 / −1.10 (<code class="mono">coding_rules</code>)</td></tr>
            <tr><td class="mono-cell">BLOCK_THREAT_BONUS / MISS_BLOCK_PENALTY</td><td>Incentivise blocking an immediate opponent win.</td><td class="mono-cell">+0.60 / −0.90 (<code class="mono">coding_rules</code>)</td></tr>
            <tr><td class="mono-cell">THREAT_REDUCE_BONUS / THREAT_EXIST_PENALTY</td><td>Shape threat counts after the agent acts.</td><td class="mono-cell">+0.1 / −0.08 per threat</td></tr>
            <tr><td class="mono-cell">CREATE_WIN_THREAT_BONUS</td><td>Bonus for increasing own immediate-win options.</td><td class="mono-cell">+0.12 per new threat</td></tr>
            <tr><td class="mono-cell">SHAPING_LAMBDA / SHAPING_GAMMA</td><td>Potential-based shaping on <code class="mono">_strategic_line_score</code>.</td><td class="mono-cell">0.25 / 0.99</td></tr>
          </tbody>
        </table>
      </div>

      <p>
        Together, the grouped tables play the same role as a spreadsheet hyper-parameter sheet: each row ties a
        symbol to its behavioural effect and the value actually exercised by the checked-in entry point. When you
        sweep a knob (learning rate, ε schedule, batch size, etc.), update the corresponding row—or regenerate the
        table from the CLI helper you use—so the PDF/HTML snapshot stays faithful to the experiment card.
      </p>
    </section>
"""

# Section 4 (Experiments / results figures): no code snippets; paths are relative to this HTML file.
EXPERIMENTS_SECTION_HTML = """
    <section id="sec-experiments">
      <h2 class="sec">4. Experiments and Results</h2>
      <p>
        This section summarises the eleven supervised training regimes used in this study. Each regime shares the same
        network and DQN loop but differs in opponent schedule (and, for <code class="mono">NoRule_*</code> runs, in the
        reward environment used while the <em>learning agent</em> is updated). Prefix <code class="mono">NoRule_</code>
        denotes checkpoints produced under the <code class="mono">coding_no_rules</code> environment (no mandatory
        tactical shaping for the trainee); all other abbreviations use the rule-augmented <code class="mono">coding_rules</code>
        environment. Whether an <em>opponent</em> applies hard win/block priors is determined solely by that opponent&rsquo;s
        mode (random, heuristic, mixed, or frozen self-policy), not by the <code class="mono">NoRule_</code> prefix on the label.
      </p>

      <h3>4.1 Eleven training modes (abbreviations)</h3>
      <p>
        The shorthand labels below match the filenames under <code class="mono">pictures/training/</code> and the legend
        used in the comparison plots. Column &ldquo;#&rdquo; aligns with the numeric suffix convention in evaluation scripts
        (<code class="mono">best_model_1</code> &hellip; <code class="mono">best_model_11</code>).
      </p>
      <div class="param-wrap" role="region" aria-label="Training mode abbreviations">
        <table class="param-table">
          <caption>Training regime shorthand &mdash; opponent schedule and reward environment for the learning agent</caption>
          <thead><tr><th scope="col">#</th><th scope="col">Abbreviation</th><th scope="col">Meaning</th></tr></thead>
          <tbody>
            <tr><td class="mono-cell">1</td><td class="mono-cell">Random</td><td>Entire run against a random opponent; agent updates under rule-augmented reward shaping.</td></tr>
            <tr><td class="mono-cell">2</td><td class="mono-cell">Heuristic</td><td>Entire run against the scripted heuristic; rule-augmented shaping for the learner.</td></tr>
            <tr><td class="mono-cell">3</td><td class="mono-cell">Full-Mix</td><td>Entire run against a mixed opponent (stochastic blend of heuristic and random); rule-augmented shaping.</td></tr>
            <tr><td class="mono-cell">4</td><td class="mono-cell">Full-Self</td><td>Entire run with the self-play snapshot bridge; rule-augmented shaping.</td></tr>
            <tr><td class="mono-cell">5</td><td class="mono-cell">Cur+Mix</td><td>Curriculum: random warm-up, then mixed opponent; rule-augmented shaping.</td></tr>
            <tr><td class="mono-cell">6</td><td class="mono-cell">Cur+Self+Heu</td><td>Curriculum configuration combining self snapshot and heuristic-related scheduling (see project scripts); rule-augmented shaping.</td></tr>
            <tr><td class="mono-cell">7</td><td class="mono-cell">Cur+Heu</td><td>Curriculum: random warm-up, then heuristic; rule-augmented shaping.</td></tr>
            <tr><td class="mono-cell">8</td><td class="mono-cell">Cur+Self</td><td>Curriculum: random warm-up, then self snapshot; rule-augmented shaping.</td></tr>
            <tr><td class="mono-cell">9</td><td class="mono-cell">NoRule_Random</td><td>Random opponent throughout; learner updates under <code class="mono">coding_no_rules</code> (no mandatory tactical shaping terms for the agent).</td></tr>
            <tr><td class="mono-cell">10</td><td class="mono-cell">NoRule_Cur+Mix</td><td>Curriculum to mixed opponent with <code class="mono">coding_no_rules</code> reward for the learner.</td></tr>
            <tr><td class="mono-cell">11</td><td class="mono-cell">NoRule_Cur+Heu</td><td>Curriculum to heuristic with <code class="mono">coding_no_rules</code> reward for the learner.</td></tr>
          </tbody>
        </table>
      </div>

      <h3>4.2 Per-mode episode return curves</h3>
      <p>
        Each panel plots per-episode return (thin trace) and a 50-episode moving average (<code class="mono">MA50</code>,
        orange). All runs use the same 750-episode horizon saved next to the corresponding checkpoint.
      </p>
      <div class="training-grid" role="group" aria-label="Training curves for eleven modes">
        <figure class="training-cell"><img src="pictures/training/Random.png" alt="Return vs episode for Random training mode" loading="eager" decoding="async" /><figcaption class="training-caption"><span class="mono">Random</span> <span class="training-cap-idx">(1)</span></figcaption></figure>
        <figure class="training-cell"><img src="pictures/training/Heuristic.png" alt="Return vs episode for Heuristic training mode" loading="eager" decoding="async" /><figcaption class="training-caption"><span class="mono">Heuristic</span> <span class="training-cap-idx">(2)</span></figcaption></figure>
        <figure class="training-cell"><img src="pictures/training/Full-Mix.png" alt="Return vs episode for Full-Mix training mode" loading="eager" decoding="async" /><figcaption class="training-caption"><span class="mono">Full-Mix</span> <span class="training-cap-idx">(3)</span></figcaption></figure>
        <figure class="training-cell"><img src="pictures/training/Full-Self.png" alt="Return vs episode for Full-Self training mode" loading="eager" decoding="async" /><figcaption class="training-caption"><span class="mono">Full-Self</span> <span class="training-cap-idx">(4)</span></figcaption></figure>
        <figure class="training-cell"><img src="pictures/training/Cur+Mix.png" alt="Return vs episode for Cur+Mix training mode" loading="eager" decoding="async" /><figcaption class="training-caption"><span class="mono">Cur+Mix</span> <span class="training-cap-idx">(5)</span></figcaption></figure>
        <figure class="training-cell"><img src="pictures/training/Cur+Self+Heu.png" alt="Return vs episode for Cur+Self+Heu training mode" loading="eager" decoding="async" /><figcaption class="training-caption"><span class="mono">Cur+Self+Heu</span> <span class="training-cap-idx">(6)</span></figcaption></figure>
        <figure class="training-cell"><img src="pictures/training/Cur+Heu.png" alt="Return vs episode for Cur+Heu training mode" loading="eager" decoding="async" /><figcaption class="training-caption"><span class="mono">Cur+Heu</span> <span class="training-cap-idx">(7)</span></figcaption></figure>
        <figure class="training-cell"><img src="pictures/training/Cur+Self.png" alt="Return vs episode for Cur+Self training mode" loading="eager" decoding="async" /><figcaption class="training-caption"><span class="mono">Cur+Self</span> <span class="training-cap-idx">(8)</span></figcaption></figure>
        <figure class="training-cell"><img src="pictures/training/NoRule_Random.png" alt="Return vs episode for NoRule_Random training mode" loading="eager" decoding="async" /><figcaption class="training-caption"><span class="mono">NoRule_Random</span> <span class="training-cap-idx">(9)</span></figcaption></figure>
        <figure class="training-cell"><img src="pictures/training/NoRule_Cur+Mix.png" alt="Return vs episode for NoRule_Cur+Mix training mode" loading="eager" decoding="async" /><figcaption class="training-caption"><span class="mono">NoRule_Cur+Mix</span> <span class="training-cap-idx">(10)</span></figcaption></figure>
        <figure class="training-cell"><img src="pictures/training/NoRule_Cur+Heu.png" alt="Return vs episode for NoRule_Cur+Heu training mode" loading="eager" decoding="async" /><figcaption class="training-caption"><span class="mono">NoRule_Cur+Heu</span> <span class="training-cap-idx">(11)</span></figcaption></figure>
      </div>
      <p class="figure-ref">Source files: <code class="mono">pictures/training/*.png</code> (eleven panels, three columns &times; four rows).</p>

      <h3>4.3 Aggregated training comparison</h3>
      <p>
        The two summary figures pool the same logs: the first plot compares rolling mean returns (last 50 episodes) as a
        function of training iteration; the second reports the simple average of per-episode returns across the entire
        750-episode window for each mode. Each figure is shown full width on its own row for readability.
      </p>
      <div class="figure-stack">
        <figure class="paper-figure paper-figure-wide">
          <div class="paper-figure-slot">
            <img src="pictures/comparison/1_return_trend.png" alt="Mean return last fifty episodes versus iteration for eleven modes" loading="eager" decoding="async" />
          </div>
          <figcaption><strong>Mean return (50-episode window) vs iteration.</strong> By iteration 750, <span class="mono">Random</span> and <span class="mono">NoRule_Random</span> sit highest (near +5), while <span class="mono">Full-Self</span> remains the lowest curve (about &minus;3). <span class="mono">NoRule_Cur+Heu</span> starts near the bottom at iteration 50 but climbs into the upper pack, indicating rapid benefit once the curriculum leaves pure random play. Several heuristic-heavy schedules stay negative on this metric despite later benchmarks looking strong, which highlights a scale mismatch between shaped training return and raw win rate.</figcaption>
        </figure>
        <figure class="paper-figure paper-figure-wide">
          <div class="paper-figure-slot">
            <img src="pictures/comparison/2_average_return.png" alt="Bar chart of average return over all training for eleven modes" loading="eager" decoding="async" />
          </div>
          <figcaption><strong>Average return over all training.</strong> <span class="mono">Random</span> leads numerically (+2.20), followed by <span class="mono">NoRule_Random</span> (+1.30) and <span class="mono">NoRule_Cur+Mix</span> (+0.59). <span class="mono">Full-Self</span> is the weakest bar (&minus;3.94), with <span class="mono">Full-Mix</span> and <span class="mono">Heuristic</span> also deeply negative. Pairwise comparisons among <span class="mono">NoRule_*</span> variants and their rule-shaped twins show mixed ordering, but removing shaping clearly inflates several aggregate return averages even when opponent difficulty is unchanged.</figcaption>
        </figure>
      </div>
      <p class="figure-ref">Source files: <code class="mono">pictures/comparison/1_return_trend.png</code>, <code class="mono">pictures/comparison/2_average_return.png</code>.</p>

      <h3>4.4 Post-training evaluation: random benchmark and pairwise matrix</h3>
      <p>
        After freezing checkpoints, each policy played 100 episodes as the first mover against a fresh random opponent.
        The same checkpoints then contested an all-pairs round robin: every ordered pairing played 100 games with the row
        agent moving first, yielding the heat-map and row-mean summaries. Each visualization below occupies a full row
        so axis labels remain legible.
      </p>
      <div class="figure-stack">
        <figure class="paper-figure paper-figure-wide">
          <div class="paper-figure-slot">
            <img src="pictures/benchmark/benchmark_win_rate.png" alt="Bar chart of win rate versus random opponent for eleven models" loading="eager" decoding="async" />
          </div>
          <figcaption><strong>Win rate vs random (<em>n</em> = 100 each).</strong> Seven policies reach a perfect 100&ndash;0 record, including every <span class="mono">NoRule_*</span> checkpoint and the strongest curriculum/self schedules. <span class="mono">Random</span>-trained weights are the weakest bar (84% wins, sixteen losses). <span class="mono">Heuristic</span> (93%) and <span class="mono">Full-Self</span> (90%) trail the pack but still dominate random most of the time.</figcaption>
        </figure>
        <figure class="paper-figure paper-figure-wide">
          <div class="paper-figure-slot">
            <img src="pictures/benchmark/benchmark_table.png" alt="Tabular summary of wins losses and win percent versus random" loading="eager" decoding="async" />
          </div>
          <figcaption><strong>Tabular benchmark.</strong> The printed table duplicates the bar chart for exact win/loss/draw counts; all listed runs report zero draws in this benchmark.</figcaption>
        </figure>
        <figure class="paper-figure paper-figure-wide">
          <div class="paper-figure-slot">
            <img src="pictures/pairwise/pairwise_winrate_heatmap.png" alt="Heatmap of pairwise win rates between eleven models" loading="eager" decoding="async" />
          </div>
          <figcaption><strong>Pairwise win-rate matrix (100 games per ordered pair, diagonal masked).</strong> <span class="mono">NoRule_Random</span> occupies the strongest row (dark green against nearly every column), while <span class="mono">Full-Self</span> is the easiest row to exploit and the softest column to attack. <span class="mono">NoRule_Cur+Mix</span> also posts very high first-player rates against weaker curricula. Mid-tier policies such as <span class="mono">Cur+Heu</span>, <span class="mono">Cur+Self</span>, and <span class="mono">Cur+Self+Heu</span> split yellow cells against each other, signalling similar strength after shaped training.</figcaption>
        </figure>
        <figure class="paper-figure paper-figure-wide">
          <div class="paper-figure-slot">
            <img src="pictures/pairwise/pairwise_row_mean_bar.png" alt="Bar chart of mean win rate as first player across opponents" loading="eager" decoding="async" />
          </div>
          <figcaption><strong>Mean first-player win rate vs the other ten checkpoints.</strong> <span class="mono">NoRule_Random</span> leads (~0.88), <span class="mono">NoRule_Cur+Mix</span> and <span class="mono">NoRule_Cur+Heu</span> follow (~0.73 / ~0.64), while <span class="mono">Full-Self</span> sits lowest (~0.28). The shaped <span class="mono">Random</span> baseline (~0.34) is only slightly ahead of the bottom bar, underscoring that high training return does not automatically translate into head-to-head dominance once opponents leave the random class.</figcaption>
        </figure>
      </div>
      <p class="figure-ref">Source files: <code class="mono">pictures/benchmark/benchmark_win_rate.png</code>, <code class="mono">pictures/benchmark/benchmark_table.png</code>, <code class="mono">pictures/pairwise/pairwise_winrate_heatmap.png</code>, <code class="mono">pictures/pairwise/pairwise_row_mean_bar.png</code>.</p>
    </section>
"""

# Section 5: one-at-a-time hyperparameter sweeps (figures only; prose in captions).
SENSITIVITY_SECTION_HTML = """
    <section id="sec-sensitivity">
      <h2 class="sec">5. Hyperparameter sensitivity analysis</h2>
      <p>
        For each knob below we trained separate agents for <strong>500 episodes</strong> while holding all other settings
        fixed, then evaluated each checkpoint for <strong>100 games</strong> against the same baseline protocol used in
        the sweep scripts. Figures are read directly from <code class="mono">pictures/sensitivity/</code>.
      </p>
      <h3 id="sec-sensitivity-params">5.0 Parameters varied</h3>
      <ul class="body-list">
        <li><strong><code class="mono">--lr</code> (learning rate).</strong> Step size of the Adam update on the value
          network; larger values change weights more aggressively each optimizer step.</li>
        <li><strong><code class="mono">--optim-steps</code>.</strong> How many gradient steps are applied on replay samples
          after each rollout segment before collecting new experience.</li>
        <li><strong><code class="mono">--eps-anneal-steps</code>.</strong> Number of environment steps over which
          &epsilon;-greedy exploration decays linearly from <code class="mono">--eps-start</code> to
          <code class="mono">--eps-end</code>.</li>
        <li><strong><code class="mono">--batch</code>.</strong> Number of transitions sampled from the replay buffer for
          each optimization mini-batch.</li>
        <li><strong><code class="mono">--gamma</code> (discount factor).</strong> Effective horizon weight in the TD target
          when bootstrapping future returns.</li>
      </ul>

      <h3>5.1 Learning rate (<code class="mono">--lr</code>)</h3>
      <div class="figure-stack">
        <figure class="paper-figure paper-figure-wide">
          <div class="paper-figure-slot">
            <img src="pictures/sensitivity/lr/lr_winrate.png" alt="Win rate versus learning rate" loading="eager" decoding="async" />
          </div>
          <figcaption><strong>Win rate.</strong> Win rate rises from roughly 0.25 at <span class="mono">1e-5</span> to a peak near <strong>0.6</strong> at <span class="mono">1e-4</span>, then falls to about <strong>0.19</strong> and <strong>0.17</strong> for the two largest rates tested.</figcaption>
        </figure>
        <figure class="paper-figure paper-figure-wide">
          <div class="paper-figure-slot">
            <img src="pictures/sensitivity/lr/lr_convergence.png" alt="50-episode mean return versus episode for five learning rates" loading="eager" decoding="async" />
          </div>
          <figcaption><strong>Convergence (50-episode moving average return).</strong> The smallest rates reach the high plateau first (near <strong>+5</strong> by roughly episodes 100&ndash;150). The curve for <span class="mono">1e-3</span> stays negative for all 500 episodes, while mid and large rates lie between these extremes with slower or noisier climbs.</figcaption>
        </figure>
        <figure class="paper-figure paper-figure-wide">
          <div class="paper-figure-slot">
            <img src="pictures/sensitivity/lr/lr_stability_variance.png" alt="Return variance versus learning rate" loading="eager" decoding="async" />
          </div>
          <figcaption><strong>Training stability (variance of return).</strong> Return variance is lowest for <span class="mono">3.16e-5</span> (shortest bar, ~58), increases through <span class="mono">1e-4</span> (~83), peaks near <span class="mono">3.16e-4</span> (~132), and drops slightly at <span class="mono">1e-3</span> (~94) but remains well above the smallest learning rates.</figcaption>
        </figure>
      </div>

      <h3>5.2 Optimiser steps per rollout (<code class="mono">--optim-steps</code>)</h3>
      <div class="figure-stack">
        <figure class="paper-figure paper-figure-wide">
          <div class="paper-figure-slot">
            <img src="pictures/sensitivity/optim_steps/optim_steps_winrate.png" alt="Win rate versus optim-steps" loading="eager" decoding="async" />
          </div>
          <figcaption><strong>Win rate.</strong> Win rate is lowest at <strong>1</strong> (~0.30), rises through <strong>2</strong> (~0.61) to a maximum at <strong>3</strong> (~0.71), then decreases at <strong>4</strong> (~0.51) with a small uptick at <strong>5</strong> (~0.53).</figcaption>
        </figure>
        <figure class="paper-figure paper-figure-wide">
          <div class="paper-figure-slot">
            <img src="pictures/sensitivity/optim_steps/optim_steps_convergence.png" alt="MA50 return versus episode for optim-steps 1 through 5" loading="eager" decoding="async" />
          </div>
          <figcaption><strong>Convergence (MA50).</strong> <span class="mono">optim_steps=5</span> reaches a high return earliest; <span class="mono">1</span> is the slowest and most jagged, including a pronounced dip mid-training. By episode 500 all five traces sit in a similar positive band (roughly return 4&ndash;6).</figcaption>
        </figure>
        <figure class="paper-figure paper-figure-wide">
          <div class="paper-figure-slot">
            <img src="pictures/sensitivity/optim_steps/optim_steps_stability_variance.png" alt="Return variance versus optim-steps" loading="eager" decoding="async" />
          </div>
          <figcaption><strong>Training stability.</strong> Variance is highest for <strong>1</strong> (~60) and falls sharply for <strong>2&ndash;4</strong> (about 44&ndash;45, with <strong>3</strong> marginally lowest). <strong>5</strong> shows a modest increase again (~50).</figcaption>
        </figure>
      </div>

      <h3>5.3 Epsilon anneal length (<code class="mono">--eps-anneal-steps</code>)</h3>
      <div class="figure-stack">
        <figure class="paper-figure paper-figure-wide">
          <div class="paper-figure-slot">
            <img src="pictures/sensitivity/eps_anneal_steps/eps_anneal_steps_winrate.png" alt="Win rate versus eps-anneal-steps" loading="eager" decoding="async" />
          </div>
          <figcaption><strong>Win rate.</strong> The bar at <strong>12000</strong> is highest (~0.52). <strong>16800</strong> forms a secondary peak (~0.31). <strong>9600</strong>, <strong>14400</strong>, and <strong>19200</strong> are lower (about 0.16, 0.13, and 0.12 respectively).</figcaption>
        </figure>
        <figure class="paper-figure paper-figure-wide">
          <div class="paper-figure-slot">
            <img src="pictures/sensitivity/eps_anneal_steps/eps_anneal_steps_convergence.png" alt="MA50 return versus episode for five eps-anneal values" loading="eager" decoding="async" />
          </div>
          <figcaption><strong>Convergence (MA50).</strong> All curves climb quickly before episode 150. <span class="mono">9600</span> ends highest (near <strong>+5</strong> at episode 500). <span class="mono">12000</span> oscillates and finishes near <strong>0</strong>. <span class="mono">16800</span> declines late and ends negative (~<strong>&minus;3</strong>). <span class="mono">14400</span> is volatile around +1; <span class="mono">19200</span> stays smoother near <strong>0</strong>.</figcaption>
        </figure>
        <figure class="paper-figure paper-figure-wide">
          <div class="paper-figure-slot">
            <img src="pictures/sensitivity/eps_anneal_steps/eps_anneal_steps_stability_variance.png" alt="Return variance versus eps-anneal-steps" loading="eager" decoding="async" />
          </div>
          <figcaption><strong>Training stability.</strong> Return variance increases from <strong>9600</strong> (~98) through <strong>16800</strong> (tallest bar, ~115), then drops for <strong>19200</strong> to the lowest value in the sweep (~82).</figcaption>
        </figure>
      </div>

      <h3>5.4 Replay batch size (<code class="mono">--batch</code>)</h3>
      <div class="figure-stack">
        <figure class="paper-figure paper-figure-wide">
          <div class="paper-figure-slot">
            <img src="pictures/sensitivity/batch/batch_winrate.png" alt="Win rate versus batch size" loading="eager" decoding="async" />
          </div>
          <figcaption><strong>Win rate.</strong> Win rate increases monotonically with batch size in this grid: about <strong>0.15</strong> at 32, <strong>0.28</strong> at 64, <strong>0.34</strong> at 128, and roughly <strong>0.54</strong> at 256.</figcaption>
        </figure>
        <figure class="paper-figure paper-figure-wide">
          <div class="paper-figure-slot">
            <img src="pictures/sensitivity/batch/batch_convergence.png" alt="MA50 return versus episode for four batch sizes" loading="eager" decoding="async" />
          </div>
          <figcaption><strong>Convergence (MA50).</strong> <span class="mono">batch=256</span> reaches the highest final average (about <strong>+6</strong>). <span class="mono">128</span> tracks near <strong>+5</strong> with little drift after episode 200. <span class="mono">32</span> settles between <strong>4</strong> and <strong>5</strong>. <span class="mono">64</span> shows a clear mid-training dip toward <strong>+1</strong> before recovering to ~<strong>4.5</strong>.</figcaption>
        </figure>
        <figure class="paper-figure paper-figure-wide">
          <div class="paper-figure-slot">
            <img src="pictures/sensitivity/batch/batch_stability_variance.png" alt="Return variance versus batch size" loading="eager" decoding="async" />
          </div>
          <figcaption><strong>Training stability.</strong> Variance is largest for batch <strong>32</strong> (~80), lower at <strong>64</strong> (~60) and <strong>128</strong> (~64), and smallest for <strong>256</strong> (~38).</figcaption>
        </figure>
      </div>

      <h3>5.5 Discount factor (<code class="mono">--gamma</code>)</h3>
      <div class="figure-stack">
        <figure class="paper-figure paper-figure-wide">
          <div class="paper-figure-slot">
            <img src="pictures/sensitivity/gamma/gamma_winrate.png" alt="Win rate versus gamma" loading="eager" decoding="async" />
          </div>
          <figcaption><strong>Win rate.</strong> The tallest bar is at <span class="mono">0.92475</span> (~<strong>0.58</strong>). <span class="mono">0.999</span> is next (~<strong>0.51</strong>). <span class="mono">0.9</span> is lowest (~<strong>0.18</strong>); <span class="mono">0.9495</span> and <span class="mono">0.97425</span> sit in between (~<strong>0.22</strong> and ~<strong>0.32</strong>).</figcaption>
        </figure>
        <figure class="paper-figure paper-figure-wide">
          <div class="paper-figure-slot">
            <img src="pictures/sensitivity/gamma/gamma_convergence.png" alt="MA50 return versus episode for five gamma values" loading="eager" decoding="async" />
          </div>
          <figcaption><strong>Convergence (MA50).</strong> Every curve moves from roughly <strong>&minus;6</strong> into positive returns within the first ~150 episodes. <span class="mono">0.97425</span> shows a sharp dip near episodes 100&ndash;150 before recovering. By the end, all five lines occupy a similar band (about return <strong>4&ndash;5.5</strong>) with overlapping fluctuations.</figcaption>
        </figure>
        <figure class="paper-figure paper-figure-wide">
          <div class="paper-figure-slot">
            <img src="pictures/sensitivity/gamma/gamma_stability_variance.png" alt="Return variance versus gamma" loading="eager" decoding="async" />
          </div>
          <figcaption><strong>Training stability.</strong> Variance is lowest for <span class="mono">0.92475</span> (~<strong>42</strong>), slightly higher at <span class="mono">0.9</span> and <span class="mono">0.9495</span> (~<strong>44&ndash;46</strong>), jumps for <span class="mono">0.97425</span> (~<strong>58</strong>), and is highest for <span class="mono">0.999</span> (~<strong>61</strong>).</figcaption>
        </figure>
      </div>
      <p class="figure-ref">Source tree: <code class="mono">pictures/sensitivity/lr/</code>, <code class="mono">optim_steps/</code>, <code class="mono">eps_anneal_steps/</code>, <code class="mono">batch/</code>, <code class="mono">gamma/</code>.</p>
    </section>
"""

# Section 6: qualitative play traces (with vs without mandatory tactical rules).
PRACTICE_SECTION_HTML = """
    <section id="sec-play">
      <h2 class="sec">6. Hands-on practice</h2>
      <p>
        This section complements the quantitative benchmarks with four saved board states from interactive sessions. The
        first group uses the <strong>rule-augmented</strong> stack (<code class="mono">coding_rules</code>): the policy is
        wrapped with hard win/block priorities so the agent must answer imminent opponent wins. The second group uses the
        <strong>plain-reward</strong> stack (<code class="mono">coding_no_rules</code>): the same network can still pursue
        its own winning geometry, but no mandatory interception layer is applied at inference.
      </p>

      <h3>6.1 With mandatory rules (<code class="mono">with_rules</code>)</h3>
      <p>
        Each panel freezes a moment where the human side is one stone away from closing a lethal pattern on the
        stochastic super-board. The cyan / blue highlight marks the cell the agent is forced to take in order to break
        the threat, illustrating that the tactical rule head recognises &ldquo;opponent about to win&rdquo; across
        different geometries.
      </p>
      <div class="figure-stack">
        <figure class="paper-figure paper-figure-wide">
          <div class="paper-figure-slot">
            <img src="pictures/play/with_rules/stop_oppo_win_hor.png" alt="Board state: horizontal four-in-a-row threat and block" loading="eager" decoding="async" />
          </div>
          <figcaption><strong>Horizontal four-in-a-row.</strong> On the wide lower band, three opposing stones already sit on one row with a single gap before a fourth would complete a horizontal line. The highlighted empty cell is exactly that gap: occupying it is the mandatory block that prevents the horizontal win.</figcaption>
        </figure>
        <figure class="paper-figure paper-figure-wide">
          <div class="paper-figure-slot">
            <img src="pictures/play/with_rules/stop_oppo_win_ver.png" alt="Board state: vertical threat across levels and block" loading="eager" decoding="async" />
          </div>
          <figcaption><strong>Vertical four spanning the stepped layout.</strong> Three opposing stones stack in one column across the raised central section, leaving only the cell above them to finish a vertical four. The marked square is the interception point the rule layer selects before the column closes.</figcaption>
        </figure>
        <figure class="paper-figure paper-figure-wide">
          <div class="paper-figure-slot">
            <img src="pictures/play/with_rules/stop_oppo_win_diag.png" alt="Board state: diagonal slant threat and block" loading="eager" decoding="async" />
          </div>
          <figcaption><strong>Diagonal / slant five.</strong> The opponent has linked several stones along a diagonal ridge toward the length-five winning pattern used in this variant. The response highlight sits on the continuation cell that cuts the slanted threat instead of letting it extend unchecked.</figcaption>
        </figure>
      </div>
      <p class="figure-ref">Files: <code class="mono">pictures/play/with_rules/stop_oppo_win_hor.png</code>, <code class="mono">stop_oppo_win_ver.png</code>, <code class="mono">stop_oppo_win_diag.png</code>.</p>

      <h3>6.2 Without mandatory rules (<code class="mono">no_rules</code>)</h3>
      <p>
        The checkpoint is still competent at shaping its own winning lines, but greedy action selection no longer rewrites
        moves to enforce defence, so the agent may ignore nearby opponent threats while it pushes its own structure forward.
      </p>
      <div class="figure-stack">
        <figure class="paper-figure paper-figure-wide">
          <div class="paper-figure-slot">
            <img src="pictures/play/no_rules/free_play.png" alt="Board state: agent extends own winning line without blocking" loading="eager" decoding="async" />
          </div>
          <figcaption><strong>Free pursuit of a winning shape.</strong> The agent concentrates on lengthening its own vertical run toward the winning span; clusters of opposing stones remain nearby without receiving a forced block. The board therefore shows effective offensive planning even though interception behaviour is absent.</figcaption>
        </figure>
      </div>
      <p class="figure-ref">File: <code class="mono">pictures/play/no_rules/free_play.png</code>.</p>

      <h3>6.3 Comparison: behaviour with vs without mandatory rules</h3>
      <ul class="body-list">
        <li><strong>Defence vs offence.</strong> Under <code class="mono">with_rules</code>, every screenshot centres on an
          imminent opponent win (horizontal four, vertical four across levels, or diagonal five) and the same story line
          appears: the highlighted move is a block dictated by the tactical wrapper, not merely a high-<span class="mono">Q</span>
          suggestion.</li>
        <li><strong>Same model, different action filter.</strong> Under <code class="mono">no_rules</code>, the agent keeps
          stacking its own winning geometry and reaches a decisive vertical span while leaving human threats unanswered.
          That contrast isolates the role of the mandatory rule layer: it is what re-injects reliable interception, whereas
          the learned policy alone still demonstrates skill, but optimises for its own lines first.</li>
        <li><strong>Takeaway.</strong> The qualitative pair shows that reward shaping during training is not a substitute
          for a hard safety-style rule at play time when instant blocks must never be missed; conversely, removing the
          wrapper makes the agent&rsquo;s offensive intent easier to see because it is no longer side-tracked by forced
          defensive moves.</li>
      </ul>
    </section>
"""


def build_html(*, title: str, organization: str) -> str:
    org_block = ""
    if organization.strip():
        org_block = f'<p class="meta org">{_escape(organization.strip())}</p>'
    author_markup = _author_chips_markup()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{_escape(title)}</title>
  <style>
    :root {{
      --text: #14201a;
      --text-soft: #3d4f44;
      --muted: #5c6b73;
      --rule: #e4e9ee;
      --rule-strong: #cfd8e3;
      --accent: #1d56c8;
      --accent-hover: #1248b0;
      --accent-tint: #e8f0fc;
      --inline-code-bg: #eef2f7;
      --inline-code-border: #dce4ee;
      --block-code-bg: #e9f0f8;
      --block-code-border: #c5d6ea;
      --block-code-edge: #6b9bd4;
      --bg: #e8ecf1;
      --bg-sheet: #ffffff;
      --font: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
      --sans: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      --radius: 12px;
      --radius-sm: 8px;
      --shadow: 0 2px 20px rgba(15, 23, 42, 0.06);
      /* Strict scale: body < h4 < h3 < h2 < h1 (all rem) */
      --fs-body: 0.875rem;
      --fs-small: 0.8125rem;
      --fs-h4: 0.96875rem;
      --fs-h3: 1.0625rem;
      --fs-h2: 1.1875rem;
      --fs-h1: clamp(1.375rem, 1.1vw + 1.1rem, 1.75rem);
    }}
    *, *::before, *::after {{ box-sizing: border-box; }}
    html {{
      scroll-behavior: smooth;
      scroll-padding-top: 1.25rem;
      font-size: 16px;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      padding: clamp(1.25rem, 4vw, 2.5rem) clamp(0.85rem, 2.5vw, 1.5rem) 3rem;
      color: var(--text);
      background: #eceff3;
      font-family: var(--sans);
      line-height: 1.58;
      font-size: var(--fs-body);
      letter-spacing: 0.01em;
      -webkit-font-smoothing: antialiased;
    }}
    .sheet {{
      max-width: min(56rem, 93vw);
      margin: 0 auto;
      background: var(--bg-sheet);
      padding: clamp(1.75rem, 3.5vw, 2.65rem) clamp(1.25rem, 3.5vw, 2.75rem) clamp(2.1rem, 4vw, 3rem);
      border-radius: 12px;
      box-shadow: var(--shadow);
      border: 1px solid var(--rule);
    }}
    .report-header {{
      margin: 0 0 2rem;
      padding-bottom: 1.65rem;
      border-bottom: 1px solid var(--rule);
      position: relative;
    }}
    .report-header::after {{
      content: "";
      position: absolute;
      left: 0;
      bottom: -1px;
      width: 7rem;
      height: 3px;
      border-radius: 2px;
      background: linear-gradient(90deg, var(--accent), #7eb0f4);
    }}
    .doc-kicker {{
      font-family: var(--sans);
      font-size: 0.65625rem;
      font-weight: 600;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: #64748b;
      margin: 0 0 0.75rem;
    }}
    .title-block {{
      position: relative;
      margin: 0 0 1.15rem;
      max-width: 44rem;
      padding: 0.65rem 0 0.85rem 1rem;
      border-left: 4px solid var(--accent);
      background: linear-gradient(90deg, var(--accent-tint), rgba(255,255,255,0));
      border-radius: 0 10px 10px 0;
    }}
    h1.report-title {{
      font-family: var(--sans);
      font-size: var(--fs-h1);
      font-weight: 700;
      line-height: 1.2;
      letter-spacing: -0.04em;
      margin: 0;
      color: #0b1220;
      background: linear-gradient(115deg, #0f172a 0%, #1e3a5f 45%, #1d56c8 100%);
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    @supports not ((-webkit-background-clip: text) or (background-clip: text)) {{
      h1.report-title {{
        color: #0f172a;
        background: none;
        -webkit-text-fill-color: unset;
      }}
    }}
    .byline {{
      font-family: var(--sans);
      display: flex;
      flex-wrap: wrap;
      align-items: baseline;
      gap: 0.35rem 1rem;
    }}
    .meta {{
      font-size: var(--fs-small);
      color: var(--muted);
      margin: 0;
      letter-spacing: 0;
    }}
    .meta.author {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.45rem 0.55rem;
      font-weight: 600;
      font-size: var(--fs-h4);
      color: var(--text);
      letter-spacing: 0;
      line-height: 1.45;
      max-width: 100%;
      margin: 0;
    }}
    .author-chip {{
      display: inline-flex;
      align-items: stretch;
      border: 1px solid #cdd7e6;
      border-radius: 9px;
      overflow: hidden;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.07);
      background: #fff;
    }}
    .author-chip-cell {{
      display: inline-flex;
      align-items: center;
      padding: 0.3rem 0.55rem 0.32rem;
      font-size: 0.8125rem;
      font-weight: 600;
      line-height: 1.25;
    }}
    .author-chip-name {{
      background: linear-gradient(180deg, #ffffff 0%, #f6f8fc 100%);
      color: #0f172a;
      border-right: 1px solid #e2e8f0;
    }}
    .author-chip-id {{
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.02em;
      color: #1e3a5f;
      background: #eef2f9;
    }}
    .meta.org {{ font-weight: 400; color: var(--muted); font-size: var(--fs-small); }}
    nav.toc {{
      font-family: var(--sans);
      margin: 0 0 2.25rem;
      padding: 0;
      border-radius: var(--radius-sm);
      overflow: hidden;
      border: 1px solid var(--rule);
      background: #fafbfc;
      box-shadow: 0 1px 0 rgba(255,255,255,.9) inset;
    }}
    .toc-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      padding: 0.85rem 1.1rem;
      background: linear-gradient(180deg, #fff 0%, #f4f6f9 100%);
      border-bottom: 1px solid var(--rule);
    }}
    nav.toc h2 {{
      margin: 0;
      padding: 0;
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.2em;
      color: #475569;
      font-weight: 700;
      border: none;
    }}
    .toc-meta {{
      font-size: 0.6875rem;
      font-weight: 600;
      color: var(--muted);
      letter-spacing: 0.04em;
    }}
    nav.toc ol.toc-list {{
      margin: 0;
      padding: 0;
      list-style: none;
      counter-reset: item;
    }}
    nav.toc ol.toc-list > li {{
      margin: 0;
      counter-increment: item;
      border-bottom: 1px solid #eef1f4;
    }}
    nav.toc ol.toc-list > li:last-child {{
      border-bottom: none;
    }}
    a.toc-link {{
      display: flex;
      align-items: center;
      gap: 0.9rem;
      padding: 0.62rem 1rem 0.62rem 1.05rem;
      color: #1e293b;
      text-decoration: none;
      border-bottom: none;
      font-size: var(--fs-small);
      font-weight: 500;
      line-height: 1.35;
      transition: background 0.14s ease, color 0.14s ease;
    }}
    a.toc-link:hover {{
      background: rgba(29, 86, 200, 0.06);
      color: var(--accent);
    }}
    .toc-idx {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 1.65rem;
      height: 1.65rem;
      padding: 0 0.35rem;
      font-size: 0.625rem;
      font-weight: 800;
      letter-spacing: 0.04em;
      color: var(--accent);
      background: #fff;
      border: 1px solid #dbe4f0;
      border-radius: 6px;
      flex-shrink: 0;
    }}
    a.toc-link:hover .toc-idx {{
      background: var(--accent-tint);
      border-color: #b8d0f0;
    }}
    .toc-text {{
      flex: 1;
      min-width: 0;
    }}
    a.toc-link,
    a.toc-link:hover {{
      border-bottom: none;
    }}
    section {{
      scroll-margin-top: 1.25rem;
    }}
    /* Late sections sit below many figures; extra margin + eager images avoid first-visit hash scroll landing short. */
    #sec-play,
    #sec-conclusion {{
      scroll-margin-top: clamp(2.5rem, 8vh, 5rem);
    }}
    section + section {{
      margin-top: 0.35rem;
    }}
    section[id^="sec-"] {{
      padding-bottom: 0.35rem;
    }}
    h2.sec {{
      font-family: var(--sans);
      font-size: var(--fs-h2);
      font-weight: 700;
      letter-spacing: -0.022em;
      line-height: 1.32;
      color: #0f172a;
      margin: 2.1rem 0 0.75rem;
      padding: 0.35rem 0 0.5rem 0.85rem;
      border: none;
      border-left: 4px solid var(--accent);
      border-bottom: 1px solid var(--rule);
      background: linear-gradient(90deg, rgba(232, 240, 252, 0.85) 0%, rgba(255,255,255,0) 65%);
    }}
    #sec-abstract h2.sec {{
      margin-top: 0;
    }}
    header.report-header + #sec-abstract {{
      margin-bottom: 0.25rem;
    }}
    #sec-abstract + nav.toc {{
      margin-top: 1.5rem;
      padding-top: 1.1rem;
      border-top: 1px solid var(--rule);
    }}
    #sec-abstract .abstract-prose-box {{
      text-align: justify;
      text-justify: inter-word;
      hyphens: auto;
      -webkit-hyphens: auto;
      text-wrap: pretty;
      margin: 0.55rem 0 0;
      padding: 0.85rem 1rem 0.9rem;
      background: #faf9fb;
      border: 1px solid #e6e4ea;
      border-top: 3px solid #6f5c8c;
      border-radius: 10px;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.055);
    }}
    #sec-abstract .abstract-prose-box > p {{
      margin: 0 0 0.75rem 0;
      padding: 0;
      background: transparent;
      border: none;
      border-radius: 0;
      box-shadow: none;
      text-align: inherit;
      hyphens: inherit;
      -webkit-hyphens: inherit;
    }}
    #sec-abstract .abstract-prose-box > p:last-child {{
      margin-bottom: 0;
    }}
    h3 {{
      font-family: var(--sans);
      font-size: var(--fs-h3);
      font-weight: 600;
      letter-spacing: -0.015em;
      line-height: 1.35;
      color: #152a45;
      margin: 1.5rem 0 0.45rem;
      padding: 0.25rem 0 0.25rem 0.75rem;
      border-left: 3px solid #6b93d6;
      background: transparent;
    }}
    h4 {{
      font-size: var(--fs-h4);
      margin: 1.15rem 0 0.35rem;
      font-weight: 600;
      font-family: var(--sans);
      letter-spacing: -0.01em;
      line-height: 1.35;
      color: #334155;
      padding: 0.15rem 0 0.15rem 0.55rem;
      border-left: 2px solid #cbd5e1;
    }}
    p {{ margin: 0.5rem 0; color: var(--text); }}
    /*
      Body paragraphs: “card” look — top accent bar, full rounded corners, outer shadow, sans prose.
      (Code blocks: left bar, right-only rounding, monospace — different silhouette.)
    */
    article.sheet section p:not(.todo):not(.snippet-caption):not(.figure-ref):not(.mono) {{
      text-align: justify;
      text-justify: inter-word;
      hyphens: auto;
      -webkit-hyphens: auto;
      text-wrap: pretty;
      margin: 0.55rem 0;
      padding: 0.72rem 1rem 0.75rem;
      background: #faf9fb;
      border: 1px solid #e6e4ea;
      border-top: 3px solid #6f5c8c;
      border-radius: 10px;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.055);
    }}
    /* Long rule bullets (e.g. §1.1): same reading width as paragraphs, dashed frame + dotted left rail */
    article.sheet section ul.body-list {{
      list-style: none;
      padding-left: 0;
      margin: 0.55rem 0 0.85rem;
    }}
    article.sheet section ul.body-list > li {{
      text-align: justify;
      text-justify: inter-word;
      hyphens: auto;
      -webkit-hyphens: auto;
      text-wrap: pretty;
      margin: 0.42rem 0;
      padding: 0.58rem 0.9rem 0.6rem 1rem;
      color: var(--text);
      background: linear-gradient(180deg, #fdfcfd 0%, #f5f2f9 100%);
      border: 1px dashed #c9bfd5;
      border-left: 2px dotted #7d6b9e;
      border-radius: 8px;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.75);
    }}
    article.sheet header p,
    article.sheet .meta,
    article.sheet .todo {{
      text-align: start;
      hyphens: manual;
      -webkit-hyphens: manual;
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
      border-bottom: 1px solid rgba(29, 86, 200, 0.25);
      transition: color 0.15s, border-color 0.15s;
    }}
    a:hover {{
      color: var(--accent-hover);
      border-bottom-color: var(--accent);
    }}
    ul {{
      margin: 0.6rem 0 0.75rem;
      padding-left: 1.35rem;
    }}
    li {{
      margin: 0.4rem 0;
      color: var(--text);
      text-align: start;
    }}
    li::marker {{ color: var(--accent); }}
    strong {{ font-weight: 600; color: #0f172a; }}
    .todo {{
      font-family: var(--sans);
      color: var(--muted);
      font-style: italic;
      font-size: var(--fs-small);
      line-height: 1.52;
      border-left: 3px solid var(--rule-strong);
      padding: 0.65rem 0 0.65rem 1rem;
      margin: 0.85rem 0;
      background: #f8fafc;
      border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    }}
    code, .mono {{
      font-family: ui-monospace, "Cascadia Code", "SF Mono", Menlo, Consolas, monospace;
      font-size: 0.86em;
      background: var(--inline-code-bg);
      border: 1px solid var(--inline-code-border);
      padding: 0.15em 0.42em;
      border-radius: 4px;
      color: #1e3a5f;
    }}
    pre code {{
      border: none;
      padding: 0;
      background: transparent;
      color: inherit;
      font-size: inherit;
    }}
    .figure-ref {{
      font-family: var(--sans);
      font-size: var(--fs-small);
      color: var(--muted);
      margin-top: 0.25rem;
    }}
    .training-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 0.75rem 1rem;
      margin: 1rem 0 1.25rem;
      align-items: stretch;
    }}
    @media (max-width: 720px) {{
      .training-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 480px) {{
      .training-grid {{ grid-template-columns: 1fr; }}
    }}
    .training-cell {{
      margin: 0;
      border: 1px solid var(--rule-strong);
      border-radius: var(--radius-sm);
      background: var(--bg-sheet);
      overflow: hidden;
      box-shadow: 0 1px 8px rgba(15, 23, 42, 0.06);
      display: flex;
      flex-direction: column;
      min-height: 0;
    }}
    .training-cell img {{
      display: block;
      width: 100%;
      height: auto;
      vertical-align: middle;
      flex: 0 0 auto;
    }}
    figcaption.training-caption {{
      flex-shrink: 0;
      text-align: center;
      font-family: var(--sans);
      font-size: 0.78rem;
      font-weight: 600;
      color: var(--text-soft);
      padding: 0.42rem 0.45rem 0.5rem;
      border-top: 1px solid var(--rule);
      background: #f8fafc;
      line-height: 1.35;
    }}
    .training-cap-idx {{
      font-weight: 500;
      font-size: 0.72rem;
      color: var(--muted);
    }}
    .figure-stack {{
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
      margin: 1.1rem 0 1.4rem;
    }}
    .paper-figure {{
      margin: 0;
      border: 1px solid var(--rule-strong);
      border-radius: var(--radius);
      background: var(--bg-sheet);
      overflow: hidden;
      box-shadow: var(--shadow);
      display: flex;
      flex-direction: column;
      min-height: 0;
    }}
    .paper-figure-wide {{
      height: auto;
    }}
    .paper-figure-slot {{
      flex: 1 1 0;
      min-height: 220px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: linear-gradient(180deg, #fafbfc 0%, #f1f5f9 100%);
    }}
    .paper-figure-wide .paper-figure-slot {{
      flex: 0 0 auto;
      min-height: 0;
      padding: 0.65rem 1rem 0.45rem;
    }}
    .paper-figure-slot img {{
      display: block;
      max-width: 100%;
      max-height: 100%;
      width: auto;
      height: auto;
      object-fit: contain;
    }}
    .paper-figure-wide .paper-figure-slot img {{
      width: 100%;
      height: auto;
      max-width: 100%;
      max-height: none;
      object-fit: contain;
    }}
    .paper-figure figcaption {{
      font-family: var(--sans);
      font-size: var(--fs-small);
      color: var(--text-soft);
      padding: 0.55rem 0.75rem 0.7rem;
      line-height: 1.48;
    }}
    pre.code-snippet {{
      font-family: ui-monospace, "Cascadia Code", "SF Mono", Menlo, Consolas, monospace;
      font-size: 0.75rem;
      line-height: 1.48;
      letter-spacing: 0.01em;
      background: linear-gradient(165deg, #edf3fb 0%, #e4eef8 100%);
      border: 1px solid #b8cce8;
      border-left: 5px solid #2a6fbf;
      border-radius: 0 10px 10px 0;
      padding: 0.95rem 1rem 1rem 1.1rem;
      overflow-x: auto;
      margin: 0.25rem 0 1.25rem;
      color: #132433;
      box-shadow: inset 0 2px 4px rgba(255, 255, 255, 0.45);
    }}
    pre.code-snippet code {{
      display: block;
      white-space: pre;
    }}
    .snippet-caption {{
      font-family: var(--sans);
      font-size: 0.7rem;
      font-weight: 500;
      color: var(--muted);
      margin: 1.1rem 0 0.35rem;
      letter-spacing: 0.01em;
    }}
    .snippet-caption code {{
      font-weight: 600;
      color: #334e68;
    }}
    .param-wrap {{
      margin: 1rem 0 1.35rem;
      overflow-x: auto;
      border-radius: 10px;
      border: 1px solid #dce3ed;
      background: #fbfcfe;
      box-shadow: 0 2px 10px rgba(15, 23, 42, 0.045);
    }}
    table.param-table {{
      width: 100%;
      border-collapse: collapse;
      font-family: var(--sans);
      font-size: 0.8125rem;
      line-height: 1.45;
    }}
    table.param-table caption {{
      text-align: left;
      font-weight: 700;
      font-size: 0.84rem;
      letter-spacing: 0.02em;
      color: #1e293b;
      padding: 0.65rem 1rem 0.55rem;
      background: linear-gradient(90deg, #eef2ff 0%, #f4f7fb 55%, #fafbfc 100%);
      border-bottom: 1px solid #d8e0ea;
    }}
    table.param-table thead th {{
      background: #f1f5f9;
      color: #475569;
      font-weight: 600;
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      text-align: left;
      padding: 0.45rem 0.85rem;
      border-bottom: 1px solid #e2e8f0;
    }}
    table.param-table td {{
      padding: 0.5rem 0.85rem;
      border-top: 1px solid #eef2f7;
      vertical-align: top;
      color: var(--text);
    }}
    table.param-table tbody tr:hover td {{
      background: rgba(29, 86, 200, 0.035);
    }}
    table.param-table td.mono-cell {{
      font-family: ui-monospace, Menlo, Consolas, monospace;
      font-size: 0.78rem;
      color: #1a3a54;
      white-space: nowrap;
    }}
    table.param-table td.mono-cell code {{
      font-size: inherit;
      background: transparent;
      border: none;
      padding: 0;
    }}
  </style>
</head>
<body>
  <article class="sheet">
    <header class="report-header">
      <p class="doc-kicker">Course project · Reinforcement learning</p>
      <div class="title-block">
        <h1 class="report-title">{_escape(title)}</h1>
      </div>
      <div class="byline">
        <p class="meta author">{author_markup}</p>
        {org_block}
      </div>
    </header>

{ABSTRACT_SECTION_HTML}

    <nav class="toc" aria-labelledby="toc-heading">
      <div class="toc-head">
        <h2 id="toc-heading">Contents</h2>
        <span class="toc-meta" aria-hidden="true">7 sections</span>
      </div>
      <ol class="toc-list">
        <li><a class="toc-link" href="#sec-intro"><span class="toc-idx">01</span><span class="toc-text">Introduction and Problem Background</span></a></li>
        <li><a class="toc-link" href="#sec-model"><span class="toc-idx">02</span><span class="toc-text">Model Architecture and Learning Objective</span></a></li>
        <li><a class="toc-link" href="#sec-training"><span class="toc-idx">03</span><span class="toc-text">Training Procedure and Hyperparameters</span></a></li>
        <li><a class="toc-link" href="#sec-experiments"><span class="toc-idx">04</span><span class="toc-text">Experiments and Results</span></a></li>
        <li><a class="toc-link" href="#sec-sensitivity"><span class="toc-idx">05</span><span class="toc-text">Hyperparameter sensitivity analysis</span></a></li>
        <li><a class="toc-link" href="#sec-play"><span class="toc-idx">06</span><span class="toc-text">Hands-on practice</span></a></li>
        <li><a class="toc-link" href="#sec-conclusion"><span class="toc-idx">07</span><span class="toc-text">Conclusion</span></a></li>
      </ol>
    </nav>
    <!-- Body: use p / ul.body-list / pre.code-snippet / p.todo as in module docstring -->

{INTRODUCTION_SECTION_HTML}

    {MODEL_SECTION_HTML}

{TRAINING_SECTION_HTML}

{EXPERIMENTS_SECTION_HTML}

{SENSITIVITY_SECTION_HTML}

{PRACTICE_SECTION_HTML}

    <section id="sec-conclusion">
      <h2 class="sec">7. Conclusion</h2>
      <p>
        This course project targets a <strong>stochastic Super Tic-Tac-Toe</strong> variant and implements a reproducible
        deep reinforcement learning pipeline: assignment rules and randomness are encoded in a Gymnasium environment, agents
        are trained with TorchRL using a <strong>Dueling Double DQN</strong> architecture, and we keep two parallel
        tracks&mdash;one with a <strong>mandatory tactical layer</strong> (<code class="mono">coding_rules</code>) and one
        that relies <strong>only on reward shaping</strong> (<code class="mono">coding_no_rules</code>). The report
        documents the model structure, training procedure, and hyperparameter choices; evaluates <strong>eleven opponent
        curricula</strong> with learning curves and match statistics; performs <strong>one-at-a-time sensitivity</strong>
        sweeps over learning rate, optimiser steps per rollout, epsilon-anneal length, batch size, and discount factor; and
        closes with board captures that contrast moves under mandatory rules versus plain greedy play.
      </p>
      <p>
        Two observations stood out. First, <strong>training return and win-rate rankings need not agree</strong>: some
        curricula look strong on shaped returns yet fare worse on the random baseline or in the round-robin matrix, while
        no-rule runs need not sit at the top of the return curves. Second, <strong>mandatory rules mainly change when a
        block is compulsory</strong>: with rules enabled, the agent reliably intercepts threats such as horizontal
        four-in-a-row, vertical four across levels, and diagonal five-in-a-row; without them, the policy focuses on
        completing its own winning geometry instead. In hands-on testing we consistently saw that, under compulsory rules,
        the agent occupies cells that would otherwise let us finish a winning line on the very next move.
      </p>
    </section>
  </article>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HTML report framework.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "report.html",
        help="Output HTML path (default: report.html next to this script)",
    )
    parser.add_argument("--title", type=str, default=TITLE)
    parser.add_argument("--organization", type=str, default=ORGANIZATION)
    args = parser.parse_args()

    html = build_html(title=args.title, organization=args.organization)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
