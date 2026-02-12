# Changes Documentation — Safe Exploration RL with CBF

This document details every modification made to the original codebase to extend it
from a 2D-only framework to an n-dimensional, multi-scenario system with full
paper-style diagnostics.

---

## Table of Contents

1. [N-Dimensional Generalization](#1-n-dimensional-generalization)
2. [Unicycle Scenario](#2-unicycle-scenario)
3. [Polar Coordinate Basis](#3-polar-coordinate-basis)
4. [Safety Filter Toggle](#4-safety-filter-toggle)
5. [Local Minima Escape](#5-local-minima-escape)
6. [Paper Metrics & Diagrams](#6-paper-metrics--diagrams)
7. [Statistics & Plots](#7-statistics--plots)
8. [Usage Examples](#8-usage-examples)
9. [File-by-File Changelist](#9-file-by-file-changelist)

---

## 1. N-Dimensional Generalization

### Problem
The original code hardcoded 2D goals (`goal_pos = np.zeros(2)`) and assumed
`state_dim == goal_dim == 2`, making it impossible to add systems with more
state variables (e.g., unicycle with heading).

### Solution

**`src/scenarios.py` — Base class additions:**
- Added `goal_dim = 2` attribute to `ScenarioStrategy` (position-only target for all scenarios).
- Added `get_error_state(state, goal_pos)` method:
  - Default: subtracts goal from the first `goal_dim` components of state, keeps the rest.
  - Example for unicycle `[x, y, θ]` with goal `[gx, gy]` → error `[x−gx, y−gy, θ]`.
  - Each scenario can override this for custom error definitions.
- Added `init_extra_state(state, rng)` method:
  - Default: no-op (returns state unchanged).
  - Unicycle overrides to randomize heading.

**`src/custom_env.py` — Environment generalization:**
- `goal_pos` dimension now reads from `scenario.goal_dim` instead of hardcoded `2`.
- Observation space: `state_dim + goal_dim` (was `state_dim + 2`).
- **Reward computation** uses `scenario.get_error_state()` so `Q` can be `(state_dim × state_dim)`:
  ```python
  x_error = self.scenario.get_error_state(self.state, self.goal_pos)
  cost_state = x_error.T @ Q @ x_error
  ```
- `reset()` calls `scenario.init_extra_state(state, rng)` to initialize heading, etc.
- Rendering adds a **heading indicator** (black line) for systems with `state_dim > 2`.

**`controllers/rl_algo.py` — Agent generalization:**
- `goal` extracted as `obs[state_dim:]` (was `obs[-2:]`).
- `x_rel` computed via `scenario.get_error_state(state, goal)` (was `state − goal`).
- Simulation-of-experience loop also uses `get_error_state`.

**Why this works:**
The key insight is that the goal is always a *position* target (2D), but the *error
state* used by the value function can be higher-dimensional. The scenario defines
how to map `(state, goal)` → `error` via `get_error_state`.

---

## 2. Unicycle Scenario

### Model

$$
\dot{x} = v \cos\theta + \theta_1 x, \quad
\dot{y} = v \sin\theta + \theta_2 y, \quad
\dot{\theta} = \omega + \theta_3 \theta
$$

| Property | Value |
|----------|-------|
| State dim | 3 (`[x, y, θ]`) |
| Action dim | 2 (`[v, ω]`) |
| Goal dim | 2 (position only) |
| θ_true | `[-0.1, -0.1, -0.05]` (drag/damping) |
| Q | `diag(1, 1, 0.1)` |
| R | `diag(0.5, 0.5)` |
| c_b | `0.5` |

### Files

**`src/scenarios.py` — `UnicycleSystem` class:**
- `get_f(x)`: Returns uncertain drag terms `[θ1·x, θ2·y, θ3·θ]`.
- `get_g(x)`: Standard unicycle input matrix `[[cos θ, 0], [sin θ, 0], [0, 1]]`.
- `get_casadi_model()`: CasADi symbolic model for CBF safety filter.
  - `h_obs` barrier uses only position `(x[0], x[1])` — heading doesn't affect
    collision geometry.
- `get_regressor(x)`: `Y = diag(x, y, θ)` so that `f = Y·θ`.
- `get_error_state()`: `[x−gx, y−gy, θ]`.
- `init_extra_state()`: Randomizes heading ∈ `[−π, π]`.

### Design choices
- **Uncertain parameters**: Position-dependent drag and heading damping. This gives
  the dynamics learner something to estimate (θ_dim = 3), making the concurrent
  learning component non-trivial.
- **Position-only goal**: The unicycle must navigate to (0,0) regardless of final
  heading. The Q matrix penalizes heading slightly (0.1) to encourage facing the
  goal direction.

---

## 3. Polar Coordinate Basis

### Motivation
Standard Cartesian polynomial features `φ(ex, ey, θ)` treat all state components
uniformly. For unicycle-like systems, the *natural* coordinates are polar:
- **ρ** = distance to goal
- **α** = heading error (angle between heading and line-of-sight to goal)

These directly relate to the control objectives: reduce ρ and align α.

### Implementation (`src/basis_functions.py` — `PolarBasis`)

**Polar transform:**
$$
\rho = \sqrt{e_x^2 + e_y^2}, \quad
\alpha = \text{atan2}(e_y, e_x) - \theta
$$

**Features:** Polynomial monomials in $(ρ, α)$ up to degree $d$:
$$
\phi_k(\rho, \alpha) = \rho^{a_k} \cdot \alpha^{b_k}, \quad a_k + b_k \leq d
$$
For degree 2: `{1, ρ, α, ρ², ρα, α²}` → L = 6 features.

**Gradient via chain rule:**
The Jacobian of `[ρ, α]` w.r.t. `[ex, ey, θ]` is:

$$
J = \begin{bmatrix}
e_x/\rho & e_y/\rho & 0 \\
-e_y/\rho^2 & e_x/\rho^2 & -1
\end{bmatrix}
$$

Then `∂φ_k/∂state = [∂φ_k/∂ρ, ∂φ_k/∂α] · J`.

**Singularity handling:** ρ is clamped to `max(ρ, 1e-6)` to avoid division by zero
when the robot is at the goal.

### Comparison experiment
Run both and compare:
```bash
python main.py --scenario unicycle --basis polynomial --episodes 100
python main.py --scenario unicycle --basis polar --episodes 100
```

---

## 4. Safety Filter Toggle

### Change
Added `use_safety_filter` parameter to `SafeMBRL.__init__()`.

**When `use_safety_filter = True` (default):**
- Actions pass through the CBF safety filter as before.
- `get_safe_action()` modifies the nominal action to ensure `h(x) > 0`.

**When `use_safety_filter = False`:**
- The nominal action from the actor is simply clipped to `[-action_max, action_max]`.
- No CBF correction is applied.
- `_last_h_val` is set to `None` (no CBF monitoring).

### Usage
```bash
# Train with safety filter (default)
python main.py --mode train --scenario single_integrator

# Train without safety filter
python main.py --mode train --scenario single_integrator --no-safety

# Compare both automatically
python main.py --mode compare --scenario single_integrator --basis paper_staf
```

The `--mode compare` flag runs both configurations back-to-back and generates
side-by-side comparison plots.

---

## 5. Local Minima Escape

### Problem
With the CBF safety filter, the robot can get **stuck** in local minima where the
RL controller pushes toward the goal but the CBF prevents forward motion (e.g.,
between two obstacles). This creates a deadlock.

### Solution — Persistent Random Perturbation

This is a **pure exploration strategy** (not an explicit controller):

1. **Stuck Detection**: Track position history over last 60 steps. If displacement
   < 0.15 and distance to goal > 0.5, the robot is "stuck".
2. **Escape Burst**: Sample a random direction in *action space* and add it to the
   nominal action for 40 consecutive steps.
3. **Persistence is key**: Unlike i.i.d. noise (which averages out), a persistent
   perturbation in one direction consistently pushes the robot, allowing it to
   escape the local basin.

### Implementation (`controllers/rl_algo.py`)

```python
# Parameters
self._stuck_window = 60       # lookback window
self._stuck_threshold = 0.15  # min displacement to not be stuck
self._escape_duration = 40    # steps of persistent perturbation
self._escape_magnitude = 2.0  # perturbation strength
```

The escape direction is sampled once per escape event and remains constant for the
full duration. The safety filter (if enabled) still constrains the final action, so
the robot won't crash during escape — it just gets a "push" to explore a new direction.

### Why this isn't cheating
- No goal-seeking or obstacle-avoidance logic.
- No PD/PID controller.
- It's an exploration strategy (like ε-greedy or Ornstein–Uhlenbeck noise) that
  kicks in only when learning stalls.

---

## 6. Paper Metrics & Diagrams

All metrics from Cohen & Belta (2021) are now tracked and plotted:

| Paper Figure | Our Metric | Tracked In |
|---|---|---|
| Fig 3: Trajectories | 2D state paths with obstacles | `stored_trajectories` |
| Fig 4: Cost J(x₀) | Integral cost per episode | `episode_cumulative_cost` |
| Fig 5: θ̂ error | ‖θ̂ − θ*‖ per episode | `episode_theta_errors` |
| Fig 6: CBF h(x) | Min h(x) per episode + full traces | `episode_min_h`, `stored_h_traces` |
| Table: Safety | Cumulative violations | `get_cumulative_violations()` |

**Additional** metrics beyond the paper:
- Episode return (reward)
- Bellman error δ convergence
- Critic/Actor weight norms ‖Wc‖, ‖Wa‖
- Rolling success rate
- Goal distance convergence

---

## 7. Statistics & Plots

### New files

**`analysis/metrics.py` — `MetricsTracker` class:**
- `start_episode()`, `record_step()`, `end_episode()` lifecycle.
- Stores both per-episode aggregates and selected full trajectories.
- `save()`/`load()` for persistence via pickle.

**`analysis/plots.py` — Plotting functions:**
Each function takes an optional `ax` parameter for embedding in multi-panel figures:
- `plot_learning_curve()` — Integral cost J(x₀)
- `plot_episode_reward()` — Episode return
- `plot_goal_distance()` — Final distance per episode
- `plot_safety_violations()` — Cumulative violations
- `plot_min_cbf()` — Minimum h(x) per episode
- `plot_parameter_error()` — ‖θ̂ − θ*‖
- `plot_bellman_error()` — Mean |δ|
- `plot_weight_norms()` — ‖Wc‖, ‖Wa‖
- `plot_trajectory_2d()` — State-space paths with obstacles
- `plot_cbf_trace()` — h(x) within episodes
- `plot_success_rate()` — Rolling success %

**Master functions:**
- `generate_all_plots(tracker, ...)` — Generates all individual plots + a dashboard.
- `generate_comparison_plots(tracker_safe, tracker_unsafe, ...)` — Side-by-side
  safety ON vs OFF comparison.

### Output
All plots are saved to `plots/<scenario>_<basis>_<tag>/` as PNG files at 150 DPI.
A summary dashboard combines all key metrics in one figure.

---

## 8. Usage Examples

### Training the paper's exact scenarios

```bash
# Single Integrator with StaF basis (Paper Section 6.1)
python main.py --scenario single_integrator --basis paper_staf --episodes 200

# Underactuated system with StaF basis (Paper Section 6.2)
python main.py --scenario underactuated --basis paper_staf --episodes 200

# Unicycle with polynomial basis
python main.py --scenario unicycle --basis polynomial --episodes 200

# Unicycle with polar basis
python main.py --scenario unicycle --basis polar --episodes 200
```

### Safety filter comparison

```bash
# Automatically trains with and without safety, generates comparison plots
python main.py --mode compare --scenario single_integrator --basis paper_staf --episodes 100
python main.py --mode compare --scenario unicycle --basis polynomial --episodes 100
```

### Evaluation

```bash
python main.py --mode eval --scenario single_integrator --basis paper_staf --episodes 50 --render
```

### Regenerate plots from saved metrics

```bash
python main.py --mode plot --scenario single_integrator --basis paper_staf
```

---

## 9. File-by-File Changelist

### Modified Files

| File | Changes |
|------|---------|
| `src/scenarios.py` | Added `goal_dim`, `get_error_state()`, `init_extra_state()` to base class. Added `UnicycleSystem`. |
| `src/basis_functions.py` | Added `PolarBasis` class. Added note to `PaperStaF` about 2D-only restriction. |
| `src/custom_env.py` | Goal dimension from `scenario.goal_dim`. Cost via `get_error_state()`. Extra state init. Heading rendering. |
| `controllers/rl_algo.py` | `use_safety_filter` flag. `goal_dim` support. `get_error_state()` for `x_rel`. Local minima escape. `reset_episode()`. `_last_h_val` for metrics. |
| `main.py` | New scenarios/bases. `--no-safety` flag. `--mode compare/plot`. `MetricsTracker` integration. Auto-plotting. `train()` function refactor. |

### New Files

| File | Purpose |
|------|---------|
| `analysis/__init__.py` | Package init |
| `analysis/metrics.py` | `MetricsTracker` — records all paper metrics |
| `analysis/plots.py` | All plotting functions (11 individual + 2 master) |
| `changes.md` | This documentation file |

### Unchanged Files

| File | Reason |
|------|--------|
| `controllers/safety_filter.py` | Already n-dim compatible via `pos_indices` and symbolic CasADi model. |
| `controllers/learning_dyn.py` | Already generic (uses `state_dim`, `theta_dim` from scenario). |
