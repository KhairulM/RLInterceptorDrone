# OmniDrones Fixes — RLInterceptorDrone

This document lists every code change made to the vendored
`source/OmniDrones/` tree (relative to its upstream baseline) and the
reason for each. Changes are grouped by file. The most recent / most
critical fix is listed first.

---

## 1. `omni_drones/robots/drone/multirotor.py` — rotor force application order (CRITICAL)

**Symptom:** Both pursuer and evader drones fell to the ground in pure
free-fall regardless of throttle. Per-step diagnostics showed `vz`
decreasing by ~`g·dt` (≈ -0.16 m/s per 1/60 s step) even when all four
rotor throttles were near max (0.8–0.9).

**Root cause:** `MultirotorBase.apply_action()` was calling

```python
self.rotors_view.apply_forces_and_torques_at_pos(...)   # rotor thrusts
self.base_link.apply_forces_and_torques_at_pos(...)     # body drag/torque
```

In Isaac Sim 5.1 / PhysX tensor API, applying forces to one rigid body
in an articulation overwrites previously-staged forces on other bodies
of that same articulation in the same step. Because `base_link` was
applied second, it silently zeroed the rotor thrust contribution from
`rotors_view`, leaving only gravity → free-fall.

**Fix:** Restored the original ordering — body forces first, rotor
thrusts last — and added a comment so it does not get reversed again.

```python
self.base_link.apply_forces_and_torques_at_pos(
    forces=self.forces.reshape(-1, 3).contiguous(),
    torques=self.torques.reshape(-1, 3).contiguous(),
    is_global=True,
)
self.rotors_view.apply_forces_and_torques_at_pos(
    forces=self.thrusts.reshape(-1, 3).contiguous(),
    is_global=False,
)
```

**Verification:** After fix, evader `vz ≈ 0` and Z holds at spawn
altitude (~5.43 m vs. target 5.43); pursuer responds to commanded
throttle changes. Logged `train/stats.evader_z` averages 2.3–4.7 m
within the configured bounds.

### Other changes in the same file

- `make()` now accepts a `name` argument so multiple drones can coexist
  in one scene without colliding in the registry.

  ```python
  def make(drone_model, controller_str=None, device="cpu", name=None):
      drone = drone_cls(name=name)
      ...
  ```

- `RigidPrimView` instances inside `initialize()` are given per-instance
  unique names: `f"{self.name}_base_link"` and `f"{self.name}_rotors"`.
  Previously these were hard-coded as `"base_link"` and `"rotors"`,
  which raised duplicate-view errors when two drones (e.g. `pursuer` and
  `evader`) were added to the same scene.

- `apply_action()` now sanitizes the input:

  ```python
  actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
  actions = actions.clamp(-1.0, 1.0).contiguous()
  ```

  Plus `nan_to_num` on `self.thrusts`, `self.torques`, and `self.forces`
  before they are dispatched to PhysX. Prevents NaN propagation from a
  bad policy output crashing the GPU pipeline.

- Type hints relaxed to `Optional[str]` / `Optional[RobotCfg]` to match
  the actual call sites in `make()`.

---

## 2. `omni_drones/envs/single/intercept.py` — task rewrite for pursuer/evader

**Symptom (original):** The upstream Intercept task spawned a Hummingbird
pursuer and a stationary `Iris` "target" with no controller; behaviour
was inconsistent at `num_envs > 1` and the evader offered no real
challenge.

**Changes:**

- Replaced the `Iris` static target with a second `MultirotorBase`
  (`evader`) that has its own controller (default
  `LeePositionController`). Both drones are now constructed via
  `MultirotorBase.make(..., name="pursuer")` and
  `MultirotorBase.make(..., name="evader")` so they get distinct
  `RigidPrimView` names (see fix #1).

- `_pre_sim_step()` now drives the evader. For `evader_trajectory_mode
  == "hover"` it calls
  `self.evader_controller.compute(evader_state, target_pos=target_pos)`
  with `target_pos = self.evader_local_pos.squeeze(1)` (the spawn
  position). Without an explicit `target_pos`, the Lee controller
  defaults `target_pos = current_pos`, producing zero restoring force
  and letting the drone drift down.

- Added `_format_action()` to clamp/sanitize and reshape both the
  pursuer policy action and the evader controller output before calling
  `apply_action`.

- Added two stats keys for diagnostics (so the bug above cannot recur
  unnoticed):

  ```python
  "pursuer_z": UnboundedContinuous(...),
  "evader_z":  UnboundedContinuous(...),
  ```

  populated each step in `_compute_reward_and_done()`. They appear as
  `train/stats.pursuer_z` / `train/stats.evader_z` in the PPO logger
  and (when enabled) in W&B.

- Observation rewrite: `evader_rel_hdg`, `pursuer_lin_vel`,
  `pursuer_rot` (flattened rotation matrix). Optional channels gated by
  config booleans:
  - `evader.use_relative_velocity` → adds `evader_rel_lin_vel`
  - `pursuer.use_ab_world_frame` → adds `pursuer_pos`
  - `pursuer.use_rot_speed` → adds `pursuer_rot_vel`
  - `task.time_encoding_dim > 0` → appends sinusoidal encoding to the
    `state` tensor (replaces upstream boolean `time_encoding`).

- New reward components (selectable; current default = `approach_reward`):
  - `_reward_distance_to_evader` — `exp(-k · ||p_e − p_p||)`
  - `_reward_align_velocity_to_heading` — cosine similarity reward
  - `_reward_approach_velocity_to_evader` — projects velocity onto
    line-of-sight, normalized by `pursuer.target_speed`
  - `_reward_success_interception` — binary at `success_radius`
  - `_reward_intercept_time` — time-to-intercept estimator

- Termination rewrite: `pursuer_z < 0.15` OR `||p_e − p_p|| > reset_thres`
  OR NaN in pursuer state OR `distance ≤ success_radius`.
  All masks reshaped to `(num_envs, 1)` to avoid broadcasting bugs that
  surfaced at `num_envs > 1`.

- Added `info_spec` with `drone_state` (13-dim) so downstream loggers
  can record the pursuer's full state without re-querying USD.

- All `*_local_pos`, `*_local_rot`, `*_local_vel` buffers gained a
  middle singleton dim `(num_envs, 1, k)` to match the
  `(env, agent, feature)` convention used everywhere else in
  OmniDrones; `_squeeze_batch()` helper drops it where needed.

---

## 3. `cfg/task/Intercept.yaml` — config schema for pursuer/evader

Replaced the flat `target_*` keys with structured `pursuer:` and
`evader:` blocks that match the new task code:

```yaml
time_encoding_dim: 4
reset_thres: 15.0
success_radius: 0.5
reward_distance_scale: 0.8
action_transform: rate

pursuer:
  target_speed: 15.0
  use_rot_speed: false
  use_ab_world_frame: false

evader:
  model: Hummingbird
  controller: LeePositionController
  speed_range: [0.8, 1.5]
  spawn_distance_range: [4.0, 7.0]
  boundary_mode: bounce
  bounds:
    min: [-6.0, -6.0, 1.0]
    max: [6.0, 6.0, 4.0]
  use_relative_velocity: false
```

The previous boolean `time_encoding` is replaced by the integer
`time_encoding_dim` (0 disables it).

---

## 4. `omni_drones/controllers/lee_position_controller.py` — PyTorch API drift

Two `torch.cross` calls used the legacy positional-`dim` form, which
PyTorch ≥ 2.0 deprecates and will eventually error on:

```python
# before
b2_des = normalize(torch.cross(b3_des, b1_des, 1))
... b2_des.cross(b3_des, 1) ...
acc_des = -rate_error * gain + angvel.cross(angvel)

# after
b2_des = normalize(torch.cross(b3_des, b1_des, dim=-1))
... torch.cross(b2_des, b3_des, dim=-1) ...
acc_des = -rate_error * gain + torch.cross(angvel, angvel, dim=-1)
```

Functionally identical for our 3-vector inputs but silences the
deprecation warning and avoids a future hard break.

---

## 5. `scripts/train.py` and `scripts/play.py` — controller lookup for Intercept

The shared rate-action transform looked up `base_env.controller`, which
the Intercept task does not expose (it has `pursuer_controller` and
`evader_controller` instead). The launcher now falls back gracefully:

```python
controller = getattr(base_env, "controller", None)
if controller is None:
    controller = getattr(base_env, "pursuer_controller", None)
if controller is None:
    raise RuntimeError("Rate action transform requires a controller ...")
transform = RateController(controller.to(base_env.device))
```

Same change in both `train.py` and `play.py`.

---

## 6. `omni_drones/controllers/cfg/lee_controller_iris.yaml` — new config file

Added because `LeePositionController` looks up gains by drone model
name and the upstream tree did not ship an Iris config. Used when the
evader is configured as `model: Iris, controller: LeePositionController`.

```yaml
position_gain:    [3, 3, 3]
velocity_gain:    [2.0, 2.0, 2.0]
attitude_gain:    [0.6, 0.6, 0.035]
angular_rate_gain:[0.1, 0.1, 0.025]
```

---

## Earlier fixes (already merged before this session)

These were applied in previous sessions and are mentioned for
completeness:

- **256-env crash workaround.** The original Intercept used a mixed
  Hummingbird+Iris scene, which triggered an articulation-view
  homogeneity assertion at `num_envs > 1`. Switching to a homogeneous
  Hummingbird+Hummingbird scene with distinct names (#1 above) was the
  permanent fix.

- **Distinct `RigidPrimView` names** (now part of #1) — required to
  spawn two drones in one environment.

- **`MultirotorBase.make(..., name=...)`** — added so two instances of
  the same drone class can be registered (`pursuer`, `evader`) without
  colliding in `RobotBase._robots`.

---

## Summary table

| File | Type of fix | Severity |
|---|---|---|
| `multirotor.py` (force order) | Physics correctness | **Critical** |
| `multirotor.py` (per-instance view names, `make(name=)`) | Multi-drone scene support | High |
| `multirotor.py` (NaN sanitization) | Robustness | Medium |
| `intercept.py` (pursuer/evader rewrite) | Task design | High |
| `intercept.py` (Z stats) | Observability | Medium |
| `Intercept.yaml` (new schema) | Config | Required by #2 |
| `lee_position_controller.py` (`torch.cross dim=`) | Future-proofing | Low |
| `train.py` / `play.py` (controller fallback) | Compatibility | Required by #2 |
| `lee_controller_iris.yaml` (new file) | Missing config | Low |
