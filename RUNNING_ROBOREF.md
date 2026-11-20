# Running RoboRef as a reward model for RL

RoboRef is a fine-tune of Robometer (Qwen3-VL-4B-Instruct + RBM heads) trained on densely
labeled robot failures with an **asymmetric loss**. This document is the recommended way to
drive downstream RL with it. It is written for someone who has a checkpoint and a cluster and
wants a run that is *comparable to the paper* rather than merely one that starts.

Everything below is either read off the code or measured. Where a number is a starting guess
rather than a measurement, it says so.

> **Read sections 3 and 4.5 before submitting anything.** A reward model whose heads failed
> to load does not crash — it returns noise, and RL looks healthy for a day. Every trap listed
> here has actually cost us a run.

---

## 1. Pick the regime first

The two heads want different RL stacks, and the choice decides everything downstream.

| | **Sparse** | **Dense** |
|---|---|---|
| Head used | `success` only | `progress` only |
| Envs | MetaWorld v3 (CoffeePush, BoxClose), Robomimic | ManiSkill (PullCube-v1) |
| Algorithm | IBRL / RLPD (BC-bootstrapped) | SAC from scratch |
| Stack | [`vlm_ibrl/`](vlm_ibrl/) | [`robometer-policy-learning/`](robometer-policy-learning/) |
| Entry point | `mw_main/train_rl_vlm_mw.py` | `scripts/train.py` |
| Reward | binarized success prob, **once at truncation** | continuous, replaces the env reward every step |
| Who terminates | the **model** (never ground truth) | nobody (fixed horizon) |
| Budget | 40k env steps, ~5 h / seed on one L40S | 300k env steps, ~10.4 h / seed on one A100 |
| Section | [§4](#4-sparse-regime-metaworld--robomimic) | [§5](#5-dense-regime-maniskill) |

The sparse regime is the one the asymmetric loss was designed for: a failure-aware success
head is exactly a better *terminal* signal. The dense regime is where reward hacking shows up
(§5.3), so treat its progress output as a potential, not as a reward level.

---

## 2. Get a loadable checkpoint

| Checkpoint | What it is |
|---|---|
| `run2_noicl_ours_step4000` | **RoboRef** — the paper checkpoint (asymmetric loss) |
| `run3_noicl_standard_step5000` | symmetric-loss control (Robometer-4B + Failures) |
| `abl_origdata_asym_step5000` | asym arm of the loss-vs-data ablation (Robometer's own data) |
| `robometer/Robometer-4B` (HF) | published baseline; useful control, **`max_frames: 8`** |

Training runs save FSDP `SHARDED_STATE_DICT`
(`pytorch_model_fsdp_0/__{0..7}_0.distcp` + `.metadata`). **`from_pretrained()` cannot read
that** — consolidate to safetensors first (`consolidate_sharded_*.py` in the handoff package).
Two things about the shards:

* they hold **fp32 master weights** — load with `torch_dtype=torch.bfloat16` or you carry
  stray fp32 params;
* consolidation is the step that can silently drop the RBM heads. That is what §3 gates.

Checkpoints are not in git. See the root [`README.md`](README.md) for where they live.

---

## 3. Gates before you spend GPU hours

Run these in order. They cost minutes; skipping them costs days.

**3.1 Can the stack compose and load it at all?**

```bash
cd robometer-policy-learning
python scripts/preflight_reward_model.py --ckpt /path/to/consolidated_ckpt
```

Front-loads the two failures that used to kill arms ~4 minutes in: Hydra override parsing,
and the checkpoint's baked-in `use_unsloth: true` replaying in an env with no unsloth.

**3.2 Always export this.** Every RoboRef checkpoint bakes `use_unsloth: true` into its
`config.yaml`; inference only ever needs plain `from_pretrained`:

```bash
export ROBOMETER_DISABLE_UNSLOTH=1
```

**3.3 The canary — re-score a handful of known trajectories.**

For a fine-tuned checkpoint, **per-episode peak `success_prob` on a true success should be
0.7–0.9**, and on a failure ~0.1. Measured references:

| setting | model | successes | failures |
|---|---|---|---|
| MetaWorld CoffeePush | FT | ~0.77 | ~0.13 |
| Robomimic Can | run2 | ~0.80 | — |
| ManiSkill PullCube | run2 | 0.81–0.85 | see calibration |
| offline eval, 473 trajs | `abl_origdata_asym` | 0.977 | 0.104 (AUC **0.996**) |

**If you see 0.01–0.4, the pipeline is wrong, not the model.** Do not calibrate a threshold on
those numbers and do not conclude the head is miscalibrated or out of distribution. Note also
that `Robometer-4B` tolerates the frame bug below and keeps scoring ~0.85 — **the baseline
looking fine does not mean your pipeline is fine.**

**3.4 Feed the collator exactly `max_frames` frames.** Robometer's
`linspace_subsample_frames` only ever *reduces*: a 7-frame clip stays 7 frames. The
upsampling must happen at the call site:

```python
idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
frames = frames[idx]          # always exactly max_frames, repeats spread evenly
```

Measured on PullCube with run2, at the ground-truth success step:

| frames fed | success_prob (true successes) | (failures) |
|---|---|---|
| 7–11 (raw prefix) | 0.053 | 0.024 |
| padded to 16 | 0.369 | 0.024 |

Failures did not move — a real fix lifts only the positives. Do **not** front-pad by repeating
frame 0: the clip then looks stuck at the start and progress is biased down. This never bit
MetaWorld/Robomimic (200–400 step episodes reach 16 frames almost immediately) but always bites
ManiSkill (50-step episodes that succeed around step 7).

Full post-mortem: [`robometer-policy-learning/REWARD_MODEL_SCORING.md`](robometer-policy-learning/REWARD_MODEL_SCORING.md).

---

## 4. Sparse regime: MetaWorld / Robomimic

Canonical template: `vlm_ibrl/jobs/followup_array.job`.

### 4.1 Environment

```bash
source set_env.sh                                  # conda env + MuJoCo + PYTHONPATH
export PYTHONPATH="$PYTHONPATH:/path/to/RoboRef/Robometer"
export MUJOCO_GL=egl                               # needs a GPU
export ROBOMETER_DISABLE_UNSLOTH=1

export ROBOMETER_FT_PATH=/path/to/consolidated_ckpt   # selected by --vlm robometer_ft
export ROBOMETER_REWARD_CAMERA=corner2_default        # reward and policy see the same view
export V3_CORNER2_ZOOM=1
```

MetaWorld **v3** matters: the reward model was trained on v3 curated frames, and the original
IBRL env rendered v2. That mismatch is the leading suspect for on-policy reward collapse
(live CoffeePush AUC ~0.53 vs curated v3 ~0.97). See
[`vlm_ibrl/README_METAWORLD.md`](vlm_ibrl/README_METAWORLD.md).

### 4.2 The autonomous protocol — copy it exactly

```bash
export AUTONOMOUS_SUCCESS=1                   # without this the reward is ~1e-4
export ROBOMETER_DETECT_HEAD=success
export ROBOMETER_SUCCESS_THRESHOLD=0.85       # per task; calibrate (§6)
export ROBOMETER_SUCCESS_CONSECUTIVE=1        # debounce
export ROBOMETER_MIN_EP_STEPS=24              # 24 CoffeePush / 45 BoxClose / 120 Robomimic
```

### 4.3 Launch

```bash
python mw_main/train_rl_vlm_mw.py \
    --config_path release/cfgs/metaworld/ibrl_long.yaml \
    --bc_policy coffeepush \
    --vlm robometer_ft \
    --episode_length 200 \
    --train_end_on_success 0 \
    --reward_at_truncation 1 \
    --robometer_beta 0.0 \
    --robometer_threshold 0.0 \
    --robometer_reward_scale 1.0 \
    --num_train_step 40000 \
    --seed 0 \
    --save_dir "$SAVE_DIR"
```

`--bc_policy` is one of `assembly`, `boxclose`, `coffeepush`, `stickpull`.
`--vlm robometer_ft` is what routes to `ROBOMETER_FT_PATH`; `robometer_4b` and `qwen35_ft`
are the other Robometer-family options.

### 4.4 Beta is a CLI flag, not an environment variable

`--robometer_beta 0.0` means *pure success head* (1.0 = pure progress); it is read from the
config, and **nothing reads `$ROBOMETER_BETA`**. Several job scripts export it, but only so it
can be interpolated into `--robometer_beta` on the command line. Exporting it on its own is a
no-op, and the default is already 0.0.

### 4.5 Five silent traps

1. **`--episode_length 200` must be passed explicitly.** `ibrl_long.yaml` sets
   `episode_length: 100`; the code default (`MainConfig`) is 200. The yaml wins if you omit the
   flag, so you silently run half-length episodes.
2. **`ROBOMETER_MIN_EP_STEPS` defaults to `0`, i.e. the gate is OFF.** Without it an
   off-policy actor-critic bootstraps value from a spurious early terminal: the policy triggers
   a false success in the first few steps, banks the reward, resets — and you get a flat, dead
   policy. Set it to ~80% of the median demo length. This mattered for *every* vision-language
   reward model we tested.
3. **`ibrl_mw_lora_cees.job` is not a safe template.** It is the only MetaWorld job in the repo
   that defaults `TRAIN_END_ON_SUCCESS=1`; 35 other jobs pass `0`. Copying it drops the
   fairness guarantee — the model must be the thing that terminates episodes, never ground truth.
4. **`--reward_at_truncation 1` is the sparse setup**: query the reward once at the end of the
   episode instead of every step.
5. **Missing `AUTONOMOUS_SUCCESS=1`** gives a reward of ~1e-4 and runs that finish ~25x too
   fast. That combination is the signature of a misconfigured run, not of a bad model.

### 4.6 Confirm the levers reached the env, not just the job header

```
[online-levers] dynamic_thr=False (... floor=<thr>) min_ep_steps=<24|45>
```

This line is printed by `env/vlm_envs.py` only when a lever is actually active. No line means
no gate.

---

## 5. Dense regime: ManiSkill

### 5.1 Use the job that bakes the recipe in

```bash
cd robometer-policy-learning
sbatch --export=ALL,ARM=dense,MODEL=run2,SEED=0,STEPS=300000 jobs/snellius_maniskill_sac.job
sbatch --export=ALL,ARM=gt,SEED=0,STEPS=300000               jobs/snellius_maniskill_sac.job
```

`MODEL` is one of `run2` (RoboRef), `run3`, `base`, `roboreward`, `robodopamine`, `lrm`.
On another cluster, override the partition and paths at submit time — the job header documents
`SCRATCH_ROOT`, `ENV_PREFIX`, `CKPT_ROOT`, `VULKAN_FIX`, `SAFEPATH`.

The paper setup: PullCube-v1, SAC from scratch, 300k env steps, the progress head **replaces**
the env reward, no VLM termination, no GT termination, 50-step episodes.

```
env.terminate_on_success=false   env.reward_shift=0.0   training.num_envs=4
online_algorithm.gamma=0.95  tau=0.01  batch_size=1024  learning_starts=4000
reward_model.add_estimated_reward=false  reward_model.use_success_detection=false
```

### 5.2 Two config traps

* **`maniskill_online_rl.yaml` alone gives a different algorithm**: it inherits
  `terminate_on_success: true`, `reward_shift: -1.0`, and `sac.yaml`'s LIBERO-tuned
  gamma 0.99 / tau 0.005 / batch 128. Every published ManiSkill result used CLI overrides that
  were never written into the config. That is exactly why the job bakes them in — pass
  `RECIPE=0` only if you intend to own every knob.
* **`jobs/hipster_maniskill_sac.job` also differs** (gamma 0.8, success detection on, GT arm
  terminates on success, Robometer group only). Fine for its own sweeps, not for paper
  comparisons.

### 5.3 If you want a dense signal, use the progress head as a potential

The progress *level* is farmable: measured on PullCube, progress rose **7x while true success
stayed at 0–2%**. Within episodes, though, the heads track real progress well (Kendall tau
0.63–0.74), which makes them good *potentials*:

```yaml
progress_beta: 1.0
progress_as_potential: true   # gamma*Phi(s') - Phi(s), Ng et al. shaping
potential_gamma: 0.99
normalize_reward: true        # running p1/p99 onto [0,1] BEFORE any cost shift
```

`normalize_reward` is not cosmetic: without it the agent sees a model-specific constant offset
(measured −0.90 for run2, −0.25 for base) with the success signal at only ~7% of the magnitude.

Reward-hacking analyses and the Goodhart curves:
[`robometer-policy-learning/MANISKILL_HANDOFF.md`](robometer-policy-learning/MANISKILL_HANDOFF.md).

---

## 6. Calibrate the threshold per task

`ROBOMETER_SUCCESS_THRESHOLD` / `success_detection_threshold` is **per task, not global**.
Expect **0.75–0.90** on the fine-tuned scale; **0.85 is a reasonable first guess** for a
checkpoint whose canary separates cleanly, but calibrate against held-out demos before the real
run:

* `vlm_ibrl/jobs/calibrate_matrix.job`, `calibrate_multitask.job`
* `post-training-analysis/reward-model-study/scripts/calibrate_threshold.py`,
  `calibrate_threshold_multitask.py`
* `vlm_ibrl/jobs/diag_causal_calib.py`

The majority vote (`ROBOMETER_CONFIRM_K`) and the min-episode gate are remedies for reward
hacking, **not defaults** — turning them on early hides whether the threshold itself is right.
The one exception is `ROBOMETER_MIN_EP_STEPS`, which you do want from the start (§4.5).

---

## 7. Logging

**Never let wandb kill a run.** `Robometer/robometer/utils/logger.py` routes every wandb call
through `_safe_wandb_log`, and `init_wandb` falls back online → offline → disabled. Scalar
`log()` calls are buffered by wandb's background service and degrade to a warning, but media
payloads (`Table`/`Image`/`Video`) register an artifact through the public GraphQL API
**synchronously, in-process**, so an auth failure raises straight into the training loop. An
expired key once took out all four arms of a sweep at step 4000 — and because the exception
unwinds inside `_maybe_log_save_evaluate` *before* the save, it ate that step's checkpoint too.

**wandb 0.16.4 cannot init online.** In the `demo2reward_v3` / vlm_ibrl env, `wandb.init()`
raises `CommError: 'NoneType' object has no attribute 'get'` against the current W&B SaaS —
a server-side schema change, not your key. Fix it with the shipped shim, first on `PYTHONPATH`:

```bash
export PYTHONPATH="/path/to/vlm_ibrl/jobs/compat:$PYTHONPATH"
python vlm_ibrl/jobs/compat/check.py     # shim + authenticated viewer() round trip
```

It must be on `PYTHONPATH` rather than monkeypatched from the training script, because the
crash happens in the **child process wandb spawns** for its sender, which only picks up
`sitecustomize.py` at interpreter startup. Do **not** `pip install -U wandb` in that env — it
drags protobuf 4.25→7.36 and urllib3 1.26→2.8 under a working torch 2.4+cu121 / MuJoCo stack.
Fallback: run offline and `python -m wandb sync` afterwards (call it as `python -m wandb`; the
`wandb` console script has a broken shebang in these envs).

---

## 8. Order of operations

```
1. consolidate the checkpoint (.distcp -> safetensors, bf16)        §2
2. preflight_reward_model.py --ckpt <dir>                           §3.1
3. export ROBOMETER_DISABLE_UNSLOTH=1                               §3.2
4. canary: peak success_prob 0.7-0.9 on true successes              §3.3   <- gate
5. confirm the call site feeds exactly max_frames                   §3.4   <- gate
6. wandb pre-flight (compat/check.py) if on the old env             §7
7. calibrate the threshold for this task                            §6
8. launch with the full protocol block                              §4.2 / §5.1
9. confirm [online-levers] in the env log, not just the job header  §4.6
```

Steps 4 and 5 are the two that are cheap to run and expensive to discover late. If the canary
comes back at AUC ~0.5 on a checkpoint whose training metrics were healthy, that is a
**consolidation** failure — re-run consolidation with the other script before considering a
retrain.

---

## 9. Reference numbers

Measured, for sanity-checking a new run against a known-good one.

| quantity | value |
|---|---|
| peak success_prob, true success (FT) | 0.77–0.85 |
| peak success_prob, failure (FT) | ~0.10–0.13 |
| offline success AUC, healthy FT checkpoint | ~0.99 |
| within-episode progress Kendall tau | 0.63–0.74 |
| success threshold, FT scale | 0.75–0.90 |
| MetaWorld reward with `AUTONOMOUS_SUCCESS` unset | ~1e-4 (broken) |
| ManiSkill 300k-step seed, 1x A100 | RoboRef ~10.4 h, base ~10.3 h, +Failures ~14 h, LRM ~46 h |
| ManiSkill GT arm (no VLM in the loop) | 18–30 steps/s, ~3–5 h |

---

## 10. Where things live

| What | Where |
|---|---|
| Reward-model library, asymmetric loss | `Robometer/robometer/trainers/rbm_heads_trainer.py` |
| Sparse RL (IBRL/RLPD) | `vlm_ibrl/`, entry `mw_main/train_rl_vlm_mw.py` |
| Reward wrapper + every `ROBOMETER_*` lever | `vlm_ibrl/env/vlm_envs.py` |
| Dense RL (SAC) | `robometer-policy-learning/`, entry `scripts/train.py` |
| Reward-model config group | `robometer-policy-learning/robometer_policy_learning/configs/reward_model/robometer.yaml` |
| Scoring-pipeline post-mortem | `robometer-policy-learning/REWARD_MODEL_SCORING.md` |
| ManiSkill recipe + reward hacking | `robometer-policy-learning/MANISKILL_HANDOFF.md` |
| Rebuild notes, cluster quirks | `robometer-policy-learning/REBUILD.md` |
| MetaWorld v3 port notes | `vlm_ibrl/README_METAWORLD.md` |
| Threshold calibration | `post-training-analysis/reward-model-study/scripts/` |
