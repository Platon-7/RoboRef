# Running RoboRef inference

One pipeline for scoring episodes with RoboRef (or any Robometer-family checkpoint). The RoboRef evaluation splits
and simulation rollouts go through the same code, so a result on one can be compared with a result on the other.

Code: `inference/roboref_infer.py` (subcommands `manifest`, `score`, `metrics`) and `inference/score.job` (Slurm).

The authoritative reference is the paper (`/home/pkarage/RoboRef_RoboPAD-Camera-Ready.pdf`, cited as "paper").
Where this repo's notes add practical detail, the note is named; `reward-model-study/RESULTS.md` and
`RL_EXPERIMENT_TRACKER.md` are obsolete and not used.

## What RoboRef takes and returns

| | |
|---|---|
| Input | RGB frames of one episode (any length, any resolution) + a task sentence |
| What the model sees | 16 frames for RoboRef, 8 for the published Robometer-4B (paper Sec. 3.1, 3.4; `data.max_frames` in each checkpoint's `config.yaml`). The collator picks them evenly from a longer clip but **never adds frames to a shorter one** |
| Output | `success_prob` and `progress` of the last frame, both in [0, 1], plus both per frame |
| `success_prob` means | "is the task done by the last frame I see". Training labels a frame successful once progress has reached completion (`robometer/data/datasets/helpers.py`). It is not a forecast. |

## The rules

1. **Hand the model exactly 16 frames.** The collator only reduces, so a 7-frame clip stays 7 frames, and the
   fine-tuned heads collapse on short clips (run2 on true successes: 0.05 with 7–11 frames, 0.37 stretched to 16).
   Stretch short clips with evenly spread repeats, `frames[np.linspace(0, T-1, 16).round().astype(int)]` ("sub16").
   Never front-pad with frame 0: that biases progress down. (`REWARD_MODEL_SCORING.md` §1;
   `reward-model-study/scripts/calibrate_threshold.py::sub16`.) The pipeline does this for you.
2. **Pick the statistic for the question.**
   - *Offline evaluation* (paper App. E.1, Table 11): one score per whole episode; rank accuracy = fraction of
     success–failure pairs ordered correctly (= AUROC). RoboRewardBench (App. E.2) reads the progress head at the
     final frame, video sampled at 1 frame/s including the final frame.
   - *Deployment / success detection in simulation* (paper App. E.3): score every frame of the growing episode; the
     success head fires when it crosses a per-model threshold (RoboRef: 0.75–0.85 across tasks and simulators,
     calibrated on oracle rollouts under a strict false-positive budget), with a minimum episode length against early
     spurious fires. Use `--prefix-every S`; `metrics` reports the per-episode
     maximum, the quantity a threshold is compared against.
3. **Sanity range.** On simulated successes, a fine-tuned checkpoint's episode-max `success_prob` should be about
   ~0.7–0.9 (`REWARD_MODEL_SCORING.md` §0: MetaWorld ~0.77, Robomimic ~0.80, ManiSkill PullCube 0.81–0.85), in line
   with the paper's 0.75–0.85 thresholds. Far lower values on true successes point to a pipeline problem.
4. **Check the pipeline against the paper first.** Score the four evaluation splits and compare with Table 11
   (see "Validation against the paper").
5. **Use the canonical evaluation data**: the HF datasets with one MP4 per episode (`step1_hf/eval_<source>_raw`),
   not the keyframe tars (`repro_canonical_eval.py` records that an earlier evaluation scored the tars by mistake).
6. **Load the model through the canonical scorer** (`vlm_ibrl/env/robometer_utils.py: get_robometer_4b`), which
   already handles: Unsloth off (checkpoints bake `use_unsloth: true`); FlashAttention hidden for Qwen3-VL (the RBM
   wrapper rejects FA2); collator arguments in the right order and `use_multi_image=True`; discrete progress for any
   C51 variant (fine-tuned checkpoints use `c51_asymmetric`, not the literal `discrete`); deterministic cuDNN and a
   warm-up loop against the all-NaN cold-start pathology; `ROBOMETER_FORCE_FP32=1` if a checkpoint still gives NaN.
   (`REWARD_MODEL_SCORING.md` §6.)
7. **Ruled out, do not chase** (`REWARD_MODEL_SCORING.md` §5): cuDNN/bf16 numerics, input resolution once the frame
   count is right (224 and 480 give identical results), the success-probability extraction, the sample construction.
   Numpy arrays and PIL images both work with this scorer (frames must be uint8; a float array would be cast to 0).
8. **Know what the model has seen.** Training clips are robot attempts of language-specified manipulation tasks from
   one third-person view (paper Sec. 3, App. A); the model is sensitive to occlusion (paper, Limitations). Scenes,
   tasks or episode structures far from that (e.g. long idle tails after the action) are out of distribution.

## Steps

### 1. Environment

```bash
PY=/home/pkarage/RoboRef/Robometer/.venv/bin/python                # torch 2.8, transformers 4.57, datasets, PyAV
export HF_HOME=/home/pkarage/.cache/huggingface HF_HUB_OFFLINE=1    # Qwen3-VL-4B base files are cached here
```

| Model | `--model` |
|---|---|
| RoboRef (`run2_noicl_ours`, step 4000) | `/home/pkarage/checkpoints/roboref_run2_noicl_ours_step4000` (default) |
| Robometer-4B (published) | `robometer/Robometer-4B` |

### 2. Manifest (one JSON line per episode)

```json
{"id": "...", "task": "Pick up the red cube ...", "label": 1, "frames": "/path/episode.mp4", "source": "failsafe"}
```

`label`: 1 success, 0 failure, null unknown. `frames`: an `.mp4`, an `.npz` with `frames` (T, H, W, 3) uint8 and
optionally `frame_t` (s), or a directory of images. Add `"fps"` if you score the growing video of an MP4.

```bash
# RoboRef evaluation splits
H=/ivi/zfs/s0/original_homes/pkarage/roboref_rebuttal/data/step1_hf
for s in droid failsafe metaworld robometer; do
  $PY inference/roboref_infer.py manifest hf --ds $H/eval_${s}_raw/robometer_frames_eval_${s} \
      --out inference/runs/manifest_eval_${s}.jsonl
done
# Simulation rollouts (MECHANICA door)
M=/home/pkarage/Mechanica/mech-wm/mechanica/results/M3_door_reward
$PY inference/roboref_infer.py manifest npz --index $M/index_test.jsonl --root $M --task "<task sentence>" \
    --out inference/runs/manifest_door.jsonl
```

### 3. Score (one A5000, ~2.5 episodes/s for 16-frame clips)

```bash
cd /home/pkarage/RoboRef
# whole episodes (offline classification)
MANIFEST=inference/runs/manifest_eval_failsafe.jsonl OUT=inference/runs/scores_failsafe.jsonl sbatch inference/score.job
# growing video every 0.5 s (deployment statistic: per-episode max)
MANIFEST=inference/runs/manifest_door.jsonl OUT=inference/runs/scores_door_prefix.jsonl PREFIX=0.5 sbatch inference/score.job
# options: MODEL=robometer/Robometer-4B   LIMIT=20 (smoke test)
```

Resumes if rerun. Output: one JSON line per (episode, cut) with `success_prob`, `progress` and per-frame arrays.

### 4. Metrics

```bash
$PY inference/roboref_infer.py metrics --scores inference/runs/scores_failsafe.jsonl --out inference/runs/metrics.json
```

Per source and pooled, both heads: AUROC success vs failure, within-task AUROC, separation. For a growing-video run:
the episode maximum, the last cut, and the canary.

## Validation against the paper

Paper reference (Table 11): RoboRef, 16 frames, **after 5,000 training steps** (the frame-budget ablation, not the
released step-4000 checkpoint), progress rank accuracy on the in-distribution splits: MetaWorld 0.861, DROID 0.622,
FailSafe 0.876, Robometer 0.867 (mean 0.806).

This pipeline, whole episodes (`runs/validation_eval_splits.json`):

| Split | n (successes) | RoboRef step 4000: progress rank acc. | paper (RoboRef, 5k steps) | RoboRef success AUROC | Robometer-4B: progress rank acc. |
|---|---|---|---|---|---|
| DROID | 963 (734) | 0.625 | 0.622 | 0.757 | 0.580 |
| FailSafe | 555 (74) | 0.837 | 0.876 | 0.889 | 0.893 |
| MetaWorld | 572 (104) | 0.867 | 0.861 | 0.961 | 0.613 |
| Robometer | 3,865 (2,478) | 0.815 | 0.867 | 0.872 | 0.721 |
| Mean | | **0.786** | **0.806** | | 0.702 |

The pipeline **approximately reproduces** the reported Table 11 values. Discrepancies (this pipeline − paper):
DROID +0.003, MetaWorld +0.006, FailSafe −0.039, Robometer −0.052, mean −0.020. The checkpoints differ (released
step 4000 here; the paper's row is the 5k-step frame-budget ablation), which *may* explain the FailSafe and Robometer
gaps, but this has **not been verified** (the 5k-step checkpoint was not scored). Other untested candidates: episode
selection or frame decoding in the evaluation splits, and bf16 nondeterminism (measured at ≤ 0.03 per score, not per
rank accuracy). On the same data RoboRef ranks above Robometer-4B (mean 0.79 vs 0.70).

## Door evaluation (MECHANICA, simulated push-to-swing door)

600 episodes (100 doors × 6 rollout sources, 193 successes), task stated with the exact rule (manifest
`runs/manifest_door_P1.jsonl`), AUROC success vs failure (`runs/metrics_door_whole_*.json`,
`runs/metrics_door_prefix_*.json`):

| Statistic | RoboRef success | RoboRef progress | Robometer-4B success | Robometer-4B progress |
|---|---|---|---|---|
| whole episode | 0.53 | 0.55 | 0.49 | 0.41 |
| growing clip every 0.5 s, per-episode max | 0.42 | 0.54 | 0.45 | 0.42 |

**What this shows.** Under the evaluated door protocols, both reward models discriminate success from failure poorly
(at or near chance on every statistic), while the same code approximately reproduces the paper on its evaluation
splits. RoboRef's episode-max success on true successes is 0.645, so scores are in a plausible range.

**What it does not show.** Chance-level performance does not establish *why* the models fail here. "The door is out of
distribution" is one hypothesis, not a finding: no experiment isolated it (e.g. an in-distribution task in the same
renderer, or a door clip with the outcome made visually salient). The result also says nothing about RoboRef's
general capabilities or its use as an RL reward (paper Sec. 5), which were not tested.

**Open evaluation limitations** (benchmark replication does not resolve these):
- *Task wording.* Results are for one instruction text (P1, stating the exact rule). Earlier wording variants
  (MECHANICA M3c, P0–P3) were not re-run through this pipeline; sensitivity to wording is unmeasured.
- *Temporal sampling.* The model sees 16 frames per clip. A 10 s episode sampled to 16 frames shows one frame per
  ~0.6 s; the release, the 25° crossing and the final rest can fall between frames. Growing-clip scoring changes what
  is visible at each cut but not the 16-frame budget.
- *Historical contact.* Success requires *no touch after 25°*. A late touch is a brief event that may be absent from
  the sampled frames, and a final-frame judgement cannot see it at all; the label depends on history the model may not
  observe.
- *Completion versus forecasting.* The success head is trained to say whether the task *is complete* in the frames
  shown, not to forecast. On the door, the outcome is decided at release but only becomes visible seconds later; early
  cuts ask for a forecast the head was not trained for, and late cuts depend on the idle tail being sampled.

## Artifacts not in git

Kept out of the repository (size or licence); locations and SHA-256 recorded here for reproducibility.

| Artifact | Location | SHA-256 / identifier |
|---|---|---|
| RoboRef checkpoint (step 4000), 16 GB | `/home/pkarage/checkpoints/roboref_run2_noicl_ours_step4000/` | `runs/CHECKSUMS.sha256` (4 shards + config.yaml) |
| Robometer-4B | HF `robometer/Robometer-4B`, cached under `$HF_HOME/hub` | snapshot `beef63bc914c5c189329d49c6d712d96d632aa34` |
| Evaluation splits (MP4, 16 frames/episode), 272 MB | `/ivi/zfs/s0/original_homes/pkarage/roboref_rebuttal/data/step1_hf/eval_{droid,failsafe,metaworld,robometer}_raw` | manifests `runs/manifest_eval_*.jsonl` list every episode |
| Door episodes (npz frames + state), 1.3 GB | `/home/pkarage/Mechanica/mech-wm/mechanica/results/M3_door_reward/episodes/` (branch `exp/mechanica-door-repro`) | `runs/manifest_door_P1.jsonl` |
| Score files (per-episode, per-frame), 34 MB | `inference/runs/scores_*.jsonl` (local only) | `runs/CHECKSUMS.sha256` |
