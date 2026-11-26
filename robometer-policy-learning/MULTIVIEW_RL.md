# Multi-camera reward for online RL

Branch `multiview-reward` (two commits on top of `inference-pipeline`).

## Why

A reward model that sees one camera cannot distinguish a solved task from an
arm parked in front of the object, and the policy's DINO features come from
that same camera, so nothing in the pipeline can notice. Measured on
PullCube-v1 with `robometer_mix_lora_ho_lam03/checkpoint-4000`, per-step
reward: **0.144 for the farming pose, 0.125 for a genuine success** — faking
paid more than solving. The farming pose inflates the trained-on view 2.1x and
leaves a second camera unmoved, so scoring both and taking the `min` removes
the exploit by construction.

## The three knobs

All default to the previous single-view behaviour; nothing changes unless you
set them.

| key | meaning |
|---|---|
| `reward_model.reward_relabeling_keys` | which obs keys are **scored** |
| `reward_model.multiview_reduce` | `mean` (old) or `min` (anti-occlusion) |
| `env.env_kwargs.second_view` | name of a 2nd camera, rendered as `obs["image2"]` |

The exact override set that produced the numbers below:

```
reward_model.reward_relabeling_keys=[image,image2] \
reward_model.multiview_reduce=min \
env.env_kwargs.second_view=base_camera \
env.extra_keys_to_drop=[language,image,image2] \
eval.eval_record_video=false
```

## Five traps, in the order they will bite you

1. **Do not add the second view to `env.dino_image_keys`.** The
   DinoEmbeddingWrapper emits one 768-d block per `dino_image_keys` entry, so a
   view added there widens the actor's input and breaks checkpoint loading.
   Reward views are deliberately a separate list; the actor is untouched.
2. **A view with no DINO chunk passes `video_embeddings=None`** and the reward
   model embeds the raw frames itself. That is the same path the calibration
   script uses, not a degraded fallback. The buffer's new `embedding_keys`
   argument exists so the chunk layout follows the DINO keys, not the reward
   keys — assuming they match splits one 768-d embedding into two 384-d halves
   and silently feeds the model garbage.
3. **Hydra struct mode.** Both config keys are *declared* in
   `configs/reward_model/robometer.yaml` and `configs/maniskill_online_rl.yaml`.
   If you add a third knob, declare it too: an undeclared key makes
   `group.key=value` a composition error, not an override.
4. **`second_view` is ManiSkill-only.** It lives in
   `envs/maniskill_wrapper.py`, which can pull from either
   `human_render_cameras` (what `env.render()` returns) or `sensors`. On
   PullCube-v1 that is `render_camera` (512², fov 1.0) and `base_camera`
   (128², fov pi/2). MetaWorld needs the same few lines in its own wrapper.
5. **`scripts/train_async.py` does not pass `multiview_reduce`** to its offline
   H5 buffer, so demos there would relabel with `mean` while live steps use
   `min`. `scripts/train.py` is wired correctly. Untested path, left alone.

## Verify before committing GPU hours

```
python jobs/demo_twoview_check.py --task PullCube-v1 --second-view base_camera \
    --reward-ckpt <ckpt> --n-episodes 6 --max-tries 25
```

It replays official demos, **keeps only the episodes your env confirms
solved**, and scores every view through the buffer's exact code path. That
guard is not optional: open-loop replay of `physx_cuda` demos diverges on
`physx_cpu` and only 3 of 25 PullCube demos reach success, so without it 22
unsolved episodes get scored as successes. Hence `--max-tries` oversampling.

What a healthy result looks like (20 verified successes, lambda=0.3):

| per-step reward | honest unsolved | farming pose | genuine success | solve/farm |
|---|---|---|---|---|
| view 1 (trained on) | 0.067 | 0.144 | 0.125 | 0.87 |
| view 2 | 0.053 | 0.053 | 0.134 | 2.50 |
| `min(v1,v2)` | 0.050 | 0.052 | 0.100 | 1.92 |

**solve/farm must be > 1.** Do not pick the reduction on separation metrics:
view 2 alone has the better progress AUC (0.907 vs min's 0.744) and `min` still
won the actual RL runs.

## What to expect

PullCube-v1, final eval over 20 episodes at 300k steps:

| arm | 1 view | 2-view `min` |
|---|---|---|
| lam01 s1 / s2 / s3 | 90% / 60% / 85% | **0%** / 70% / **5%** |
| lam03 s1 / s2 / s3 | 0% / 0% / 0% | **100%** / 90%\* / 85%\* |
| lam07 s1 / s2 | 70% / 0% | 90-100% / 85-90% |

\* truncated at ~120k/300k by the wall clock.

`min` repairs lambda=0.3 (0/3 to 3/3 seeds) and lambda=0.7, and costs
lambda=0.1 two of three seeds — `min` is dominated by the weaker camera, and
that model's weaker view carries little task signal (its critic values solving
and not-solving identically, q~0.5 either way). **If an arm goes flat under
`min`, try `mean` before concluding the reward is broken:** on the measured
numbers `mean` still flips solve/farm to 1.32 while keeping about twice the
signal magnitude.

Scoring a second view costs no wall time (3.44 step/s, same as single-view).

## Reading the run

Per-view head outputs are recorded per step in the episode log
(`score_per_step_per_view`, `sp_per_step_per_view`, `view_keys`,
`multiview_reduce`). The disagreement between views *is* the occlusion
measurement and cannot be recovered from the reduced value, so keep them.

Judge on the environment's own success rate, never the reward: the progress
level is farmable, which is the whole point of this change.
