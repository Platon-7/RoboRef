"""One inference pipeline for RoboRef / Robometer checkpoints, for any episodes: the RoboRef evaluation splits and
simulation rollouts alike. See INFERENCE.md next to this file.

Three steps:
  manifest  turn a data source into a manifest: one JSON line per episode
            {"id", "task", "label" (1 success / 0 failure / null), "frames", "source", optional "fps", "meta"}
            where "frames" is an .mp4, an .npz (array "frames", optional "frame_t"), or a directory of images.
  score     run the checkpoint on every manifest episode -> one JSON line per (episode, cut time) with the success
            probability and progress of the last frame and the per-frame arrays.
  metrics   success / progress AUROC (global, within-task), separation, per source.

Scoring uses the canonical scorer of this repo (vlm_ibrl/env/robometer_utils.py: get_robometer_4b), the same one the
reward-model study and RL use: deterministic cuDNN, warm-up until outputs are finite, Unsloth and FlashAttention off for
Qwen3-VL checkpoints, discrete (C51) progress decoding read from the checkpoint's config.

Frames (robometer-policy-learning/REWARD_MODEL_SCORING.md): the model reduces any clip to its max_frames (16) evenly,
but never adds frames, and the fine-tuned heads collapse on short clips. So a clip shorter than max_frames is stretched
here to exactly max_frames with evenly spread repeats ("sub16", reward-model-study/scripts/calibrate_threshold.py).

Two ways to score an episode:
  whole episode (default)    one score on the whole episode: the canonical offline evaluation
  --prefix-every S           the growing video, cut every S seconds (and at the end); the per-episode MAXIMUM of
                             success_prob is the deployment statistic (detectors fire once on the running video)

Examples
  python roboref_infer.py manifest hf   --ds <step1_hf>/eval_failsafe_raw/robometer_frames_eval_failsafe --out m.jsonl
  python roboref_infer.py manifest npz  --index <rollouts>/index_test.jsonl --root <rollouts> --task "..." --out m.jsonl
  python roboref_infer.py score   --manifest m.jsonl --model <ckpt or hf id> --out s.jsonl [--prefix-every 0.5] [--limit N]
  python roboref_infer.py metrics --scores s.jsonl [s2.jsonl ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "/home/pkarage/checkpoints/roboref_run2_noicl_ours_step4000"


# ---------------------------------------------------------------- frames

def load_frames(spec: str, cut_s: float | None = None, fps: float | None = None) -> np.ndarray:
    """(T, H, W, 3) uint8 from an .mp4, an .npz ("frames" [+ "frame_t"]) or an image directory.
    With cut_s, only frames up to that time (s) are kept: by "frame_t" if present, else by fps."""
    p = Path(spec)
    t = None
    if p.suffix == ".npz":
        z = np.load(p)
        frames = z["frames"]
        t = z["frame_t"] if "frame_t" in z.files else None
    elif p.suffix in (".mp4", ".avi", ".mov", ".webm"):
        import imageio.v3 as iio
        try:
            frames = np.stack([np.asarray(f) for f in iio.imread(p, plugin="pyav")])
        except Exception:
            frames = np.stack([np.asarray(f) for f in iio.imread(p)])
    elif p.is_dir():
        from PIL import Image
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
        frames = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files])
    else:
        raise ValueError(f"unsupported frames: {spec}")
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"{spec}: expected (T, H, W, 3), got {frames.shape}")
    if cut_s is not None:
        if t is None:
            if fps is None:
                raise ValueError(f"{spec}: cutting at {cut_s} s needs frame_t in the npz or an fps in the manifest")
            t = np.arange(len(frames)) / fps
        frames = frames[np.asarray(t) <= cut_s + 1e-9]
    return frames.astype(np.uint8, copy=False)


# ---------------------------------------------------------------- manifests

def manifest_hf(args):
    """A Robometer-format HF dataset (one row per episode, "frames" = mp4 path relative to the dataset's parent)."""
    from datasets import load_from_disk
    ds = load_from_disk(args.ds)
    root = Path(args.ds).parent
    with open(args.out, "w") as f:
        for r in ds:
            label = {"successful": 1, "failure": 0}.get(r["quality_label"])
            f.write(json.dumps({"id": r["id"], "task": r["task"], "label": label, "frames": str(root / r["frames"]),
                                "source": r.get("data_source")}) + "\n")
    print(f"{len(ds)} episodes -> {args.out}")


def manifest_npz(args):
    """Simulation rollouts: an index JSONL with "file" (npz relative to --root), "success", "instance_id", "source"."""
    rows = [json.loads(l) for l in open(args.index)]
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps({"id": f"{r['instance_id']}:{r['source']}", "task": args.task,
                                "label": int(bool(r["success"])), "frames": str(Path(args.root) / r["file"]),
                                "source": r["source"],
                                "meta": {k: r[k] for k in ("final_deg", "touch_late", "t_last_contact") if k in r}})
                    + "\n")
    print(f"{len(rows)} episodes -> {args.out}")


# ---------------------------------------------------------------- scoring

def sub16(frames: np.ndarray, n: int) -> np.ndarray:
    """Exactly n frames for a clip shorter than n: evenly spread repeats (never front-pad with frame 0)."""
    if len(frames) >= n:
        return frames
    return frames[np.linspace(0, len(frames) - 1, n).round().astype(int)]


def episode_length_s(spec: str, fps: float | None):
    p = Path(spec)
    if p.suffix == ".npz":
        z = np.load(p)
        if "frame_t" in z.files:
            return float(z["frame_t"][-1])
        n = len(z["frames"])
    else:
        n = len(load_frames(spec))
    if fps is None:
        raise ValueError(f"{spec}: --prefix-every needs frame times (npz frame_t) or an fps in the manifest")
    return (n - 1) / fps


def load_scorer(model: str):
    sys.path.insert(0, str(REPO / "Robometer"))
    sys.path.insert(0, str(REPO / "vlm_ibrl"))     # first: its unsloth stub must shadow the real package
    from env.robometer_utils import get_robometer_4b
    return get_robometer_4b(model_path=model)


def score(args):
    rows = [json.loads(l) for l in open(args.manifest)]
    if args.limit:
        rows = rows[: args.limit]
    fixed_cuts = [None] if not args.cuts else [float(c) for c in args.cuts.split(",")]
    done = set()
    if os.path.exists(args.out):
        done = {(json.loads(l)["id"], json.loads(l)["cut_s"]) for l in open(args.out)}
    scorer = load_scorer(args.model)
    print(f"model {args.model}: max_frames {scorer.max_frames}; {len(rows)} episodes; "
          f"{'growing video every ' + str(args.prefix_every) + ' s' if args.prefix_every else 'cuts ' + str(fixed_cuts)}",
          flush=True)
    t0, n = time.time(), 0
    with open(args.out, "a") as f:
        for i, r in enumerate(rows):
            cuts = fixed_cuts
            if args.prefix_every:
                end = episode_length_s(r["frames"], r.get("fps"))
                cuts = sorted(set(np.round(np.arange(args.prefix_every, end + 1e-9, args.prefix_every), 3).tolist()
                                  + [round(end, 3)]))
            for cut in cuts:
                if (r["id"], cut) in done:
                    continue
                frames = sub16(load_frames(r["frames"], cut, r.get("fps")), scorer.max_frames)
                out = scorer(frames, task=r["task"], icl_frames=None, detailed=True)
                f.write(json.dumps({"id": r["id"], "source": r.get("source"), "task": r["task"], "label": r["label"],
                                    "cut_s": cut, "n_frames_in": int(len(frames)), "model": args.model,
                                    "prefix_mode": bool(args.prefix_every),
                                    "success_prob": out["success_prob"], "progress": out["progress_reward"],
                                    "success_per_frame": out.get("success_probs_per_frame"),
                                    "progress_per_frame": out.get("progress_per_frame"),
                                    "meta": r.get("meta")}) + "\n")
                n += 1
            if (i + 1) % 100 == 0:
                f.flush()
                print(f"  {i + 1}/{len(rows)} episodes, {n / (time.time() - t0):.2f} scores/s", flush=True)
    print(f"done: {n} scores in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- metrics

def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg])
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a)); ranks[order] = np.arange(1, len(a) + 1)
    for v in np.unique(a):                      # average ranks for ties
        m = a == v
        ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def summarise(rows, head):
    y = np.array([r["label"] for r in rows]); s = np.array([r[head] for r in rows], float)
    tasks = np.array([r["task"] for r in rows])
    within = [auroc(s[(tasks == t) & (y == 1)], s[(tasks == t) & (y == 0)]) for t in np.unique(tasks)]
    within = [w for w in within if w == w]
    return {"n": int(len(y)), "n_success": int(y.sum()), "auroc": auroc(s[y == 1], s[y == 0]),
            "auroc_within_task": float(np.mean(within)) if within else float("nan"), "n_tasks_with_both": len(within),
            "separation": float(s[y == 1].mean() - s[y == 0].mean()) if 0 < y.sum() < len(y) else float("nan"),
            "mean": float(s.mean())}


def metrics(args):
    rows = [json.loads(l) for p in args.scores for l in open(p)]
    rows = [r for r in rows if r["label"] is not None]
    out = {}
    if any(r.get("prefix_mode") for r in rows):   # growing video: per-episode max (and the last cut) per head
        ep = {}
        for r in rows:
            e = ep.setdefault(r["id"], dict(r, success_prob=-1.0, progress=-1.0, last_cut=-1.0))
            e["success_prob"] = max(e["success_prob"], r["success_prob"]); e["progress"] = max(e["progress"], r["progress"])
            if r["cut_s"] > e["last_cut"]:
                e["last_cut"], e["success_last"], e["progress_last"] = r["cut_s"], r["success_prob"], r["progress"]
        eps = list(ep.values())
        res = {"episode max": {h: summarise(eps, h) for h in ("success_prob", "progress")},
               "last cut": {h: summarise([dict(e, **{h: e[k]}) for e in eps], h)
                            for h, k in (("success_prob", "success_last"), ("progress", "progress_last"))}}
        pos = [e["success_prob"] for e in eps if e["label"] == 1]
        res["canary: episode-max success_prob on true successes (median; expect ~0.7-0.9)"] = \
            float(np.median(pos)) if pos else None
        for src in sorted({e["source"] for e in eps}):
            res[f"episode max, {src}"] = {h: summarise([e for e in eps if e["source"] == src], h)
                                          for h in ("success_prob", "progress")}
        print(json.dumps(res, indent=1))
        if args.out:
            json.dump(res, open(args.out, "w"), indent=1)
        return
    for cut in sorted({r["cut_s"] for r in rows}, key=lambda c: (c is not None, c or 0)):
        rc = [r for r in rows if r["cut_s"] == cut]
        key = "full" if cut is None else f"cut {cut}s"
        out[key] = {"all": {h: summarise(rc, h) for h in ("success_prob", "progress")}}
        for src in sorted({r["source"] for r in rc}):
            out[key][src] = {h: summarise([r for r in rc if r["source"] == src], h) for h in ("success_prob", "progress")}
    print(json.dumps(out, indent=1))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest"); msub = m.add_subparsers(dest="kind", required=True)
    h = msub.add_parser("hf"); h.add_argument("--ds", required=True); h.add_argument("--out", required=True)
    z = msub.add_parser("npz"); z.add_argument("--index", required=True); z.add_argument("--root", required=True)
    z.add_argument("--task", required=True); z.add_argument("--out", required=True)
    s = sub.add_parser("score"); s.add_argument("--manifest", required=True); s.add_argument("--out", required=True)
    s.add_argument("--model", default=DEFAULT_MODEL); s.add_argument("--limit", type=int)
    s.add_argument("--cuts", help="comma-separated cut times in s (default: the whole episode)")
    s.add_argument("--prefix-every", type=float, help="score the growing video every S seconds (deployment protocol)")
    q = sub.add_parser("metrics"); q.add_argument("--scores", nargs="+", required=True); q.add_argument("--out")
    args = ap.parse_args()
    if args.cmd == "manifest":
        {"hf": manifest_hf, "npz": manifest_npz}[args.kind](args)
    else:
        {"score": score, "metrics": metrics}[args.cmd](args)


if __name__ == "__main__":
    main()
