"""POSITIVE CONTROL: do BOTH cameras agree that a genuinely solved demo is solved?

The occlusion fix pays reward only where every view supports it (multiview_reduce=min,
and success needs a majority of views = both). That is only safe if view2 actually
FIRES on a real success. If base_camera scores a true success low, min() suppresses
reward on genuine successes and the fix breaks learning instead of fixing it.

Method:
  * replay ManiSkill's official PullCube demos (actions, recorded seed)
  * keep only episodes that OUR env reports as success -- the demo's own label is
    not trusted, because the demos were generated on physx_cuda and we run physx_cpu
  * score every prefix with the SAME call the replay buffer uses
    (raw_dict_to_sample -> process_batch_helper -> extract_*), so the numbers are
    the ones training would actually see
  * both views are scored with video_embeddings=None, i.e. symmetrically, which is
    also exactly what the online path does for image2

Reported: per-view progress and success-prob at the solved steps, and whether each
view clears the threshold. Verdict is about view2 -- view1 is the known-good control.
"""
import argparse, json, os, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-dir", required=True)
    ap.add_argument("--control-mode", default="pd_ee_delta_pos")
    ap.add_argument("--reward-ckpt", required=True)
    ap.add_argument("--second-view", default="base_camera")
    ap.add_argument("--task", default="PullCube-v1")
    ap.add_argument("--n-episodes", type=int, default=6)
    ap.add_argument("--max-tries", type=int, default=25)
    ap.add_argument("--threshold", type=float, default=0.75,
                    help="success_detection_threshold to judge 'fires' (lam03 BoxClose gate)")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    import h5py, torch
    from robometer_policy_learning.envs.maniskill_wrapper import make_maniskill_env
    from robometer.evals.eval_utils import raw_dict_to_sample
    from robometer.evals.eval_server import process_batch_helper
    from robometer.utils.setup_utils import setup_batch_collator
    from robometer_policy_learning.utils.robometer_utils import (
        extract_rewards_from_output, extract_success_probs_from_output,
    )
    from robometer.utils.save import load_model_from_hf

    stem = f"trajectory.none.{args.control_mode}.physx_cuda"
    meta = json.load(open(os.path.join(args.demo_dir, stem + ".json")))
    h5 = h5py.File(os.path.join(args.demo_dir, stem + ".h5"), "r")
    eps = meta["episodes"]
    print(f"[demo] {len(eps)} episodes, control_mode={args.control_mode}", flush=True)

    env, instruction = make_maniskill_env(
        env_id=args.task, num_envs=1, image_size=224, use_full_state=False,
        max_episode_steps=meta["env_info"].get("max_episode_steps", 50),
        reward_mode="sparse", obs_mode="state", sim_backend="physx_cpu",
        terminate_on_success=False, second_view=args.second_view,
        control_mode=args.control_mode,
    )
    print(f"[env] instruction={instruction!r}  second_view={args.second_view}", flush=True)

    def frame(obs, key):
        v = np.asarray(obs[key])
        return v[0] if v.ndim == 4 else v

    # ---- replay until we have n_episodes that OUR env calls a success ----------
    kept = []
    for ep in eps[: args.max_tries]:
        if len(kept) >= args.n_episodes:
            break
        if not ep.get("success", False):
            continue
        tid = f"traj_{ep['episode_id']}"
        if tid not in h5:
            continue
        actions = np.asarray(h5[tid]["actions"])
        obs, _ = env.reset(seed=int(ep["episode_seed"]))
        v1 = [frame(obs, "image")]
        v2 = [frame(obs, "image2")]
        succ_flags = [False]
        for a in actions:
            obs, _r, term, trunc, info = env.step(np.asarray(a, dtype=np.float32)[None, :]
                                                 if np.asarray(a).ndim == 1 else np.asarray(a))
            s = info.get("success", info.get("is_success", False))
            s = bool(np.asarray(s).reshape(-1)[0]) if s is not None else False
            v1.append(frame(obs, "image")); v2.append(frame(obs, "image2"))
            succ_flags.append(s)
            if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
                break
        if any(succ_flags):
            kept.append(dict(ep_id=ep["episode_id"], v1=np.stack(v1), v2=np.stack(v2),
                             succ=np.array(succ_flags, dtype=bool)))
            print(f"  ep {ep['episode_id']}: replayed, OUR env says success at step "
                  f"{int(np.argmax(succ_flags))}/{len(succ_flags)-1}", flush=True)
        else:
            print(f"  ep {ep['episode_id']}: replay did NOT reach success in our env -- skipped",
                  flush=True)
    h5.close()
    if not kept:
        print("[fatal] no demo replayed to a success in our env; cannot run the control")
        return 2
    print(f"[demo] using {len(kept)} verified-successful episodes", flush=True)

    # ---- reward model, exactly as the buffer builds it -------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_cfg, tokenizer, processor, reward_model = load_model_from_hf(
        model_path=args.reward_ckpt, device=device)
    if not exp_cfg.data.use_multi_image:
        exp_cfg.data.use_multi_image = True
    collator = setup_batch_collator(processor, tokenizer, exp_cfg, is_eval=True)
    max_frames = getattr(exp_cfg.data, "max_frames", 16)
    plt_ = exp_cfg.loss.progress_loss_type.lower()
    is_discrete = plt_ == "discrete" or "c51" in plt_
    nbins = exp_cfg.loss.progress_discrete_bins
    print(f"[rm] loss={plt_} discrete={is_discrete} bins={nbins} max_frames={max_frames}",
          flush=True)

    def score(frames, ep_id):
        """Progress + success prob for every prefix of this view's frames."""
        raw = [dict(frames=frames[: t + 1], task=instruction, id=ep_id,
                    metadata=dict(subsequence_length=t + 1),
                    video_embeddings=None, text_embedding=None)
               for t in range(len(frames))]
        samples = [raw_dict_to_sample(r, max_frames=max_frames, sample_type="progress")
                   for r in raw]
        out = process_batch_helper(
            model_type=exp_cfg.model.model_type, model=reward_model, tokenizer=tokenizer,
            batch_collator=collator, device=reward_model.device,
            batch_data=[s.model_dump() for s in samples], job_id=0,
            is_discrete_mode=is_discrete, num_bins=nbins)
        return (np.asarray(extract_rewards_from_output(out), dtype=float),
                np.asarray(extract_success_probs_from_output(out), dtype=float))

    rows = []
    for k in kept:
        p1, s1 = score(k["v1"], k["ep_id"])
        p2, s2 = score(k["v2"], k["ep_id"])
        rows.append(dict(ep_id=k["ep_id"], succ=k["succ"].tolist(),
                         prog1=p1.tolist(), prog2=p2.tolist(),
                         sp1=s1.tolist(), sp2=s2.tolist()))
        m = k["succ"]
        print(f"\n--- episode {k['ep_id']} ({len(m)} steps, solved from step {int(np.argmax(m))}) ---",
              flush=True)
        print("  step :  " + " ".join(f"{i:6d}" for i in range(len(m))))
        print("  gt   :  " + " ".join(f"{'  SOLV' if x else '     .'}" for x in m))
        print("  prog1:  " + " ".join(f"{x:6.3f}" for x in p1))
        print("  prog2:  " + " ".join(f"{x:6.3f}" for x in p2))
        print("  min  :  " + " ".join(f"{min(a,b):6.3f}" for a, b in zip(p1, p2)))
        print("  sp1  :  " + " ".join(f"{x:6.3f}" for x in s1))
        print("  sp2  :  " + " ".join(f"{x:6.3f}" for x in s2))

    # ---- verdict ---------------------------------------------------------------
    thr = args.threshold
    P1 = np.concatenate([np.asarray(r["prog1"])[np.asarray(r["succ"])] for r in rows])
    P2 = np.concatenate([np.asarray(r["prog2"])[np.asarray(r["succ"])] for r in rows])
    S1 = np.concatenate([np.asarray(r["sp1"])[np.asarray(r["succ"])] for r in rows])
    S2 = np.concatenate([np.asarray(r["sp2"])[np.asarray(r["succ"])] for r in rows])
    print("\n" + "=" * 68)
    print(f"ON GENUINELY SOLVED STEPS (n={len(P1)} across {len(rows)} episodes)")
    print(f"  view1 render_camera: progress {P1.mean():.3f}  success_prob {S1.mean():.3f} "
          f"fires@{thr}: {(S1 > thr).mean():.0%}")
    print(f"  view2 base_camera  : progress {P2.mean():.3f}  success_prob {S2.mean():.3f} "
          f"fires@{thr}: {(S2 > thr).mean():.0%}")
    print(f"  min(progress)      : {np.minimum(P1, P2).mean():.3f}   "
          f"BOTH fire: {((S1 > thr) & (S2 > thr)).mean():.0%}")
    print("-" * 68)
    # NOTE: an earlier version of this verdict required BOTH views to clear `thr`
    # and declared failure when they did not. That was wrong and it fired on the
    # 2026-11-23 run: `thr` is the MetaWorld SUCCESS-head gate from BoxClose, but
    # these ManiSkill runs use the PROGRESS head with success detection OFF, so
    # NEITHER view clears it -- view1 fires 0% too. It indicted view2 for a
    # failure that is equally view1's, against a threshold that is not in play.
    # Judge on separation (solved vs unsolved) within each view instead, and
    # report the level cost of min() rather than gating on it.
    U1 = np.concatenate([np.asarray(r["prog1"])[~np.asarray(r["succ"])] for r in rows])
    U2 = np.concatenate([np.asarray(r["prog2"])[~np.asarray(r["succ"])] for r in rows])
    sep1, sep2 = P1.mean() - U1.mean(), P2.mean() - U2.mean()
    MN = np.minimum(P1, P2)
    print(f"  separation (solved - unsolved): view1 {sep1:+.3f}  view2 {sep2:+.3f}")
    print(f"  cost of min() on real successes: {P1.mean():.3f} -> {MN.mean():.3f} "
          f"({(MN.mean()-P1.mean())/P1.mean():+.1%})")
    print("-" * 68)
    if sep2 <= 0 or P2.std() < 1e-6:
        print("VERDICT: view2 is BLIND (no separation / no variance). min() would clamp")
        print("         the reward flat -- do NOT launch; use a MATCHED second camera.")
    elif sep2 >= sep1:
        print("VERDICT: view2 separates solved from unsolved AT LEAST AS WELL as view1.")
        print("         min() is safe on real successes; whether it is WORTH its level")
        print("         cost depends on how much farming reward it removes -- that comes")
        print("         from the policy replay, not from this control.")
    else:
        print("VERDICT: view2 is weaker than view1 but not blind. Judgement call --")
        print("         weigh the level cost above against the farming reward removed.")
    print("=" * 68)

    if args.out_json:
        json.dump(rows, open(args.out_json, "w"))
        print(f"[out] wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
