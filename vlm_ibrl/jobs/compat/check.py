"""Pre-flight for online wandb logging in the demo2reward_v3 env.

Exit 0 only if BOTH hold, so a job never dies at wandb.init() 11 minutes in:
  1. the sitecustomize ServerSettings shim is installed (see sitecustomize.py)
  2. an authenticated GraphQL round trip to the W&B server actually works
Any failure exits 1 and the caller logs offline, to be synced later.
"""

import sys
import traceback


def main() -> int:
    try:
        import wandb
        import wandb.sdk.internal.internal_api as ia
    except Exception:
        traceback.print_exc()
        print("wandb-preflight: import failed")
        return 1

    fn = ia.Api.server_settings_introspection
    shim = getattr(fn, "__module__", "") == "sitecustomize"

    viewer = None
    try:
        viewer = ia.Api().viewer()
    except Exception as exc:
        print(f"wandb-preflight: viewer query failed: {type(exc).__name__}: {exc}")

    entity = (viewer or {}).get("entity") if isinstance(viewer, dict) else None
    ok = shim and bool(viewer)
    print(
        f"wandb-preflight: wandb={wandb.__version__} "
        f"shim={'ACTIVE' if shim else 'MISSING'} "
        f"server={'reachable' if viewer else 'UNREACHABLE'} "
        f"entity={entity} -> {'ONLINE' if ok else 'OFFLINE'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
