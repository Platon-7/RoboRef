"""Compatibility shim for the wandb 0.16.4 pinned in the demo2reward_v3 env.

Put this directory first on PYTHONPATH and Python imports it automatically at
interpreter startup (via site.py) -- including in the subprocess wandb spawns
for its internal sender, which is where the crash actually happens. Patching
from the training script would not help: the sender runs in a *spawned* child
that re-imports everything fresh.

Problem: Api.upsert_run() calls server_settings_introspection(), which does
    res.get("ServerSettingsType", {}).get("fields", [{}])
The current W&B SaaS GraphQL schema returns null for __type(name:"ServerSettings"),
so the chained .get() runs on None and wandb.init() dies with
    CommError: 'NoneType' object has no attribute 'get'
Not a network or credential failure -- it reproduces on the login node.

Fix: resolve that probe to [] without querying. wandb then omits the optional
`serverSettings { serverMessages }` block from the upsert mutation and returns
server_messages=None, exactly as it does against any server lacking the type.
Only visible effect: no server broadcast/MOTD banners in the console.

ROBOREF_WANDB_COMPAT=0 disables the shim. Jobs can verify it is really active
with jobs/compat/check.py before committing a GPU allocation.
"""

import os
import sys

_TARGET = "wandb.sdk.internal.internal_api"


def _apply(module):
    api = getattr(module, "Api", None)
    if api is None or not hasattr(api, "server_settings_introspection"):
        return False

    def server_settings_introspection(self):
        if self._server_settings_type is None:
            self._server_settings_type = []

    api.server_settings_introspection = server_settings_introspection
    return True


def _install():
    if os.environ.get("ROBOREF_WANDB_COMPAT", "1") == "0":
        return
    if _TARGET in sys.modules:
        _apply(sys.modules[_TARGET])
        return

    import importlib.util
    from importlib.abc import MetaPathFinder

    class _Finder(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != _TARGET:
                return None
            sys.meta_path.remove(self)  # don't recurse into ourselves
            try:
                spec = importlib.util.find_spec(fullname)
            except Exception:
                return None
            finally:
                if self not in sys.meta_path:
                    sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            inner = spec.loader.exec_module

            def exec_module(module, _inner=inner):
                _inner(module)
                _apply(module)

            try:
                spec.loader.exec_module = exec_module
            except Exception:
                return None
            return spec

    sys.meta_path.insert(0, _Finder())


try:
    _install()
except Exception:  # never break the interpreter over a logging shim
    pass
