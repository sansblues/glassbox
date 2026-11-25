"""WSL pinned-memory shim for old vLLM builds (enable with PYTHONPATH=compat).

vLLM before ~mid-2026 returned is_pin_memory_available() == False whenever it detected WSL, so Model Runner V2
(which needs pinned UVA buffers) refused to start ("UVA is not available"). Newer vLLM verifies pinned memory on
WSL instead, and vLLM 0.34 runs MRV2 with pinned UVA buffers on this machine. This shim makes the old builds answer
True as well. It changes no scheduling, KV-cache or attention code, and is applied identically to pre-fix and fix.
"""
import importlib.abc, importlib.machinery, sys


class _Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "vllm.platforms.interface":
            return None
        sys.meta_path.remove(self)
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        orig = spec.loader.exec_module

        def exec_module(module):
            orig(module)
            module.Platform.is_pin_memory_available = classmethod(lambda cls: True)
            print("[compat] WSL: vllm Platform.is_pin_memory_available forced True", file=sys.stderr)
        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _Hook())
