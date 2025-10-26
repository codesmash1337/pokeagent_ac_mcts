"""Expose vendored test utilities under metamon.test.* for local development."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

__all__: list[str] = []

_TEST_DIR = Path(__file__).resolve().parents[2] / "test"
if _TEST_DIR.is_dir():
    if str(_TEST_DIR) not in sys.path:
        sys.path.insert(0, str(_TEST_DIR))

    for module_name in [
        "ac_inference",
        "test_inference",
        "test_teampreview_conversion",
        "test_forward_conditionals",
        "test_observation",
    ]:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        sys.modules[f"{__name__}.{module_name}"] = module
        globals()[module_name] = module
        __all__.append(module_name)
else:
    raise ImportError(
        "Could not locate vendored 'test' directory for metamon; ensure PYTHONPATH includes vendor/metamon"
    )
