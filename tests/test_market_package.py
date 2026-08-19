from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_market_clone_root_exposes_semantic_sticker_plugin() -> None:
    script = textwrap.dedent(
        """
        import importlib.util
        import os
        import runpy
        import sys
        import types
        from pathlib import Path

        repository_root = Path(os.environ["PLUGIN_REPOSITORY_ROOT"])
        runpy.run_path(str(repository_root / "tests" / "conftest.py"))

        packages = types.ModuleType("packages")
        packages.__path__ = [str(repository_root.parent)]
        sys.modules["packages"] = packages

        module_name = "packages.nekro_plugin_semantic_sticker"
        entrypoint = repository_root / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            module_name,
            entrypoint,
            submodule_search_locations=[str(repository_root)],
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        assert module.plugin.module_name == "semantic_sticker"
        assert module.plugin.version == "1.2.4"
        assert Path(module.__file__).resolve() == entrypoint.resolve()
        assert module.router.__name__ == (
            "packages.nekro_plugin_semantic_sticker.router"
        )
        assert not (repository_root / "source" / "nekro_plugin_semantic_sticker").exists()
        """
    )
    env = os.environ.copy()
    env["PLUGIN_REPOSITORY_ROOT"] = str(REPOSITORY_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
