import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path


def test_root_entrypoints_import_without_pythonpath():
    runtime_dir = Path("test-runtime") / uuid.uuid4().hex
    runtime_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["VOICESCRIPT_DATA_DIR"] = str(runtime_dir / "data")
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import mcp_server; import api_server; print('ok')"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    finally:
        shutil.rmtree(runtime_dir, ignore_errors=True)

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
