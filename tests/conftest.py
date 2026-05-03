from pathlib import Path
import shutil
import uuid

import pytest


@pytest.fixture()
def runtime_dir():
    path = Path("test-runtime") / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)
