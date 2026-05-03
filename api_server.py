from pathlib import Path
import sys

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from voicescript.api import create_app
from voicescript.config import Settings


app = create_app()


if __name__ == "__main__":
    settings = Settings.from_env()
    uvicorn.run(app, host="127.0.0.1", port=settings.api_port)
