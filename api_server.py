from pathlib import Path
import sys

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from voicescript.api import create_app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
