import httpx
from pathlib import Path
import logging
from voicescript.config import Settings

logger = logging.getLogger("uvicorn.error")

def download_file(url: str, target_path: Path, max_size_bytes: int | None = None) -> Path:
    """Download a file from a URL to a local path with optional size limit."""
    logger.info("Downloading %s to %s", url, target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    with httpx.stream("GET", url, follow_redirects=True) as response:
        response.raise_for_status()
        
        # Check Content-Length if available
        content_length = response.headers.get("Content-Length")
        if content_length and max_size_bytes and int(content_length) > max_size_bytes:
            raise ValueError(f"URL content exceeds maximum size of {max_size_bytes} bytes.")
            
        bytes_written = 0
        with open(target_path, "wb") as f:
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                bytes_written += len(chunk)
                if max_size_bytes and bytes_written > max_size_bytes:
                    target_path.unlink(missing_ok=True)
                    raise ValueError(f"Download exceeded maximum size of {max_size_bytes} bytes.")
                f.write(chunk)
    
    logger.info("Download complete: %s (%d bytes)", target_path, bytes_written)
    return target_path

def is_url(path_or_url: str) -> bool:
    """Check if a string looks like a URL."""
    return path_or_url.startswith(("http://", "https://"))
