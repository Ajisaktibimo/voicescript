"""Backward-compatible re-export of the public Pydantic schemas.

New code should import from ``voicescript.schemas`` or one of its focused
submodules. This module remains so existing integrations keep working.
"""

from .schemas import *  # noqa: F401,F403
