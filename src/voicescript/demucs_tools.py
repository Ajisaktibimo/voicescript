"""Backward-compatible source-separation provider exports.

New code should import from ``voicescript.providers.separation``.
"""

from .providers.separation import (  # noqa: F401
    DisabledSourceSeparator,
    LocalDemucsSeparator,
    SourceSeparator,
    _demucs_available,
    _find_vocals_file,
    create_source_separator,
)

DemucsSeparator = LocalDemucsSeparator
