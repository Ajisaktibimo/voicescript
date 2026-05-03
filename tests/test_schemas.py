from voicescript import models
from voicescript.schemas import AudioMetadata, ForensicReport, SourceSeparationResult, to_jsonable
from voicescript.schemas.audio import SilenceSummary


def test_schemas_package_is_public_source_for_pydantic_models():
    assert AudioMetadata is models.AudioMetadata
    assert ForensicReport is models.ForensicReport
    assert SourceSeparationResult is models.SourceSeparationResult
    assert to_jsonable(SilenceSummary()) == {"segments": [], "silence_ratio": 0.0}
