import pathlib

def clean_file(filepath, replacements):
    path = pathlib.Path(filepath)
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8")
    for old, new in replacements:
        content = content.replace(old, new)
    path.write_text(content, encoding="utf-8")

clean_file("tests/test_analyzer.py", [
    ("    SourceSeparationResult,\n", ""),
    ("        demucs_separator=RaisingSourceSeparator(),\n", ""),
    ("        demucs_separator=FixedSourceSeparator(\n            SourceSeparationResult(available=False, enabled=True, limitations=[])\n        ),\n", ""),
    ("        demucs_separator=FixedSourceSeparator(\n            SourceSeparationResult(available=False, enabled=True, limitations=[\"Demucs is not installed.\"])\n        ),\n", ""),
    ("        source_separation_provider=\"disabled\",\n", ""),
])

clean_file("tests/test_api.py", [
    ("    SourceSeparationResult,\n", ""),
])

clean_file("tests/test_config.py", [
    ("        source_separation_provider=\"local\",\n", ""),
    ("    assert settings.source_separation_provider == \"local\"\n", ""),
    ("        source_separation_provider=\"huggingface\",\n", ""),
    ("    assert settings.source_separation_provider == \"huggingface\"\n", ""),
    ("        source_separation_provider=\"disabled\",\n", ""),
    ("    assert settings.source_separation_provider == \"disabled\"\n", ""),
])

clean_file("tests/test_evaluation.py", [
    ("    SourceSeparationResult,\n", ""),
])

clean_file("tests/test_forensic_rules.py", [
    ("    SourceSeparationResult,\n", ""),
])

clean_file("tests/test_fusion.py", [
    ("    SourceSeparationResult,\n", ""),
])

clean_file("tests/test_mcp_tools.py", [
    ("    SourceSeparationResult,\n", ""),
])

clean_file("tests/test_parsers.py", [
    ("    SourceSeparationResult,\n", ""),
])

clean_file("tests/test_providers.py", [
    ("from voicescript.providers import separation\n", ""),
    ("from voicescript.providers.separation import DisabledSourceSeparator, create_source_separator\n", ""),
    ("            source_separation_provider=provider_name,\n", ""),
    ("    separator = create_source_separator(settings)\n", ""),
    ("    assert type(separator).__name__ == expected_separation\n", ""),
    ("[local-local-faster-whisper-local-pyannote-local-demucs]", "[local-local-faster-whisper-local-pyannote]"),
    ("[disabled-disabled-disabled-disabled]", "[disabled-disabled-disabled]"),
    ("[huggingface-huggingface-huggingface-huggingface]", "[huggingface-huggingface-huggingface]"),
    ("[api-api-api-api]", "[api-api-api]"),
    ("[other-other-other-other]", "[other-other-other]"),
    (", separation: str", ""),
    ("    separation_allowed = \"Allowed values: api, disabled, huggingface, local, other\"\n", ""),
    ("    with pytest.raises(ValueError, match=separation_allowed):\n        create_source_separator(Settings(source_separation_provider=\"demuc\"))\n", ""),
])

clean_file("tests/test_schemas.py", [
    ("    SourceSeparationResult,\n", ""),
    ("    assert SourceSeparationResult is models.SourceSeparationResult\n", ""),
])

print("Replaced simple text")
