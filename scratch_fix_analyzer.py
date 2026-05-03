import pathlib
import re

content = pathlib.Path('tests/test_analyzer.py').read_text(encoding='utf-8')

# Delete mock classes completely
content = re.sub(r'class RaisingSourceSeparator:.*?(?=\n\nclass |\Z)', '', content, flags=re.DOTALL)
content = re.sub(r'class FixedSourceSeparator:.*?(?=\n\nclass |\n\ndef |\Z)', '', content, flags=re.DOTALL)
content = re.sub(r'class RecordingSourceSeparator:.*?(?=\n\nclass |\n\ndef |\Z)', '', content, flags=re.DOTALL)

# Delete demucs tests completely
content = re.sub(r'def test_demucs_vocals_path_is_used_for_speech_when_available.*?(\n\ndef |\Z)', r'\1', content, flags=re.DOTALL)
content = re.sub(r'def test_demucs_artifacts_are_isolated_per_run_for_same_filename.*?(\n\ndef |\Z)', r'\1', content, flags=re.DOTALL)
content = re.sub(r'def test_direct_isolate_vocals_uses_unique_default_output_dirs.*?(\n\ndef |\Z)', r'\1', content, flags=re.DOTALL)
content = re.sub(r'def test_demucs_unavailable_falls_back_to_original_for_speech.*?(\n\ndef |\Z)', r'\1', content, flags=re.DOTALL)

# Remove artifacts check
content = re.sub(r'    assert artifacts\["demucs_vocals"\].*?\n', '', content)

# Also remove 'stage=source_separation' from pipeline test
content = re.sub(r'\s*"stage=source_separation",\n', '\n', content)

pathlib.Path('tests/test_analyzer.py').write_text(content, encoding='utf-8')
print('Removed mocks and tests from test_analyzer.py')
