from pathlib import Path
import sys

from fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from voicescript.mcp_tools import McpToolset
from voicescript.agent import ForensicAgent
from voicescript.models import to_jsonable


mcp = FastMCP("CourtAudioForensicMCP")
tools = McpToolset()
agent = ForensicAgent()


@mcp.tool()
def get_audio_metadata(input_file: str) -> dict[str, object]:
    """Extract ffprobe metadata while preserving stream/channel distinctions."""
    return tools.get_audio_metadata(input_file)


@mcp.tool()
def analyze_audio_quality(input_file: str) -> dict[str, object]:
    """Analyze silence, volume, low-volume, and clipping indicators."""
    return tools.analyze_audio_quality(input_file)


@mcp.tool()
def detect_silence(input_file: str) -> dict[str, object]:
    """Detect silence segments and silence ratio."""
    return tools.detect_silence(input_file)


@mcp.tool()
def detect_clipping(input_file: str) -> dict[str, object]:
    """Detect clipping or hard-limiting indicators from volume analysis."""
    return tools.detect_clipping(input_file)


@mcp.tool()
def analyze_channels(input_file: str) -> dict[str, object]:
    """Estimate channel layout and microphone setup from measurable evidence."""
    return tools.analyze_channels(input_file)


@mcp.tool()
def estimate_speakers(input_file: str) -> dict[str, object]:
    """Estimate speaker count from diarization evidence when available."""
    return tools.estimate_speakers(input_file)


@mcp.tool()
def estimate_microphone_setup(input_file: str) -> dict[str, object]:
    """Estimate microphone count from streams, channels, and channel evidence."""
    return tools.estimate_microphone_setup(input_file)


@mcp.tool()
def detect_forensic_indicators(input_file: str) -> list[dict[str, object]]:
    """Return forensic triage indicators with evidence and confidence."""
    return tools.detect_forensic_indicators(input_file)


@mcp.tool()
def generate_spectrogram(input_file: str, output_image: str) -> dict[str, object]:
    """Generate a spectrogram image for visual forensic review."""
    return tools.generate_spectrogram(input_file, output_image)



@mcp.tool()
def analyze_audio_file(input_file: str) -> dict[str, object]:
    """Run the complete forensic triage analysis for one audio file."""
    return tools.analyze_audio_file(input_file)


@mcp.tool()
def analyze_with_agent(input_file: str) -> dict[str, object]:
    """Perform an agentic audio analysis with human-readable insights using an LLM."""
    return to_jsonable(agent.analyze_file(input_file))


@mcp.tool()
def analyze_audio_batch(input_files: list[str]) -> dict[str, object]:
    """Run forensic triage analysis for multiple audio files and aggregate results."""
    return tools.analyze_audio_batch(input_files)


@mcp.tool()
def evaluate_report(report: dict[str, object], reference: dict[str, object]) -> dict[str, object]:
    """Evaluate a forensic report against a human reference fixture."""
    return tools.evaluate_report(report, reference)


if __name__ == "__main__":
    mcp.run()
