import json
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from voicescript.agent import ForensicAgent
from voicescript.config import Settings
from voicescript.utils.network import download_file
from langchain_core.messages import AIMessage, ToolMessage

@pytest.fixture
def mock_settings():
    return Settings(
        openai_api_key="fake-key",
        llm_model="gpt-5.4-mini"
    )

@patch("voicescript.agent.ChatOpenAI")
@patch("voicescript.agent.McpToolset")
def test_agent_full_workflow(mock_toolset_cls, mock_llm_cls, mock_settings):
    # Setup mocks
    mock_toolset = mock_toolset_cls.return_value
    mock_toolset.get_audio_metadata.return_value = {"duration": 120}
    mock_toolset.analyze_audio_quality.return_value = {"silence_ratio": 0.05}
    mock_toolset.estimate_speakers.return_value = {"speaker_count": 2}
    
    mock_llm = mock_llm_cls.return_value
    
    # First response: call a tool
    msg1 = AIMessage(
        content="", 
        tool_calls=[{"name": "get_metadata", "args": {"input_file": "test.wav"}, "id": "call_1"}],
        response_metadata={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
    )
    
    # Second response: transition to end
    msg2 = AIMessage(
        content="I have all the data. Generating report.",
        response_metadata={"token_usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}}
    )
    
    # Third response: actual summary call
    msg3 = AIMessage(
        content='{"file_name": "test.wav", "duration_seconds": 120, "audio_quality": {"silence_ratio": 0.05}, "issues": [], "recommendations": "Good quality."}',
        response_metadata={"token_usage": {"prompt_tokens": 15, "completion_tokens": 20, "total_tokens": 35}}
    )

    mock_llm.invoke.side_effect = [msg1, msg2, msg3]
    mock_llm.bind_tools.return_value = mock_llm

    agent = ForensicAgent(mock_settings)
    report = agent.analyze_file("test.wav")
    
    assert report["file_name"] == "test.wav"
    assert report["duration_seconds"] == 120
    assert "llm_usage" in report
    # 15 + 20 + 35 = 70
    assert report["llm_usage"]["total_tokens"] == 70


def test_generate_summary_extracts_tool_messages():
    """_generate_summary must pass tool outputs as structured data, not just history."""
    # Build a state with one ToolMessage
    tool_output = {"duration_seconds": 120.0, "bitrate": 128000}
    tool_msg = ToolMessage(
        content=json.dumps(tool_output),
        tool_call_id="call_123",
        name="get_metadata",
    )
    state = {
        "messages": [tool_msg],
        "input_file": "court.wav",
        "analysis_results": {},
        "final_report": None,
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    # Mock the LLM to capture what prompt it receives
    fake_report = {"file_name": "court.wav", "duration_seconds": 120.0}
    mock_response = MagicMock()
    mock_response.content = json.dumps(fake_report)
    mock_response.response_metadata = {"token_usage": {}}

    agent = ForensicAgent.__new__(ForensicAgent)  # skip __init__
    agent.settings = MagicMock()
    agent.llm = MagicMock(return_value=mock_response)
    agent.llm.invoke = MagicMock(return_value=mock_response)

    result = agent._generate_summary(state)

    # Verify final_report was produced
    assert result["final_report"]["file_name"] == "court.wav"

    # Verify tool data appeared in the prompt as an *explicit structured JSON block*
    # (not merely as raw ToolMessage history). The fixed code injects a HumanMessage
    # that starts with "Tool results collected:" followed by a ```json block.
    assert agent.llm.invoke.called, "llm.invoke should have been called"
    call_args = agent.llm.invoke.call_args
    # call_args[0][0] is the list of messages passed to llm.invoke
    prompt_messages = call_args[0][0]
    human_texts = [
        msg.content
        for msg in prompt_messages
        if hasattr(msg, "content") and isinstance(msg.content, str)
    ]
    combined = "\n".join(human_texts)
    assert "Tool results collected:" in combined, (
        "Expected 'Tool results collected:' header in the summary prompt; "
        "the dead loop was not replaced. Prompt content: " + combined[:600]
    )
    assert '"get_metadata"' in combined, (
        "Expected tool name key 'get_metadata' in the structured JSON block. "
        "Prompt content: " + combined[:600]
    )
    # technical_data is now dict[str, list] — values must be arrays
    assert '"duration_seconds"' in combined, (
        "Expected the tool output contents in the JSON block. "
        "Prompt content: " + combined[:600]
    )


def test_generate_summary_deduplicates_tool_calls():
    """Calling the same tool twice must not lose the first result."""
    tool_output_1 = {"duration_seconds": 60.0, "bitrate": 64000}
    tool_output_2 = {"duration_seconds": 120.0, "bitrate": 128000}
    tool_msg_1 = ToolMessage(
        content=json.dumps(tool_output_1),
        tool_call_id="call_a",
        name="get_metadata",
    )
    tool_msg_2 = ToolMessage(
        content=json.dumps(tool_output_2),
        tool_call_id="call_b",
        name="get_metadata",
    )
    state = {
        "messages": [tool_msg_1, tool_msg_2],
        "input_file": "court.wav",
        "analysis_results": {},
        "final_report": None,
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    fake_report = {"file_name": "court.wav", "duration_seconds": 120.0}
    mock_response = MagicMock()
    mock_response.content = json.dumps(fake_report)
    mock_response.response_metadata = {"token_usage": {}}

    agent = ForensicAgent.__new__(ForensicAgent)
    agent.settings = MagicMock()
    agent.llm = MagicMock(return_value=mock_response)
    agent.llm.invoke = MagicMock(return_value=mock_response)

    result = agent._generate_summary(state)

    assert result["final_report"]["file_name"] == "court.wav"

    call_args = agent.llm.invoke.call_args
    prompt_messages = call_args[0][0]
    human_texts = [
        msg.content
        for msg in prompt_messages
        if hasattr(msg, "content") and isinstance(msg.content, str)
    ]
    combined = "\n".join(human_texts)

    # Both results must appear; neither should have been silently overwritten
    assert "64000" in combined, (
        "First tool call result (bitrate 64000) was lost — overwrite bug still present. "
        "Prompt content: " + combined[:800]
    )
    assert "128000" in combined, (
        "Second tool call result (bitrate 128000) is missing from the prompt. "
        "Prompt content: " + combined[:800]
    )
    # The 'get_metadata' value in the tool_block must be a JSON array — verify both
    # entries are present inside the structured ```json block (not just in raw history).
    assert combined.count('"get_metadata"') >= 1, (
        "Expected 'get_metadata' key in the structured tool_block. "
        "Prompt content: " + combined[:800]
    )
    # Both bitrate values must appear together inside the tool_block array
    tool_block_start = combined.find("Tool results collected:")
    assert tool_block_start != -1, "tool_block header not found in prompt"
    tool_block_section = combined[tool_block_start:]
    assert "64000" in tool_block_section, (
        "First call's bitrate (64000) missing from tool_block — accumulation failed. "
        "tool_block section: " + tool_block_section[:600]
    )
    assert "128000" in tool_block_section, (
        "Second call's bitrate (128000) missing from tool_block. "
        "tool_block section: " + tool_block_section[:600]
    )
