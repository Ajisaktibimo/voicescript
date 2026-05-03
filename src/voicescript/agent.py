from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from voicescript.config import Settings
from voicescript.mcp_tools import McpToolset
from voicescript.utils.network import is_url, download_file


class AgentState(TypedDict):
    """The state of the forensic analysis agent."""
    messages: Annotated[list[BaseMessage], add_messages]
    analysis_results: dict[str, Any]
    input_file: str
    final_report: dict[str, Any] | None
    token_usage: dict[str, int]


class ForensicAgent:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.tools = McpToolset()
        self.llm = ChatOpenAI(
            model=getattr(self.settings, "llm_model", "gpt-5.4-mini"),
            api_key=getattr(self.settings, "openai_api_key", ""),
        )
        
        # Define tools for the LLM
        @tool
        def get_metadata(input_file: str) -> dict[str, Any]:
            """Extract audio metadata like duration, bitrate, sample rate, etc."""
            return self.tools.get_audio_metadata(input_file)

        @tool
        def analyze_quality(input_file: str) -> dict[str, Any]:
            """Analyze audio quality (silence, clipping, volume)."""
            return self.tools.analyze_audio_quality(input_file)

        @tool
        def estimate_speakers(input_file: str) -> dict[str, Any]:
            """Estimate the number of speakers in the audio."""
            return self.tools.estimate_speakers(input_file)

        self.tool_list = [get_metadata, analyze_quality, estimate_speakers]
        self.llm_with_tools = self.llm.bind_tools(self.tool_list)
        
        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        # Nodes
        workflow.add_node("orchestrator", self._call_model)
        workflow.add_node("tools", ToolNode(self.tool_list))
        workflow.add_node("summarizer", self._generate_summary)

        # Edges
        workflow.add_edge(START, "orchestrator")
        workflow.add_conditional_edges(
            "orchestrator",
            self._should_continue,
            {
                "continue": "tools",
                "end": "summarizer",
            },
        )
        workflow.add_edge("tools", "orchestrator")
        workflow.add_edge("summarizer", END)

        return workflow.compile()

    def _call_model(self, state: AgentState) -> dict[str, Any]:
        """Node to decide which tool to call."""
        messages = state["messages"]
        if not messages:
            # Initialize with the analysis request
            file_path = state["input_file"]
            messages = [
                SystemMessage(content=(
                    "You are a Forensic Audio Analysis Orchestrator. "
                    "Your goal is to perform a complete triage of an audio file. "
                    "Call the necessary tools to get metadata, quality info, and speaker estimates. "
                    "Once you have all technical data, finish to generate the final report."
                )),
                HumanMessage(content=f"Please analyze the file: {file_path}")
            ]
        
        response = self.llm_with_tools.invoke(messages)
        usage = response.response_metadata.get("token_usage", {})
        
        current_usage = state.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        new_usage = {
            "prompt_tokens": current_usage["prompt_tokens"] + usage.get("prompt_tokens", 0),
            "completion_tokens": current_usage["completion_tokens"] + usage.get("completion_tokens", 0),
            "total_tokens": current_usage["total_tokens"] + usage.get("total_tokens", 0),
        }
        
        return {"messages": [response], "token_usage": new_usage}

    def _should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        """Determine if we should continue tool calling or finalize."""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        return "end"

    def _generate_summary(self, state: AgentState) -> dict[str, Any]:
        """Final node to produce structured insights and human-readable summary."""
        # Collect all tool outputs from message history; accumulate into lists so
        # repeated calls to the same tool are never silently overwritten.
        technical_data: dict[str, list] = {}
        for msg in state["messages"]:
            if isinstance(msg, ToolMessage):
                key = msg.name or msg.tool_call_id
                try:
                    technical_data.setdefault(key, []).append(json.loads(msg.content))
                except (json.JSONDecodeError, TypeError):
                    technical_data.setdefault(key, []).append(msg.content)

        tool_block = (
            f"Tool results collected:\n```json\n{json.dumps(technical_data, indent=2)}\n```\n\n"
            if technical_data else ""
        )

        summary_prompt = [
            SystemMessage(content=(
                "You are an expert Forensic Audio Analyst. "
                "Review the technical data collected by the tools and produce a final structured report. "
                "Include a executive summary and suggested remedial actions. "
                "Format your response as a valid JSON object matching the requested schema."
            )),
            *state["messages"],
            HumanMessage(content=(
                tool_block
                + "Please generate the final forensic report as JSON. "
                "Include: file_name, duration_seconds, audio_quality (silence_ratio, clipping_detected, avg_volume_db), "
                "issues (list of strings), and recommendations (human-readable string)."
            ))
        ]
        
        report_response = self.llm.invoke(summary_prompt)
        usage = report_response.response_metadata.get("token_usage", {})
        
        current_usage = state.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        total_usage = {
            "prompt_tokens": current_usage["prompt_tokens"] + usage.get("prompt_tokens", 0),
            "completion_tokens": current_usage["completion_tokens"] + usage.get("completion_tokens", 0),
            "total_tokens": current_usage["total_tokens"] + usage.get("total_tokens", 0),
        }
        
        try:
            # Try to parse JSON from the response
            report_content = report_response.content
            if "```json" in report_content:
                report_content = report_content.split("```json")[1].split("```")[0].strip()
            final_report = json.loads(report_content)
        except Exception:
            final_report = {"error": "Failed to parse LLM report", "raw": report_response.content}

        # Add token usage to the final report
        final_report["llm_usage"] = total_usage
        
        return {"final_report": final_report, "token_usage": total_usage}

    def analyze_file(self, input_file: str | Path) -> dict[str, Any]:
        """Public entry point to analyze a file using the agentic workflow."""
        input_str = str(input_file)
        
        # Handle URLs
        if is_url(input_str):
            import uuid
            run_id = uuid.uuid4().hex
            filename = input_str.split("/")[-1].split("?")[0] or "download.audio"
            target_path = self.settings.data_dir / "uploads" / run_id / filename
            input_str = str(download_file(input_str, target_path, max_size_bytes=self.settings.max_upload_size_bytes))

        result = self.graph.invoke({
            "messages": [],
            "input_file": input_str,
            "analysis_results": {},
            "final_report": None,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        })
        return result["final_report"]
