"""Tests for Copilot fallback behavior when LangChain tool calling is unavailable."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class _FakeTool:
    def __init__(self, name, result):
        self.name = name
        self._result = result
        self.calls = []

    def invoke(self, args):
        self.calls.append(args)
        return self._result


class _FakeCopilotLlm:
    supports_langchain_tool_calls = False

    def __init__(self, response_text):
        self.response_text = response_text
        self.calls = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=self.response_text)


class TestIndicatorAgentCopilotFallback(unittest.TestCase):
    """Tests for indicator agent fallback without tool calls."""

    def test_indicator_agent_runs_tools_directly_when_tool_calls_are_unsupported(self):
        from indicator_agent import create_indicator_agent

        llm = _FakeCopilotLlm("Reporte de indicadores con fallback.")
        toolkit = SimpleNamespace(
            compute_macd=_FakeTool("compute_macd", {"macd": [1.0]}),
            compute_rsi=_FakeTool("compute_rsi", {"rsi": [55.0]}),
            compute_roc=_FakeTool("compute_roc", {"roc": [3.0]}),
            compute_stoch=_FakeTool("compute_stoch", {"stoch_k": [20.0], "stoch_d": [25.0]}),
            compute_willr=_FakeTool("compute_willr", {"willr": [-15.0]}),
        )
        agent = create_indicator_agent(llm, toolkit)

        state = {
            "time_frame": "1hour",
            "kline_data": {
                "Datetime": ["2026-04-18 10:00:00", "2026-04-18 11:00:00"],
                "Open": [1.0, 2.0],
                "High": [1.5, 2.5],
                "Low": [0.5, 1.5],
                "Close": [1.2, 2.2],
            },
            "messages": [],
        }

        result = agent(state)

        self.assertEqual(result["indicator_report"], "Reporte de indicadores con fallback.")
        self.assertEqual(len(toolkit.compute_macd.calls), 1)
        self.assertEqual(len(toolkit.compute_rsi.calls), 1)
        self.assertEqual(len(toolkit.compute_roc.calls), 1)
        self.assertEqual(len(toolkit.compute_stoch.calls), 1)
        self.assertEqual(len(toolkit.compute_willr.calls), 1)
        invoked_messages = llm.calls[0]
        combined_content = "\n".join(
            str(getattr(message, "content", "")) for message in invoked_messages
        )
        self.assertIn("compute_macd", combined_content)
        self.assertIn('"rsi"', combined_content)
        self.assertIn("55.0", combined_content)


class TestPatternAgentCopilotFallback(unittest.TestCase):
    """Tests for pattern agent fallback without tool calls."""

    def test_pattern_agent_generates_image_directly_when_tool_calls_are_unsupported(self):
        from pattern_agent import create_pattern_agent

        tool_llm = _FakeCopilotLlm("No deberia usarse para tool calling.")
        graph_llm = _FakeCopilotLlm("Patron detectado.")
        generate_kline_image = _FakeTool(
            "generate_kline_image",
            {"pattern_image": "ZmFrZS1wYXR0ZXJu", "pattern_image_description": "chart"},
        )
        toolkit = SimpleNamespace(generate_kline_image=generate_kline_image)
        agent = create_pattern_agent(tool_llm, graph_llm, toolkit)

        state = {
            "time_frame": "1hour",
            "kline_data": {
                "Datetime": ["2026-04-18 10:00:00", "2026-04-18 11:00:00"],
                "Open": [1.0, 2.0],
                "High": [1.5, 2.5],
                "Low": [0.5, 1.5],
                "Close": [1.2, 2.2],
            },
            "messages": [],
        }

        result = agent(state)

        self.assertEqual(result["pattern_report"], "Patron detectado.")
        self.assertEqual(len(generate_kline_image.calls), 1)
        self.assertEqual(len(tool_llm.calls), 0)
        self.assertEqual(len(graph_llm.calls), 1)


class TestTrendAgentCopilotFallback(unittest.TestCase):
    """Tests for trend agent fallback without tool calls."""

    def test_trend_agent_generates_image_directly_when_tool_calls_are_unsupported(self):
        from trend_agent import create_trend_agent

        tool_llm = _FakeCopilotLlm("No deberia usarse para tool calling.")
        graph_llm = _FakeCopilotLlm("Tendencia alcista.")
        generate_trend_image = _FakeTool(
            "generate_trend_image",
            {"trend_image": "ZmFrZS10cmVuZA==", "trend_image_description": "trend"},
        )
        toolkit = SimpleNamespace(generate_trend_image=generate_trend_image)
        agent = create_trend_agent(tool_llm, graph_llm, toolkit)

        state = {
            "time_frame": "1hour",
            "kline_data": {
                "Datetime": ["2026-04-18 10:00:00", "2026-04-18 11:00:00"],
                "Open": [1.0, 2.0],
                "High": [1.5, 2.5],
                "Low": [0.5, 1.5],
                "Close": [1.2, 2.2],
            },
            "messages": [],
        }

        result = agent(state)

        self.assertEqual(result["trend_report"], "Tendencia alcista.")
        self.assertEqual(result["trend_image"], "ZmFrZS10cmVuZA==")
        self.assertEqual(len(generate_trend_image.calls), 1)
        self.assertEqual(len(tool_llm.calls), 0)
        self.assertEqual(len(graph_llm.calls), 1)

    def test_trend_agent_falls_back_to_graph_reasoning_when_image_generation_returns_no_image(self):
        from trend_agent import create_trend_agent

        tool_llm = _FakeCopilotLlm("No deberia usarse para tool calling.")
        graph_llm = _FakeCopilotLlm("Tendencia lateral sin imagen.")
        generate_trend_image = _FakeTool(
            "generate_trend_image",
            {"trend_image_description": "trend"},
        )
        toolkit = SimpleNamespace(generate_trend_image=generate_trend_image)
        agent = create_trend_agent(tool_llm, graph_llm, toolkit)

        state = {
            "time_frame": "1hour",
            "kline_data": {
                "Datetime": ["2026-04-18 10:00:00", "2026-04-18 11:00:00"],
                "Open": [1.0, 2.0],
                "High": [1.5, 2.5],
                "Low": [0.5, 1.5],
                "Close": [1.2, 2.2],
            },
            "messages": [],
        }

        result = agent(state)

        self.assertEqual(result["trend_report"], "Tendencia lateral sin imagen.")
        self.assertIsNone(result["trend_image"])
        self.assertEqual(len(generate_trend_image.calls), 1)
        self.assertEqual(len(tool_llm.calls), 0)
        self.assertEqual(len(graph_llm.calls), 1)
