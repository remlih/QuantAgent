"""Tests for Spanish localization in agent prompts."""

import os
import sys
import unittest
from unittest.mock import MagicMock


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

for mod_name in ["talib", "langchain_qwq"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()


class TestAgentPromptLocalization(unittest.TestCase):
    """Ensure agent prompts request Spanish output without breaking contracts."""

    def test_indicator_prompt_requests_spanish_output(self):
        """Indicator prompt should instruct the model to answer in Spanish."""
        from indicator_agent import build_indicator_system_prompt

        prompt = build_indicator_system_prompt("1h")

        self.assertIn("Responde en español", prompt)
        self.assertIn("datos OHLC", prompt)

    def test_pattern_prompts_request_spanish_output(self):
        """Pattern prompts should keep reference text and analysis instructions in Spanish."""
        from pattern_agent import (
            build_pattern_image_prompt_text,
            build_pattern_reference_text,
        )

        pattern_reference = build_pattern_reference_text()
        image_prompt = build_pattern_image_prompt_text("1h", pattern_reference)

        self.assertIn("patrones clásicos de velas", pattern_reference)
        self.assertIn("Responde en español", image_prompt)

    def test_trend_prompts_request_spanish_output(self):
        """Trend prompts should ask for Spanish analysis and Spanish trend labels."""
        from trend_agent import (
            build_trend_image_prompt_text,
            build_trend_tool_system_prompt,
        )

        tool_prompt = build_trend_tool_system_prompt()
        image_prompt = build_trend_image_prompt_text("1h")

        self.assertIn("Responde en español", tool_prompt)
        self.assertIn("alcista", image_prompt)
        self.assertIn("bajista", image_prompt)
        self.assertIn("lateral", image_prompt)

    def test_decision_prompt_keeps_json_keys_and_requests_spanish_reasoning(self):
        """Decision prompt should require Spanish prose while preserving parser keys."""
        from decision_agent import build_decision_prompt

        prompt = build_decision_prompt(
            time_frame="1h",
            stock_name="BTC",
            indicator_report="Indicadores en español",
            pattern_report="Patrones en español",
            trend_report="Tendencia en español",
        )

        self.assertIn("Responde completamente en español", prompt)
        self.assertIn('"forecast_horizon"', prompt)
        self.assertIn('"decision"', prompt)
        self.assertIn('"justification"', prompt)
        self.assertIn('"risk_reward_ratio"', prompt)
        self.assertIn("LONG o SHORT", prompt)


if __name__ == "__main__":
    unittest.main()
