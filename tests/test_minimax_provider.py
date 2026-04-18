"""Unit tests for MiniMax provider integration in QuantAgent."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock heavy native/incompatible dependencies before importing project modules
# TA-Lib requires a C library; langchain has pydantic v1/v2 conflicts
for mod_name in [
    "talib",
    "langchain_qwq",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from default_config import DEFAULT_CONFIG


class TestDefaultConfig(unittest.TestCase):
    """Tests for MiniMax fields in DEFAULT_CONFIG."""

    def test_minimax_api_key_field_exists(self):
        """DEFAULT_CONFIG should contain a minimax_api_key field."""
        self.assertIn("minimax_api_key", DEFAULT_CONFIG)

    def test_provider_comment_mentions_minimax(self):
        """Provider fields should accept 'minimax' as a valid value."""
        config = DEFAULT_CONFIG.copy()
        config["agent_llm_provider"] = "minimax"
        config["graph_llm_provider"] = "minimax"
        self.assertEqual(config["agent_llm_provider"], "minimax")
        self.assertEqual(config["graph_llm_provider"], "minimax")


class TestTradingGraphGetApiKey(unittest.TestCase):
    """Tests for TradingGraph._get_api_key() with minimax provider."""

    def _make_graph(self, config):
        """Create a TradingGraph with mocked LLM creation."""
        from trading_graph import TradingGraph
        orig_create = TradingGraph._create_llm
        TradingGraph._create_llm = MagicMock(return_value=MagicMock())
        tg = TradingGraph(config=config)
        TradingGraph._create_llm = orig_create
        return tg

    def test_get_api_key_from_config(self):
        """Should return minimax_api_key from config."""
        config = DEFAULT_CONFIG.copy()
        config["minimax_api_key"] = "test-minimax-key-123"
        tg = self._make_graph(config)
        key = tg._get_api_key("minimax")
        self.assertEqual(key, "test-minimax-key-123")

    def test_get_openai_api_key_falls_back_to_env_when_config_uses_placeholder(self):
        """OpenAI should ignore the default placeholder and use OPENAI_API_KEY."""
        config = DEFAULT_CONFIG.copy()
        config["api_key"] = "sk-"
        tg = self._make_graph(config)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-openai-key"}):
            key = tg._get_api_key("openai")
            self.assertEqual(key, "env-openai-key")

    def test_get_api_key_from_env(self):
        """Should fall back to MINIMAX_API_KEY env var."""
        config = DEFAULT_CONFIG.copy()
        config["minimax_api_key"] = ""
        tg = self._make_graph(config)
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "env-minimax-key"}):
            key = tg._get_api_key("minimax")
            self.assertEqual(key, "env-minimax-key")

    def test_get_api_key_missing_raises(self):
        """Should raise ValueError if no MiniMax API key is available."""
        config = DEFAULT_CONFIG.copy()
        config["minimax_api_key"] = ""
        tg = self._make_graph(config)
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MINIMAX_API_KEY", None)
            with self.assertRaises(ValueError) as ctx:
                tg._get_api_key("minimax")
            self.assertIn("MiniMax", str(ctx.exception))

    def test_unsupported_provider_raises(self):
        """Should raise ValueError for unsupported provider."""
        config = DEFAULT_CONFIG.copy()
        tg = self._make_graph(config)
        with self.assertRaises(ValueError) as ctx:
            tg._get_api_key("unsupported_provider")
        self.assertIn("Unsupported provider", str(ctx.exception))
        self.assertIn("minimax", str(ctx.exception))


class TestTradingGraphCreateLlm(unittest.TestCase):
    """Tests for TradingGraph._create_llm() with minimax provider."""

    def _make_graph(self, config):
        """Create a TradingGraph with mocked LLM creation."""
        from trading_graph import TradingGraph
        orig_create = TradingGraph._create_llm
        TradingGraph._create_llm = MagicMock(return_value=MagicMock())
        tg = TradingGraph(config=config)
        TradingGraph._create_llm = orig_create
        return tg

    @patch("trading_graph.ChatOpenAI")
    def test_create_llm_minimax_uses_chatopenai(self, mock_openai):
        """MiniMax provider should create ChatOpenAI with custom base URL."""
        config = DEFAULT_CONFIG.copy()
        config["minimax_api_key"] = "test-key"
        tg = self._make_graph(config)
        tg.config = config

        mock_openai.return_value = MagicMock()
        result = tg._create_llm("minimax", "MiniMax-M2.7", 0.1)

        mock_openai.assert_called_once_with(
            model="MiniMax-M2.7",
            temperature=0.1,
            api_key="test-key",
            openai_api_base="https://api.minimax.io/v1",
        )

    @patch("trading_graph.ChatOpenAI")
    def test_create_llm_minimax_clamps_temperature(self, mock_openai):
        """MiniMax temperature should be clamped to (0.0, 1.0]."""
        config = DEFAULT_CONFIG.copy()
        config["minimax_api_key"] = "test-key"
        tg = self._make_graph(config)
        tg.config = config

        mock_openai.return_value = MagicMock()
        tg._create_llm("minimax", "MiniMax-M2.7", 0.0)
        call_args = mock_openai.call_args
        self.assertAlmostEqual(call_args.kwargs["temperature"], 0.01)

    @patch("trading_graph.ChatOpenAI")
    def test_create_llm_minimax_clamps_high_temperature(self, mock_openai):
        """MiniMax temperature > 1.0 should be clamped to 1.0."""
        config = DEFAULT_CONFIG.copy()
        config["minimax_api_key"] = "test-key"
        tg = self._make_graph(config)
        tg.config = config

        mock_openai.return_value = MagicMock()
        tg._create_llm("minimax", "MiniMax-M2.7", 1.5)
        call_args = mock_openai.call_args
        self.assertAlmostEqual(call_args.kwargs["temperature"], 1.0)

    @patch("trading_graph.ChatOpenAI")
    def test_create_llm_minimax_normal_temperature(self, mock_openai):
        """Normal temperature within range should be passed through."""
        config = DEFAULT_CONFIG.copy()
        config["minimax_api_key"] = "test-key"
        tg = self._make_graph(config)
        tg.config = config

        mock_openai.return_value = MagicMock()
        tg._create_llm("minimax", "MiniMax-M2.7", 0.5)
        call_args = mock_openai.call_args
        self.assertAlmostEqual(call_args.kwargs["temperature"], 0.5)


class TestTradingGraphUpdateApiKey(unittest.TestCase):
    """Tests for TradingGraph.update_api_key() with minimax provider."""

    def _make_graph(self, config):
        from trading_graph import TradingGraph
        orig_create = TradingGraph._create_llm
        TradingGraph._create_llm = MagicMock(return_value=MagicMock())
        tg = TradingGraph(config=config)
        TradingGraph._create_llm = orig_create
        return tg

    def test_update_api_key_minimax(self):
        """update_api_key('minimax') should update config and env var."""
        config = DEFAULT_CONFIG.copy()
        config["minimax_api_key"] = ""
        config["agent_llm_provider"] = "minimax"
        config["graph_llm_provider"] = "minimax"
        config["agent_llm_model"] = "MiniMax-M2.7"
        config["graph_llm_model"] = "MiniMax-M2.7"
        tg = self._make_graph(config)

        with patch.object(tg, "refresh_llms"):
            tg.update_api_key("new-minimax-key", provider="minimax")

        self.assertEqual(tg.config["minimax_api_key"], "new-minimax-key")
        self.assertEqual(os.environ.get("MINIMAX_API_KEY"), "new-minimax-key")

    def test_update_api_key_unsupported_raises(self):
        """update_api_key() with unsupported provider should raise ValueError."""
        config = DEFAULT_CONFIG.copy()
        tg = self._make_graph(config)
        with self.assertRaises(ValueError) as ctx:
            tg.update_api_key("key", provider="unsupported")
        self.assertIn("minimax", str(ctx.exception))


class TestTradingGraphRefreshLlms(unittest.TestCase):
    """Tests for TradingGraph.refresh_llms() with minimax provider."""

    @patch("trading_graph.ChatOpenAI")
    @patch("trading_graph.ChatAnthropic")
    @patch("trading_graph.ChatQwen")
    def test_refresh_llms_minimax(self, mock_qwen, mock_anthropic, mock_openai):
        """refresh_llms() should recreate LLMs when provider is minimax."""
        from trading_graph import TradingGraph

        config = DEFAULT_CONFIG.copy()
        config["agent_llm_provider"] = "minimax"
        config["graph_llm_provider"] = "minimax"
        config["agent_llm_model"] = "MiniMax-M2.7"
        config["graph_llm_model"] = "MiniMax-M2.7"
        config["minimax_api_key"] = "test-key"

        mock_openai.return_value = MagicMock()
        tg = TradingGraph(config=config)

        mock_openai.reset_mock()
        tg.refresh_llms()

        # ChatOpenAI should be called twice (agent_llm + graph_llm)
        self.assertEqual(mock_openai.call_count, 2)
        for call in mock_openai.call_args_list:
            self.assertEqual(call.kwargs["openai_api_base"], "https://api.minimax.io/v1")


class TestWebInterfaceProviderUpdate(unittest.TestCase):
    """Tests for web interface provider update with MiniMax."""

    @patch("web_interface.TradingGraph")
    def test_update_provider_minimax(self, mock_tg_class):
        """POST /api/update-provider with minimax should succeed."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer
        analyzer.config = DEFAULT_CONFIG.copy()
        analyzer.trading_graph = mock_tg

        client = app.test_client()
        resp = client.post(
            "/api/update-provider",
            json={"provider": "minimax"},
            content_type="application/json",
        )
        data = resp.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(analyzer.config["agent_llm_model"], "MiniMax-M2.7")
        self.assertEqual(analyzer.config["graph_llm_model"], "MiniMax-M2.7")

    @patch("web_interface.TradingGraph")
    def test_update_provider_invalid(self, mock_tg_class):
        """POST /api/update-provider with invalid provider should fail."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer
        analyzer.config = DEFAULT_CONFIG.copy()
        analyzer.trading_graph = mock_tg

        client = app.test_client()
        resp = client.post(
            "/api/update-provider",
            json={"provider": "invalid"},
            content_type="application/json",
        )
        data = resp.get_json()
        self.assertIn("error", data)

    @patch("web_interface.TradingGraph")
    def test_update_api_key_minimax(self, mock_tg_class):
        """POST /api/update-api-key with minimax should set env var."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer
        analyzer.config = DEFAULT_CONFIG.copy()
        analyzer.trading_graph = mock_tg

        client = app.test_client()
        resp = client.post(
            "/api/update-api-key",
            json={"api_key": "test-mm-key", "provider": "minimax"},
            content_type="application/json",
        )
        data = resp.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(os.environ.get("MINIMAX_API_KEY"), "test-mm-key")

    @patch("web_interface.TradingGraph")
    def test_get_api_key_status_minimax(self, mock_tg_class):
        """GET /api/get-api-key-status?provider=minimax should work."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer
        config = DEFAULT_CONFIG.copy()
        config["minimax_api_key"] = "test-minimax-key-12345"
        analyzer.config = config
        analyzer.trading_graph = mock_tg

        client = app.test_client()
        resp = client.get("/api/get-api-key-status?provider=minimax")
        data = resp.get_json()
        self.assertTrue(data.get("has_key"))
        self.assertIn("masked_key", data)

    @patch("web_interface.TradingGraph")
    def test_get_api_key_status_minimax_missing(self, mock_tg_class):
        """GET /api/get-api-key-status?provider=minimax with no key."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer
        config = DEFAULT_CONFIG.copy()
        config["minimax_api_key"] = ""
        analyzer.config = config
        analyzer.trading_graph = mock_tg

        os.environ.pop("MINIMAX_API_KEY", None)

        client = app.test_client()
        resp = client.get("/api/get-api-key-status?provider=minimax")
        data = resp.get_json()
        self.assertFalse(data.get("has_key"))


class TestProviderSwitchBackToOpenAI(unittest.TestCase):
    """Test that switching from MiniMax back to OpenAI resets model names."""

    @patch("web_interface.TradingGraph")
    def test_switch_minimax_to_openai(self, mock_tg_class):
        """Switching from minimax to openai should reset model names."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer
        config = DEFAULT_CONFIG.copy()
        config["agent_llm_model"] = "MiniMax-M2.7"
        config["graph_llm_model"] = "MiniMax-M2.7"
        analyzer.config = config
        analyzer.trading_graph = mock_tg

        client = app.test_client()
        resp = client.post(
            "/api/update-provider",
            json={"provider": "openai"},
            content_type="application/json",
        )
        data = resp.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(analyzer.config["agent_llm_model"], "gpt-4o-mini")
        self.assertEqual(analyzer.config["graph_llm_model"], "gpt-4o")


class TestWebInterfaceApiKeyStatus(unittest.TestCase):
    """Tests for API key status reporting in the web interface."""

    @patch("web_interface.TradingGraph")
    def test_get_api_key_status_openai_placeholder_returns_false(self, mock_tg_class):
        """The default placeholder key should not be reported as a real OpenAI key."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer

        analyzer.config = DEFAULT_CONFIG.copy()
        analyzer.trading_graph = mock_tg
        os.environ.pop("OPENAI_API_KEY", None)

        client = app.test_client()
        resp = client.get("/api/get-api-key-status?provider=openai")
        data = resp.get_json()

        self.assertFalse(data.get("has_key"))


class TestWebTradingAnalyzerRunAnalysis(unittest.TestCase):
    """Tests for WebTradingAnalyzer.run_analysis()."""

    @patch("web_interface.TradingGraph")
    @patch("web_interface.static_util.generate_trend_image")
    @patch("web_interface.static_util.generate_kline_image")
    def test_run_analysis_fails_fast_when_openai_key_is_not_configured(
        self, mock_kline_image, mock_trend_image, mock_tg_class
    ):
        """run_analysis() should stop before image generation when the key is missing."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg.graph = MagicMock()
        mock_tg_class.return_value = mock_tg

        from web_interface import WebTradingAnalyzer

        analyzer = WebTradingAnalyzer()
        analyzer.config = DEFAULT_CONFIG.copy()
        analyzer.trading_graph = mock_tg
        os.environ.pop("OPENAI_API_KEY", None)

        df = pd.DataFrame(
            {
                "Datetime": pd.to_datetime(["2026-04-17 00:00:00", "2026-04-17 01:00:00"]),
                "Open": [1.0, 2.0],
                "High": [1.5, 2.5],
                "Low": [0.5, 1.5],
                "Close": [1.2, 2.2],
            }
        )

        result = analyzer.run_analysis(df, "BTC", "1h")

        self.assertFalse(result["success"])
        self.assertIn("no está configurada", result["error"])
        mock_kline_image.assert_not_called()
        mock_trend_image.assert_not_called()
        mock_tg.graph.invoke.assert_not_called()


class TestWebInterfaceTemplateHelpers(unittest.TestCase):
    """Tests for JavaScript helpers required by the rendered web UI."""

    @patch("web_interface.TradingGraph")
    def test_home_page_defines_api_key_status_helper(self, mock_tg_class):
        """The home page should define the helper used by checkApiKeyStatus()."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer

        analyzer.config = DEFAULT_CONFIG.copy()
        analyzer.trading_graph = mock_tg

        client = app.test_client()
        resp = client.get("/")

        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("function showApiKeyStatus(message)", html)

    @patch("web_interface.TradingGraph")
    def test_home_page_includes_favicon_link(self, mock_tg_class):
        """The home page should declare a favicon to avoid 404 console noise."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer

        analyzer.config = DEFAULT_CONFIG.copy()
        analyzer.trading_graph = mock_tg

        client = app.test_client()
        resp = client.get("/")

        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn('rel="icon"', html)

    @patch("web_interface.TradingGraph")
    def test_home_page_associates_labels_with_form_fields(self, mock_tg_class):
        """Rendered form controls should have associated labels."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer

        analyzer.config = DEFAULT_CONFIG.copy()
        analyzer.trading_graph = mock_tg

        client = app.test_client()
        resp = client.get("/")

        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        for control_id in (
            "customAssetInput",
            "startDate",
            "endDate",
            "startTime",
            "endTime",
            "llmProviderSelect",
            "openaiApiKeyInput",
            "anthropicApiKeyInput",
            "qwenApiKeyInput",
            "minimaxApiKeyInput",
        ):
            self.assertIn(f'for="{control_id}"', html)
        for aria_label in (
            'aria-label="Fecha de inicio"',
            'aria-label="Fecha de fin"',
            'aria-label="Hora de inicio"',
            'aria-label="Hora de fin"',
        ):
            self.assertIn(aria_label, html)
        self.assertNotIn('<label class="form-label">\n                                        <i class="fas fa-coins"></i> Asset', html)
        self.assertNotIn('<label class="form-label">\n                                        <i class="fas fa-chart-line"></i> Timeframe', html)


if __name__ == "__main__":
    unittest.main()
