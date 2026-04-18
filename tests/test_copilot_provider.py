"""Unit tests for GitHub Copilot provider integration in QuantAgent."""

import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import HumanMessage, SystemMessage

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock heavy native/incompatible dependencies before importing project modules
for mod_name in [
    "talib",
    "langchain_qwq",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from default_config import DEFAULT_CONFIG


class TestDefaultConfig(unittest.TestCase):
    """Tests for Copilot fields in DEFAULT_CONFIG."""

    def test_copilot_github_token_field_exists(self):
        """DEFAULT_CONFIG should contain a copilot_github_token field."""
        self.assertIn("copilot_github_token", DEFAULT_CONFIG)

    def test_provider_fields_accept_copilot(self):
        """Provider fields should accept 'copilot' as a valid value."""
        config = DEFAULT_CONFIG.copy()
        config["agent_llm_provider"] = "copilot"
        config["graph_llm_provider"] = "copilot"
        self.assertEqual(config["agent_llm_provider"], "copilot")
        self.assertEqual(config["graph_llm_provider"], "copilot")


class TestTradingGraphGetApiKey(unittest.TestCase):
    """Tests for TradingGraph._get_api_key() with copilot provider."""

    def _make_graph(self, config):
        from trading_graph import TradingGraph

        orig_create = TradingGraph._create_llm
        TradingGraph._create_llm = MagicMock(return_value=MagicMock())
        tg = TradingGraph(config=config)
        TradingGraph._create_llm = orig_create
        return tg

    def test_get_copilot_token_from_config(self):
        """Should return copilot_github_token from config."""
        config = DEFAULT_CONFIG.copy()
        config["copilot_github_token"] = "github_pat_config"
        tg = self._make_graph(config)
        key = tg._get_api_key("copilot")
        self.assertEqual(key, "github_pat_config")

    def test_get_copilot_token_from_env_priority_order(self):
        """Should prefer COPILOT_GITHUB_TOKEN, then GH_TOKEN, then GITHUB_TOKEN."""
        config = DEFAULT_CONFIG.copy()
        config["copilot_github_token"] = ""
        tg = self._make_graph(config)

        with patch.dict(
            os.environ,
            {
                "COPILOT_GITHUB_TOKEN": "copilot-token",
                "GH_TOKEN": "gh-token",
                "GITHUB_TOKEN": "github-token",
            },
            clear=True,
        ):
            self.assertEqual(tg._get_api_key("copilot"), "copilot-token")

        with patch.dict(
            os.environ,
            {
                "GH_TOKEN": "gh-token",
                "GITHUB_TOKEN": "github-token",
            },
            clear=True,
        ):
            self.assertEqual(tg._get_api_key("copilot"), "gh-token")

        with patch.dict(
            os.environ,
            {
                "GITHUB_TOKEN": "github-token",
            },
            clear=True,
        ):
            self.assertEqual(tg._get_api_key("copilot"), "github-token")

    def test_get_copilot_token_missing_returns_empty_for_logged_in_user_auth(self):
        """Should allow Copilot CLI logged-in user auth when no explicit token is set."""
        config = DEFAULT_CONFIG.copy()
        config["copilot_github_token"] = ""
        tg = self._make_graph(config)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(tg._get_api_key("copilot"), "")


class TestTradingGraphCreateLlm(unittest.TestCase):
    """Tests for TradingGraph._create_llm() with copilot provider."""

    def _make_graph(self, config):
        from trading_graph import TradingGraph

        orig_create = TradingGraph._create_llm
        TradingGraph._create_llm = MagicMock(return_value=MagicMock())
        tg = TradingGraph(config=config)
        TradingGraph._create_llm = orig_create
        return tg

    @patch("trading_graph.CopilotChatModel")
    def test_create_llm_copilot_uses_copilot_chat_model(self, mock_copilot_model):
        """Copilot provider should create CopilotChatModel with the selected model."""
        config = DEFAULT_CONFIG.copy()
        config["copilot_github_token"] = "github_pat_test"
        tg = self._make_graph(config)
        tg.config = config

        mock_copilot_model.return_value = MagicMock()
        tg._create_llm("copilot", "gpt-5.4", 0.2)

        mock_copilot_model.assert_called_once_with(
            model="gpt-5.4",
            temperature=0.2,
            github_token="github_pat_test",
        )


class TestTradingGraphUpdateApiKey(unittest.TestCase):
    """Tests for TradingGraph.update_api_key() with copilot provider."""

    def _make_graph(self, config):
        from trading_graph import TradingGraph

        orig_create = TradingGraph._create_llm
        TradingGraph._create_llm = MagicMock(return_value=MagicMock())
        tg = TradingGraph(config=config)
        TradingGraph._create_llm = orig_create
        return tg

    def test_update_api_key_copilot(self):
        """update_api_key('copilot') should update config and env var."""
        config = DEFAULT_CONFIG.copy()
        config["copilot_github_token"] = ""
        config["agent_llm_provider"] = "copilot"
        config["graph_llm_provider"] = "copilot"
        config["agent_llm_model"] = "gpt-5.4"
        config["graph_llm_model"] = "claude-opus-4.6"
        tg = self._make_graph(config)

        with patch.object(tg, "refresh_llms"):
            tg.update_api_key("github_pat_new", provider="copilot")

        self.assertEqual(tg.config["copilot_github_token"], "github_pat_new")
        self.assertEqual(os.environ.get("COPILOT_GITHUB_TOKEN"), "github_pat_new")


class TestTradingGraphRefreshLlms(unittest.TestCase):
    """Tests for TradingGraph.refresh_llms() with copilot provider."""

    @patch("trading_graph.CopilotChatModel")
    @patch("trading_graph.ChatOpenAI")
    @patch("trading_graph.ChatAnthropic")
    @patch("trading_graph.ChatQwen")
    def test_refresh_llms_copilot(
        self, mock_qwen, mock_anthropic, mock_openai, mock_copilot_model
    ):
        """refresh_llms() should recreate Copilot LLMs when provider is copilot."""
        from trading_graph import TradingGraph

        config = DEFAULT_CONFIG.copy()
        config["agent_llm_provider"] = "copilot"
        config["graph_llm_provider"] = "copilot"
        config["agent_llm_model"] = "gpt-5.4"
        config["graph_llm_model"] = "claude-opus-4.6"
        config["copilot_github_token"] = "github_pat_test"

        mock_copilot_model.return_value = MagicMock()
        tg = TradingGraph(config=config)

        mock_copilot_model.reset_mock()
        tg.refresh_llms()

        self.assertEqual(mock_copilot_model.call_count, 2)
        self.assertEqual(
            [call.kwargs["model"] for call in mock_copilot_model.call_args_list],
            ["gpt-5.4", "claude-opus-4.6"],
        )


class TestCopilotChatModel(unittest.TestCase):
    """Tests for CopilotChatModel request conversion helpers."""

    def test_prepare_sdk_request_converts_data_url_image_to_blob_attachment(self):
        """Should convert LangChain image_url data URLs to Copilot SDK blob attachments."""
        from copilot_provider import CopilotChatModel

        model = CopilotChatModel(model="claude-opus-4.6", github_token="github_pat_test")
        prompt, system_message, attachments = model._prepare_sdk_request(
            [
                SystemMessage(content="Responde en espanol."),
                HumanMessage(
                    content=[
                        {"type": "text", "text": "Analiza este grafico"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,ZmFrZS1pbWFnZS1ieXRlcw=="
                            },
                        },
                    ]
                ),
            ]
        )

        self.assertEqual(system_message, "Responde en espanol.")
        self.assertEqual(prompt, "Analiza este grafico")
        self.assertEqual(
            attachments,
            [
                {
                    "type": "blob",
                    "data": "ZmFrZS1pbWFnZS1ieXRlcw==",
                    "mimeType": "image/png",
                }
            ],
        )


class TestCopilotSdkHelpers(unittest.TestCase):
    """Tests for Copilot SDK helper functions."""

    @patch("copilot_provider.CopilotClient")
    def test_list_available_copilot_models_returns_model_ids(self, mock_client_class):
        """Should return the model IDs exposed by the Copilot SDK."""
        from copilot_provider import list_available_copilot_models

        mock_client = MagicMock()
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock()
        mock_client.list_models = AsyncMock(
            return_value=[
                type("ModelInfo", (), {"id": "gpt-5.4"})(),
                type("ModelInfo", (), {"id": "claude-opus-4.6"})(),
            ]
        )
        mock_client_class.return_value = mock_client

        models = list_available_copilot_models(github_token="github_pat_test")
        self.assertEqual(models, ["gpt-5.4", "claude-opus-4.6"])


class TestWebInterfaceCopilotRoutes(unittest.TestCase):
    """Tests for web interface provider update with Copilot."""

    @patch("web_interface.TradingGraph")
    def test_update_provider_copilot(self, mock_tg_class):
        """POST /api/update-provider with copilot should succeed."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer

        analyzer.config = DEFAULT_CONFIG.copy()
        analyzer.trading_graph = mock_tg

        client = app.test_client()
        resp = client.post(
            "/api/update-provider",
            json={"provider": "copilot"},
            content_type="application/json",
        )
        data = resp.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(analyzer.config["agent_llm_model"], "gpt-5.4")
        self.assertEqual(analyzer.config["graph_llm_model"], "claude-opus-4.6")

    @patch("web_interface.validate_copilot_auth")
    @patch("web_interface.TradingGraph")
    def test_validate_api_key_copilot_uses_sdk_auth_check(
        self, mock_tg_class, mock_validate_copilot_auth
    ):
        """POST /api/validate-api-key with copilot should use SDK auth validation."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg
        mock_validate_copilot_auth.return_value = (True, ["gpt-5.4", "claude-opus-4.6"])

        from web_interface import app, analyzer

        analyzer.config = DEFAULT_CONFIG.copy()
        analyzer.config["agent_llm_provider"] = "copilot"
        analyzer.trading_graph = mock_tg

        client = app.test_client()
        resp = client.post(
            "/api/validate-api-key",
            json={"provider": "copilot"},
            content_type="application/json",
        )
        data = resp.get_json()
        self.assertTrue(data.get("valid"))
        self.assertIn("GitHub Copilot", data.get("message", ""))

