"""Tests for Spanish localization in backend-facing web messages."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

for mod_name in ["talib", "langchain_qwq"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from default_config import DEFAULT_CONFIG


class TestWebBackendLocalization(unittest.TestCase):
    """Tests for user-facing backend messages exposed by the Flask app."""

    @patch("web_interface.TradingGraph")
    def test_analyze_route_rejects_non_live_data_source_in_spanish(self, mock_tg_class):
        """POST /api/analyze should reject unsupported data sources in Spanish."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer

        analyzer.config = DEFAULT_CONFIG.copy()
        analyzer.trading_graph = mock_tg

        client = app.test_client()
        resp = client.post(
            "/api/analyze",
            json={"data_source": "csv", "asset": "BTC", "timeframe": "1h"},
            content_type="application/json",
        )
        data = resp.get_json()

        self.assertEqual(data["error"], "Solo se admiten datos en vivo de Yahoo Finance.")

    @patch("web_interface.TradingGraph")
    def test_validate_date_range_returns_spanish_errors(self, mock_tg_class):
        """validate_date_range() should report invalid ranges in Spanish."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import WebTradingAnalyzer

        analyzer = WebTradingAnalyzer()
        analyzer.config = DEFAULT_CONFIG.copy()
        analyzer.trading_graph = mock_tg

        result = analyzer.validate_date_range("2026-04-18", "2026-04-17", "1h")

        self.assertFalse(result["valid"])
        self.assertIn("La fecha/hora de inicio debe ser anterior", result["error"])

    @patch("web_interface.TradingGraph")
    def test_validate_api_key_reports_missing_key_in_spanish(self, mock_tg_class):
        """validate_api_key() should explain missing provider keys in Spanish."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import WebTradingAnalyzer

        analyzer = WebTradingAnalyzer()
        analyzer.config = DEFAULT_CONFIG.copy()
        analyzer.trading_graph = mock_tg

        with patch.dict(os.environ, {}, clear=True):
            result = analyzer.validate_api_key(provider="openai")

        self.assertFalse(result["valid"])
        self.assertIn("no está configurada", result["error"])

    @patch("web_interface.TradingGraph")
    def test_update_provider_invalid_message_is_in_spanish(self, mock_tg_class):
        """POST /api/update-provider should reject unsupported providers in Spanish."""
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

        self.assertEqual(
            data["error"],
            "El proveedor debe ser 'openai', 'anthropic', 'qwen' o 'minimax'",
        )

    @patch("web_interface.TradingGraph")
    def test_update_api_key_requires_key_in_spanish(self, mock_tg_class):
        """POST /api/update-api-key should require an API key in Spanish."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer

        analyzer.config = DEFAULT_CONFIG.copy()
        analyzer.trading_graph = mock_tg

        client = app.test_client()
        resp = client.post(
            "/api/update-api-key",
            json={"provider": "openai"},
            content_type="application/json",
        )
        data = resp.get_json()

        self.assertEqual(data["error"], "La clave API es obligatoria")


if __name__ == "__main__":
    unittest.main()
