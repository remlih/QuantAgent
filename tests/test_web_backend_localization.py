"""Tests for Spanish localization in backend-facing web messages."""

import pandas as pd
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
    def test_health_route_returns_ok(self, mock_tg_class):
        """GET /health should return a simple ok payload."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app

        client = app.test_client()
        resp = client.get("/health")
        data = resp.get_json()

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(data, {"status": "ok"})

    @patch("web_interface.TradingGraph")
    def test_index_route_sets_basic_security_headers(self, mock_tg_class):
        """GET / should include baseline security headers for the web UI."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app

        client = app.test_client()
        resp = client.get("/")

        self.assertEqual(resp.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(resp.headers["X-Frame-Options"], "DENY")
        self.assertEqual(resp.headers["Referrer-Policy"], "no-referrer")
        self.assertIn("default-src 'self'", resp.headers["Content-Security-Policy"])

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
            "El proveedor debe ser 'openai', 'anthropic', 'qwen', 'minimax' o 'copilot'",
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

    @patch("web_interface.TradingGraph")
    def test_analyze_route_hides_unexpected_exception_details(self, mock_tg_class):
        """POST /api/analyze should not expose raw internal exception text."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer

        analyzer.config = DEFAULT_CONFIG.copy()
        analyzer.trading_graph = mock_tg
        analyzer.fetch_yfinance_data_with_datetime = MagicMock(
            return_value=pd.DataFrame(
                [
                    {
                        "Datetime": "2026-04-17 00:00:00",
                        "Open": 1.0,
                        "High": 2.0,
                        "Low": 0.5,
                        "Close": 1.5,
                    }
                ]
            )
        )
        analyzer.run_analysis = MagicMock(side_effect=RuntimeError("internal boom"))

        client = app.test_client()
        resp = client.post(
            "/api/analyze",
            json={
                "data_source": "live",
                "asset": "BTC",
                "timeframe": "1h",
                "start_date": "2026-04-16",
                "end_date": "2026-04-17",
            },
            content_type="application/json",
        )
        data = resp.get_json()

        self.assertEqual(
            data["error"],
            "Ocurrió un error inesperado al procesar el análisis. Inténtalo de nuevo.",
        )
        self.assertNotIn("internal boom", data["error"])

    @patch("web_interface.TradingGraph")
    def test_update_provider_hides_internal_exception_details(self, mock_tg_class):
        """POST /api/update-provider should return a localized generic error on failure."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer

        analyzer.config = DEFAULT_CONFIG.copy()
        analyzer.trading_graph = mock_tg
        analyzer.trading_graph.refresh_llms.side_effect = RuntimeError("provider boom")

        client = app.test_client()
        resp = client.post(
            "/api/update-provider",
            json={"provider": "openai"},
            content_type="application/json",
        )
        data = resp.get_json()

        self.assertEqual(
            data["error"],
            "No se pudo actualizar el proveedor en este momento. Inténtalo de nuevo.",
        )
        self.assertNotIn("provider boom", data["error"])

    @patch("web_interface.TradingGraph")
    def test_update_api_key_hides_internal_exception_details(self, mock_tg_class):
        """POST /api/update-api-key should return a localized generic error on failure."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer

        analyzer.config = DEFAULT_CONFIG.copy()
        analyzer.trading_graph = mock_tg
        analyzer.trading_graph.update_api_key.side_effect = RuntimeError("key boom")

        client = app.test_client()
        resp = client.post(
            "/api/update-api-key",
            json={"provider": "openai", "api_key": "sk-test-1234"},
            content_type="application/json",
        )
        data = resp.get_json()

        self.assertEqual(
            data["error"],
            "No se pudo actualizar la clave API en este momento. Inténtalo de nuevo.",
        )
        self.assertNotIn("key boom", data["error"])

    @patch("web_interface.TradingGraph")
    def test_validate_api_key_hides_unexpected_provider_exception_details(self, mock_tg_class):
        """validate_api_key() should avoid echoing unexpected provider errors."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import WebTradingAnalyzer

        analyzer = WebTradingAnalyzer()
        analyzer.config = DEFAULT_CONFIG.copy()
        analyzer.trading_graph = mock_tg

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-1234"}, clear=True):
            with patch("openai.OpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_client.chat.completions.create.side_effect = RuntimeError(
                    "socket boom"
                )
                mock_openai.return_value = mock_client

                result = analyzer.validate_api_key(provider="openai")

        self.assertFalse(result["valid"])
        self.assertEqual(
            result["error"],
            "❌ Error de clave API: no fue posible validar la clave API en este momento. Inténtalo de nuevo.",
        )
        self.assertNotIn("socket boom", result["error"])


if __name__ == "__main__":
    unittest.main()
