"""Tests for Spanish localization on the results page."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

for mod_name in ["talib", "langchain_qwq"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from default_config import DEFAULT_CONFIG


class TestOutputPageLocalization(unittest.TestCase):
    """Tests for Spanish copy on the default results page."""

    @patch("web_interface.TradingGraph")
    def test_output_page_declares_favicon(self, mock_tg_class):
        """The results page should declare a favicon to avoid 404 console noise."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer

        analyzer.config = DEFAULT_CONFIG.copy()
        analyzer.trading_graph = mock_tg

        client = app.test_client()
        resp = client.get("/output")

        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn('rel="icon"', html)
        self.assertIn('<html lang="es">', html)

    @patch("web_interface.TradingGraph")
    def test_output_page_renders_core_sections_in_spanish(self, mock_tg_class):
        """GET /output should render the main results labels in Spanish."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer

        analyzer.config = DEFAULT_CONFIG.copy()
        analyzer.trading_graph = mock_tg

        client = app.test_client()
        resp = client.get("/output")

        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        self.assertIn("Volver a la demo", html)
        self.assertIn("Resumen del análisis", html)
        self.assertIn("Decisión final de trading", html)
        self.assertIn("Horizonte de pronóstico", html)
        self.assertIn("Relación riesgo/beneficio", html)
        self.assertIn("Agente de indicadores", html)
        self.assertIn("Agente de patrones", html)
        self.assertIn("Agente de tendencia", html)

        self.assertNotIn("Back to Demo", html)

    @patch("web_interface.TradingGraph")
    def test_output_page_default_content_is_localized_to_spanish(self, mock_tg_class):
        """The fallback decision and helper content should also be localized."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer

        analyzer.config = DEFAULT_CONFIG.copy()
        analyzer.trading_graph = mock_tg

        client = app.test_client()
        resp = client.get("/output")

        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        self.assertIn("LARGO", html)
        self.assertIn("Patrón identificado", html)
        self.assertIn("Confiabilidad del patrón", html)
        self.assertIn("Indicadores de fuerza de la tendencia", html)
        self.assertIn("Líneas de soporte y resistencia", html)
        self.assertIn("LARGA en BTC", html)
        self.assertIn("24-48 horas", html)

        self.assertNotIn("Back to Demo", html)

    @patch("web_interface.TradingGraph")
    def test_output_page_shows_placeholders_when_no_chart_images_are_available(self, mock_tg_class):
        """Default output should render placeholders instead of broken image URLs."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer

        analyzer.config = DEFAULT_CONFIG.copy()
        analyzer.trading_graph = mock_tg

        client = app.test_client()
        resp = client.get("/output")

        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        self.assertIn("Visualización del patrón no disponible.", html)
        self.assertIn("Visualización de tendencia no disponible.", html)
        self.assertNotIn('src="/api/images/pattern"', html)
        self.assertNotIn('src="/api/images/trend"', html)
        self.assertIn('data-analysis-image-container="pattern"', html)
        self.assertIn('data-analysis-image-container="trend"', html)
        self.assertIn("restoreChartFromSessionStorage(", html)
        self.assertIn('data-analysis-chart="${chartType}"', html)


if __name__ == "__main__":
    unittest.main()
