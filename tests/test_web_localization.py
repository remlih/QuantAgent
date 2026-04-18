"""Tests for Spanish localization in the Flask web interface."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock heavy native/incompatible dependencies before importing project modules.
for mod_name in ["talib", "langchain_qwq"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from default_config import DEFAULT_CONFIG


class TestDemoPageLocalization(unittest.TestCase):
    """Tests for Spanish copy on the main analysis page."""

    @patch("web_interface.TradingGraph")
    def test_demo_page_renders_key_analysis_flow_copy_in_spanish(self, mock_tg_class):
        """The demo page should present the core analysis flow in Spanish."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer

        analyzer.config = DEFAULT_CONFIG.copy()
        analyzer.trading_graph = mock_tg

        client = app.test_client()
        resp = client.get("/demo")

        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        self.assertIn("Iniciar análisis", html)
        self.assertIn("Configuración del análisis", html)
        self.assertIn("Selección de datos", html)
        self.assertIn("Activo", html)
        self.assertIn("Intervalo", html)
        self.assertIn("Configuración de fecha y hora", html)
        self.assertIn("Usar la fecha y hora actuales como fin", html)
        self.assertIn("Configuración", html)
        self.assertIn("Clave API de OpenAI", html)

        self.assertNotIn("Start Analysis", html)
        self.assertNotIn("Use current date & time for end", html)

    @patch("web_interface.TradingGraph")
    def test_demo_page_javascript_messages_are_in_spanish(self, mock_tg_class):
        """Front-end alerts and button states should be localized to Spanish."""
        mock_tg = MagicMock()
        mock_tg.config = DEFAULT_CONFIG.copy()
        mock_tg_class.return_value = mock_tg

        from web_interface import app, analyzer

        analyzer.config = DEFAULT_CONFIG.copy()
        analyzer.trading_graph = mock_tg

        client = app.test_client()
        resp = client.get("/demo")

        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        self.assertIn("Por favor, ingresa un símbolo de activo personalizado.", html)
        self.assertIn("Por favor, selecciona tanto un activo como un intervalo.", html)
        self.assertIn("Por favor, selecciona las fechas de inicio y fin.", html)
        self.assertIn("Analizando...", html)
        self.assertIn("El análisis falló:", html)
        self.assertIn("¡Análisis completado con éxito!", html)
        self.assertIn("Ocurrió un error durante el análisis. Inténtalo de nuevo.", html)

        self.assertNotIn("Please enter a custom asset symbol.", html)
        self.assertNotIn("Please select both asset and timeframe.", html)
        self.assertNotIn("Please select both start and end dates.", html)
        self.assertNotIn("Analyzing...", html)


if __name__ == "__main__":
    unittest.main()
