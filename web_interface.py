import json
import os
import re
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, render_template, request, send_file
from openai import OpenAI

import static_util
from trading_graph import TradingGraph, resolve_api_key

app = Flask(__name__)

GENERIC_ANALYSIS_ERROR = (
    "❌ Error de análisis: no se pudo completar el análisis en este momento. "
    "Inténtalo de nuevo."
)
GENERIC_ANALYSIS_ROUTE_ERROR = (
    "Ocurrió un error inesperado al procesar el análisis. Inténtalo de nuevo."
)
GENERIC_API_KEY_ERROR = (
    "❌ Error de clave API: no fue posible validar la clave API en este momento. "
    "Inténtalo de nuevo."
)
GENERIC_API_KEY_ROUTE_ERROR = (
    "No se pudo validar la clave API en este momento. Inténtalo de nuevo."
)
GENERIC_PROVIDER_UPDATE_ERROR = (
    "No se pudo actualizar el proveedor en este momento. Inténtalo de nuevo."
)
GENERIC_API_KEY_UPDATE_ERROR = (
    "No se pudo actualizar la clave API en este momento. Inténtalo de nuevo."
)


class WebTradingAnalyzer:
    def __init__(self):
        """Initialize the web trading analyzer."""
        from default_config import DEFAULT_CONFIG
        # Start with default config (OpenAI)
        self.config = DEFAULT_CONFIG.copy()
        self.trading_graph = TradingGraph(config=self.config)
        self.data_dir = Path("data")

        # Ensure data dir exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Available assets and their display names
        self.asset_mapping = {
            "SPX": "S&P 500",
            "BTC": "Bitcoin",
            "GC": "Gold Futures",
            "NQ": "Nasdaq Futures",
            "CL": "Crude Oil",
            "ES": "E-mini S&P 500",
            "DJI": "Dow Jones",
            "QQQ": "Invesco QQQ Trust",
            "VIX": "Volatility Index",
            "DXY": "US Dollar Index",
            "AAPL": "Apple Inc.",  # New asset
            "TSLA": "Tesla Inc.",  # New asset
        }

        # Yahoo Finance symbol mapping
        self.yfinance_symbols = {
            "SPX": "^GSPC",  # S&P 500
            "BTC": "BTC-USD",  # Bitcoin
            "GC": "GC=F",  # Gold Futures
            "NQ": "NQ=F",  # Nasdaq Futures
            "CL": "CL=F",  # Crude Oil
            "ES": "ES=F",  # E-mini S&P 500
            "DJI": "^DJI",  # Dow Jones
            "QQQ": "QQQ",  # Invesco QQQ Trust
            "VIX": "^VIX",  # Volatility Index
            "DXY": "DX-Y.NYB",  # US Dollar Index
        }

        # Yahoo Finance interval mapping
        self.yfinance_intervals = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",  # yfinance supports 4h natively!
            "1d": "1d",
            "1w": "1wk",
            "1mo": "1mo",
        }

        # Load persisted custom assets
        self.custom_assets_file = self.data_dir / "custom_assets.json"
        self.custom_assets = self.load_custom_assets()

    def fetch_yfinance_data(
        self, symbol: str, interval: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance."""
        try:
            yf_symbol = self.yfinance_symbols.get(symbol, symbol)
            yf_interval = self.yfinance_intervals.get(interval, interval)

            df = yf.download(
                tickers=yf_symbol, start=start_date, end=end_date, interval=yf_interval
            )

            if df is None or df.empty:
                return pd.DataFrame()

            # Ensure df is a DataFrame, not a Series
            if isinstance(df, pd.Series):
                df = df.to_frame()

            # Reset index to ensure we have a clean DataFrame
            df = df.reset_index()

            # Ensure we have a DataFrame
            if not isinstance(df, pd.DataFrame):
                return pd.DataFrame()

            # Handle potential MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Rename columns if needed
            column_mapping = {
                "Date": "Datetime",
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume",
            }

            # Only rename columns that exist
            existing_columns = {
                old: new for old, new in column_mapping.items() if old in df.columns
            }
            df = df.rename(columns=existing_columns)

            # Ensure we have the required columns
            required_columns = ["Datetime", "Open", "High", "Low", "Close"]
            if not all(col in df.columns for col in required_columns):
                print(f"Warning: Missing columns. Available: {list(df.columns)}")
                return pd.DataFrame()

            # Select only the required columns
            df = df[required_columns]
            df["Datetime"] = pd.to_datetime(df["Datetime"])

            return df

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_yfinance_data_with_datetime(
        self,
        symbol: str,
        interval: str,
        start_datetime: datetime,
        end_datetime: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance using datetime objects for exact time precision."""
        try:
            yf_symbol = self.yfinance_symbols.get(symbol, symbol)
            yf_interval = self.yfinance_intervals.get(interval, interval)

            print(
                f"Fetching {yf_symbol} from {start_datetime} to {end_datetime} with interval {yf_interval}"
            )

            # Use datetime objects directly for yfinance
            df = yf.download(
                tickers=yf_symbol,
                start=start_datetime,
                end=end_datetime,
                interval=yf_interval,
                auto_adjust=True,
                prepost=False,
            )

            if df is None or df.empty:
                print(f"No data returned for {symbol}")
                return pd.DataFrame()

            # Ensure df is a DataFrame, not a Series
            if isinstance(df, pd.Series):
                df = df.to_frame()

            # Reset index to ensure we have a clean DataFrame
            df = df.reset_index()

            # Ensure we have a DataFrame
            if not isinstance(df, pd.DataFrame):
                return pd.DataFrame()

            # Handle potential MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Rename columns if needed
            column_mapping = {
                "Date": "Datetime",
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume",
            }

            # Only rename columns that exist
            existing_columns = {
                old: new for old, new in column_mapping.items() if old in df.columns
            }
            df = df.rename(columns=existing_columns)

            # Ensure we have the required columns
            required_columns = ["Datetime", "Open", "High", "Low", "Close"]
            if not all(col in df.columns for col in required_columns):
                print(f"Warning: Missing columns. Available: {list(df.columns)}")
                return pd.DataFrame()

            # Select only the required columns
            df = df[required_columns]
            df["Datetime"] = pd.to_datetime(df["Datetime"])

            print(f"Successfully fetched {len(df)} data points for {symbol}")
            print(f"Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")

            return df

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def get_available_assets(self) -> list:
        """Get list of available assets from the asset mapping dictionary."""
        return sorted(list(self.asset_mapping.keys()))

    def get_available_files(self, asset: str, timeframe: str) -> list:
        """Get available data files for a specific asset and timeframe."""
        asset_dir = self.data_dir / asset.lower()
        if not asset_dir.exists():
            return []

        pattern = f"{asset}_{timeframe}_*.csv"
        files = list(asset_dir.glob(pattern))
        return sorted(files)

    def run_analysis(
        self, df: pd.DataFrame, asset_name: str, timeframe: str
    ) -> Dict[str, Any]:
        """Run the trading analysis on the provided DataFrame."""
        try:
            # Debug: Check DataFrame structure
            print(f"DataFrame columns: {df.columns}")
            print(f"DataFrame index: {type(df.index)}")
            print(f"DataFrame shape: {df.shape}")

            # Prepare data for analysis
            # if len(df) > 49:
            #     df_slice = df.tail(49).iloc[:-3]
            # else:
            #     df_slice = df.tail(45)

            df_slice = df.tail(45)

            # Ensure DataFrame has the expected structure
            required_columns = ["Datetime", "Open", "High", "Low", "Close"]
            if not all(col in df_slice.columns for col in required_columns):
                return {
                    "success": False,
                    "error": f"Faltan columnas obligatorias. Disponibles: {list(df_slice.columns)}",
                }

            # Reset index to avoid any MultiIndex issues
            df_slice = df_slice.reset_index(drop=True)

            # Debug: Check the slice before conversion
            print(f"Slice columns: {df_slice.columns}")
            print(f"Slice index: {type(df_slice.index)}")

            # Convert to dict for tool input - use explicit conversion to avoid tuple keys
            df_slice_dict = {}
            for col in required_columns:
                if col == "Datetime":
                    # Convert datetime objects to strings for JSON serialization
                    df_slice_dict[col] = (
                        df_slice[col].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
                    )
                else:
                    df_slice_dict[col] = df_slice[col].tolist()

            # Debug: Check the resulting dictionary
            print(f"Dictionary keys: {list(df_slice_dict.keys())}")
            print(f"Dictionary key types: {[type(k) for k in df_slice_dict.keys()]}")

            # Format timeframe for display
            display_timeframe = timeframe
            if timeframe.endswith("h"):
                display_timeframe += "our"
            elif timeframe.endswith("m"):
                display_timeframe += "in"
            elif timeframe.endswith("d"):
                display_timeframe += "ay"
            elif timeframe == "1w":
                display_timeframe = "1 week"
            elif timeframe == "1mo":
                display_timeframe = "1 month"

            provider_names = {
                "openai": "OpenAI",
                "anthropic": "Anthropic",
                "qwen": "Qwen",
                "minimax": "MiniMax",
            }
            providers_to_check = {
                self.config.get("agent_llm_provider", "openai"),
                self.config.get("graph_llm_provider", "openai"),
            }
            for provider in providers_to_check:
                if not resolve_api_key(self.config, provider):
                    provider_name = provider_names.get(provider, provider)
                    return {
                        "success": False,
                        "error": f"❌ Clave API inválida: la clave API de {provider_name} no está configurada. Actualízala en la sección Configuración.",
                    }

            p_image = static_util.generate_kline_image(df_slice_dict)
            t_image = static_util.generate_trend_image(df_slice_dict)

            # Create initial state
            initial_state = {
                "kline_data": df_slice_dict,
                "analysis_results": None,
                "messages": [],
                "time_frame": display_timeframe,
                "stock_name": asset_name,
                "pattern_image": p_image["pattern_image"],
                "trend_image": t_image["trend_image"],
            }

            # Run the trading graph
            final_state = self.trading_graph.graph.invoke(initial_state)

            return {
                "success": True,
                "final_state": final_state,
                "asset_name": asset_name,
                "timeframe": display_timeframe,
                "data_length": len(df_slice),
            }

        except Exception as e:
            error_msg = str(e)
            
            # Get current provider from config
            provider = self.config.get("agent_llm_provider", "openai")
            if provider == "openai":
                provider_name = "OpenAI"
            elif provider == "anthropic":
                provider_name = "Anthropic"
            elif provider == "minimax":
                provider_name = "MiniMax"
            else:
                provider_name = "Qwen"

            # Check for specific API key authentication errors
            if (
                "authentication" in error_msg.lower()
                or "invalid api key" in error_msg.lower()
                or "401" in error_msg
                or "invalid_api_key" in error_msg.lower()
            ):
                return {
                    "success": False,
                    "error": f"❌ Clave API inválida: la clave API de {provider_name} que proporcionaste es inválida o ha vencido. Revísala en la sección Configuración e inténtalo de nuevo.",
                }
            elif "rate limit" in error_msg.lower() or "429" in error_msg:
                return {
                    "success": False,
                    "error": f"⚠️ Límite de tasa excedido: alcanzaste el límite de la API de {provider_name}. Espera un momento e inténtalo de nuevo.",
                }
            elif "quota" in error_msg.lower() or "billing" in error_msg.lower():
                return {
                    "success": False,
                    "error": f"💳 Problema de facturación: tu cuenta de {provider_name} no tiene créditos suficientes o presenta problemas de facturación. Revisa tu cuenta de {provider_name}.",
                }
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                return {
                    "success": False,
                    "error": f"🌐 Error de red: no fue posible conectar con los servidores de {provider_name}. Verifica tu conexión a internet e inténtalo de nuevo.",
                }
            else:
                return {"success": False, "error": GENERIC_ANALYSIS_ERROR}

    def extract_analysis_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format analysis results for web display."""
        if not results["success"]:
            return {"error": results["error"]}

        final_state = results["final_state"]

        # Extract analysis results from state fields
        technical_indicators = final_state.get("indicator_report", "")
        pattern_analysis = final_state.get("pattern_report", "")
        trend_analysis = final_state.get("trend_report", "")
        final_decision_raw = final_state.get("final_trade_decision", "")

        # Extract chart data if available
        pattern_chart = final_state.get("pattern_image", "")
        trend_chart = final_state.get("trend_image", "")
        pattern_image_filename = final_state.get("pattern_image_filename", "")
        trend_image_filename = final_state.get("trend_image_filename", "")

        # Parse final decision
        final_decision = ""
        if final_decision_raw:
            try:
                # Try to extract JSON from the decision
                start = final_decision_raw.find("{")
                end = final_decision_raw.rfind("}") + 1
                if start != -1 and end != 0:
                    json_str = final_decision_raw[start:end]
                    decision_data = json.loads(json_str)
                    final_decision = {
                        "decision": decision_data.get("decision", "N/A"),
                        "risk_reward_ratio": decision_data.get(
                            "risk_reward_ratio", "N/A"
                        ),
                        "forecast_horizon": decision_data.get(
                            "forecast_horizon", "N/A"
                        ),
                        "justification": decision_data.get("justification", "N/A"),
                    }
                else:
                    # If no JSON found, return the raw text
                    final_decision = {"raw": final_decision_raw}
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw text
                final_decision = {"raw": final_decision_raw}

        return {
            "success": True,
            "asset_name": results["asset_name"],
            "timeframe": results["timeframe"],
            "data_length": results["data_length"],
            "technical_indicators": technical_indicators,
            "pattern_analysis": pattern_analysis,
            "trend_analysis": trend_analysis,
            "pattern_chart": pattern_chart,
            "trend_chart": trend_chart,
            "pattern_image_filename": pattern_image_filename,
            "trend_image_filename": trend_image_filename,
            "final_decision": final_decision,
        }

    def get_timeframe_date_limits(self, timeframe: str) -> Dict[str, Any]:
        """Get valid date range limits for a given timeframe."""
        limits = {
            "1m": {"max_days": 7, "description": "datos de 1 minuto: máximo 7 días"},
            "2m": {"max_days": 60, "description": "datos de 2 minutos: máximo 60 días"},
            "5m": {"max_days": 60, "description": "datos de 5 minutos: máximo 60 días"},
            "15m": {"max_days": 60, "description": "datos de 15 minutos: máximo 60 días"},
            "30m": {"max_days": 60, "description": "datos de 30 minutos: máximo 60 días"},
            "60m": {"max_days": 730, "description": "datos de 1 hora: máximo 730 días"},
            "90m": {"max_days": 60, "description": "datos de 90 minutos: máximo 60 días"},
            "1h": {"max_days": 730, "description": "datos de 1 hora: máximo 730 días"},
            "4h": {"max_days": 730, "description": "datos de 4 horas: máximo 730 días"},
            "1d": {"max_days": 730, "description": "datos de 1 día: máximo 730 días"},
            "5d": {"max_days": 60, "description": "datos de 5 días: máximo 60 días"},
            "1w": {"max_days": 730, "description": "datos de 1 semana: máximo 730 días"},
            "1wk": {"max_days": 730, "description": "datos de 1 semana: máximo 730 días"},
            "1mo": {"max_days": 730, "description": "datos de 1 mes: máximo 730 días"},
            "3mo": {"max_days": 730, "description": "datos de 3 meses: máximo 730 días"},
        }

        return limits.get(
            timeframe, {"max_days": 730, "description": "predeterminado: máximo 730 días"}
        )

    def validate_date_range(
        self,
        start_date: str,
        end_date: str,
        timeframe: str,
        start_time: str = "00:00",
        end_time: str = "23:59",
    ) -> Dict[str, Any]:
        """Validate date and time range for the given timeframe."""
        try:
            # Create datetime objects with time
            start_datetime_str = f"{start_date} {start_time}"
            end_datetime_str = f"{end_date} {end_time}"

            start = datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M")
            end = datetime.strptime(end_datetime_str, "%Y-%m-%d %H:%M")

            if start >= end:
                return {
                    "valid": False,
                    "error": "La fecha/hora de inicio debe ser anterior a la fecha/hora de fin",
                }

            # Get timeframe limits
            limits = self.get_timeframe_date_limits(timeframe)
            max_days = limits["max_days"]

            # Calculate time difference in days (including fractional days)
            time_diff = end - start
            days_diff = time_diff.total_seconds() / (24 * 3600)  # Convert to days

            if days_diff > max_days:
                return {
                    "valid": False,
                    "error": f"El rango de tiempo es demasiado grande. {limits['description']}. Selecciona un rango menor.",
                    "max_days": max_days,
                    "current_days": round(days_diff, 2),
                }

            return {"valid": True, "days": round(days_diff, 2)}

        except ValueError as e:
            return {"valid": False, "error": f"Formato de fecha/hora inválido: {str(e)}"}

    def validate_api_key(self, provider: str = None) -> Dict[str, Any]:
        """Validate the current API key by making a simple test call."""
        try:
            # Get provider from config if not provided
            if provider is None:
                provider = self.config.get("agent_llm_provider", "openai")
            
            if provider == "openai":
                from openai import OpenAI
                api_key = resolve_api_key(self.config, provider)
                if not api_key:
                    return {
                        "valid": False,
                        "error": "❌ Clave API inválida: la clave API de OpenAI no está configurada. Actualízala en la sección Configuración.",
                    }

                client = OpenAI(api_key=api_key)
                
                # Make a simple test call
                _ = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5,
                )
                
                provider_name = "OpenAI"
            elif provider == "anthropic":
                from anthropic import Anthropic
                api_key = resolve_api_key(self.config, provider)
                if not api_key:
                    return {
                        "valid": False,
                        "error": "❌ Clave API inválida: la clave API de Anthropic no está configurada. Actualízala en la sección Configuración.",
                    }
                
                client = Anthropic(api_key=api_key)
                
                # Make a simple test call
                _ = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=5,
                    messages=[{"role": "user", "content": "Hello"}],
                )
                
                provider_name = "Anthropic"
            elif provider == "qwen":
                from langchain_qwq import ChatQwen
                api_key = resolve_api_key(self.config, provider)
                if not api_key:
                    return {
                        "valid": False,
                        "error": "❌ Clave API inválida: la clave API de Qwen no está configurada. Actualízala en la sección Configuración.",
                    }

                # Make a simple test call using LangChain
                llm = ChatQwen(model="qwen-flash", api_key=api_key)
                _ = llm.invoke([("user", "Hello")])

                provider_name = "Qwen"
            else:  # minimax
                from openai import OpenAI as _OpenAI
                api_key = resolve_api_key(self.config, provider)
                if not api_key:
                    return {
                        "valid": False,
                        "error": "❌ Clave API inválida: la clave API de MiniMax no está configurada. Actualízala en la sección Configuración.",
                    }

                client = _OpenAI(api_key=api_key, base_url="https://api.minimax.io/v1")
                _ = client.chat.completions.create(
                    model="MiniMax-M2.7-highspeed",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5,
                )

                provider_name = "MiniMax"
            return {"valid": True, "message": f"La clave API de {provider_name} es válida"}

        except Exception as e:
            error_msg = str(e)
            
            # Determine provider name for error messages
            if provider is None:
                provider = self.config.get("agent_llm_provider", "openai")
            if provider == "openai":
                provider_name = "OpenAI"
            elif provider == "anthropic":
                provider_name = "Anthropic"
            elif provider == "minimax":
                provider_name = "MiniMax"
            else:
                provider_name = "Qwen"

            if (
                "authentication" in error_msg.lower()
                or "invalid api key" in error_msg.lower()
                or "401" in error_msg
                or "invalid_api_key" in error_msg.lower()
            ):
                return {
                    "valid": False,
                    "error": f"❌ Clave API inválida: la clave API de {provider_name} es inválida o ha vencido. Actualízala en la sección Configuración.",
                }
            elif "rate limit" in error_msg.lower() or "429" in error_msg:
                return {
                    "valid": False,
                    "error": f"⚠️ Límite de tasa excedido: alcanzaste el límite de la API de {provider_name}. Espera un momento e inténtalo de nuevo.",
                }
            elif "quota" in error_msg.lower() or "billing" in error_msg.lower():
                return {
                    "valid": False,
                    "error": f"💳 Problema de facturación: tu cuenta de {provider_name} no tiene créditos suficientes o presenta problemas de facturación. Revisa tu cuenta de {provider_name}.",
                }
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                return {
                    "valid": False,
                    "error": f"🌐 Error de red: no fue posible conectar con los servidores de {provider_name}. Verifica tu conexión a internet.",
                }
            else:
                return {"valid": False, "error": GENERIC_API_KEY_ERROR}

    def load_custom_assets(self) -> list:
        """Load custom assets from persistent JSON file."""
        try:
            if self.custom_assets_file.exists():
                with open(self.custom_assets_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            return []
        except Exception as e:
            print(f"Error loading custom assets: {e}")
            return []

    def save_custom_asset(self, symbol: str) -> bool:
        """Save a custom asset symbol persistently (avoid duplicates)."""
        try:
            symbol = symbol.strip()
            if not symbol:
                return False
            if symbol in self.custom_assets:
                return True  # already present
            self.custom_assets.append(symbol)
            # write to file
            with open(self.custom_assets_file, "w", encoding="utf-8") as f:
                json.dump(self.custom_assets, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving custom asset '{symbol}': {e}")
            return False


# Initialize the analyzer
analyzer = WebTradingAnalyzer()


@app.route("/")
def index():
    """Main landing page - redirect to demo."""
    return render_template("demo_new.html")


@app.route("/demo")
def demo():
    """Demo page with new interface."""
    return render_template("demo_new.html")


@app.route("/output")
def output():
    """Output page with analysis results."""
    # Get results from session or query parameters
    results = request.args.get("results")
    if results:
        try:
            # Handle URL-encoded results
            results = urllib.parse.unquote(results)
            results_data = json.loads(results)
            return render_template("output.html", results=results_data)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error parsing results: {e}")
            # Fall back to default results

    # Default results if none provided
    default_results = {
        "asset_name": "BTC",
        "timeframe": "1h",
        "data_length": 1247,
        "technical_indicators": "RSI (14): 65.4 - Impulso neutral a alcista\nMACD: Cruce alcista con histograma en aumento\nMedias móviles: Precio por encima de las MA de 50 y 200 días\nBandas de Bollinger: Precio en la banda superior, mostrando fortaleza\nVolumen: Volumen por encima del promedio, respaldando la acción del precio",
        "pattern_analysis": "Patrón de bandera alcista: consolidación tras un movimiento fuerte al alza\nCruce dorado: la MA de 50 días cruza por encima de la MA de 200 días\nMáximos y mínimos crecientes: confirmación de tendencia alcista\nPatrón de volumen: aumento del volumen durante los avances del precio",
        "trend_analysis": "Tendencia principal: alcista (largo plazo)\nTendencia secundaria: alcista (mediano plazo)\nTendencia de corto plazo: consolidación con sesgo alcista\nADX: 28.5 - Fuerza de tendencia moderada\nAcción del precio: se mantienen máximos y mínimos crecientes\nImpulso: divergencia positiva en RSI",
        "pattern_chart": "",
        "trend_chart": "",
        "pattern_image_filename": "",
        "trend_image_filename": "",
        "final_decision": {
            "decision": "LONG",
            "risk_reward_ratio": "1:2.5",
            "forecast_horizon": "24-48 horas",
            "justification": "Con base en el análisis integral de indicadores técnicos, reconocimiento de patrones y análisis de tendencia, el sistema recomienda una posición LARGA en BTC. El análisis muestra un fuerte impulso alcista, niveles clave de soporte defendidos y múltiples indicadores técnicos confirmando un movimiento ascendente.",
        },
    }

    return render_template("output.html", results=default_results)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        data_source = data.get("data_source")
        asset = data.get("asset")
        timeframe = data.get("timeframe")
        redirect_to_output = data.get("redirect_to_output", False)

        if data_source != "live":
            return jsonify({"error": "Solo se admiten datos en vivo de Yahoo Finance."})

        # Live Yahoo Finance data only
        start_date = data.get("start_date")
        start_time = data.get("start_time", "00:00")
        end_date = data.get("end_date")
        end_time = data.get("end_time", "23:59")
        use_current_time = data.get("use_current_time", False)

        # Create datetime objects for validation
        if start_date:
            start_datetime_str = f"{start_date} {start_time}"
            try:
                start_dt = datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M")
            except ValueError:
                return jsonify({"error": "Formato inválido para la fecha/hora de inicio."})

            if start_dt > datetime.now():
                return jsonify({"error": "La fecha/hora de inicio no puede estar en el futuro."})

        if end_date:
            if use_current_time:
                end_dt = datetime.now()
            else:
                end_datetime_str = f"{end_date} {end_time}"
                try:
                    end_dt = datetime.strptime(end_datetime_str, "%Y-%m-%d %H:%M")
                except ValueError:
                    return jsonify({"error": "Formato inválido para la fecha/hora de fin."})

                if end_dt > datetime.now():
                    return jsonify({"error": "La fecha/hora de fin no puede estar en el futuro."})

            if start_date and start_dt and end_dt and end_dt < start_dt:
                return jsonify(
                    {"error": "La fecha/hora de fin no puede ser anterior a la fecha/hora de inicio."}
                )

        # Fetch data with datetime objects
        df = analyzer.fetch_yfinance_data_with_datetime(
            asset, timeframe, start_dt, end_dt
        )
        if df.empty:
            return jsonify({"error": "No hay datos disponibles para los parámetros especificados"})

        display_name = analyzer.asset_mapping.get(asset, asset)
        if display_name is None:
            display_name = asset
        results = analyzer.run_analysis(df, display_name, timeframe)
        formatted_results = analyzer.extract_analysis_results(results)

        # If redirect is requested, return redirect URL with results
        if redirect_to_output:
            if formatted_results.get("success", False):
                # Create a version without base64 images for URL encoding
                # Base64 images are too large for URL parameters
                url_safe_results = formatted_results.copy()
                url_safe_results["pattern_chart"] = ""  # Remove base64 data
                url_safe_results["trend_chart"] = ""  # Remove base64 data

                # Encode results for URL
                results_json = json.dumps(url_safe_results)
                encoded_results = urllib.parse.quote(results_json)
                redirect_url = f"/output?results={encoded_results}"

                # Store full results (with images) in session or temporary storage
                # For now, we'll pass them back in the response for the frontend to handle
                return jsonify(
                    {
                        "redirect": redirect_url,
                        "full_results": formatted_results,  # Include images in response body
                    }
                )
            else:
                return jsonify(
                    {"error": formatted_results.get("error", "El análisis falló")}
                )

        return jsonify(formatted_results)
    except Exception as e:
        print(f"Error inesperado en /api/analyze: {str(e)}")
        return jsonify({"error": GENERIC_ANALYSIS_ROUTE_ERROR})


@app.route("/api/files/<asset>/<timeframe>")
def get_files(asset, timeframe):
    """API endpoint to get available files for an asset/timeframe."""
    try:
        files = analyzer.get_available_files(asset, timeframe)
        file_list = []

        for i, file_path in enumerate(files):
            match = re.search(r"_(\d+)\.csv$", file_path.name)
            file_number = match.group(1) if match else "N/A"
            file_list.append(
                {"index": i, "number": file_number, "name": file_path.name}
            )

        return jsonify({"files": file_list})

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/save-custom-asset", methods=["POST"])
def save_custom_asset():
    """Save a custom asset symbol server-side for persistence."""
    try:
        data = request.get_json()
        symbol = (data.get("symbol") or "").strip()
        if not symbol:
            return jsonify({"success": False, "error": "El símbolo es obligatorio"}), 400

        ok = analyzer.save_custom_asset(symbol)
        if not ok:
            return jsonify({"success": False, "error": "No se pudo guardar el símbolo"}), 500

        return jsonify({"success": True, "symbol": symbol})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/custom-assets", methods=["GET"])
def custom_assets():
    """Return server-persisted custom assets."""
    try:
        return jsonify({"custom_assets": analyzer.custom_assets or []})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/assets")
def get_assets():
    """API endpoint to get available assets."""
    try:
        assets = analyzer.get_available_assets()
        asset_list = []

        for asset in assets:
            asset_list.append(
                {"code": asset, "name": analyzer.asset_mapping.get(asset, asset)}
            )

        # Include server-persisted custom assets at the end
        for custom in analyzer.custom_assets:
            asset_list.append({"code": custom, "name": custom})

        return jsonify({"assets": asset_list})

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/timeframe-limits/<timeframe>")
def get_timeframe_limits(timeframe):
    """API endpoint to get date range limits for a timeframe."""
    try:
        limits = analyzer.get_timeframe_date_limits(timeframe)
        return jsonify(limits)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/validate-date-range", methods=["POST"])
def validate_date_range():
    """API endpoint to validate date and time range for a timeframe."""
    try:
        data = request.get_json()
        start_date = data.get("start_date")
        end_date = data.get("end_date")
        timeframe = data.get("timeframe")
        start_time = data.get("start_time", "00:00")
        end_time = data.get("end_time", "23:59")

        if not all([start_date, end_date, timeframe]):
            return jsonify({"error": "Faltan parámetros obligatorios"})

        validation = analyzer.validate_date_range(
            start_date, end_date, timeframe, start_time, end_time
        )
        return jsonify(validation)

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/update-provider", methods=["POST"])
def update_provider():
    """API endpoint to update LLM provider."""
    try:
        data = request.get_json()
        provider = data.get("provider", "openai")

        if provider not in ["openai", "anthropic", "qwen", "minimax"]:
            return jsonify({"error": "El proveedor debe ser 'openai', 'anthropic', 'qwen' o 'minimax'"})

        print(f"Updating provider to: {provider}")

        # Update config in both analyzer and trading_graph
        analyzer.config["agent_llm_provider"] = provider
        analyzer.config["graph_llm_provider"] = provider
        analyzer.trading_graph.config["agent_llm_provider"] = provider
        analyzer.trading_graph.config["graph_llm_provider"] = provider
        
        # Update model names if switching providers
        if provider == "anthropic":
            # Set default Claude models if not already set to Anthropic models
            if not analyzer.config["agent_llm_model"].startswith("claude"):
                analyzer.config["agent_llm_model"] = "claude-haiku-4-5-20251001"
            if not analyzer.config["graph_llm_model"].startswith("claude"):
                analyzer.config["graph_llm_model"] = "claude-haiku-4-5-20251001"
        elif provider == "qwen":
            # Set default Qwen models if not already set to Qwen models
            if not analyzer.config["agent_llm_model"].startswith("qwen"):
                analyzer.config["agent_llm_model"] = "qwen3-max"
            if not analyzer.config["graph_llm_model"].startswith("qwen"):
                analyzer.config["graph_llm_model"] = "qwen3-vl-plus"
        elif provider == "minimax":
            # Set default MiniMax models if not already set to MiniMax models
            if not analyzer.config["agent_llm_model"].startswith("MiniMax"):
                analyzer.config["agent_llm_model"] = "MiniMax-M2.7"
            if not analyzer.config["graph_llm_model"].startswith("MiniMax"):
                analyzer.config["graph_llm_model"] = "MiniMax-M2.7"

        else:
            # Set default OpenAI models if not already set to OpenAI models
            if analyzer.config["agent_llm_model"].startswith(("claude", "qwen", "MiniMax")):
                analyzer.config["agent_llm_model"] = "gpt-4o-mini"
            if analyzer.config["graph_llm_model"].startswith(("claude", "qwen", "MiniMax")):
                analyzer.config["graph_llm_model"] = "gpt-4o"
        
        analyzer.trading_graph.config.update(analyzer.config)

        # Refresh the trading graph with new provider
        analyzer.trading_graph.refresh_llms()

        print(f"Provider updated to {provider} successfully")
        print(f"graph_llm_model updated to {analyzer.config['graph_llm_model']} successfully")
        print(f"agent_llm updated to {analyzer.config['agent_llm_model']} successfully")
        return jsonify({"success": True, "message": f"Proveedor actualizado a {provider}"})

    except Exception as e:
        print(f"Error in update_provider: {str(e)}")
        return jsonify({"error": GENERIC_PROVIDER_UPDATE_ERROR})


@app.route("/api/update-api-key", methods=["POST"])
def update_api_key():
    """API endpoint to update API key for OpenAI or Anthropic."""
    try:
        data = request.get_json()
        new_api_key = data.get("api_key")
        provider = data.get("provider", "openai")  # Default to "openai" for backward compatibility

        if not new_api_key:
            return jsonify({"error": "La clave API es obligatoria"})

        if provider not in ["openai", "anthropic", "qwen", "minimax"]:
            return jsonify({"error": "El proveedor debe ser 'openai', 'anthropic', 'qwen' o 'minimax'"})

        print(f"Updating {provider} API key to: {new_api_key[:8]}...{new_api_key[-4:]}")

        # Update the environment variable
        if provider == "openai":
            os.environ["OPENAI_API_KEY"] = new_api_key
        elif provider == "anthropic":
            os.environ["ANTHROPIC_API_KEY"] = new_api_key
        elif provider == "qwen":
            os.environ["DASHSCOPE_API_KEY"] = new_api_key
        elif provider == "minimax":
            os.environ["MINIMAX_API_KEY"] = new_api_key

        # Update the API key in the trading graph
        analyzer.trading_graph.update_api_key(new_api_key, provider=provider)

        print(f"{provider} API key updated successfully")
        return jsonify({"success": True, "message": f"La clave API de {provider.capitalize()} se actualizó correctamente"})

    except Exception as e:
        print(f"Error in update_api_key: {str(e)}")
        return jsonify({"error": GENERIC_API_KEY_UPDATE_ERROR})


@app.route("/api/get-api-key-status")
def get_api_key_status():
    """API endpoint to check if API key is set for a provider."""
    try:
        provider = request.args.get("provider", "openai")

        config = analyzer.config if hasattr(analyzer, "config") else {}
        try:
            api_key = resolve_api_key(config, provider)
        except ValueError:
            api_key = ""

        if api_key:
            # Return masked version for security
            masked_key = (
                api_key[:3] + "..." + api_key[-3:] if len(api_key) > 12 else "***"
            )
            return jsonify({"has_key": True, "masked_key": masked_key})
        else:
            return jsonify({"has_key": False})
    except Exception as e:
        print(f"Error in get_api_key_status: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "has_key": False})


@app.route("/api/images/<image_type>")
def get_image(image_type):
    """API endpoint to serve generated images."""
    try:
        if image_type == "pattern":
            image_path = "kline_chart.png"
        elif image_type == "trend":
            image_path = "trend_graph.png"
        elif image_type == "pattern_chart":
            image_path = "pattern_chart.png"
        elif image_type == "trend_chart":
            image_path = "trend_chart.png"
        else:
            return jsonify({"error": "Tipo de imagen inválido"})

        if not os.path.exists(image_path):
            return jsonify({"error": "Imagen no encontrada"})

        return send_file(image_path, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/validate-api-key", methods=["POST"])
def validate_api_key():
    """API endpoint to validate the current API key."""
    try:
        data = request.get_json() or {}
        provider = data.get("provider") or analyzer.config.get("agent_llm_provider", "openai")
        validation = analyzer.validate_api_key(provider=provider)
        return jsonify(validation)
    except Exception as e:
        print(f"Error in validate_api_key route: {str(e)}")
        return jsonify({"valid": False, "error": GENERIC_API_KEY_ROUTE_ERROR})


@app.route("/assets/<path:filename>")
def serve_assets(filename):
    """Serve static assets from the assets folder."""
    try:
        return send_file(f"assets/{filename}")
    except FileNotFoundError:
        return jsonify({"error": "Activo no encontrado"}), 404


if __name__ == "__main__":
    # Create templates directory if it doesn't exist
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)

    # Create static directory if it doesn't exist
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)

    app.run(debug=True, host="127.0.0.1", port=5000)
