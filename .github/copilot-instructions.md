# Copilot Instructions for QuantAgent

## Commands

### Environment setup
- Create the Conda environment used by this repo: `conda create -n quantagents python=3.11`
- Activate it: `conda activate quantagents`
- Install dependencies: `pip install -r requirements.txt`
- If `TA-Lib` installation fails on Windows or native builds are missing: `conda install -c conda-forge ta-lib`

### Run the app
- Start the Flask web UI: `python web_interface.py`

### Tests
- Run the full test suite: `conda run -n quantagents python -m unittest discover -s tests -p "test_*.py"`
- Run a single test module: `conda run -n quantagents python -m unittest tests.test_minimax_provider`
- Run a single test case: `conda run -n quantagents python -m unittest tests.test_minimax_provider.TestTradingGraphGetApiKey.test_get_api_key_from_env`
- Run live MiniMax integration tests (requires `MINIMAX_API_KEY`; optional `MINIMAX_TEST_MODEL`): `conda run -n quantagents python -m unittest tests.test_minimax_integration`

## High-level architecture

- `web_interface.py` is the main runtime entrypoint. `WebTradingAnalyzer` fetches live OHLC data from Yahoo Finance, normalizes the frame into a dict-of-lists, trims analysis input to the last 45 candles, precomputes `pattern_image` and `trend_image` with `static_util.py`, then invokes `TradingGraph.graph` with an `IndicatorAgentState`-shaped state object.
- `trading_graph.py` is the orchestration boundary. It owns provider-specific API key lookup, creates two LLM clients (`agent_llm` and `graph_llm`), constructs `TechnicalTools`, and rebuilds the LangGraph when config changes via `refresh_llms()`.
- `graph_setup.py` wires a strictly linear LangGraph pipeline: `Indicator Agent -> Pattern Agent -> Trend Agent -> Decision Maker`. Changes to agent ordering or adding/removing agents usually require matching edits in `agent_state.py`, `graph_setup.py`, and any UI/result extraction code that expects specific state fields.
- `graph_util.py` contains the LangChain `@tool` implementations used inside the graph. Indicator tools compute TA-Lib series; the pattern and trend tools generate images plus ignored local artifacts (`kline_chart.png`, `trend_graph.png`, `record.csv`) in the repository root.
- `static_util.py` duplicates the chart generation path for the web flow. The Flask path prefers precomputed images in the initial state; `pattern_agent.py` and `trend_agent.py` only fall back to calling the image-generation tools when those state fields are missing.
- `decision_agent.py` emits the final trade decision as model text that is expected to contain a JSON object. `web_interface.py` extracts the first `{...}` block from `final_trade_decision`, so any change to the decision format must be coordinated with that parser.
- `.github/workflows/deploy.yml` only publishes the repository root to GitHub Pages. It is not a build/test workflow for the Flask application.

## Key conventions

- The graph state contract is important. `kline_data` should remain a dict of lists with exact keys `Datetime`, `Open`, `High`, `Low`, and `Close`, and the web path serializes datetimes as `%Y-%m-%d %H:%M:%S`.
- Pattern and trend analysis assume a vision-capable model because both agents send base64 chart images to the LLM. Provider changes that remove image support will break those paths even if plain text calls still work.
- Provider support is cross-cutting. When adding or changing a provider, keep `default_config.py`, `trading_graph.py`, and the provider update/API-key routes in `web_interface.py` in sync.
- The Flask UI updates both `analyzer.config` and `analyzer.trading_graph.config` before calling `refresh_llms()`. Preserve that double update when changing runtime configuration behavior or the UI and graph can drift out of sync.
- MiniMax is implemented through `ChatOpenAI` with the MiniMax OpenAI-compatible endpoint (`https://api.minimax.io/v1`). The code clamps MiniMax temperature into `(0.0, 1.0]`, and integration tests read `MINIMAX_API_KEY` plus optional `MINIMAX_TEST_MODEL`.
- Existing tests use `unittest`, not pytest, even though one integration test docstring mentions pytest-style skipping. Follow the actual `unittest` entrypoints above.
- Tests that import project modules often mock heavy/native dependencies (`talib`, `langchain_qwq`) before importing `TradingGraph` or `web_interface`. Reuse that pattern when adding tests around provider setup or web routes.
