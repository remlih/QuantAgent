"""
Agent for technical indicator analysis in high-frequency trading (HFT) context.
Uses LLM and toolkit to compute and interpret indicators like MACD, RSI, ROC, Stochastic, and Williams %R.
"""

import copy
import json

from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def _supports_langchain_tool_calls(llm) -> bool:
    """Return whether the model can surface LangChain-style tool calls."""
    return getattr(llm, "supports_langchain_tool_calls", True)


def build_indicator_system_prompt(time_frame: str) -> str:
    """Build the Spanish system prompt for indicator analysis."""
    return (
        "Eres un asistente analista de trading de alta frecuencia (HFT) que opera bajo condiciones sensibles al tiempo. "
        "Debes analizar indicadores técnicos para respaldar una ejecución rápida de trading.\n\n"
        "Tienes acceso a las herramientas: compute_rsi, compute_macd, compute_roc, compute_stoch y compute_willr. "
        "Úsalas proporcionando argumentos adecuados como `kline_data` y los períodos correspondientes.\n\n"
        f"⚠️ Los datos OHLC proporcionados corresponden a intervalos de {time_frame} y reflejan el comportamiento reciente del mercado. "
        "Debes interpretarlos con rapidez y precisión.\n\n"
        "Estos son los datos OHLC:\n{kline_data}.\n\n"
        "Llama a las herramientas necesarias, analiza los resultados y Responde en español.\n"
    )


def create_indicator_agent(llm, toolkit):
    """
    Create an indicator analysis agent node for HFT. The agent uses LLM and indicator tools to analyze OHLCV data.
    """

    def indicator_agent_node(state):
        # --- Tool definitions ---
        tools = [
            toolkit.compute_macd,
            toolkit.compute_rsi,
            toolkit.compute_roc,
            toolkit.compute_stoch,
            toolkit.compute_willr,
        ]
        time_frame = state["time_frame"]
        supports_tool_calls = _supports_langchain_tool_calls(llm)
        # --- System prompt for LLM ---
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    build_indicator_system_prompt(time_frame),
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        ).partial(kline_data=json.dumps(state["kline_data"], indent=2))

        if not supports_tool_calls:
            precomputed_indicators = {}
            for tool_fn in tools:
                precomputed_indicators[tool_fn.name] = tool_fn.invoke(
                    {"kline_data": copy.deepcopy(state["kline_data"])}
                )

            fallback_messages = [
                HumanMessage(
                    content=(
                        "Inicia el análisis de indicadores y responde en español. "
                        "Los indicadores ya fueron calculados manualmente; no intentes llamar herramientas.\n\n"
                        f"Resultados:\n{json.dumps(precomputed_indicators, indent=2)}"
                    )
                )
            ]
            final_response = llm.invoke(fallback_messages)
            return {
                "messages": fallback_messages + [final_response],
                "indicator_report": final_response.content
                if getattr(final_response, "content", "")
                else "Análisis de indicadores completado.",
            }

        chain = prompt | llm.bind_tools(tools)
        # messages = state["messages"]
        messages = state.get("messages", [])
        if not messages:
            messages = [HumanMessage(content="Inicia el análisis de indicadores y responde en español.")]


        # --- Step 1: Ask for tool calls ---
        ai_response = chain.invoke(messages)
        messages.append(ai_response)
        
        # --- Step 2: Collect tool results ---
        if hasattr(ai_response, "tool_calls") and ai_response.tool_calls:
            for call in ai_response.tool_calls:
                tool_name = call["name"]
                tool_args = call["args"]
                # Always provide kline_data
                tool_args["kline_data"] = copy.deepcopy(state["kline_data"])
                # Lookup tool by name
                tool_fn = next(t for t in tools if t.name == tool_name)
                tool_result = tool_fn.invoke(tool_args)
                # Append result as ToolMessage
                messages.append(
                    ToolMessage(
                        tool_call_id=call["id"], content=json.dumps(tool_result)
                    )
                )

        # --- Step 3: Re-run the chain with tool results ---
        # Keep invoking until we get a text response (not another tool call)
        # This is important for Claude which may make multiple tool calls
        max_iterations = 5  # Prevent infinite loops
        iteration = 0
        final_response = None
        
        while iteration < max_iterations:
            iteration += 1
            final_response = chain.invoke(messages)
            messages.append(final_response)
            
            # If there are no tool calls, we have the final answer
            if not hasattr(final_response, "tool_calls") or not final_response.tool_calls:
                break
            
            # If there are more tool calls, execute them
            for call in final_response.tool_calls:
                tool_name = call["name"]
                tool_args = call["args"]
                tool_args["kline_data"] = copy.deepcopy(state["kline_data"])
                tool_fn = next(t for t in tools if t.name == tool_name)
                tool_result = tool_fn.invoke(tool_args)
                messages.append(
                    ToolMessage(
                        tool_call_id=call["id"], content=json.dumps(tool_result)
                    )
                )

        # Extract content - handle both string and empty content cases
        if final_response:
            report_content = final_response.content
            # If content is empty or None, try to get text from recent messages
            if not report_content or (isinstance(report_content, str) and not report_content.strip()):
                # Check if there's any text content in the messages (skip tool calls)
                for msg in reversed(messages):
                    if (hasattr(msg, 'content') and msg.content and 
                        isinstance(msg.content, str) and msg.content.strip() and 
                        not hasattr(msg, 'tool_calls')):
                        report_content = msg.content
                        break
        else:
            report_content = "El análisis de indicadores se completó, pero no se generó un reporte detallado."

        return {
            "messages": messages,
            "indicator_report": report_content if report_content else "Análisis de indicadores completado.",
        }

    return indicator_agent_node
