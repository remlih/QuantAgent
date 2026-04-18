import copy
import json
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from openai import RateLimitError


def _supports_langchain_tool_calls(llm) -> bool:
    """Return whether the model can surface LangChain-style tool calls."""
    return getattr(llm, "supports_langchain_tool_calls", True)


def build_pattern_reference_text() -> str:
    """Return the Spanish reference list of classic candlestick patterns."""
    return """
        Consulta los siguientes patrones clásicos de velas:

        1. Hombro-cabeza-hombro invertido: tres mínimos donde el central es el más bajo; estructura simétrica que suele indicar una próxima tendencia alcista.
        2. Doble suelo: dos mínimos similares con un rebote intermedio, formando una "W".
        3. Suelo redondeado: caída gradual seguida de una subida gradual, formando una "U".
        4. Base oculta: consolidación horizontal seguida de una ruptura alcista repentina.
        5. Cuña descendente: el precio se estrecha a la baja y suele romper al alza.
        6. Cuña ascendente: el precio sube lentamente pero converge, y suele romper a la baja.
        7. Triángulo ascendente: línea de soporte ascendente con resistencia plana arriba; la ruptura suele ser alcista.
        8. Triángulo descendente: resistencia descendente con soporte plano abajo; normalmente rompe a la baja.
        9. Bandera alcista: tras una subida brusca, el precio consolida brevemente a la baja antes de continuar al alza.
        10. Bandera bajista: tras una caída brusca, el precio consolida brevemente al alza antes de continuar a la baja.
        11. Rectángulo: el precio fluctúa entre soporte y resistencia horizontales.
        12. Reversión en isla: dos gaps en direcciones opuestas que forman una isla aislada de precio.
        13. Reversión en V: caída brusca seguida de recuperación brusca, o viceversa.
        14. Techo / suelo redondeado: formación gradual de techo o suelo en forma de arco.
        15. Triángulo expansivo: máximos y mínimos cada vez más amplios, indicando oscilaciones volátiles.
        16. Triángulo simétrico: máximos y mínimos convergen hacia el vértice y suelen anteceder una ruptura.
        """


def build_pattern_tool_system_prompt() -> str:
    """Build the Spanish tool prompt for pattern generation."""
    return (
        "Eres un asistente de reconocimiento de patrones de trading encargado de identificar patrones clásicos de alta frecuencia. "
        "Tienes acceso a la herramienta generate_kline_image. "
        "Úsala con argumentos apropiados como `kline_data`.\n\n"
        "Una vez generado el gráfico, compáralo con las descripciones de patrones clásicos, determina si hay algún patrón conocido presente y Responde en español."
    )


def build_pattern_image_prompt_text(time_frame: str, pattern_text: str) -> str:
    """Build the Spanish image-analysis prompt for the pattern agent."""
    return (
        f"Este es un gráfico de velas de {time_frame} generado a partir de datos OHLC recientes del mercado.\n\n"
        f"{pattern_text}\n\n"
        "Determina si el gráfico coincide con alguno de los patrones enumerados. "
        "Indica claramente el patrón o patrones detectados y explica tu razonamiento según la estructura, la tendencia y la simetría. "
        "Responde en español."
    )


def invoke_tool_with_retry(tool_fn, tool_args, retries=3, wait_sec=4):
    """
    Invoke a tool function with retries if the result is missing an image.
    """
    for attempt in range(retries):
        result = tool_fn.invoke(tool_args)
        img_b64 = result.get("pattern_image")
        if img_b64:
            return result
        print(
            f"Tool returned no image, retrying in {wait_sec}s (attempt {attempt + 1}/{retries})..."
        )
        time.sleep(wait_sec)
    raise RuntimeError("Tool failed to generate image after multiple retries")


def create_pattern_agent(tool_llm, graph_llm, toolkit):
    """
    Create a pattern recognition agent node for candlestick pattern analysis.
    The agent uses precomputed images from state or falls back to tool generation.
    """

    def pattern_agent_node(state):
        # --- Tool and pattern definitions ---
        tools = [toolkit.generate_kline_image]
        time_frame = state["time_frame"]
        pattern_text = build_pattern_reference_text()
        supports_tool_calls = _supports_langchain_tool_calls(tool_llm)

        # --- Check for precomputed image in state ---
        pattern_image_b64 = state.get("pattern_image")

        # --- Retry wrapper for LLM invocation ---
        def invoke_with_retry(call_fn, *args, retries=3, wait_sec=8):
            for attempt in range(retries):
                try:
                    return call_fn(*args)
                except RateLimitError:
                    print(
                        f"Rate limit hit, retrying in {wait_sec}s (attempt {attempt + 1}/{retries})..."
                    )
                    time.sleep(wait_sec)
                except Exception as e:
                    print(
                        f"Other error: {e}, retrying in {wait_sec}s (attempt {attempt + 1}/{retries})..."
                    )
                    time.sleep(wait_sec)
            raise RuntimeError("Max retries exceeded")

        messages = state.get("messages", [])

        # --- If no precomputed image, fall back to tool generation ---
        if not pattern_image_b64:
            print(
                "No precomputed pattern image found in state, generating with tool..."
            )

            # --- System prompt setup for tool generation ---
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        build_pattern_tool_system_prompt(),
                    ),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            ).partial(kline_data=json.dumps(state["kline_data"], indent=2))

            chain = prompt | tool_llm.bind_tools(tools) if supports_tool_calls else None

            if not supports_tool_calls:
                tool_result = invoke_tool_with_retry(
                    toolkit.generate_kline_image,
                    {"kline_data": copy.deepcopy(state["kline_data"])},
                )
                pattern_image_b64 = tool_result.get("pattern_image")
            else:
                # --- Step 1: First LLM call to determine tool usage ---
                ai_response = invoke_with_retry(chain.invoke, messages)
                messages.append(ai_response)

                # --- Step 2: Handle tool call (generate_kline_image) ---
                if hasattr(ai_response, "tool_calls"):
                    for call in ai_response.tool_calls:
                        tool_name = call["name"]
                        tool_args = call["args"]
                        # Always provide kline_data
                        tool_args["kline_data"] = copy.deepcopy(state["kline_data"])
                        tool_fn = next(t for t in tools if t.name == tool_name)
                        tool_result = invoke_tool_with_retry(tool_fn, tool_args)
                        pattern_image_b64 = tool_result.get("pattern_image")
                        messages.append(
                            ToolMessage(
                                tool_call_id=call["id"], content=json.dumps(tool_result)
                            )
                        )
        else:
            print("Using precomputed pattern image from state")

        # --- Step 3: Vision analysis with image (precomputed or generated) ---
        if pattern_image_b64:
            image_prompt = [
                {
                    "type": "text",
                    "text": build_pattern_image_prompt_text(time_frame, pattern_text),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{pattern_image_b64}"},
                },
            ]

            # Create messages - ensure HumanMessage has valid content
            # For Anthropic, SystemMessage is extracted separately, but messages array must have at least one message
            human_msg = HumanMessage(content=image_prompt)
            
            # Verify HumanMessage content is valid
            if not human_msg.content:
                raise ValueError("HumanMessage content is empty")
            if isinstance(human_msg.content, list) and len(human_msg.content) == 0:
                raise ValueError("HumanMessage content list is empty")
            
            messages = [
                SystemMessage(
                    content="Eres un asistente de reconocimiento de patrones de trading encargado de analizar gráficos de velas. Responde en español."
                ),
                human_msg,
            ]
            
            try:
                final_response = invoke_with_retry(
                    graph_llm.invoke,
                    messages,
                )
            except Exception as e:
                error_str = str(e)
                # Handle Anthropic's "at least one message is required" error
                # This can happen when SystemMessage extraction leaves empty messages array
                if "at least one message" in error_str.lower():
                    # Retry with only HumanMessage (SystemMessage will be lost but Anthropic should work)
                    print("Retrying with HumanMessage only due to Anthropic message conversion issue...")
                    final_response = invoke_with_retry(
                        graph_llm.invoke,
                        [human_msg],
                    )
                else:
                    raise
        else:
            # If no image was generated, fall back to reasoning with messages
            final_response = invoke_with_retry(chain.invoke, messages)

        return {
            "messages": messages + [final_response],
            "pattern_report": final_response.content,
        }

    return pattern_agent_node
