"""
Agent for making final trade decisions in high-frequency trading (HFT) context.
Combines indicator, pattern, and trend reports to issue a LONG or SHORT order.
"""


def build_decision_prompt(
    time_frame: str,
    stock_name: str,
    indicator_report: str,
    pattern_report: str,
    trend_report: str,
) -> str:
    """Build the Spanish decision prompt while preserving the JSON contract."""
    return f"""Eres un analista cuantitativo de trading de alta frecuencia (HFT) que opera sobre el gráfico K-line actual de {time_frame} para {stock_name}. Tu tarea es emitir una **orden de ejecución inmediata**: **LONG** o **SHORT**. ⚠️ HOLD está prohibido por las restricciones HFT.

            Tu decisión debe pronosticar el movimiento del mercado durante las **próximas N velas**, donde:
            - Ejemplo: TIME_FRAME = 15min, N = 1 → Pronostica los próximos 15 minutos.
            - TIME_FRAME = 4hour, N = 1 → Pronostica las próximas 4 horas.

            Basa tu decisión en la fuerza combinada, la alineación y el timing de los tres reportes siguientes:

            ---

            ### 1. Reporte de indicadores técnicos:
            - Evalúa el impulso (por ejemplo, MACD, ROC) y los osciladores (por ejemplo, RSI, Stochastic, Williams %R).
            - Da **más peso a señales direccionales fuertes** como cruces de MACD, divergencias de RSI y niveles extremos de sobrecompra/sobreventa.
            - **Ignora o reduce el peso** de señales neutrales o mixtas, salvo que coincidan entre varios indicadores.

            ---

            ### 2. Reporte de patrones:
            - Solo actúa sobre patrones alcistas o bajistas si:
            - El patrón es **claramente reconocible y está mayormente completo**, y
            - Una **ruptura alcista o bajista ya está en marcha** o es altamente probable según el precio y el impulso (por ejemplo, mecha fuerte, pico de volumen, vela envolvente).
            - **No actúes** sobre patrones tempranos o especulativos. No trates configuraciones en consolidación como operables salvo que exista **confirmación de ruptura** desde los otros reportes.

            ---

            ### 3. Reporte de tendencia:
            - Analiza cómo interactúa el precio con el soporte y la resistencia:
            - Una **línea de soporte con pendiente ascendente** sugiere interés comprador.
            - Una **línea de resistencia con pendiente descendente** sugiere presión vendedora.
            - Si el precio se comprime entre líneas de tendencia:
            - Predice ruptura **solo cuando haya confluencia con velas fuertes o confirmación de indicadores**.
            - **No asumas la dirección de ruptura** solo por la geometría.

            ---

            ### ✅ Estrategia de decisión

            1. Actúa únicamente sobre señales **confirmadas**; evita señales emergentes, especulativas o conflictivas.
            2. Prioriza decisiones donde **los tres reportes** (Indicadores, Patrones y Tendencia) **apunten en la misma dirección**.
            3. Da más peso a:
            - Impulso fuerte reciente (por ejemplo, cruce de MACD, ruptura de RSI)
            - Acción decisiva del precio (por ejemplo, vela de ruptura, mechas de rechazo, rebote en soporte)
            4. Si los reportes discrepan:
            - Elige la dirección con **confirmación más fuerte y reciente**
            - Prefiere señales respaldadas por impulso frente a pistas débiles de osciladores.
            5. ⚖️ Si el mercado está en consolidación o los reportes están mezclados:
            - Usa por defecto la **pendiente dominante de la tendencia** (por ejemplo, SHORT en un canal descendente).
            - No adivines la dirección; elige el lado **más defendible**.
            6. Sugiere una **relación riesgo/beneficio** razonable entre **1.2 y 1.8**, según la volatilidad actual y la fortaleza de la tendencia.

            ---
            ### 🧠 Formato de salida en JSON (para parsing del sistema)

            Responde completamente en español, pero conserva **exactamente** estas claves JSON en inglés y usa `decision` con valor **LONG o SHORT**:

            ```
            {{
            "forecast_horizon": "Pronosticando las próximas 3 velas (15 minutos, 1 hora, etc.)",
            "decision": "<LONG or SHORT>",
            "justification": "<Razón concisa y confirmada basada en los reportes>",
            "risk_reward_ratio": "<float between 1.2 and 1.8>",
            }}

            --------
            **Reporte de indicadores técnicos**  
            {indicator_report}

            **Reporte de patrones**  
            {pattern_report}

            **Reporte de tendencia**  
            {trend_report}

        """


def create_final_trade_decider(llm):
    """
    Create a trade decision agent node. The agent uses LLM to synthesize indicator, pattern, and trend reports
    and outputs a final trade decision (LONG or SHORT) with justification and risk-reward ratio.
    """

    def trade_decision_node(state) -> dict:
        indicator_report = state["indicator_report"]
        pattern_report = state["pattern_report"]
        trend_report = state["trend_report"]
        time_frame = state["time_frame"]
        stock_name = state["stock_name"]

        # --- System prompt for LLM ---
        prompt = build_decision_prompt(
            time_frame=time_frame,
            stock_name=stock_name,
            indicator_report=indicator_report,
            pattern_report=pattern_report,
            trend_report=trend_report,
        )

        # --- LLM call for decision ---
        response = llm.invoke(prompt)

        return {
            "final_trade_decision": response.content,
            "messages": [response],
            "decision_prompt": prompt,
        }

    return trade_decision_node
