"""GitHub Copilot SDK-backed chat model helpers for QuantAgent."""

from __future__ import annotations

import asyncio
from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

try:
    from copilot import CopilotClient, SubprocessConfig
    from copilot.session import PermissionHandler
except ImportError:  # pragma: no cover - exercised in integration environments
    CopilotClient = None
    SubprocessConfig = None
    PermissionHandler = None

DEFAULT_COPILOT_AGENT_MODEL = "gpt-5.4"
DEFAULT_COPILOT_GRAPH_MODEL = "claude-opus-4.6"


def _parse_data_url(url: str) -> tuple[str, str] | None:
    """Return (mime_type, base64_data) for a data URL, or None if unsupported."""
    if not url.startswith("data:") or ";base64," not in url:
        return None

    header, data = url.split(",", 1)
    mime_type = header[5:].split(";", 1)[0]
    return mime_type, data


def _build_client_config(github_token: str | None) -> Any:
    """Build an SDK client config when an explicit GitHub token is provided."""
    if not github_token:
        return None
    if SubprocessConfig is None:
        raise ImportError(
            "github-copilot-sdk is required to use the 'copilot' provider. "
            "Install it with 'pip install github-copilot-sdk'."
        )
    return SubprocessConfig(
        github_token=github_token,
        use_logged_in_user=False,
    )


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from sync entrypoints."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _alist_available_copilot_models(github_token: str | None = None) -> list[str]:
    """List available Copilot model IDs through the SDK."""
    if CopilotClient is None:
        raise ImportError(
            "github-copilot-sdk is required to use the 'copilot' provider. "
            "Install it with 'pip install github-copilot-sdk'."
        )

    client = CopilotClient(_build_client_config(github_token))
    await client.start()
    try:
        models = await client.list_models()
        return [model.id for model in models]
    finally:
        await client.stop()


def list_available_copilot_models(github_token: str | None = None) -> list[str]:
    """Synchronously list available Copilot model IDs."""
    return _run_async(_alist_available_copilot_models(github_token))


def validate_copilot_auth(github_token: str | None = None) -> tuple[bool, list[str] | str]:
    """Validate that Copilot SDK authentication works and return visible model IDs."""
    try:
        models = list_available_copilot_models(github_token=github_token)
    except Exception as exc:
        return False, str(exc)

    return True, models


class CopilotChatModel(BaseChatModel):
    """Minimal LangChain adapter for GitHub Copilot SDK chat sessions."""

    model: str
    temperature: float = 0.1
    github_token: str | None = None
    use_logged_in_user: bool = True
    supports_langchain_tool_calls: bool = False

    @property
    def _llm_type(self) -> str:
        return "copilot"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "github_token": bool(self.github_token),
            "use_logged_in_user": self.use_logged_in_user,
        }

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "CopilotChatModel":
        """Return self; Copilot tool fallback is handled by the agents for now."""
        return self

    def _prepare_sdk_request(
        self, messages: Sequence[BaseMessage]
    ) -> tuple[str, str, list[dict[str, str]]]:
        """Convert LangChain messages into prompt text plus Copilot SDK attachments."""
        system_parts: list[str] = []
        prompt_parts: list[str] = []
        attachments: list[dict[str, str]] = []

        has_non_human_history = any(
            not isinstance(message, (HumanMessage, SystemMessage)) for message in messages
        )

        for message in messages:
            text_segments: list[str] = []
            content = message.content

            if isinstance(content, str):
                if content.strip():
                    text_segments.append(content.strip())
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue

                    item_type = item.get("type")
                    if item_type == "text":
                        text = item.get("text", "").strip()
                        if text:
                            text_segments.append(text)
                    elif item_type == "image_url":
                        image_url = item.get("image_url", {}).get("url", "")
                        parsed = _parse_data_url(image_url)
                        if parsed:
                            mime_type, data = parsed
                            attachments.append(
                                {
                                    "type": "blob",
                                    "data": data,
                                    "mimeType": mime_type,
                                }
                            )

            if isinstance(message, SystemMessage):
                system_parts.extend(text_segments)
                continue

            if not text_segments:
                continue

            text = "\n".join(text_segments)
            if has_non_human_history:
                prompt_parts.append(f"{message.type}: {text}")
            else:
                prompt_parts.append(text)

        prompt = "\n\n".join(prompt_parts).strip()
        system_message = "\n\n".join(system_parts).strip()
        return prompt, system_message, attachments

    async def _ainvoke_copilot(
        self, prompt: str, system_message: str, attachments: list[dict[str, str]]
    ) -> str:
        """Send a prompt through the Copilot SDK and return the final assistant content."""
        if CopilotClient is None or PermissionHandler is None:
            raise ImportError(
                "github-copilot-sdk is required to use the 'copilot' provider. "
                "Install it with 'pip install github-copilot-sdk'."
            )

        async with CopilotClient(_build_client_config(self.github_token)) as client:
            session_kwargs: dict[str, Any] = {
                "on_permission_request": PermissionHandler.approve_all,
                "model": self.model,
                "available_tools": [],
            }
            if system_message:
                session_kwargs["system_message"] = {"content": system_message}

            async with await client.create_session(**session_kwargs) as session:
                done = asyncio.Event()
                final_content: dict[str, str] = {"value": ""}

                def on_event(event: Any) -> None:
                    event_type = getattr(event, "type", None)
                    event_type_value = getattr(event_type, "value", event_type)
                    data = getattr(event, "data", None)
                    if event_type_value == "assistant.message":
                        content = getattr(data, "content", "") or ""
                        if content:
                            final_content["value"] = content
                    if event_type_value == "session.idle":
                        done.set()

                session.on(on_event)

                send_kwargs: dict[str, Any] = {}
                if attachments:
                    send_kwargs["attachments"] = attachments

                await session.send(prompt, **send_kwargs)
                await done.wait()
                return final_content["value"]

    def _run_async(self, coro: Any) -> Any:
        """Run an async coroutine from sync LangChain entrypoints."""
        return _run_async(coro)

    def _generate(
        self,
        messages: Sequence[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt, system_message, attachments = self._prepare_sdk_request(messages)
        content = self._run_async(
            self._ainvoke_copilot(prompt, system_message, attachments)
        )
        message = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=message)])
