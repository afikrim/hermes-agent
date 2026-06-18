"""OpenAI-compatible facade that talks to Google's Antigravity API backend.

Antigravity is Google's internal IDE platform that provides access to Gemini
models with significantly higher rate limits than the standard Code Assist API
(``cloudcode-pa.googleapis.com``).  This adapter lets Hermes use the
``google-antigravity`` provider as if it were a standard OpenAI-shaped chat
completion endpoint, while the underlying HTTP traffic goes to
``daily-cloudcode-pa.sandbox.googleapis.com/v1internal:{generateContent,
streamGenerateContent}`` with a Bearer access token obtained via Antigravity
OAuth.

Architecture
------------
- ``AntigravityClient`` exposes ``.chat.completions.create(**kwargs)``
  mirroring the subset of the OpenAI SDK that ``run_agent.py`` uses.
- Incoming OpenAI ``messages[]`` / ``tools[]`` / ``tool_choice`` are translated
  to Gemini's native ``contents[]`` / ``tools[].functionDeclarations`` /
  ``toolConfig`` / ``systemInstruction`` shape — identical to the Cloud Code
  Assist translation.
- The request body is wrapped ``{project, model, user_prompt_id, request}``
  per Antigravity API expectations (same envelope as Cloud Code Assist).
- Responses (``candidates[].content.parts[]``) are converted back to
  OpenAI ``choices[0].message`` shape with ``content`` + ``tool_calls``.
- Streaming uses SSE (``?alt=sse``) and yields OpenAI-shaped delta chunks.

Derived from:
- agent/gemini_cloudcode_adapter.py  (Hermes built-in)
- opencode-antigravity-auth (MIT) by NoeFabris
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional

import httpx

from agent import antigravity_oauth
from agent.google_code_assist import (
    CODE_ASSIST_ENDPOINT,
    CodeAssistError,
    ProjectContext,
    resolve_project_context,
)
from agent.gemini_schema import sanitize_gemini_tool_parameters

logger = logging.getLogger(__name__)


# =============================================================================
# Antigravity endpoint configuration
# =============================================================================
# Antigravity uses the daily sandbox endpoint as primary, with fallbacks.
# See opencode-antigravity-auth constants.ts for the full list.

ANTIGRAVITY_ENDPOINT_DAILY = "https://daily-cloudcode-pa.sandbox.googleapis.com"
ANTIGRAVITY_ENDPOINT_AUTOPUSH = "https://autopush-cloudcode-pa.sandbox.googleapis.com"
ANTIGRAVITY_ENDPOINT_PROD = CODE_ASSIST_ENDPOINT  # "https://cloudcode-pa.googleapis.com"

# Default: start with the sandbox (Antigravity / daily channel)
ANTIGRAVITY_DEFAULT_ENDPOINT = ANTIGRAVITY_ENDPOINT_DAILY

# Fallback endpoint for when daily is down — mirrors opencode-antigravity-auth
ANTIGRAVITY_FALLBACK_ENDPOINTS = [
    ANTIGRAVITY_ENDPOINT_DAILY,
    ANTIGRAVITY_ENDPOINT_AUTOPUSH,
    ANTIGRAVITY_ENDPOINT_PROD,
]

# Default project ID used when Antigravity doesn't return one
DEFAULT_PROJECT_ID = "rising-fact-p41fc"

# Antigravity version for User-Agent header
ANTIGRAVITY_VERSION = "2.0.6"


# =============================================================================
# Request translation: OpenAI → Gemini
# =============================================================================
# This is identical to gemini_cloudcode_adapter.py's translation — both APIs
# use the same Gemini-native content format inside the {project, model, request}
# envelope.

_ROLE_MAP_OPENAI_TO_GEMINI = {
    "user": "user",
    "assistant": "model",
    "system": "user",   # handled separately via systemInstruction
    "tool": "user",     # functionResponse is wrapped in a user-role turn
    "function": "user",
}


def _coerce_content_to_text(content: Any) -> str:
    """OpenAI content may be str or a list of parts; reduce to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: List[str] = []
        for p in content:
            if isinstance(p, str):
                pieces.append(p)
            elif isinstance(p, dict):
                if p.get("type") == "text" and isinstance(p.get("text"), str):
                    pieces.append(p["text"])
                elif p.get("type") in {"image_url", "input_audio"}:
                    logger.debug("Dropping multimodal part (not yet supported): %s", p.get("type"))
        return "\n".join(pieces)
    return str(content)


def _translate_tool_call_to_gemini(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """OpenAI tool_call -> Gemini functionCall part."""
    fn = tool_call.get("function") or {}
    args_raw = fn.get("arguments", "")
    try:
        args = json.loads(args_raw) if isinstance(args_raw, str) and args_raw else {}
    except json.JSONDecodeError:
        args = {"_raw": args_raw}
    if not isinstance(args, dict):
        args = {"_value": args}
    return {
        "functionCall": {
            "name": fn.get("name") or "",
            "args": args,
        },
        "thoughtSignature": "skip_thought_signature_validator",
    }


def _translate_tool_result_to_gemini(message: Dict[str, Any]) -> Dict[str, Any]:
    """OpenAI tool-role message -> Gemini functionResponse part."""
    name = str(message.get("name") or message.get("tool_call_id") or "tool")
    content = _coerce_content_to_text(message.get("content"))
    try:
        parsed = json.loads(content) if content.strip().startswith(("{", "[")) else None
    except json.JSONDecodeError:
        parsed = None
    response = parsed if isinstance(parsed, dict) else {"output": content}
    return {
        "functionResponse": {
            "name": name,
            "response": response,
        },
    }


def _build_gemini_contents(
    messages: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Convert OpenAI messages[] to Gemini contents[] + systemInstruction."""
    system_text_parts: List[str] = []
    contents: List[Dict[str, Any]] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user")

        if role == "system":
            system_text_parts.append(_coerce_content_to_text(msg.get("content")))
            continue

        if role == "tool" or role == "function":
            contents.append({
                "role": "user",
                "parts": [_translate_tool_result_to_gemini(msg)],
            })
            continue

        gemini_role = _ROLE_MAP_OPENAI_TO_GEMINI.get(role, "user")
        parts: List[Dict[str, Any]] = []

        text = _coerce_content_to_text(msg.get("content"))
        if text:
            parts.append({"text": text})

        tool_calls = msg.get("tool_calls") or []
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict):
                    parts.append(_translate_tool_call_to_gemini(tc))

        if not parts:
            continue

        contents.append({"role": gemini_role, "parts": parts})

    system_instruction: Optional[Dict[str, Any]] = None
    joined_system = "\n".join(p for p in system_text_parts if p).strip()
    if joined_system:
        system_instruction = {
            "role": "system",
            "parts": [{"text": joined_system}],
        }

    return contents, system_instruction


def _translate_tools_to_gemini(tools: Any) -> List[Dict[str, Any]]:
    """OpenAI tools[] -> Gemini tools[].functionDeclarations[]."""
    if not isinstance(tools, list) or not tools:
        return []
    declarations: List[Dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") or {}
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not name:
            continue
        decl = {"name": str(name)}
        if fn.get("description"):
            decl["description"] = str(fn["description"])
        params = fn.get("parameters")
        if isinstance(params, dict):
            decl["parameters"] = sanitize_gemini_tool_parameters(params)
        declarations.append(decl)
    if not declarations:
        return []
    return [{"functionDeclarations": declarations}]


def _translate_tool_choice_to_gemini(tool_choice: Any) -> Optional[Dict[str, Any]]:
    """OpenAI tool_choice -> Gemini toolConfig.functionCallingConfig."""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice == "auto":
            return {"functionCallingConfig": {"mode": "AUTO"}}
        if tool_choice == "required":
            return {"functionCallingConfig": {"mode": "ANY"}}
        if tool_choice == "none":
            return {"functionCallingConfig": {"mode": "NONE"}}
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function") or {}
        name = fn.get("name")
        if name:
            return {
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowedFunctionNames": [str(name)],
                },
            }
    return None


def _normalize_thinking_config(config: Any) -> Optional[Dict[str, Any]]:
    """Accept thinkingBudget / thinkingLevel / includeThoughts (+ snake_case)."""
    if not isinstance(config, dict) or not config:
        return None
    budget = config.get("thinkingBudget", config.get("thinking_budget"))
    level = config.get("thinkingLevel", config.get("thinking_level"))
    include = config.get("includeThoughts", config.get("include_thoughts"))
    normalized: Dict[str, Any] = {}
    if isinstance(budget, (int, float)):
        normalized["thinkingBudget"] = int(budget)
    if isinstance(level, str) and level.strip():
        normalized["thinkingLevel"] = level.strip().lower()
    if isinstance(include, bool):
        normalized["includeThoughts"] = include
    return normalized or None


def _build_antigravity_request(
    *,
    messages: List[Dict[str, Any]],
    tools: Any = None,
    tool_choice: Any = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    stop: Any = None,
    thinking_config: Any = None,
) -> Dict[str, Any]:
    """Build the inner Gemini request body (goes inside ``request`` wrapper)."""
    contents, system_instruction = _build_gemini_contents(messages)

    body: Dict[str, Any] = {"contents": contents}
    if system_instruction is not None:
        body["systemInstruction"] = system_instruction

    gemini_tools = _translate_tools_to_gemini(tools)
    if gemini_tools:
        body["tools"] = gemini_tools
    tool_cfg = _translate_tool_choice_to_gemini(tool_choice)
    if tool_cfg is not None:
        body["toolConfig"] = tool_cfg

    generation_config: Dict[str, Any] = {}
    if isinstance(temperature, (int, float)):
        generation_config["temperature"] = float(temperature)
    if isinstance(max_tokens, int) and max_tokens > 0:
        generation_config["maxOutputTokens"] = max_tokens
    if isinstance(top_p, (int, float)):
        generation_config["topP"] = float(top_p)
    if isinstance(stop, str) and stop:
        generation_config["stopSequences"] = [stop]
    elif isinstance(stop, list) and stop:
        generation_config["stopSequences"] = [str(s) for s in stop if s]
    normalized_thinking = _normalize_thinking_config(thinking_config)
    if normalized_thinking:
        generation_config["thinkingConfig"] = normalized_thinking
    if generation_config:
        body["generationConfig"] = generation_config

    return body


def _normalize_antigravity_model(model: str) -> str:
    """Normalize model name for the Antigravity API.

    - Strip ``antigravity-`` prefix (marks explicit Antigravity quota)
    - Strip ``-preview`` suffix (Antigravity doesn't use preview suffixes)
    - For Gemini 3 Pro without a thinking tier, append ``-low``
    """
    normalized = model.strip()
    # Remove antigravity- prefix
    if normalized.lower().startswith("antigravity-"):
        normalized = normalized[len("antigravity-"):]
    # Remove -preview suffix
    normalized = normalized.removesuffix("-preview")
    # Gemini 3 Pro needs a tier suffix
    import re as _re
    if _re.match(r"^gemini-3(?:\.\d+)?-pro$", normalized, _re.I):
        normalized = f"{normalized}-low"
    return normalized


def _wrap_antigravity_request(
    *,
    project_id: str,
    model: str,
    inner_request: Dict[str, Any],
    user_prompt_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Wrap the inner Gemini request in the Antigravity envelope."""
    return {
        "project": project_id,
        "model": _normalize_antigravity_model(model),
        "user_prompt_id": user_prompt_id or str(uuid.uuid4()),
        "request": inner_request,
    }


# =============================================================================
# Antigravity headers
# =============================================================================

def _get_antigravity_headers(access_token: str) -> Dict[str, str]:
    """Build the Antigravity-specific request headers.

    These match what the official Antigravity IDE sends:
    - Electron/Chrome-style User-Agent
    - Client-Metadata with ideType=ANTIGRAVITY
    - Randomised api-client header
    """
    import platform as _platform
    platform_str = "WINDOWS" if _platform.system() == "Windows" else "MACOS"
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
        "User-Agent": (
            f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            f"AppleWebKit/537.36 (KHTML, like Gecko) "
            f"Antigravity/{ANTIGRAVITY_VERSION} "
            f"Chrome/138.0.7204.235 Electron/37.3.1 Safari/537.36"
        ),
        "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
        "Client-Metadata": (
            f'{{"ideType":"ANTIGRAVITY","platform":"{platform_str}","pluginType":"GEMINI"}}'
        ),
        "x-activity-request-id": str(uuid.uuid4()),
    }


# =============================================================================
# Response translation: Gemini → OpenAI
# =============================================================================

def _translate_antigravity_response(
    resp: Dict[str, Any],
    model: str,
) -> SimpleNamespace:
    """Non-streaming Antigravity response -> OpenAI-shaped SimpleNamespace.

    Antigravity wraps the actual Gemini response inside ``response``, just
    like Cloud Code Assist.
    """
    inner = resp.get("response") if isinstance(resp.get("response"), dict) else resp

    candidates = inner.get("candidates") or []
    if not isinstance(candidates, list) or not candidates:
        return _empty_response(model)

    cand = candidates[0]
    content_obj = cand.get("content") if isinstance(cand, dict) else {}
    parts = content_obj.get("parts") if isinstance(content_obj, dict) else []

    text_pieces: List[str] = []
    reasoning_pieces: List[str] = []
    tool_calls: List[SimpleNamespace] = []

    for i, part in enumerate(parts or []):
        if not isinstance(part, dict):
            continue
        if part.get("thought") is True:
            if isinstance(part.get("text"), str):
                reasoning_pieces.append(part["text"])
            continue
        if isinstance(part.get("text"), str):
            text_pieces.append(part["text"])
            continue
        fc = part.get("functionCall")
        if isinstance(fc, dict) and fc.get("name"):
            try:
                args_str = json.dumps(fc.get("args") or {}, ensure_ascii=False)
            except (TypeError, ValueError):
                args_str = "{}"
            tool_calls.append(SimpleNamespace(
                id=f"call_{uuid.uuid4().hex[:12]}",
                type="function",
                index=i,
                function=SimpleNamespace(name=str(fc["name"]), arguments=args_str),
            ))

    finish_reason = "tool_calls" if tool_calls else _map_gemini_finish_reason(
        str(cand.get("finishReason") or "")
    )

    usage_meta = inner.get("usageMetadata") or {}
    usage = SimpleNamespace(
        prompt_tokens=int(usage_meta.get("promptTokenCount") or 0),
        completion_tokens=int(usage_meta.get("candidatesTokenCount") or 0),
        total_tokens=int(usage_meta.get("totalTokenCount") or 0),
        prompt_tokens_details=SimpleNamespace(
            cached_tokens=int(usage_meta.get("cachedContentTokenCount") or 0),
        ),
    )

    message = SimpleNamespace(
        role="assistant",
        content="".join(text_pieces) if text_pieces else None,
        tool_calls=tool_calls or None,
        reasoning="".join(reasoning_pieces) or None,
        reasoning_content="".join(reasoning_pieces) or None,
        reasoning_details=None,
    )
    choice = SimpleNamespace(
        index=0,
        message=message,
        finish_reason=finish_reason,
    )
    return SimpleNamespace(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[choice],
        usage=usage,
    )


def _empty_response(model: str) -> SimpleNamespace:
    message = SimpleNamespace(
        role="assistant", content="", tool_calls=None,
        reasoning=None, reasoning_content=None, reasoning_details=None,
    )
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    usage = SimpleNamespace(
        prompt_tokens=0, completion_tokens=0, total_tokens=0,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
    )
    return SimpleNamespace(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[choice],
        usage=usage,
    )


def _map_gemini_finish_reason(reason: str) -> str:
    mapping = {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
        "OTHER": "stop",
    }
    return mapping.get(reason.upper(), "stop")


# =============================================================================
# Streaming SSE iterator
# =============================================================================

class _AntigravityStreamChunk(SimpleNamespace):
    """Mimics an OpenAI ChatCompletionChunk with .choices[0].delta."""
    pass


def _make_stream_chunk(
    *,
    model: str,
    content: str = "",
    tool_call_delta: Optional[Dict[str, Any]] = None,
    finish_reason: Optional[str] = None,
    reasoning: str = "",
) -> _AntigravityStreamChunk:
    delta_kwargs: Dict[str, Any] = {
        "role": "assistant",
        "content": None,
        "tool_calls": None,
        "reasoning": None,
        "reasoning_content": None,
    }
    if content:
        delta_kwargs["content"] = content
    if tool_call_delta is not None:
        delta_kwargs["tool_calls"] = [SimpleNamespace(
            index=tool_call_delta.get("index", 0),
            id=tool_call_delta.get("id") or f"call_{uuid.uuid4().hex[:12]}",
            type="function",
            function=SimpleNamespace(
                name=tool_call_delta.get("name") or "",
                arguments=tool_call_delta.get("arguments") or "",
            ),
        )]
    if reasoning:
        delta_kwargs["reasoning"] = reasoning
        delta_kwargs["reasoning_content"] = reasoning
    delta = SimpleNamespace(**delta_kwargs)
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return _AntigravityStreamChunk(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        object="chat.completion.chunk",
        created=int(time.time()),
        model=model,
        choices=[choice],
        usage=None,
    )


def _iter_sse_events(response: httpx.Response) -> Iterator[Dict[str, Any]]:
    """Parse Server-Sent Events from an httpx streaming response."""
    buffer = ""
    for chunk in response.iter_text():
        if not chunk:
            continue
        buffer += chunk
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")
            if not line:
                continue
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    return
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    logger.debug("Non-JSON SSE line: %s", data[:200])


def _translate_stream_event(
    event: Dict[str, Any],
    model: str,
    tool_call_counter: List[int],
) -> List[_AntigravityStreamChunk]:
    """Unwrap Antigravity envelope and emit OpenAI-shaped chunk(s)."""
    inner = event.get("response") if isinstance(event.get("response"), dict) else event
    candidates = inner.get("candidates") or []
    if not candidates:
        return []
    cand = candidates[0]
    if not isinstance(cand, dict):
        return []

    chunks: List[_AntigravityStreamChunk] = []

    content = cand.get("content") or {}
    parts = content.get("parts") if isinstance(content, dict) else []
    for part in parts or []:
        if not isinstance(part, dict):
            continue
        if part.get("thought") is True and isinstance(part.get("text"), str):
            chunks.append(_make_stream_chunk(
                model=model, reasoning=part["text"],
            ))
            continue
        if isinstance(part.get("text"), str) and part["text"]:
            chunks.append(_make_stream_chunk(model=model, content=part["text"]))
        fc = part.get("functionCall")
        if isinstance(fc, dict) and fc.get("name"):
            name = str(fc["name"])
            idx = tool_call_counter[0]
            tool_call_counter[0] += 1
            try:
                args_str = json.dumps(fc.get("args") or {}, ensure_ascii=False)
            except (TypeError, ValueError):
                args_str = "{}"
            chunks.append(_make_stream_chunk(
                model=model,
                tool_call_delta={
                    "index": idx,
                    "name": name,
                    "arguments": args_str,
                },
            ))

    finish_reason_raw = str(cand.get("finishReason") or "")
    if finish_reason_raw:
        mapped = _map_gemini_finish_reason(finish_reason_raw)
        if tool_call_counter[0] > 0:
            mapped = "tool_calls"
        chunks.append(_make_stream_chunk(model=model, finish_reason=mapped))
    return chunks


# =============================================================================
# AntigravityClient — OpenAI-compatible facade
# =============================================================================

MARKER_BASE_URL = "antigravity://google"


class _AntigravityChatCompletions:
    def __init__(self, client: "AntigravityClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _AntigravityChatNamespace:
    def __init__(self, client: "AntigravityClient"):
        self.completions = _AntigravityChatCompletions(client)


class AntigravityClient:
    """Minimal OpenAI-SDK-compatible facade over Antigravity v1internal."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        project_id: str = "",
        **_: Any,
    ):
        # `api_key` is the OAuth access token (refreshed on each call)
        self.api_key = api_key or "antigravity-oauth"
        self.base_url = base_url or MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._configured_project_id = project_id
        self.chat = _AntigravityChatNamespace(self)
        self.is_closed = False
        self._http = httpx.Client(timeout=httpx.Timeout(connect=15.0, read=600.0, write=30.0, pool=30.0))
        # Track the endpoint we're using (with fallback support)
        self._effective_endpoint = ANTIGRAVITY_DEFAULT_ENDPOINT
        self._endpoint_attempted = False

    def close(self) -> None:
        self.is_closed = True
        try:
            self._http.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _resolve_project_id(self) -> str:
        """Resolve the project ID to use for requests."""
        # First: configured project ID
        if self._configured_project_id:
            return self._configured_project_id
        # Second: stored in Antigravity OAuth credentials
        try:
            creds = antigravity_oauth.load_credentials()
            if creds and creds.project_id:
                return creds.project_id
        except Exception:
            pass
        # Third: default project from opencode-antigravity-auth
        return DEFAULT_PROJECT_ID

    def _send_request_with_fallback(
        self,
        wrapped: Dict[str, Any],
        headers: Dict[str, str],
        streaming: bool = False,
    ) -> Any:
        """Send a request with endpoint fallback support.

        Tries the primary antigravity endpoint first (daily sandbox), then
        falls back to autopush and prod if the primary returns an error.
        """
        last_error: Optional[Exception] = None

        endpoints_to_try = ANTIGRAVITY_FALLBACK_ENDPOINTS
        # Rotate so we start from the last successful endpoint
        if self._endpoint_attempted:
            start_idx = ANTIGRAVITY_FALLBACK_ENDPOINTS.index(self._effective_endpoint) if self._effective_endpoint in ANTIGRAVITY_FALLBACK_ENDPOINTS else 0
            endpoints_to_try = ANTIGRAVITY_FALLBACK_ENDPOINTS[start_idx:] + ANTIGRAVITY_FALLBACK_ENDPOINTS[:start_idx]

        for endpoint in endpoints_to_try:
            try:
                result = self._do_request(endpoint, wrapped, headers, streaming)
                self._effective_endpoint = endpoint
                self._endpoint_attempted = True
                return result
            except (CodeAssistError, httpx.HTTPError) as exc:
                last_error = exc
                logger.warning("Antigravity endpoint %s failed: %s", endpoint, exc)
                continue

        if last_error:
            raise last_error
        raise CodeAssistError("All Antigravity endpoints failed", code="antigravity_all_endpoints_failed")

    def _do_request(
        self,
        endpoint: str,
        wrapped: Dict[str, Any],
        headers: Dict[str, str],
        streaming: bool,
    ) -> Any:
        """Send a single request to a specific endpoint."""
        if streaming:
            url = f"{endpoint}/v1internal:streamGenerateContent?alt=sse"
            return self._stream_from_url(url, wrapped, headers)
        else:
            url = f"{endpoint}/v1internal:generateContent"
            response = self._http.post(url, json=wrapped, headers=headers)
            if response.status_code != 200:
                raise _antigravity_http_error(response)
            try:
                payload = response.json()
            except ValueError as exc:
                raise CodeAssistError(
                    f"Invalid JSON from Antigravity: {exc}",
                    code="antigravity_invalid_json",
                ) from exc
            return _translate_antigravity_response(payload, model=wrapped.get("model", ""))

    def _stream_from_url(
        self,
        url: str,
        wrapped: Dict[str, Any],
        headers: Dict[str, str],
    ) -> Iterator[_AntigravityStreamChunk]:
        """Generator that yields OpenAI-shaped streaming chunks from a URL."""
        stream_headers = dict(headers)
        stream_headers["Accept"] = "text/event-stream"

        def _generator() -> Iterator[_AntigravityStreamChunk]:
            try:
                with self._http.stream("POST", url, json=wrapped, headers=stream_headers) as response:
                    if response.status_code != 200:
                        response.read()
                        raise _antigravity_http_error(response)
                    tool_call_counter: List[int] = [0]
                    for event in _iter_sse_events(response):
                        for chunk in _translate_stream_event(event, wrapped.get("model", ""), tool_call_counter):
                            yield chunk
            except httpx.HTTPError as exc:
                raise CodeAssistError(
                    f"Antigravity streaming request failed: {exc}",
                    code="antigravity_stream_error",
                ) from exc

        return _generator()

    def _create_chat_completion(
        self,
        *,
        model: str = "gemini-2.5-flash",
        messages: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        tools: Any = None,
        tool_choice: Any = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Any = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Any = None,
        **_: Any,
    ) -> Any:
        try:
            access_token = antigravity_oauth.get_valid_access_token()
        except antigravity_oauth.AntigravityOAuthError as exc:
            raise CodeAssistError(
                f"Antigravity OAuth error: {exc}",
                code="antigravity_oauth_error",
            ) from exc

        project_id = self._resolve_project_id()

        thinking_config = None
        if isinstance(extra_body, dict):
            thinking_config = extra_body.get("thinking_config") or extra_body.get("thinkingConfig")

        inner = _build_antigravity_request(
            messages=messages or [],
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            thinking_config=thinking_config,
        )
        wrapped = _wrap_antigravity_request(
            project_id=project_id,
            model=model,
            inner_request=inner,
        )

        headers = _get_antigravity_headers(access_token)
        headers.update(self._default_headers)

        return self._send_request_with_fallback(wrapped, headers, streaming=stream)


# =============================================================================
# Error handling
# =============================================================================

def _antigravity_http_error(response: httpx.Response) -> CodeAssistError:
    """Translate an httpx response into a CodeAssistError with rich metadata."""
    status = response.status_code
    body_text = ""
    body_json: Dict[str, Any] = {}
    try:
        body_text = response.text
    except Exception:
        body_text = ""
    if body_text:
        try:
            parsed = json.loads(body_text)
            if isinstance(parsed, dict):
                body_json = parsed
        except (ValueError, TypeError):
            body_json = {}

    err_obj = body_json.get("error") if isinstance(body_json, dict) else None
    if not isinstance(err_obj, dict):
        err_obj = {}
    err_status = str(err_obj.get("status") or "").strip()
    err_message = str(err_obj.get("message") or "").strip()
    _raw_details = err_obj.get("details")
    err_details_list = _raw_details if isinstance(_raw_details, list) else []

    # Determine error sub-code
    error_code = "antigravity_api_error"
    if status == 429 or err_status == "RESOURCE_EXHAUSTED":
        error_code = "antigravity_rate_limit"
    elif status in (401, 403):
        error_code = "antigravity_auth_error"
    elif status in (404,):
        error_code = "antigravity_model_not_found"
    elif status >= 500:
        error_code = "antigravity_server_error"

    # Build a human-readable message
    reason = ""
    retry_delay = ""
    for detail in err_details_list:
        if isinstance(detail, dict):
            d_reason = detail.get("reason", "")
            if d_reason:
                reason = d_reason
            d_retry = (detail.get("retryDelay") or detail.get("metadata", {}).get("retryDelay", ""))
            if d_retry:
                retry_delay = d_retry
        if reason and retry_delay:
            break

    if err_message and reason:
        summary = f"{err_message} ({reason})"
    elif err_message:
        summary = err_message
    elif reason:
        summary = f"Antigravity error: {reason} (HTTP {status})"
    else:
        summary = f"Antigravity API error (HTTP {status})"

    if retry_delay:
        summary += f" [retry after {retry_delay}]"

    exc = CodeAssistError(summary, code=error_code, response=response)
    exc.status_code = status
    return exc
