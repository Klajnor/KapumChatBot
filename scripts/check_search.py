"""Live web-search probe — run it on the bot's server.

    python scripts/check_search.py [--query "курс биткоина"] [--api URL]

Steps:
  1) config summary — is the tool even offered to the model;
  2) raw request to the provider endpoint — reachability, HTTP status,
     content type (an HTML body with status 200 is the classic "wrong URL"
     failure: the bot then logs "returned invalid JSON" and answers from
     memory);
  3) the bot's own ``search_web()`` — the exact path the chat handler uses,
     so a green result here means the tool really can return snippets.

The key is never printed in full. Exit code 0 = search works end to end.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import aiohttp

# ``python scripts/check_search.py`` puts scripts/ on sys.path, not the repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.config import config
from bot.services.ai_client import ai_client
from bot.services.search import (
    PROVIDERS,
    SEARCH_TOOL,
    _looks_like_block_page,
    _redact_secrets,
    _safe_host,
    _search_brave,
    _search_duckduckgo,
    _search_exa,
    _search_tavily,
    apply_search_hint,
    duckduckgo_available,
    format_search_results,
    provider_name,
    search_configured,
    search_web,
)


def mask(key: str) -> str:
    """Enough to tell keys apart, not enough to use one."""
    if len(key) <= 8:
        return "*" * len(key) if key else "(empty)"
    return f"{key[:4]}…{key[-2:]} ({len(key)} chars)"


def proxy_url() -> str:
    """First configured standard proxy URL (credentials are never printed)."""
    for name in (
        "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
        "ALL_PROXY", "all_proxy",
    ):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def endpoint() -> str:
    if config.search_api_url:
        return config.search_api_url
    provider = provider_name()
    if provider == "brave":
        return "https://api.search.brave.com/res/v1/web/search"
    if provider == "exa":
        return "https://api.exa.ai/search"
    if provider == "duckduckgo":
        return "https://html.duckduckgo.com/html/ (through ddgs)"
    return "https://api.tavily.com/search"


async def raw_probe(query: str) -> bool:
    """One hand-made request: is the body JSON at all?"""
    provider = provider_name()
    url = endpoint()
    headers = {"Accept": "application/json"}
    if provider == "brave":
        headers["X-Subscription-Token"] = config.search_api_key
        method, params, payload = "GET", {"q": query, "count": 1}, None
    elif provider == "exa":
        headers["x-api-key"] = config.search_api_key
        headers["Content-Type"] = "application/json"
        method, params = "POST", None
        payload = {
            "query": query,
            "type": "auto",
            "numResults": 1,
            "contents": {"highlights": True},
        }
    else:
        headers["Authorization"] = f"Bearer {config.search_api_key}"
        headers["Content-Type"] = "application/json"
        method = "POST"
        params = None
        payload = {
            "query": query,
            "max_results": 1,
            "search_depth": "basic",
            "include_answer": False,
        }

    timeout = aiohttp.ClientTimeout(total=config.search_timeout)
    print(f"--- {method} {url}")
    try:
        async with (
            aiohttp.ClientSession(timeout=timeout, trust_env=True) as session,
            session.request(
                method,
                url,
                headers=headers,
                params=params,
                json=payload,
                allow_redirects=False,
            ) as resp,
        ):
            body = await resp.text()
            ctype = resp.headers.get("Content-Type", "?")
            server = resp.headers.get("Server", "")
            print(f"    HTTP {resp.status} | Content-Type: {ctype}")
            print(f"    body: {_redact_secrets(body[:300], headers)!r}")
            if _looks_like_block_page(body, ctype, server):
                print(
                    "    CLOUDFLARE BLOCK: the provider refused this network or "
                    "region (Exa blocks Russia, Ukraine and others). Set "
                    "HTTPS_PROXY to an allowed network."
                )
                return False
            if resp.status != 200:
                return False
            try:
                json.loads(body)
            except json.JSONDecodeError:
                print(
                    "    NOT JSON — the URL is wrong (an HTML page answered "
                    "with 200). Set SEARCH_API_URL or fix the default."
                )
                return False
            return True
    except TimeoutError:
        print(f"    TIMEOUT after {config.search_timeout}s (blocked or unreachable?)")
    except aiohttp.ClientConnectorError as exc:
        print(
            "    CONNECT ERROR (DNS/network): "
            f"{_redact_secrets(str(exc.os_error or exc), headers)}"
        )
    except aiohttp.ClientError as exc:
        print(f"    CLIENT ERROR: {_redact_secrets(repr(exc), headers)}")
    return False


async def probe(query: str) -> int:
    provider = provider_name()
    print("=== configuration ===")
    print(f"ENABLE_WEB_SEARCH : {config.enable_web_search}")
    print(f"SEARCH_PROVIDER   : {provider or '(empty)'}")
    print(f"SEARCH_API_KEY    : {mask(config.search_api_key)}")
    if provider == "duckduckgo":
        print(f"ddgs installed    : {duckduckgo_available()}")
        print(f"ddgs backend      : {config.search_ddgs_backend or 'duckduckgo'}")
    print(f"endpoint          : {endpoint()}")
    proxy = proxy_url()
    print(f"proxy             : {_safe_host(proxy) if proxy else '(not set)'}")
    print(f"timeout / results : {config.search_timeout}s / {config.search_max_results}")
    print(f"max rounds        : {config.max_search_rounds}")

    if not config.enable_web_search:
        print("\nDISABLED: set ENABLE_WEB_SEARCH=true and restart the bot.")
        return 1
    if provider not in PROVIDERS:
        print(
            f"\nUNKNOWN PROVIDER {provider!r}: use 'duckduckgo', 'tavily', "
            "'brave' or 'exa'."
        )
        return 1
    if provider == "duckduckgo":
        if not duckduckgo_available():
            print("\nNO PACKAGE: pip install ddgs (or switch to tavily/brave/exa).")
            return 1
    elif not config.search_api_key:
        print(
            "\nNO KEY: set SEARCH_API_KEY (tvly-… for Tavily, BSA… for Brave, "
            "an Exa dashboard key for Exa)."
        )
        return 1
    print(f"\nsearch_configured() -> {search_configured()}")

    print("\n=== step 1: raw endpoint ===")
    raw_ok: bool | None
    if provider == "duckduckgo":
        # The provider parses HTML through ddgs; there is no JSON API to probe.
        print("    skipped: ddgs builds its own request to html.duckduckgo.com")
        raw_ok = None
    else:
        raw_ok = await raw_probe(query)

    print("\n=== step 2: provider parser ===")
    try:
        if provider == "duckduckgo":
            results = await _search_duckduckgo(query, 3)
        elif provider == "brave":
            results = await _search_brave(query, 3)
        elif provider == "exa":
            results = await _search_exa(query, 3)
        else:
            results = await _search_tavily(query, 3)
    except Exception as exc:  # noqa: BLE001 - diagnostics must not crash
        print(f"    parser raised: {exc!r}")
        results = []
    print(f"    parsed {len(results)} result(s)")
    if getattr(results, "error", ""):
        print(f"    provider failure: {results.error}")
    if provider == "duckduckgo" and not results:
        print(
            "    note: ddgs also returns an empty list when html.duckduckgo.com "
            "answers 202 (rate limit) instead of a result page"
        )

    print("\n=== step 3: search_web() (the handler's path) ===")
    live = await search_web(query, 3)
    print(f"    search_web() -> {len(live)} result(s)")
    if getattr(live, "error", ""):
        print(f"    provider failure: {live.error}")
    for index, item in enumerate(live, start=1):
        print(f"    [{index}] {item.title[:70]}")
        print(f"         {item.url}")
        print(f"         {item.snippet[:100]}…")

    print()
    if live:
        print(format_search_results(query, live)[:400])
        print("\nOK: the model will receive this as a tool result.")
        if raw_ok is False:
            print(
                "Note: the raw probe failed but parsing worked — check the "
                "content-type/URL override later."
            )
        return 0
    if getattr(live, "error", "") == "blocked":
        print(
            "FAIL: the provider blocked this network/region (Cloudflare). "
            "Set HTTPS_PROXY to an allowed network; the key itself is not "
            "the problem."
        )
    elif raw_ok is None:
        print(
            "FAIL: DuckDuckGo returned no hits — the query found nothing or the "
            "server IP is being throttled. Try SEARCH_DDGS_BACKEND=auto."
        )
    elif raw_ok:
        print(
            "FAIL: the API answered valid JSON but no results were parsed — "
            "response shape changed or the query found nothing."
        )
    else:
        print("FAIL: the endpoint never returned JSON. Fix the URL/key first.")
    return 1


async def model_probe(query: str) -> bool:
    """Does the model itself call the tool? The last link in the chain.

    Mirrors ``bot/handlers/chat.py``: same system hint, same tool schema, one
    non-streaming request. Costs a few tokens of the configured model.
    """
    print(f"\n=== step 4: model tool call ({config.default_model}) ===")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "Ты полезный ассистент."},
        {"role": "user", "content": query},
    ]
    apply_search_hint(messages)
    try:
        result = await ai_client.chat_with_tools(messages=messages, tools=[SEARCH_TOOL])
    except Exception as exc:  # noqa: BLE001 - diagnostics must not crash
        print(f"    model request failed: {exc!r}")
        return False
    calls = [call for call in result.tool_calls if call.name == "search_web"]
    if calls:
        print(f"    model asked for search_web: {calls[0].arguments}")
        return True
    reply = (result.content or "").strip().replace("\n", " ")[:300]
    print(f"    model answered WITHOUT a tool call: {reply!r}")
    return False


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # Windows consoles are cp1251
        except (AttributeError, OSError):
            pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="latest AI news", help="test query")
    parser.add_argument(
        "--ask",
        action="store_true",
        help="also ask the configured model and report whether it calls search_web",
    )
    args = parser.parse_args()
    return asyncio.run(run(args.query, ask=args.ask))


async def run(query: str, *, ask: bool) -> int:
    try:
        code = await probe(query)
        if code != 0 or not ask:
            return code
        return 0 if await model_probe(query) else 1
    finally:
        await ai_client.close()


if __name__ == "__main__":
    raise SystemExit(main())
