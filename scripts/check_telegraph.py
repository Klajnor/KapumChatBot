"""Telegraph reachability probe — run it on the bot's server.

    python scripts/check_telegraph.py [--api https://api.telegra.ph]

Steps:
  1) GET  /            — is the host reachable/does TLS work at all;
  2) POST createAccount — can an account be created (the bot needs it because
                          anonymous createPage answers ACCESS_TOKEN_INVALID);
  3) POST createPage    — once with the new token, once anonymously, so the
                          real API policy is visible.

Raw responses are printed (truncated); tokens are masked everywhere except in
the final ready-to-paste ``TELEGRAPH_*`` lines — keep that output private.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from typing import Any
from uuid import uuid4

import aiohttp


def mask(text: str) -> str:
    text = re.sub(r'("(?:access|auth)_token"\s*:\s*")[^"]*(")', r"\1***\2", text)
    # auth_url grants editing of the account's pages — a secret as well
    return re.sub(r'("auth_url"\s*:\s*")[^"]*(")', r"\1***\2", text)


def unwrap(parsed: Any) -> Any:
    """Normalise the live API's {"ok":true,"result":{...}} wrapper to flat."""
    if (
        isinstance(parsed, dict)
        and parsed.get("ok")
        and isinstance(parsed.get("result"), dict)
    ):
        flat = {key: value for key, value in parsed.items() if key != "result"}
        flat.update(parsed["result"])
        return flat
    return parsed


async def _show(
    session: aiohttp.ClientSession, method: str, url: str, data: dict[str, str] | None
) -> dict[str, Any] | None:
    label = f"{method} {url.rsplit('/', 1)[-1]}"
    try:
        async with session.request(method, url, data=data) as resp:
            body = await resp.text()
            print(f"--- {label}: HTTP {resp.status}")
            print(mask(body[:400]))
            try:
                parsed = unwrap(json.loads(body))
                if isinstance(parsed, dict):
                    return parsed
                return {"ok": False, "error": "not an object"}
            except json.JSONDecodeError:
                return {"ok": False, "error": "non-JSON body (see above)"}
    except TimeoutError:
        print(f"--- {label}: TIMEOUT (blocked or unreachable?)")
    except aiohttp.ClientConnectorSSLError as exc:
        print(f"--- {label}: TLS ERROR — handshake refused: {exc}")
    except aiohttp.ClientConnectorError as exc:
        print(f"--- {label}: CONNECT ERROR (DNS/network): {exc.os_error or exc}")
    except aiohttp.ClientError as exc:
        print(f"--- {label}: CLIENT ERROR: {exc!r}")
    return None


async def probe(api: str, *, show_token: bool = False) -> int:
    timeout = aiohttp.ClientTimeout(total=20)
    # trust_env honours HTTPS_PROXY if you exported one
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        print(f"HTTPS_PROXY from env: {proxy or '(not set)'}")
        await _show(session, "GET", f"{api}/", None)

        short = f"diag{uuid4().hex[:10]}"
        account = await _show(
            session,
            "POST",
            f"{api}/createAccount",
            {"short_name": short, "author_name": "diagnostics"},
        )
        token = ""
        if account and account.get("ok"):
            token = str(account.get("access_token") or account.get("auth_token") or "")

        nodes = json.dumps(
            [{"tag": "h3", "children": ["probe"]},
             {"tag": "p", "children": ["$E=mc^2$"]}],
            ensure_ascii=False,
        )
        page_fields = {"title": "probe", "author_name": "diagnostics", "content": nodes}
        if token:
            await _show(
                session,
                "POST",
                f"{api}/createPage",
                {"short_name": short, "access_token": token, **page_fields},
            )
        await _show(session, "POST", f"{api}/createPage", page_fields)

        print()
        if token:
            print("createAccount works. Ready for .env (secret — keep private):")
            print(f"TELEGRAPH_SHORT_NAME={short}")
            if show_token:
                print(f"TELEGRAPH_ACCESS_TOKEN={token}")
            else:
                print("TELEGRAPH_ACCESS_TOKEN=<hidden: pass --show-token to reveal>")
            return 0
        print("No usable token. Reading the raw responses above:")
        print(" * TLS/CONNECT/TIMEOUT  -> telegra.ph is blocked from this host:")
        print("   run the bot via HTTPS_PROXY or move it to an unblocked network.")
        print(" * ok=true w/o token    -> API response shape changed; share the")
        print("   printed body — the bot masks/labels it the same way now.")
        print(" * ACCESS_TOKEN_INVALID for the anonymous createPage is expected")
        print("   (that is exactly why the bot needs an account).")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api", default="https://api.telegra.ph", help="Telegraph API base URL"
    )
    parser.add_argument(
        "--show-token",
        action="store_true",
        help="Print the access token in the final ready-to-paste lines "
             "(default: print only the short_name and a placeholder)",
    )
    args = parser.parse_args()
    return asyncio.run(probe(args.api.rstrip("/"), show_token=args.show_token))


if __name__ == "__main__":
    raise SystemExit(main())
