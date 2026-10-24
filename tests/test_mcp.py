"""Offline tests for the MCP stdio client and its config parsing.

No real Obsidian server is needed: a tiny Python stdio server is generated in
a temporary directory and speaks JSON-RPC with the client.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

# Keep local credentials out of the tests and never contact configured services.
os.environ.setdefault("BOT_TOKEN", "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi")
os.environ.setdefault("AI_API_KEY", "test-key")

from bot.config import _load_mcp_servers, config
from bot.services.mcp import (
    MCPClient,
    MCPServerConfig,
    _child_environment,
    _expand_env,
    _exposed_name,
    _format_tool_result,
    _normalize_schema,
    mcp_available,
)

# A minimal stdio MCP server: initialize, tools/list, tools/call.
FAKE_SERVER = textwrap.dedent(
    """
    import json
    import sys

    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")

    def reply(request_id, result):
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}) + "\\n")
        sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        message = json.loads(line)
        if "id" not in message:
            continue  # notification
        request_id = message["id"]
        method = message.get("method")
        if method == "initialize":
            reply(request_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "1.0"},
            })
        elif method == "tools/list":
            reply(request_id, {"tools": [
                {
                    "name": "echo",
                    "description": "Echo the text back",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                },
                {"name": "fail", "description": "Always fails", "inputSchema": {}},
            ]})
        elif method == "tools/call":
            params = message.get("params") or {}
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if name == "fail":
                reply(request_id, {
                    "content": [{"type": "text", "text": "boom"}],
                    "isError": True,
                })
            else:
                reply(request_id, {
                    "content": [{"type": "text", "text": "echo: " + str(arguments.get("text", ""))}]
                })
        else:
            sys.stdout.write(json.dumps({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": "unknown method"},
            }) + "\\n")
            sys.stdout.flush()
    """
)


class MCPConfigTests(unittest.TestCase):
    def test_loads_mcp_json_and_inline_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".mcp.json"
            path.write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "brain": {
                                "command": "python",
                                "args": ["mcp_server.py"],
                                "env": {"VAULT_PATH": "D:/Vault"},
                            },
                            "broken": {"args": []},
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertLogs("bot.config", level="WARNING"):
                servers = _load_mcp_servers("", str(path))
                # A directory is accepted too.
                from_dir = _load_mcp_servers("", tmp)
            self.assertEqual(set(servers), {"brain"})
            self.assertEqual(servers["brain"]["args"], ["mcp_server.py"])
            self.assertEqual(set(from_dir), {"brain"})

            with self.assertLogs("bot.config", level="WARNING"):
                overridden = _load_mcp_servers(
                    '{"brain": {"command": "py", "args": ["other.py"]}}',
                    str(path),
                )
            self.assertEqual(overridden["brain"]["command"], "py")
            self.assertEqual(overridden["brain"]["args"], ["other.py"])

    def test_invalid_file_does_not_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".mcp.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertLogs("bot.config", level="WARNING"):
                self.assertEqual(_load_mcp_servers("", str(path)), {})

    def test_expand_env(self) -> None:
        old = dict(os.environ)
        try:
            os.environ["MCP_TEST_VALUE"] = "vault"
            self.assertEqual(_expand_env("${MCP_TEST_VALUE}/notes"), "vault/notes")
            self.assertEqual(_expand_env("${MCP_MISSING:-fallback}"), "fallback")
            self.assertEqual(_expand_env("${MCP_MISSING}"), "")
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_exposed_names_are_safe(self) -> None:
        name = _exposed_name("My server!", "search notes")
        self.assertEqual(name, "mcp__My_server__search_notes")
        long_name = _exposed_name("server", "x" * 100)
        self.assertLessEqual(len(long_name), 64)
        self.assertRegex(long_name, r"^[A-Za-z0-9_-]+$")

    def test_schema_and_result_formatting(self) -> None:
        self.assertEqual(
            _normalize_schema({}),
            {"type": "object", "properties": {}},
        )
        self.assertEqual(
            _format_tool_result(
                {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
            ),
            "a\nb",
        )
        self.assertIn(
            "Ошибка",
            _format_tool_result(
                {"content": [{"type": "text", "text": "boom"}], "isError": True}
            ),
        )
        structured = _format_tool_result(
            {"content": [], "structuredContent": {"answer": 42}}
        )
        self.assertIn('"answer": 42', structured)


class MCPAccessTests(unittest.TestCase):
    def test_admin_only_by_default(self) -> None:
        saved = (
            config.enable_mcp,
            config.mcp_admin_only,
            config.mcp_servers,
            config.admin_ids,
        )
        try:
            config.enable_mcp = True
            config.mcp_admin_only = True
            config.mcp_servers = {"brain": {"command": "python", "args": []}}
            config.admin_ids = [42]
            self.assertTrue(mcp_available(42))
            self.assertFalse(mcp_available(43))
            self.assertFalse(mcp_available(None))

            config.mcp_admin_only = False
            self.assertTrue(mcp_available(43))
        finally:
            (
                config.enable_mcp,
                config.mcp_admin_only,
                config.mcp_servers,
                config.admin_ids,
            ) = saved


class MCPEnvironmentTests(unittest.TestCase):
    """An MCP server must never inherit the bot's credentials."""

    SECRETS = (
        "BOT_TOKEN",
        "AI_API_KEY",
        "QWEN_API_KEY",
        "DASHSCOPE_API_KEY",
        "SEARCH_API_KEY",
        "TAVILY_API_KEY",
        "TELEGRAPH_ACCESS_TOKEN",
        "DATABASE_URL",
    )

    def test_bot_secrets_are_not_inherited(self) -> None:
        with patch.dict(
            os.environ,
            {name: f"secret-{name.lower()}" for name in self.SECRETS},
        ):
            environment = _child_environment()

        for name in self.SECRETS:
            self.assertNotIn(name, environment, name)
            self.assertNotIn(name.lower(), environment, name)

    def test_allowlisted_and_explicit_variables_survive(self) -> None:
        with patch.dict(os.environ, {"PATH": "/usr/bin:/bin", "UNRELATED": "x"}):
            environment = _child_environment(
                {"VAULT_PATH": "/vault", "COMBINED": "${PATH}/extra"}
            )

        # PATH is needed to start the interpreter; unrelated variables are not
        # forwarded just because they happen to be set.
        self.assertEqual(environment.get("PATH"), "/usr/bin:/bin")
        self.assertNotIn("UNRELATED", environment)
        # Explicit env block wins and supports ${VAR} expansion.
        self.assertEqual(environment["VAULT_PATH"], "/vault")
        self.assertEqual(environment["COMBINED"], "/usr/bin:/bin/extra")

    def test_explicit_values_override_inherited_ones(self) -> None:
        with patch.dict(os.environ, {"LANG": "C"}):
            environment = _child_environment({"LANG": "ru_RU.UTF-8"})
        self.assertEqual(environment["LANG"], "ru_RU.UTF-8")

    def test_env_expansion_cannot_pull_bot_secrets(self) -> None:
        """Even an explicit ${VAR} in the config must not expose credentials."""
        with patch.dict(os.environ, {"BOT_TOKEN": "secret-token"}):
            with self.assertLogs("bot.services.mcp", level="WARNING"):
                self.assertEqual(_expand_env("${BOT_TOKEN}"), "")
            with self.assertLogs("bot.services.mcp", level="WARNING"):
                self.assertEqual(_expand_env("${bot_token}"), "")
        with (
            patch.dict(os.environ, {"AI_API_KEY": "secret-key"}),
            self.assertLogs("bot.services.mcp", level="WARNING"),
        ):
            # A default is harmless and still honoured.
            self.assertEqual(_expand_env("${AI_API_KEY:-fallback}"), "fallback")
        # Non-secret variables keep expanding as before.
        with patch.dict(os.environ, {"MCP_TEST_PATH": "/vault"}):
            self.assertEqual(_expand_env("${MCP_TEST_PATH}/notes"), "/vault/notes")


class MCPEnvironmentIsolationTests(unittest.IsolatedAsyncioTestCase):
    """End-to-end: a real started server sees the filtered environment only."""

    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dump = Path(self._tmp.name) / "env.json"
        script = Path(self._tmp.name) / "dump_server.py"
        # Snapshot the environment first, then behave like a normal MCP server.
        script.write_text(
            "import json, os\n"
            f"with open({str(self.dump)!r}, 'w', encoding='utf-8') as fh:\n"
            "    json.dump(dict(os.environ), fh)\n" + FAKE_SERVER,
            encoding="utf-8",
        )
        self.client = MCPClient(
            MCPServerConfig(
                name="dump",
                command=sys.executable,
                args=[str(script)],
                env={"MCP_ENV_DUMP": str(self.dump)},
            ),
            timeout=20,
        )

    async def asyncTearDown(self) -> None:
        await self.client.close()
        self._tmp.cleanup()

    async def test_started_server_cannot_read_bot_secrets(self) -> None:
        old = dict(os.environ)
        os.environ["BOT_TOKEN"] = "secret-bot-token"
        os.environ["AI_API_KEY"] = "secret-ai-key"
        os.environ["TELEGRAPH_ACCESS_TOKEN"] = "secret-telegraph-token"
        os.environ["DATABASE_URL"] = "postgresql+asyncpg://u:p@localhost/db"
        try:
            await self.client.list_tools()  # starts the subprocess
        finally:
            os.environ.clear()
            os.environ.update(old)

        seen = json.loads(self.dump.read_text(encoding="utf-8"))
        names = {name.upper() for name in seen}
        for name in MCPEnvironmentTests.SECRETS:
            self.assertNotIn(name, names)
        self.assertIn("PATH", names)
        if os.name == "nt":
            self.assertIn("SYSTEMROOT", names)
        # The explicit env block is still delivered.
        self.assertEqual(seen["MCP_ENV_DUMP"], str(self.dump))


class MCPClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        script = Path(self._tmp.name) / "fake_mcp_server.py"
        script.write_text(FAKE_SERVER, encoding="utf-8")
        self.client = MCPClient(
            MCPServerConfig(
                name="fake",
                command=sys.executable,
                args=[str(script)],
            ),
            timeout=20,
        )

    async def asyncTearDown(self) -> None:
        await self.client.close()
        self._tmp.cleanup()

    async def test_handshake_list_and_call(self) -> None:
        tools = await self.client.list_tools()
        self.assertEqual([tool.name for tool in tools], ["echo", "fail"])
        self.assertEqual(tools[0].exposed_name, "mcp__fake__echo")
        self.assertEqual(tools[0].openai_schema()["function"]["name"], "mcp__fake__echo")

        text = await self.client.call_tool("echo", {"text": "привет"})
        self.assertEqual(text, "echo: привет")

        failed = await self.client.call_tool("fail", {})
        self.assertIn("Ошибка", failed)
        self.assertIn("boom", failed)


if __name__ == "__main__":
    unittest.main()