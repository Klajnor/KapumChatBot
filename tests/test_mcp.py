"""Offline tests for the MCP stdio client and its config parsing.

No real Obsidian server is needed: a tiny Python stdio server is generated in
a temporary directory and speaks JSON-RPC with the client.
"""
from __future__ import annotations

import hashlib
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
    MCP_SYSTEM_HINT,
    MCPClient,
    MCPError,
    MCPServerConfig,
    _child_environment,
    _expand_env,
    _exposed_name,
    _format_tool_result,
    _normalize_schema,
    apply_mcp_hint,
    mcp_available,
    mcp_write_available,
    remove_mcp_hint,
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
                servers = _load_mcp_servers("", str(path), allow_unpinned_config=True)
                # A directory is accepted too.
                from_dir = _load_mcp_servers(
                    "", tmp, allow_unpinned_config=True
                )
            self.assertEqual(set(servers), {"brain"})
            self.assertEqual(servers["brain"]["args"], ["mcp_server.py"])
            self.assertEqual(set(from_dir), {"brain"})

            with self.assertLogs("bot.config", level="WARNING"):
                overridden = _load_mcp_servers(
                    '{"brain": {"command": "py", "args": ["other.py"]}}',
                    str(path),
                    allow_unpinned_config=True,
                )
            self.assertEqual(overridden["brain"]["command"], "py")
            self.assertEqual(overridden["brain"]["args"], ["other.py"])

    def test_invalid_file_does_not_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".mcp.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertLogs("bot.config", level="WARNING"):
                self.assertEqual(
                    _load_mcp_servers(
                        "", str(path), allow_unpinned_config=True
                    ),
                    {},
                )

    def test_unpinned_external_config_is_ignored_by_default(self) -> None:
        """A config another program can rewrite must not become code execution."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".mcp.json"
            path.write_text(
                json.dumps(
                    {"mcpServers": {"brain": {"command": "/bin/true", "args": []}}}
                ),
                encoding="utf-8",
            )
            # tempfile is outside the working directory in the test runner.
            with self.assertLogs("bot.config", level="ERROR"):
                servers = _load_mcp_servers("", str(path))
            self.assertEqual(servers, {})
            # ...unless the operator explicitly trusts unpinned configs.
            self.assertEqual(
                set(_load_mcp_servers("", str(path), allow_unpinned_config=True)),
                {"brain"},
            )

    def test_config_hash_pin_accepts_matching_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".mcp.json"
            raw = json.dumps(
                {"mcpServers": {"brain": {"command": "python", "args": []}}}
            ).encode("utf-8")
            path.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            servers = _load_mcp_servers("", str(path), digest)
        self.assertEqual(set(servers), {"brain"})

    def test_config_hash_pin_rejects_a_modified_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".mcp.json"
            path.write_text(
                json.dumps(
                    {"mcpServers": {"brain": {"command": "python", "args": []}}}
                ),
                encoding="utf-8",
            )
            with self.assertLogs("bot.config", level="ERROR"):
                servers = _load_mcp_servers("", str(path), "0" * 64)
        self.assertEqual(servers, {})

    def test_inline_servers_survive_a_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".mcp.json"
            path.write_text('{"mcpServers": {}}', encoding="utf-8")
            with self.assertLogs("bot.config", level="ERROR"):
                servers = _load_mcp_servers(
                    '{"inline": {"command": "/bin/true", "args": []}}',
                    str(path),
                    "0" * 64,
                )
        self.assertEqual(set(servers), {"inline"})

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


class MCPAllowlistTests(unittest.TestCase):
    def test_default_allows_only_the_read_only_second_brain_tools(self) -> None:
        server = MCPServerConfig(name="brain", command="x")
        for name in ("search_notes", "get_chunk", "get_note", "list_notes"):
            self.assertTrue(server.permits(name), name)
        for name in ("write_note", "delete_note", "update_note", "run_shell"):
            self.assertFalse(server.permits(name), name)

    def test_env_setting_and_per_server_override(self) -> None:
        saved = config.mcp_allowed_tools
        try:
            config.mcp_allowed_tools = "a, b"
            self.assertTrue(MCPServerConfig(name="s", command="x").permits("b"))
            self.assertFalse(MCPServerConfig(name="s", command="x").permits("c"))
            own = MCPServerConfig(name="s", command="x", allowed_tools=["c"])
            self.assertTrue(own.permits("c"))
            self.assertFalse(own.permits("a"))
            config.mcp_allowed_tools = "*"
            self.assertTrue(MCPServerConfig(name="s", command="x").permits("anything"))
            self.assertFalse(
                MCPServerConfig(name="s", command="x", allowed_tools=[]).permits("a")
            )
        finally:
            config.mcp_allowed_tools = saved

    def test_allowed_tools_is_read_from_mcp_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".mcp.json"
            path.write_text(
                json.dumps({"mcpServers": {
                    "brain": {"command": "/x/python", "allowed_tools": ["get_note", " "]},
                    "plain": {"command": "/x/python"},
                    "bad": {"command": "/x/python", "allowed_tools": "get_note"},
                }}),
                encoding="utf-8",
            )
            servers = _load_mcp_servers("", str(path), allow_unpinned_config=True)
        self.assertEqual(servers["brain"]["allowed_tools"], ["get_note"])
        self.assertNotIn("allowed_tools", servers["plain"])
        self.assertNotIn("allowed_tools", servers["bad"])  # invalid → default


class MCPAccessTests(unittest.TestCase):
    def test_read_for_everyone_write_only_for_owner(self) -> None:
        saved = (
            config.enable_mcp,
            config.mcp_owner_only,
            config.mcp_servers,
            config.owner_ids,
            config.extra_admin_ids,
        )
        try:
            config.enable_mcp = True
            config.mcp_owner_only = True
            config.mcp_servers = {"brain": {"command": "python", "args": []}}
            config.owner_ids = [42]
            config.extra_admin_ids = {44}  # promoted admin: not an owner
            self.assertTrue(mcp_available(42))
            self.assertTrue(mcp_available(43))
            self.assertTrue(mcp_available(44))
            self.assertFalse(mcp_available(None))
            self.assertTrue(mcp_write_available(42))
            self.assertFalse(mcp_write_available(43))
            self.assertFalse(mcp_write_available(44))
            self.assertFalse(mcp_write_available(None))

            config.mcp_owner_only = False
            self.assertTrue(mcp_available(43))
        finally:
            (
                config.enable_mcp,
                config.mcp_owner_only,
                config.mcp_servers,
                config.owner_ids,
                config.extra_admin_ids,
            ) = saved

    def test_mcp_hint_is_applied_once_and_removable(self) -> None:
        messages = [{"role": "system", "content": "base"}]
        apply_mcp_hint(messages)
        apply_mcp_hint(messages)
        self.assertEqual(messages[0]["content"].count(MCP_SYSTEM_HINT), 1)
        remove_mcp_hint(messages)
        self.assertEqual(messages[0]["content"], "base")


class MCPEnvironmentTests(unittest.TestCase):
    """An MCP server must never inherit the bot's credentials."""

    SECRETS = (
        "BOT_TOKEN",
        "AI_API_KEY",
        "QWEN_API_KEY",
        "DASHSCOPE_API_KEY",
        "SEARCH_API_KEY",
        "TAVILY_API_KEY",
        "EXA_API_KEY",
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
        with (
            patch.dict(os.environ, {"EXA_API_KEY": "secret-exa-key"}),
            self.assertLogs("bot.services.mcp", level="WARNING"),
        ):
            self.assertEqual(_expand_env("${EXA_API_KEY}"), "")
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
        os.environ["EXA_API_KEY"] = "secret-exa-key"
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
                allowed_tools=["*"],  # the fake server's tools are not in the default list
            ),
            timeout=20,
        )

    async def asyncTearDown(self) -> None:
        await self.client.close()
        self._tmp.cleanup()

    async def test_allowlist_hides_and_blocks_other_tools(self) -> None:
        script = Path(self._tmp.name) / "fake_mcp_server.py"
        client = MCPClient(
            MCPServerConfig(
                name="fake", command=sys.executable, args=[str(script)],
                allowed_tools=["echo"],
            ),
            timeout=20,
        )
        try:
            tools = await client.list_tools()
            self.assertEqual([tool.name for tool in tools], ["echo"])
            self.assertEqual(client.hidden_tools, ["fail"])
            self.assertEqual(await client.call_tool("echo", {"text": "x"}), "echo: x")
            with self.assertRaises(MCPError):
                await client.call_tool("fail", {})  # refused before the server
        finally:
            await client.close()

    async def test_default_allowlist_is_read_only_set(self) -> None:
        script = Path(self._tmp.name) / "fake_mcp_server.py"
        client = MCPClient(
            MCPServerConfig(name="fake", command=sys.executable, args=[str(script)]),
            timeout=20,
        )
        try:
            self.assertEqual(await client.list_tools(), [])  # echo/fail not listed
            self.assertEqual(client.hidden_tools, ["echo", "fail"])
        finally:
            await client.close()

    async def test_relative_command_is_refused_by_default(self) -> None:
        """A bare ``python`` would be resolved through PATH: fail closed."""
        client = MCPClient(
            MCPServerConfig(name="relative", command="python", args=[]),
            timeout=5,
        )
        saved = config.mcp_allow_path_command
        try:
            config.mcp_allow_path_command = False
            with self.assertRaises(MCPError):
                await client.start()
        finally:
            config.mcp_allow_path_command = saved
            await client.close()

    async def test_absolute_command_still_starts(self) -> None:
        # The regular handshake test uses ``sys.executable``, an absolute path;
        # this pins the invariant explicitly.
        self.assertTrue(os.path.isabs(sys.executable))
        tools = await self.client.list_tools()
        self.assertIn("echo", [tool.name for tool in tools])

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
