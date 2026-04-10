"""Unit tests for oas_cli/providers/ — engine routing and provider behaviour."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from oas_cli.providers import EngineNotSupportedError, ProviderError, get_provider
from oas_cli.providers.anthropic_http import AnthropicProvider
from oas_cli.providers.codex import CodexProvider
from oas_cli.providers.custom import CustomProvider
from oas_cli.providers.openai_http import OpenAIProvider
from oas_cli.providers.registry import _OPENAI_COMPAT_DEFAULTS, invoke_intelligence

# ---------------------------------------------------------------------------
# get_provider routing
# ---------------------------------------------------------------------------


class TestGetProvider:
    def test_openai(self):
        assert isinstance(
            get_provider({"engine": "openai", "model": "gpt-4o"}), OpenAIProvider
        )

    def test_anthropic(self):
        assert isinstance(
            get_provider(
                {"engine": "anthropic", "model": "claude-3-5-sonnet-20241022"}
            ),
            AnthropicProvider,
        )

    def test_codex(self):
        assert isinstance(
            get_provider({"engine": "codex", "model": "gpt-4.1-codex"}), CodexProvider
        )

    @pytest.mark.parametrize("engine", ["grok", "xai", "local", "cortex"])
    def test_openai_compat_engines_route_to_openai_provider(self, engine):
        assert isinstance(
            get_provider({"engine": engine, "model": "any"}), OpenAIProvider
        )

    def test_custom_routes_to_custom_provider(self):
        assert isinstance(
            get_provider({"engine": "custom", "model": "any"}), CustomProvider
        )

    def test_unknown_engine_raises(self):
        with pytest.raises(EngineNotSupportedError):
            get_provider({"engine": "banana"})

    def test_default_engine_is_openai_when_missing(self):
        assert isinstance(get_provider({"model": "gpt-4o"}), OpenAIProvider)

    def test_engine_name_is_case_insensitive(self):
        assert isinstance(
            get_provider({"engine": "OPENAI", "model": "gpt-4o"}), OpenAIProvider
        )
        assert isinstance(
            get_provider({"engine": "Grok", "model": "grok-3-latest"}), OpenAIProvider
        )


# ---------------------------------------------------------------------------
# Engine defaults applied by invoke_intelligence
# ---------------------------------------------------------------------------


class TestEngineDefaults:
    """Verify _OPENAI_COMPAT_DEFAULTS are merged correctly (spec wins)."""

    def _captured_invoke(self, engine: str, extra: dict | None = None):
        """Call invoke_intelligence and capture the resolved config passed to the provider."""
        captured: dict = {}

        def fake_invoke(*, system, user, config):
            captured.update(config)
            return '{"ok": true}'

        base = {"engine": engine, "model": "test-model", **(extra or {})}
        with patch("oas_cli.providers.registry.OpenAIProvider") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.invoke.side_effect = fake_invoke
            mock_cls.return_value = mock_instance
            invoke_intelligence("sys", "usr", base)

        return mock_instance.invoke.call_args[1]["config"]

    def test_grok_default_endpoint(self):
        cfg = self._captured_invoke("grok")
        assert cfg["endpoint"] == "https://api.x.ai/v1"

    def test_grok_default_api_key_env(self):
        cfg = self._captured_invoke("grok")
        assert cfg["api_key_env"] == "XAI_API_KEY"

    def test_grok_default_model_in_defaults_table(self):
        assert _OPENAI_COMPAT_DEFAULTS["grok"]["model"] == "grok-3-latest"

    def test_spec_model_overrides_grok_default(self):
        cfg = self._captured_invoke("grok", {"model": "grok-2"})
        assert cfg["model"] == "grok-2"

    def test_xai_same_defaults_as_grok(self):
        cfg = self._captured_invoke("xai")
        assert cfg["endpoint"] == _OPENAI_COMPAT_DEFAULTS["grok"]["endpoint"]
        assert cfg["api_key_env"] == _OPENAI_COMPAT_DEFAULTS["grok"]["api_key_env"]

    def test_local_has_no_api_key_env(self):
        cfg = self._captured_invoke("local")
        assert cfg.get("api_key_env") is None

    def test_local_default_endpoint(self):
        cfg = self._captured_invoke("local")
        assert "localhost" in cfg["endpoint"]

    def test_local_endpoint_overridable(self):
        cfg = self._captured_invoke("local", {"endpoint": "http://localhost:1234/v1"})
        assert cfg["endpoint"] == "http://localhost:1234/v1"

    def test_cortex_uses_openai_key_env_by_default(self):
        cfg = self._captured_invoke("cortex")
        assert cfg["api_key_env"] == "OPENAI_API_KEY"

    def test_cortex_api_key_env_overridable(self):
        cfg = self._captured_invoke("cortex", {"api_key_env": "CORTEX_API_KEY"})
        assert cfg["api_key_env"] == "CORTEX_API_KEY"


# ---------------------------------------------------------------------------
# OpenAIProvider — auth header behaviour
# ---------------------------------------------------------------------------


class TestOpenAIProviderAuth:
    def _make_provider_call(self, *, api_key_env: str | None, env_vars: dict):
        """Run OpenAIProvider.invoke with a mocked HTTP call and return the request headers."""
        captured_headers: dict = {}

        def fake_http_post(url, payload, headers, timeout=60):
            captured_headers.update(headers)
            return '{"choices": [{"message": {"content": "hi"}}]}'

        with patch.dict("os.environ", env_vars, clear=False):
            with patch(
                "oas_cli.providers.openai_http._http_post", side_effect=fake_http_post
            ):
                provider = OpenAIProvider()
                config: dict = {
                    "model": "gpt-4o",
                    "endpoint": "https://api.openai.com/v1",
                }
                if api_key_env is not None:
                    config["api_key_env"] = api_key_env
                provider.invoke(system="sys", user="usr", config=config)

        return captured_headers

    def test_auth_header_present_when_key_set(self):
        headers = self._make_provider_call(
            api_key_env="OPENAI_API_KEY",
            env_vars={"OPENAI_API_KEY": "sk-test"},
        )
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer sk-test"

    def test_no_auth_header_when_api_key_env_is_none(self):
        """Local/anonymous endpoints — explicit api_key_env=None skips auth entirely."""
        captured_headers: dict = {}

        def fake_http_post(url, payload, headers, timeout=60):
            captured_headers.update(headers)
            return '{"choices": [{"message": {"content": "hi"}}]}'

        with patch(
            "oas_cli.providers.openai_http._http_post", side_effect=fake_http_post
        ):
            provider = OpenAIProvider()
            provider.invoke(
                system="sys",
                user="usr",
                config={
                    "model": "llama3.2",
                    "endpoint": "http://localhost:11434/v1",
                    "api_key_env": None,  # explicit None — local server, no auth
                },
            )

        assert "Authorization" not in captured_headers

    def test_error_when_key_env_set_but_missing(self):
        import os

        os.environ.pop("MISSING_KEY_ENV_XYZ", None)
        with pytest.raises(ProviderError, match="MISSING_KEY_ENV_XYZ"):
            provider = OpenAIProvider()
            provider.invoke(
                system="s",
                user="u",
                config={"model": "gpt-4o", "api_key_env": "MISSING_KEY_ENV_XYZ"},
            )


# ---------------------------------------------------------------------------
# CustomProvider
# ---------------------------------------------------------------------------


class TestCustomProvider:
    def test_no_module_falls_back_to_openai_http(self):
        # The import is local inside invoke(), so patch at the source module.
        with patch("oas_cli.providers.openai_http.OpenAIProvider") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.invoke.return_value = '{"result": "ok"}'
            mock_cls.return_value = mock_instance

            provider = CustomProvider()
            result = provider.invoke(
                system="s",
                user="u",
                config={
                    "engine": "custom",
                    "model": "my-model",
                    "endpoint": "http://x/v1",
                },
            )

        assert result == '{"result": "ok"}'

    def test_module_class_is_called(self, tmp_path, monkeypatch):
        """A user-supplied class in sys.path is instantiated and run."""

        # Write a tiny router class to a temp file on sys.path
        router_file = tmp_path / "my_router.py"
        router_file.write_text(
            "import json\n"
            "class MyRouter:\n"
            "    def __init__(self, endpoint, model, config): self.called = True\n"
            "    def run(self, prompt, **kwargs): return json.dumps({'echo': prompt})\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        provider = CustomProvider()
        result = provider.invoke(
            system="hello",
            user="world",
            config={"engine": "custom", "model": "m", "module": "my_router.MyRouter"},
        )
        data = json.loads(result)
        assert "echo" in data
        assert "hello" in data["echo"]
        assert "world" in data["echo"]

    def test_bad_module_raises_provider_error(self):
        provider = CustomProvider()
        with pytest.raises(ProviderError, match="nonexistent_module_xyz"):
            provider.invoke(
                system="s",
                user="u",
                config={"module": "nonexistent_module_xyz.MyClass"},
            )

    def test_missing_dot_in_module_path_raises(self):
        provider = CustomProvider()
        with pytest.raises(ProviderError, match="module.ClassName"):
            provider.invoke(system="s", user="u", config={"module": "NoDotsHere"})

    def test_class_not_found_raises(self, tmp_path, monkeypatch):

        mod_file = tmp_path / "empty_mod.py"
        mod_file.write_text("# nothing here\n")
        monkeypatch.syspath_prepend(str(tmp_path))

        provider = CustomProvider()
        with pytest.raises(ProviderError, match="GhostClass"):
            provider.invoke(
                system="s",
                user="u",
                config={"module": "empty_mod.GhostClass"},
            )
