"""Tests for prompt resolution order in runner._build_prompt and BCE contracts."""

import pytest

from oas_cli.runner import (
    CONTRACTS_ENABLED,
    OARunError,
    _build_prompt,
    _merge_contracts,
    _resolve_contract,
    run_task_from_spec,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spec(
    *,
    global_system: str | None = None,
    global_user: str | None = None,
    style_b_system: str | None = None,
    style_b_user: str | None = None,
    style_a_system: str | None = None,
    style_a_user: str | None = None,
) -> dict:
    """Build a minimal spec dict with the requested prompt configuration."""
    task_prompts: dict = {}
    if style_a_system is not None:
        task_prompts["system"] = style_a_system
    if style_a_user is not None:
        task_prompts["user"] = style_a_user

    task_def: dict = {
        "description": "test task",
        "output": {"type": "object", "properties": {"result": {"type": "string"}}},
    }
    if task_prompts:
        task_def["prompts"] = task_prompts

    prompts: dict = {}
    if global_system is not None:
        prompts["system"] = global_system
    if global_user is not None:
        prompts["user"] = global_user
    if style_b_system is not None or style_b_user is not None:
        b: dict = {}
        if style_b_system is not None:
            b["system"] = style_b_system
        if style_b_user is not None:
            b["user"] = style_b_user
        prompts["mytask"] = b

    return {
        "open_agent_spec": "1.3.0",
        "agent": {"name": "test-agent", "description": "test"},
        "intelligence": {"type": "llm", "engine": "openai", "model": "gpt-4o"},
        "tasks": {"mytask": task_def},
        "prompts": prompts,
    }


# ---------------------------------------------------------------------------
# Layer 4 — global fallback
# ---------------------------------------------------------------------------


class TestGlobalFallback:
    def test_uses_global_system(self):
        spec = _spec(global_system="global sys", global_user="{{ name }}")
        result = _build_prompt(spec, "mytask", {"name": "Alice"})
        assert result.startswith("global sys")
        assert "Alice" in result

    def test_uses_global_user_template(self):
        spec = _spec(global_system="sys", global_user="hello {{ name }}")
        result = _build_prompt(spec, "mytask", {"name": "Bob"})
        assert "hello Bob" in result


# ---------------------------------------------------------------------------
# Layer 3 — Style B  (prompts.<task_name>)
# ---------------------------------------------------------------------------


class TestStyleB:
    def test_style_b_overrides_global_system(self):
        spec = _spec(
            global_system="global sys",
            global_user="{{ q }}",
            style_b_system="style-b sys",
        )
        result = _build_prompt(spec, "mytask", {"q": "hi"})
        assert "style-b sys" in result
        assert "global sys" not in result

    def test_style_b_overrides_global_user(self):
        spec = _spec(
            global_system="sys",
            global_user="global {{ q }}",
            style_b_user="style-b {{ q }}",
        )
        result = _build_prompt(spec, "mytask", {"q": "hi"})
        assert "style-b hi" in result
        assert "global hi" not in result


# ---------------------------------------------------------------------------
# Layer 2 — Style A  (tasks.<name>.prompts)
# ---------------------------------------------------------------------------


class TestStyleA:
    def test_style_a_overrides_global(self):
        spec = _spec(
            global_system="global sys",
            global_user="{{ q }}",
            style_a_system="inline sys",
            style_a_user="inline {{ q }}",
        )
        result = _build_prompt(spec, "mytask", {"q": "test"})
        assert "inline sys" in result
        assert "inline test" in result
        assert "global sys" not in result

    def test_style_a_overrides_style_b(self):
        spec = _spec(
            global_system="global sys",
            global_user="{{ q }}",
            style_b_system="style-b sys",
            style_b_user="style-b {{ q }}",
            style_a_system="inline sys",
            style_a_user="inline {{ q }}",
        )
        result = _build_prompt(spec, "mytask", {"q": "x"})
        assert "inline sys" in result
        assert "inline x" in result
        assert "style-b" not in result
        assert "global sys" not in result

    def test_style_a_partial_fallthrough(self):
        """Style A system with no Style A user should fall back to global user."""
        spec = _spec(
            global_system="global sys",
            global_user="global {{ q }}",
            style_a_system="inline sys",
        )
        result = _build_prompt(spec, "mytask", {"q": "y"})
        assert "inline sys" in result
        assert "global y" in result


# ---------------------------------------------------------------------------
# Layer 1 — CLI overrides
# ---------------------------------------------------------------------------


class TestCliOverrides:
    def test_override_system_replaces_all(self):
        spec = _spec(
            global_system="global sys",
            style_a_system="inline sys",
            global_user="{{ q }}",
        )
        result = _build_prompt(spec, "mytask", {"q": "z"}, override_system="cli sys")
        assert "cli sys" in result
        assert "global sys" not in result
        assert "inline sys" not in result

    def test_override_user_replaces_all(self):
        spec = _spec(
            global_system="sys",
            global_user="global {{ q }}",
            style_a_user="inline {{ q }}",
        )
        result = _build_prompt(spec, "mytask", {"q": "z"}, override_user="cli {{ q }}")
        assert "cli z" in result
        assert "global z" not in result
        assert "inline z" not in result

    def test_both_overrides(self):
        spec = _spec(global_system="global sys", global_user="{{ q }}")
        result = _build_prompt(
            spec,
            "mytask",
            {"q": "ignored"},
            override_system="cli sys",
            override_user="cli user",
        )
        assert result == "cli sys\n\ncli user"

    def test_override_system_none_does_not_replace(self):
        """Passing override_system=None must not suppress the spec prompt."""
        spec = _spec(global_system="global sys", global_user="{{ q }}")
        result = _build_prompt(spec, "mytask", {"q": "hi"}, override_system=None)
        assert "global sys" in result


# ---------------------------------------------------------------------------
# Template interpolation
# ---------------------------------------------------------------------------


class TestTemplateInterpolation:
    def test_double_brace_placeholder(self):
        spec = _spec(global_system="sys", global_user="Hello {{ name }}")
        result = _build_prompt(spec, "mytask", {"name": "Alice"})
        assert "Hello Alice" in result

    def test_input_dot_prefix(self):
        spec = _spec(global_system="sys", global_user="Hi {{ input.name }}")
        result = _build_prompt(spec, "mytask", {"name": "Bob"})
        assert "Hi Bob" in result

    def test_multiple_placeholders(self):
        spec = _spec(global_system="sys", global_user="{{ a }} and {{ b }}")
        result = _build_prompt(spec, "mytask", {"a": "foo", "b": "bar"})
        assert "foo and bar" in result


# ---------------------------------------------------------------------------
# run_task_from_spec: overrides flow end-to-end (no real LLM call)
# ---------------------------------------------------------------------------


class TestRunTaskFromSpecOverrides:
    """Verify overrides reach _build_prompt through run_task_from_spec.

    We mock invoke_intelligence to capture the prompt without a real API call.
    """

    def test_override_system_reaches_invoke(self, monkeypatch):
        captured: dict = {}

        def fake_invoke(system: str, user: str, config: dict) -> str:
            captured["prompt"] = f"{system}\n\n{user}"
            return '{"result": "ok"}'

        monkeypatch.setattr("oas_cli.runner.invoke_intelligence", fake_invoke)

        spec = _spec(global_system="original sys", global_user="{{ q }}")
        run_task_from_spec(
            spec,
            task_name="mytask",
            input_data={"q": "hi"},
            override_system="injected sys",
        )
        assert "injected sys" in captured["prompt"]
        assert "original sys" not in captured["prompt"]

    def test_no_override_uses_spec_prompt(self, monkeypatch):
        captured: dict = {}

        def fake_invoke(system: str, user: str, config: dict) -> str:
            captured["prompt"] = f"{system}\n\n{user}"
            return '{"result": "ok"}'

        monkeypatch.setattr("oas_cli.runner.invoke_intelligence", fake_invoke)

        spec = _spec(global_system="spec sys", global_user="{{ q }}")
        run_task_from_spec(spec, task_name="mytask", input_data={"q": "hi"})
        assert "spec sys" in captured["prompt"]


# ---------------------------------------------------------------------------
# Gap 2 — response_format: text
# ---------------------------------------------------------------------------


def _text_spec(response_format: str = "text") -> dict:
    """Minimal spec with response_format on the task."""
    return {
        "open_agent_spec": "1.3.0",
        "agent": {"name": "ta", "description": "ta"},
        "intelligence": {"type": "llm", "engine": "openai", "model": "gpt-4o"},
        "tasks": {
            "prose": {
                "description": "write prose",
                "response_format": response_format,
                "output": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            }
        },
        "prompts": {"system": "you write prose", "user": "{{ topic }}"},
    }


class TestResponseFormatText:
    def test_text_mode_returns_raw_string(self, monkeypatch):
        monkeypatch.setattr(
            "oas_cli.runner.invoke_intelligence",
            lambda s, u, c: "This is plain prose output.",
        )
        result = run_task_from_spec(
            _text_spec(), task_name="prose", input_data={"topic": "cats"}
        )
        assert result["output"] == "This is plain prose output."

    def test_text_mode_skips_json_parsing(self, monkeypatch):
        """Raw output that is NOT valid JSON must be returned unchanged in text mode."""
        monkeypatch.setattr(
            "oas_cli.runner.invoke_intelligence",
            lambda s, u, c: "Not JSON at all { broken",
        )
        result = run_task_from_spec(
            _text_spec(), task_name="prose", input_data={"topic": "x"}
        )
        assert result["output"] == "Not JSON at all { broken"

    def test_text_mode_does_not_parse_fenced_json(self, monkeypatch):
        """Even fenced JSON should be returned as-is in text mode (no fence stripping)."""
        raw = '```json\n{"key": "val"}\n```'
        monkeypatch.setattr("oas_cli.runner.invoke_intelligence", lambda s, u, c: raw)
        result = run_task_from_spec(
            _text_spec(), task_name="prose", input_data={"topic": "x"}
        )
        assert result["output"] == raw

    def test_json_mode_still_parses(self, monkeypatch):
        """response_format: json (default) still parses JSON as before."""
        monkeypatch.setattr(
            "oas_cli.runner.invoke_intelligence",
            lambda s, u, c: '{"result": "ok"}',
        )
        result = run_task_from_spec(
            _text_spec("json"), task_name="prose", input_data={"topic": "x"}
        )
        assert result["output"] == {"result": "ok"}

    def test_json_mode_default_when_field_absent(self, monkeypatch):
        """A task with no response_format behaves as json mode."""
        spec = _spec(global_system="sys", global_user="{{ q }}")
        monkeypatch.setattr(
            "oas_cli.runner.invoke_intelligence",
            lambda s, u, c: '{"result": "fine"}',
        )
        result = run_task_from_spec(spec, task_name="mytask", input_data={"q": "hi"})
        assert result["output"] == {"result": "fine"}


# ---------------------------------------------------------------------------
# Gap 5 — OARunError.to_dict()
# ---------------------------------------------------------------------------


class TestOARunError:
    def test_to_dict_includes_required_fields(self):
        err = OARunError(
            "Task 'x' not found", code="TASK_NOT_FOUND", stage="routing", task="x"
        )
        d = err.to_dict()
        assert d["error"] == "Task 'x' not found"
        assert d["code"] == "TASK_NOT_FOUND"
        assert d["stage"] == "routing"
        assert d["task"] == "x"

    def test_to_dict_omits_task_when_none(self):
        err = OARunError("Spec load failed", code="SPEC_LOAD_ERROR", stage="load")
        d = err.to_dict()
        assert "task" not in d

    def test_task_not_found_raises_oa_run_error(self):
        spec = _spec(global_system="s", global_user="{{ q }}")
        with pytest.raises(OARunError) as exc_info:
            run_task_from_spec(spec, task_name="missing", input_data={})
        err = exc_info.value
        assert err.code == "TASK_NOT_FOUND"
        assert err.stage == "routing"
        assert err.task == "missing"

    def test_invoke_error_wrapped_in_oa_run_error(self, monkeypatch):
        def bad_invoke(s, u, c):
            raise RuntimeError("network timeout")

        monkeypatch.setattr("oas_cli.runner.invoke_intelligence", bad_invoke)
        spec = _spec(global_system="s", global_user="{{ q }}")
        with pytest.raises(OARunError) as exc_info:
            run_task_from_spec(spec, task_name="mytask", input_data={"q": "x"})
        err = exc_info.value
        assert err.code == "RUN_ERROR"
        assert err.stage == "run"
        assert "network timeout" in str(err)


# ---------------------------------------------------------------------------
# Gap 6 — depends_on linear chaining
# ---------------------------------------------------------------------------


def _chain_spec(*, add_text_format: bool = False) -> dict:
    """Spec with two tasks where summarize depends_on extract."""
    extract_task: dict = {
        "description": "extract facts",
        "output": {
            "type": "object",
            "properties": {"facts": {"type": "string"}},
            "required": ["facts"],
        },
    }
    summarize_task: dict = {
        "description": "summarize facts",
        "depends_on": ["extract"],
        "output": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
        },
    }
    if add_text_format:
        summarize_task["response_format"] = "text"
    return {
        "open_agent_spec": "1.3.0",
        "agent": {"name": "chain-agent", "description": "test"},
        "intelligence": {"type": "llm", "engine": "openai", "model": "gpt-4o"},
        "tasks": {"extract": extract_task, "summarize": summarize_task},
        "prompts": {"system": "you summarize", "user": "{{ facts }}"},
    }


class TestDependsOn:
    def test_chain_executes_dependency_first(self, monkeypatch):
        calls: list[str] = []

        def fake_invoke(system: str, user: str, config: dict) -> str:
            if "extract" in user or calls == []:
                calls.append("extract")
                return '{"facts": "the sky is blue"}'
            calls.append("summarize")
            return '{"summary": "sky=blue"}'

        monkeypatch.setattr("oas_cli.runner.invoke_intelligence", fake_invoke)
        spec = _chain_spec()
        result = run_task_from_spec(spec, task_name="summarize", input_data={})
        assert result["task"] == "summarize"
        assert "chain" in result
        assert "extract" in result["chain"]

    def test_dep_output_merged_into_next_input(self, monkeypatch):
        """Facts from extract must appear in summarize's prompt."""
        prompts_seen: list[str] = []

        def fake_invoke(system: str, user: str, config: dict) -> str:
            prompts_seen.append(f"{system}\n\n{user}")
            if len(prompts_seen) == 1:
                return '{"facts": "the sky is blue"}'
            return '{"summary": "sky"}'

        monkeypatch.setattr("oas_cli.runner.invoke_intelligence", fake_invoke)
        spec = _chain_spec()
        run_task_from_spec(spec, task_name="summarize", input_data={})
        # The second prompt (summarize) should contain the extracted facts.
        assert "the sky is blue" in prompts_seen[1]

    def test_dep_output_wins_on_key_collision(self, monkeypatch):
        """When both base input and dep output have the same key, dep output wins."""
        prompts_seen: list[str] = []

        def fake_invoke(system: str, user: str, config: dict) -> str:
            prompts_seen.append(f"{system}\n\n{user}")
            if len(prompts_seen) == 1:
                return '{"facts": "from dep"}'
            return '{"summary": "ok"}'

        monkeypatch.setattr("oas_cli.runner.invoke_intelligence", fake_invoke)
        spec = _chain_spec()
        # Caller provides facts; dep should win.
        run_task_from_spec(
            spec, task_name="summarize", input_data={"facts": "from caller"}
        )
        assert "from dep" in prompts_seen[1]
        assert "from caller" not in prompts_seen[1]

    def test_chain_result_includes_intermediate(self, monkeypatch):
        calls = {"n": 0}

        def fake_invoke(system: str, user: str, config: dict) -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                return '{"facts": "extracted facts"}'
            return '{"summary": "final summary"}'

        monkeypatch.setattr("oas_cli.runner.invoke_intelligence", fake_invoke)
        spec = _chain_spec()
        result = run_task_from_spec(spec, task_name="summarize", input_data={})
        assert result["chain"]["extract"]["output"]["facts"] == "extracted facts"
        assert result["output"]["summary"] == "final summary"

    def test_no_chain_key_without_depends_on(self, monkeypatch):
        """Tasks with no depends_on must NOT add a chain key."""
        monkeypatch.setattr(
            "oas_cli.runner.invoke_intelligence",
            lambda s, u, c: '{"facts": "x"}',
        )
        spec = _chain_spec()
        result = run_task_from_spec(spec, task_name="extract", input_data={})
        assert "chain" not in result

    def test_unknown_dep_raises(self, monkeypatch):
        """depends_on referencing a non-existent task must raise OARunError."""
        spec = _chain_spec()
        spec["tasks"]["summarize"]["depends_on"] = ["nonexistent"]
        monkeypatch.setattr(
            "oas_cli.runner.invoke_intelligence",
            lambda s, u, c: '{"facts": "x"}',
        )
        with pytest.raises(OARunError) as exc_info:
            run_task_from_spec(spec, task_name="summarize", input_data={})
        assert exc_info.value.code == "TASK_NOT_FOUND"

    def test_cycle_detection(self, monkeypatch):
        """Circular depends_on must raise CHAIN_CYCLE_ERROR."""
        spec = _chain_spec()
        spec["tasks"]["extract"]["depends_on"] = ["summarize"]
        monkeypatch.setattr(
            "oas_cli.runner.invoke_intelligence",
            lambda s, u, c: "{}",
        )
        with pytest.raises(OARunError) as exc_info:
            run_task_from_spec(spec, task_name="summarize", input_data={})
        assert exc_info.value.code == "CHAIN_CYCLE_ERROR"

    def test_missing_required_input_after_merge_raises(self, monkeypatch):
        """If required fields are still missing after merge, fail fast."""
        monkeypatch.setattr(
            "oas_cli.runner.invoke_intelligence",
            lambda s, u, c: "{}",  # dep returns empty output
        )
        spec = _chain_spec()
        # extract output missing 'facts', summarize needs nothing required — swap:
        # Make summarize require 'facts' but dep returns nothing
        spec["tasks"]["summarize"]["input"] = {
            "type": "object",
            "properties": {"facts": {"type": "string"}},
            "required": ["facts"],
        }
        # Dep (extract) returns empty output so facts won't be in merged input
        with pytest.raises(OARunError) as exc_info:
            run_task_from_spec(spec, task_name="summarize", input_data={})
        assert exc_info.value.code == "CHAIN_INPUT_MISSING"
        assert exc_info.value.stage == "input_validation"


# ---------------------------------------------------------------------------
# Behavioural contract — _merge_contracts and _resolve_contract unit tests
# ---------------------------------------------------------------------------


class TestMergeContracts:
    def test_no_overlap_combines_keys(self):
        base = {"version": "1.0", "description": "base"}
        override = {"policy": {"pii": False}}
        merged = _merge_contracts(base, override)
        assert merged["version"] == "1.0"
        assert merged["policy"] == {"pii": False}

    def test_arrays_are_unioned(self):
        base = {
            "response_contract": {"output_format": {"required_fields": ["confidence"]}}
        }
        override = {
            "response_contract": {"output_format": {"required_fields": ["summary"]}}
        }
        merged = _merge_contracts(base, override)
        fields = merged["response_contract"]["output_format"]["required_fields"]
        assert set(fields) == {"confidence", "summary"}

    def test_array_union_preserves_order_no_duplicates(self):
        base = {"response_contract": {"output_format": {"required_fields": ["a", "b"]}}}
        override = {
            "response_contract": {"output_format": {"required_fields": ["b", "c"]}}
        }
        merged = _merge_contracts(base, override)
        fields = merged["response_contract"]["output_format"]["required_fields"]
        assert fields == ["a", "b", "c"]

    def test_scalar_override_wins(self):
        base = {"version": "1.0", "description": "base desc"}
        override = {"version": "2.0"}
        merged = _merge_contracts(base, override)
        assert merged["version"] == "2.0"
        assert merged["description"] == "base desc"

    def test_empty_base_returns_override(self):
        assert _merge_contracts({}, {"version": "1.0"}) == {"version": "1.0"}

    def test_empty_override_returns_base(self):
        assert _merge_contracts({"version": "1.0"}, {}) == {"version": "1.0"}


class TestResolveContract:
    def _spec_with_contracts(
        self,
        *,
        global_fields: list[str] | None = None,
        task_fields: list[str] | None = None,
    ) -> dict:
        spec: dict = {
            "open_agent_spec": "1.3.0",
            "agent": {"name": "ta", "description": "ta"},
            "intelligence": {"type": "llm", "engine": "openai", "model": "gpt-4o"},
            "tasks": {
                "mytask": {
                    "description": "test",
                    "output": {
                        "type": "object",
                        "properties": {"r": {"type": "string"}},
                    },
                }
            },
            "prompts": {"system": "sys", "user": "{{ q }}"},
        }
        if global_fields is not None:
            spec["behavioural_contract"] = {
                "version": "1.0",
                "description": "global",
                "response_contract": {
                    "output_format": {"required_fields": global_fields}
                },
            }
        if task_fields is not None:
            spec["tasks"]["mytask"]["behavioural_contract"] = {
                "version": "1.0",
                "description": "task",
                "response_contract": {
                    "output_format": {"required_fields": task_fields}
                },
            }
        return spec

    def test_no_contract_returns_none(self):
        spec = self._spec_with_contracts()
        assert _resolve_contract(spec, "mytask") is None

    def test_global_only(self):
        spec = self._spec_with_contracts(global_fields=["confidence"])
        contract = _resolve_contract(spec, "mytask")
        assert contract is not None
        fields = contract["response_contract"]["output_format"]["required_fields"]
        assert fields == ["confidence"]

    def test_task_only(self):
        spec = self._spec_with_contracts(task_fields=["summary"])
        contract = _resolve_contract(spec, "mytask")
        fields = contract["response_contract"]["output_format"]["required_fields"]
        assert fields == ["summary"]

    def test_global_plus_task_merged(self):
        spec = self._spec_with_contracts(
            global_fields=["confidence"], task_fields=["summary"]
        )
        contract = _resolve_contract(spec, "mytask")
        fields = set(contract["response_contract"]["output_format"]["required_fields"])
        assert fields == {"confidence", "summary"}


# ---------------------------------------------------------------------------
# Behavioural contract — runtime enforcement in _run_single_task
# (only run when the library is installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CONTRACTS_ENABLED, reason="behavioural-contracts not installed")
class TestContractEnforcementLive:
    """Tests that require the actual behavioural-contracts library."""

    def _contract_spec(
        self, required_fields: list[str], response_format: str = "json"
    ) -> dict:
        task: dict = {
            "description": "test",
            "response_format": response_format,
            "output": {"type": "object", "properties": {"r": {"type": "string"}}},
            "behavioural_contract": {
                "version": "1.0",
                "description": "test contract",
                "response_contract": {
                    "output_format": {"required_fields": required_fields}
                },
            },
        }
        return {
            "open_agent_spec": "1.3.0",
            "agent": {"name": "ta", "description": "ta"},
            "intelligence": {"type": "llm", "engine": "openai", "model": "gpt-4o"},
            "tasks": {"mytask": task},
            "prompts": {"system": "sys", "user": "{{ q }}"},
        }

    def test_valid_output_passes(self, monkeypatch):
        monkeypatch.setattr(
            "oas_cli.runner.invoke_intelligence",
            lambda s, u, c: '{"summary": "all good", "confidence": "high"}',
        )
        spec = self._contract_spec(["summary", "confidence"])
        result = run_task_from_spec(spec, task_name="mytask", input_data={"q": "hi"})
        assert result["output"]["summary"] == "all good"

    def test_missing_required_field_raises_contract_violation(self, monkeypatch):
        monkeypatch.setattr(
            "oas_cli.runner.invoke_intelligence",
            lambda s, u, c: '{"summary": "ok"}',  # missing 'confidence'
        )
        spec = self._contract_spec(["summary", "confidence"])
        with pytest.raises(OARunError) as exc_info:
            run_task_from_spec(spec, task_name="mytask", input_data={"q": "hi"})
        err = exc_info.value
        assert err.code == "CONTRACT_VIOLATION"
        assert err.stage == "contract"
        assert err.task == "mytask"
        assert "confidence" in str(err)

    def test_text_mode_skips_contract(self, monkeypatch):
        """response_format: text must never raise CONTRACT_VIOLATION."""
        monkeypatch.setattr(
            "oas_cli.runner.invoke_intelligence",
            lambda s, u, c: "plain prose — no fields at all",
        )
        spec = self._contract_spec(["summary"], response_format="text")
        result = run_task_from_spec(spec, task_name="mytask", input_data={"q": "hi"})
        assert result["output"] == "plain prose — no fields at all"

    def test_global_contract_enforced_on_task(self, monkeypatch):
        """Top-level contract applies when task has none of its own."""
        monkeypatch.setattr(
            "oas_cli.runner.invoke_intelligence",
            lambda s, u, c: '{"summary": "ok"}',  # missing global 'confidence'
        )
        spec = {
            "open_agent_spec": "1.3.0",
            "agent": {"name": "ta", "description": "ta"},
            "intelligence": {"type": "llm", "engine": "openai", "model": "gpt-4o"},
            "tasks": {
                "mytask": {
                    "description": "test",
                    "output": {
                        "type": "object",
                        "properties": {"r": {"type": "string"}},
                    },
                }
            },
            "prompts": {"system": "sys", "user": "{{ q }}"},
            "behavioural_contract": {
                "version": "1.0",
                "description": "global",
                "response_contract": {
                    "output_format": {"required_fields": ["confidence"]}
                },
            },
        }
        with pytest.raises(OARunError) as exc_info:
            run_task_from_spec(spec, task_name="mytask", input_data={"q": "hi"})
        assert exc_info.value.code == "CONTRACT_VIOLATION"

    def test_chain_dep_violation_stops_chain(self, monkeypatch):
        """Contract violation on a dependency must halt the chain before the main task runs."""
        calls: list[str] = []

        def fake_invoke(system: str, user: str, config: dict) -> str:
            calls.append(f"{system}\n\n{user}")
            return '{"wrong_field": "oops"}'  # missing 'facts'

        monkeypatch.setattr("oas_cli.runner.invoke_intelligence", fake_invoke)
        spec = {
            "open_agent_spec": "1.3.0",
            "agent": {"name": "ta", "description": "ta"},
            "intelligence": {"type": "llm", "engine": "openai", "model": "gpt-4o"},
            "tasks": {
                "extract": {
                    "description": "extract",
                    "output": {
                        "type": "object",
                        "properties": {"facts": {"type": "string"}},
                    },
                    "behavioural_contract": {
                        "version": "1.0",
                        "description": "extract contract",
                        "response_contract": {
                            "output_format": {"required_fields": ["facts"]}
                        },
                    },
                },
                "summarize": {
                    "description": "summarize",
                    "depends_on": ["extract"],
                    "output": {
                        "type": "object",
                        "properties": {"summary": {"type": "string"}},
                    },
                },
            },
            "prompts": {"system": "sys", "user": "{{ q }}"},
        }
        with pytest.raises(OARunError) as exc_info:
            run_task_from_spec(spec, task_name="summarize", input_data={"q": "hi"})
        err = exc_info.value
        assert err.code == "CONTRACT_VIOLATION"
        assert err.task == "extract"
        # Only one LLM call was made — summarize never ran
        assert len(calls) == 1
