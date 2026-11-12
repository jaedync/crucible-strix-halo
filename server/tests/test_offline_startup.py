"""Exercise tokenizer and CLI boundaries without a checkpoint or listener."""
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

import server.__main__ as startup
from server.api import ApiError, STOP_IDS, validate_request
from server.template import ChatTemplate
from server.tokenization import ModelTokenizer


@pytest.fixture
def checkpoint(tmp_path):
    fixture = Path(__file__).parent / "fixtures/qwen_chat_template.jinja"
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": fixture.read_text()}))
    return tmp_path


def request(kind="completion", **options):
    body = {"prompt": "hello world"} if kind == "completion" else {"messages": [{"role": "user", "content": "hello"}]}
    return validate_request(body | options, kind, "crucible")


def test_constructor_checks_every_special_id(checkpoint):
    expected = {"<|im_end|>": 248046, "<|endoftext|>": 248044, "<tool_call>": 248058, "</tool_call>": 248059}
    source = MagicMock(padding=None, truncation=None)
    source.token_to_id.side_effect = expected.get
    with patch("tokenizers.Tokenizer.from_file", return_value=source) as load:
        adapter = ModelTokenizer(checkpoint)
    load.assert_called_once_with(str(checkpoint / "tokenizer.json"))
    assert adapter.tokenizer is source
    assert source.token_to_id.call_count == 4
    for wrong in expected:
        source.token_to_id.side_effect = lambda name: -1 if name == wrong else expected[name]
        with patch("tokenizers.Tokenizer.from_file", return_value=source), pytest.raises(ValueError, match="special-token ID"):
            ModelTokenizer(checkpoint)


def test_offline_encode_decode_and_unicode_stream(checkpoint):
    source = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2, "café": 3, "日本語": 4}, unk_token="[UNK]"))
    source.pre_tokenizer = pre_tokenizers.Whitespace()
    from tokenizers.decoders import DecodeStream
    adapter = ModelTokenizer.__new__(ModelTokenizer)
    adapter.tokenizer, adapter.template, adapter.decode_stream = source, ChatTemplate(checkpoint), DecodeStream
    for text in ("hello world", "café 日本語", "hello café world"):
        ids = adapter.encode(request(prompt=text))
        stream = adapter.stream()
        assert "".join(adapter.step(stream, token) for token in ids) == adapter.decode(ids)
        assert adapter.step(stream, STOP_IDS[0]) == ""
        assert adapter.decode(ids + list(STOP_IDS)) == text
    ids = adapter.encode(request(prompt=[1, 2]))
    ids.append(3)
    assert adapter.encode(request(prompt=[1, 2])) == [1, 2]
    assert adapter.cache_prefix_tokens(request(), [1, 2]) == 0


@pytest.mark.parametrize("kind,param", [("chat", "messages"), ("completion", "prompt")])
def test_tokenization_failure_is_an_api_error(kind, param):
    adapter = ModelTokenizer.__new__(ModelTokenizer)
    adapter.tokenizer = MagicMock()
    adapter.tokenizer.encode.side_effect = ValueError("private internal details")
    adapter.render_prompt = lambda *args: ("hello", {})
    with pytest.raises(ApiError) as error:
        adapter.encode(request(kind))
    assert error.value.param == param
    assert str(error.value) == "The prompt could not be tokenized."


def test_template_file_override_and_forced_call_preserve_history(checkpoint):
    (checkpoint / "chat_template.jinja").write_text("{{ messages|tojson }}|{{ tools|length }}")
    template = ChatTemplate(checkpoint)
    messages = ({"role": "system", "content": "rules"}, {"role": "user", "content": "question"})
    choice = {"type": "function", "function": {"name": "read"}}
    rendered = template.render(messages, tools=({"name": "read"},), tool_choice=choice)
    assert "Call the function read." in rendered
    assert messages[0]["content"] == "rules"
    assert template.render(messages, tools=({"name": "read"},), tool_choice="none").endswith("|0")


def test_template_invalid_configuration_and_explicit_error(checkpoint):
    for template in (None, [], ""):
        (checkpoint / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))
        with pytest.raises(ValueError, match="string chat template"):
            ChatTemplate(checkpoint)
    (checkpoint / "chat_template.jinja").write_text("{{ raise_exception('bad history') }}")
    with pytest.raises(ApiError, match="bad history") as error:
        ChatTemplate(checkpoint).render(())
    assert error.value.param == "messages"


@pytest.mark.parametrize("options", [
    ["--ctx", "0"], ["--ctx", "524289"], ["--conversation-cache-mib", "-1"],
    ["--conversation-cache-mib", "131073"], ["--prefill", "-1"], ["--prefill", "4097"],
    ["--model-id", ""], ["--model-id", "x" * 201], ["--streams", "0"], ["--streams", "9"],
])
def test_startup_rejects_invalid_options_before_loading(options):
    with patch.object(sys, "argv", ["crucible", "checkpoint"] + options), patch.object(startup, "ModelTokenizer") as load:
        with pytest.raises(SystemExit) as error:
            startup.main()
    assert error.value.code == 2
    load.assert_not_called()


def test_serial_startup_omitted_prefill_keeps_native_default():
    listener = MagicMock()
    listener.serve_forever.side_effect = KeyboardInterrupt
    with patch.object(sys, "argv", ["crucible", "checkpoint", "--serial"]), patch.object(startup.signal, "signal"), \
            patch.object(startup, "ModelTokenizer"), patch.object(startup, "EngineBackend") as backend, \
            patch.object(startup, "ThreadingHTTPServer", return_value=listener):
        assert startup.main() == 0
    assert backend.call_args.args[4] is None


def test_serial_startup_options_and_interrupt_cleanup():
    listener = MagicMock()
    listener.serve_forever.side_effect = KeyboardInterrupt
    args = ["crucible", "checkpoint", "--serial", "--engine", "fake", "--ctx", "128", "--prefill", "64",
            "--conversation-cache-mib", "32", "--model-id", "local-model"]
    with patch.object(sys, "argv", args), patch.object(startup.signal, "signal") as register, \
            patch.object(startup, "ModelTokenizer"), patch.object(startup, "EngineBackend") as backend, \
            patch.object(startup, "ThreadingHTTPServer", return_value=listener) as server:
        assert startup.main() == 0
    backend.assert_called_once_with(Path("fake"), Path("checkpoint"), 128, 32, 64,
                                    engine_options=("--session-replay", "1"), lookup=7)
    assert server.call_args.args[0] == ("127.0.0.1", 8790)
    listener.server_close.assert_called_once()
    backend.return_value.close.assert_called_once()
    with pytest.raises(KeyboardInterrupt):
        register.call_args.args[1](None, None)


@pytest.mark.parametrize("options, mode", [([], "text"), (["--content-special-tokens", "parse"], "parse")])
def test_content_special_tokens_switch_reaches_the_tokenizer_and_the_ready_line(options, mode, capsys):
    # RA-8: the mode changes prompt IDs for content with special-token strings, so the ready line names it.
    listener = MagicMock()
    listener.serve_forever.side_effect = KeyboardInterrupt
    with patch.object(sys, "argv", ["crucible", "checkpoint", "--serial"] + options), patch.object(startup.signal, "signal"), \
            patch.object(startup, "ModelTokenizer") as tokenizer, patch.object(startup, "EngineBackend"), \
            patch.object(startup, "ThreadingHTTPServer", return_value=listener):
        assert startup.main() == 0
    tokenizer.assert_called_once_with(Path("checkpoint"), content_special_tokens=mode)
    assert f"Crucible is ready at http://127.0.0.1:8790/v1 (content special tokens: {mode})" in capsys.readouterr().err


def test_content_special_tokens_switch_rejects_other_values():
    with patch.object(sys, "argv", ["crucible", "checkpoint", "--content-special-tokens", "raw"]), \
            patch.object(startup, "ModelTokenizer") as load, pytest.raises(SystemExit):
        startup.main()
    load.assert_not_called()


@pytest.mark.parametrize("failure", ["tokenizer", "listener", "serve"])
def test_startup_failure_closes_only_created_resources(failure, capsys):
    listener, backend = MagicMock(), MagicMock()
    with patch.object(sys, "argv", ["crucible", "checkpoint", "--serial"]), patch.object(startup.signal, "signal"), \
            patch.object(startup, "ModelTokenizer") as tokenizer, \
            patch.object(startup, "EngineBackend", return_value=backend) as initialize, \
            patch.object(startup, "ThreadingHTTPServer", return_value=listener) as server:
        if failure == "tokenizer": tokenizer.side_effect = ValueError("invalid tokenizer")
        if failure == "listener": server.side_effect = OSError("address in use")
        if failure == "serve": listener.serve_forever.side_effect = RuntimeError("listener failed")
        assert startup.main() == 1
    assert "server error:" in capsys.readouterr().err
    assert backend.close.call_count == (failure != "tokenizer")
    assert listener.server_close.call_count == (failure == "serve")
    if failure == "tokenizer": initialize.assert_not_called()


def test_module_entry_rejects_options_before_loading():
    with patch.object(sys, "argv", ["crucible", "checkpoint", "--ctx", "0"]), pytest.raises(SystemExit) as error:
        # runpy reaches the module guard without binding a port or loading an engine.
        with pytest.warns(RuntimeWarning, match="found in sys.modules"):
            runpy.run_module("server.__main__", run_name="__main__")
    assert error.value.code == 2
