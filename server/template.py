"""Render the checkpoint's real text chat template with its existing Jinja dependency."""
import json
from pathlib import Path
from jinja2.sandbox import ImmutableSandboxedEnvironment
from .api import ApiError
from . import hy4_format

def raise_template_error(message):
    raise ApiError(str(message), "messages")

def effort_option(request):
    """The render option of the client's effort word (api.Request.effort_word), empty when the request has none."""
    word = getattr(request, "effort_word", None)
    return {} if word is None else dict(effort_word=word)

class ChatTemplate:
    def __init__(self, checkpoint):
        root = Path(checkpoint)
        config = json.loads((root / "tokenizer_config.json").read_text())
        path = root / "chat_template.jinja"
        template = path.read_text() if path.exists() else config.get("chat_template")
        if not isinstance(template, str) or not template:
            raise ValueError("The checkpoint must contain a string chat template.")
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        environment.globals["raise_exception"] = raise_template_error
        # HF's tojson filter preserves Unicode and does not HTML-escape the JSON.
        # The Hy4-flash template passes ensure_ascii=False itself.
        environment.filters["tojson"] = lambda value, ensure_ascii=False, **kwargs: json.dumps(
            value, ensure_ascii=ensure_ascii, **kwargs)
        self.template = environment.from_string(template)
        self.hy4 = hy4_format.is_hy4(root)
        self.special_tokens = {key: value for key, value in config.items()
                               if key.endswith("_token") and isinstance(value, str)}

    def render(self, messages, enable_thinking=False, reasoning_effort="xhigh", tools=(), tool_choice="auto",
               add_generation_prompt=True, preserve_thinking=None, effort_word=None):
        """reasoning_effort: the Crucible level (api.EFFORTS). effort_word: the client's word (api.Request.effort_word),
        which only the Hy4-flash template reads."""
        prepared = tuple(messages)
        instruction = None
        if tool_choice == "required":
            instruction = "Call one of the available tools. Do not give a text-only answer."
        elif isinstance(tool_choice, dict):
            instruction = "Call the function " + tool_choice["function"]["name"] + ". Do not call another function."
        if instruction:
            if prepared and prepared[0]["role"] == "system":
                prepared = (prepared[0] | dict(content=prepared[0]["content"] + "\n\n" + instruction),) + prepared[1:]
            else:
                prepared = (dict(role="system", content=instruction),) + prepared
        available = () if tool_choice == "none" else tools
        # The template keeps earlier reasoning unless preserve_thinking is false, so leave it undefined by default.
        preserve = {} if preserve_thinking is None else dict(preserve_thinking=preserve_thinking)
        if self.hy4:
            # The Hy4-flash template reads one effort word (no_think, high, max) and the name preserved_thinking.
            reasoning_effort = hy4_format.template_effort(enable_thinking, effort_word)
            preserve = {} if preserve_thinking is None else dict(preserved_thinking=preserve_thinking)
        return self.template.render(messages=list(prepared), tools=list(available), add_generation_prompt=add_generation_prompt,
                                    enable_thinking=enable_thinking, reasoning_effort=reasoning_effort,
                                    **preserve, **self.special_tokens)
