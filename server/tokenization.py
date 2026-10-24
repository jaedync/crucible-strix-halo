"""Checkpoint tokenization and Unicode-safe incremental decoding."""
from pathlib import Path
from .api import ApiError, STOP_IDS
from .content_tokens import ContentEncoder, MODES
from .template import ChatTemplate
from .vision import expand_pads

SHARED_SENTINEL = "\u2063crucible-shared-prefix\u2063"

class ModelTokenizer:
    def __init__(self, checkpoint, content_special_tokens="text"):
        """content_special_tokens: "text" encodes special-token strings in client content as text (RA-8);
        "parse" keeps the old rule, where they become control tokens."""
        from tokenizers import Tokenizer
        from tokenizers.decoders import DecodeStream
        if content_special_tokens not in MODES:
            raise ValueError("content_special_tokens must be text or parse.")
        path = Path(checkpoint) / "tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(path))
        self.content = ContentEncoder(self.tokenizer, path) if content_special_tokens == "text" else None
        self.template = ChatTemplate(checkpoint)
        self.decode_stream = DecodeStream
        expected = {"<|im_end|>": 248046, "<|endoftext|>": 248044,
                    "<tool_call>": 248058, "</tool_call>": 248059}
        for token, token_id in expected.items():
            if self.tokenizer.token_to_id(token) != token_id:
                raise ValueError(f"The checkpoint has an unexpected special-token ID for {token}.")

    def render_prompt(self, request, add_generation_prompt=True, messages=None):
        """(rendered text, {sentinel: special-token string}). The table is empty, and the text is the old render,
        unless client content holds a special-token string (RA-8)."""
        messages = request.messages if messages is None else messages
        tools, restore = request.tools, {}
        if getattr(self, "content", None) is not None:
            (messages, tools), restore = self.content.escape(messages, tools)
        preserve = getattr(request, "preserve_thinking", None)
        options = {} if preserve is None else dict(preserve_thinking=preserve)
        return self.template.render(messages, request.enable_thinking, request.reasoning_effort, tools,
                                    request.tool_choice, add_generation_prompt=add_generation_prompt,
                                    **options), restore

    def render_chat(self, request, add_generation_prompt=True):
        return self.render_prompt(request, add_generation_prompt)[0]

    def encode_prompt(self, text, restore=None):
        # The text mode encodes the pieces between added tokens in one parallel encode_batch: the same IDs as one
        # encode, and about 17 ms instead of 160 ms for a 128K prompt on the test machine.
        if getattr(self, "content", None) is None:
            return self.tokenizer.encode(text, add_special_tokens=False).ids
        return self.content.encode(text, restore)

    def encode(self, request):
        # A completion prompt is raw model input, so its special-token strings stay control tokens.
        restore = None
        if request.kind == "chat":
            text, restore = self.render_prompt(request)
        elif isinstance(request.prompt, tuple):
            return list(request.prompt)
        else:
            text = request.prompt
        try:
            return self.encode_prompt(text, restore)
        except Exception as error:
            raise ApiError("The prompt could not be tokenized.", "messages" if request.kind == "chat" else "prompt") from error

    def encode_images(self, request):
        """(IDs with each image placeholder expanded, first-token offset of each image)."""
        return expand_pads(self.encode(request), request.images)

    def cache_prefix_tokens(self, request, ids):
        if request.kind != "chat":
            return 0
        prefix = self.encode_prompt(*self.render_prompt(request, add_generation_prompt=False))
        images = getattr(request, "images", ())
        if images:
            prefix = expand_pads(prefix, images)[0]
        # Tokenizer merges at the suffix boundary can invalidate a text prefix.
        return len(prefix) if ids[:len(prefix)] == prefix else 0

    def shared_prefix_tokens(self, request, ids):
        """Token count of the system-and-tools block that sessions with the same setup share."""
        if request.kind != "chat":
            return 0
        system = request.messages[:1] if request.messages and request.messages[0]["role"] == "system" else ()
        # The template needs a user turn. A sentinel turn marks where the shared block ends.
        sentinel = dict(role="user", content=SHARED_SENTINEL)
        try:
            text, restore = self.render_prompt(request, add_generation_prompt=False, messages=system + (sentinel,))
        except ApiError:
            return 0  # The shared frontier is an optimization; a template that rejects the sentinel turn skips it.
        end = text.rfind("<|im_start|>", 0, max(text.find(SHARED_SENTINEL), 0))
        if end <= 0:
            return 0
        prefix = self.encode_prompt(text[:end], restore)
        # Only an exact token prefix of this prompt is useful; the engine also matches cached tokens.
        return len(prefix) if ids[:len(prefix)] == prefix else 0

    def think_close_ids(self):
        """The forced close of a think block (see think_budget), tokenized once."""
        if not hasattr(self, "_think_close"):
            from .think_budget import close_ids
            self._think_close = close_ids(self.tokenizer)
        return self._think_close

    def closed_think_ids(self, reasoning, ids=()):
        """The reasoning and forced close as the next turn renders them.

        The template trims reasoning_content and writes "\\n</think>\\n\\n" after it. The tokenizer can split that
        text in another way than the model did (":" "<" against ":<", or a cut after a newline that joins the close's
        "\\n\\n"), and then the next prompt leaves the replay tail inside the reasoning. `ids` are the generated
        IDs so far: an added token that the model wrote in its reasoning keeps its ID, as on the next turn.
        """
        from .think_budget import THINK_END
        text, content = reasoning.strip(), getattr(self, "content", None)
        if content is not None and content.has_special(text):
            # RA-8: other special-token strings in the reasoning are text, as on the next turn.
            body = content.encode_text(text + "\n", content.generated_strings(self.decode, ids, text) if ids else
                                       frozenset())
        else:
            body = self.tokenizer.encode(text + "\n", add_special_tokens=False).ids
        close = self.tokenizer.encode("\n\n", add_special_tokens=False).ids
        return tuple(body) + (self.tokenizer.token_to_id(THINK_END),) + tuple(close)

    def remember_output(self, ids, message):
        """Text mode: remember which special-token strings of the returned answer and reasoning the model wrote as
        added tokens, so the next turn that sends the same text back keeps their IDs and its replay tail."""
        content = getattr(self, "content", None)
        if content is None:
            return
        from .content_tokens import REPLAY_FIELDS
        for field in REPLAY_FIELDS:
            text = message.get(field)
            if isinstance(text, str) and content.has_special(text):
                # Also with nothing kept: a later generation of the same text with added tokens is then ambiguous.
                content.remember(text, content.generated_strings(self.decode, ids, text))

    def stream(self):
        return self.decode_stream(skip_special_tokens=False)

    def step(self, stream, token_id):
        if token_id in STOP_IDS:
            return ""
        return stream.step(self.tokenizer, token_id) or ""

    def decode(self, ids):
        return self.tokenizer.decode([token for token in ids if token not in STOP_IDS], skip_special_tokens=False)
