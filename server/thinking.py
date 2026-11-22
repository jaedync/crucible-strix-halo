"""Split generated thinking text from the answer when a chat request enables thinking.

With thinking on, the chat template ends the prompt with "<think>\\n", so the model writes its reasoning first
and closes it with "</think>". Clients (Pi, Hermes) expect that text in reasoning_content, as Halogen sends it,
and send it back as assistant reasoning_content, which the template renders inside <think> again.
With thinking off the splitter passes every character through unchanged.
"""

END = "</think>"


class ThinkSplitter:
    def __init__(self, enabled, end=END, end_id=None):
        """end: the close tag of the model family (Hy4-flash: hy4_format.THINK_END). end_id: its token ID. With an ID,
        only that token ends the reasoning, as in the engine grammar and the forced close (think_close_ids). The same
        text as plain pieces (a quoted end tag; RA-8 encodes a client "</think>" so) stays reasoning. With no ID
        (a stub tokenizer) the text ends it."""
        self.enabled, self.end, self.end_id = enabled, end, end_id
        self.thinking = enabled
        self.pending = ""
        self.reasoning = ""
        self.answer = ""  # the answer text after the end tag, as returned (response.calls_from_reasoning reads it)
        self.started = False
        self.answer_started = False

    def feed(self, text, token=None):
        """Return (reasoning_delta, content_delta) for one piece of decoded text, the text of `token` if one is given."""
        if not self.thinking:
            return "", self._answer(text)
        combined = self.pending + text
        if self.end_id is not None:
            if token != self.end_id:
                self.pending = ""
                return self._reason(combined), ""
            # The end token's text can follow the held bytes of earlier tokens: split at its last tag.
            self.thinking, self.pending, end = False, "", combined.rfind(self.end)
            if end < 0:
                return self._reason(combined), ""
            return self._reason(combined[:end]), self._answer(combined[end + len(self.end):])
        end = combined.find(self.end)
        if end >= 0:
            self.thinking, self.pending = False, ""
            return self._reason(combined[:end]), self._answer(combined[end + len(self.end):])
        # Hold back a suffix that could start the end tag in the next piece.
        keep = next((length for length in range(min(len(combined), len(self.end) - 1), 0, -1)
                     if self.end.startswith(combined[-length:])), 0)
        self.pending = combined[len(combined) - keep:]
        return self._reason(combined[:len(combined) - keep]), ""

    def flush(self):
        """Text held at the end of generation: reasoning that never reached the end tag."""
        if not self.thinking:
            return "", ""
        text, self.pending = self.pending, ""
        return self._reason(text), ""

    def _reason(self, text):
        if not self.started:
            text = text.lstrip("\n")
            self.started = bool(text)
        self.reasoning += text
        return text

    def _answer(self, text):
        if self.enabled and not self.answer_started:
            # The template writes "</think>\n\n" before the answer and trims it on replay.
            # Strip newlines only, so an answer that starts with indented code keeps its indentation.
            text = text.lstrip("\n")
            self.answer_started = bool(text)
        self.answer += text
        return text
