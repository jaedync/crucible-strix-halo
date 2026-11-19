"""A handler stand-in for CompletionResponse that writes another API's shapes (Anthropic Messages, OpenAI Responses).

CompletionResponse builds the chat reply from the engine tokens. A sink takes its place as the handler: the HTTP
status, the headers and the connection go to the real handler, but each chat chunk (sse) and the final chat body
(send_json) go to the sink, which writes the same text, reasoning, tool calls and counts in its own shape. So the
tokens, the stop rules, the tool parser and the think split are those of /v1/chat/completions.
"""
import json
import time


class Sink:
    def __init__(self, handler):
        self.handler = handler
        self.output = None  # the CompletionResponse that writes to this sink (set after it is made)

    def send_response(self, *args):
        self.handler.send_response(*args)

    def send_header(self, *args):
        self.handler.send_header(*args)

    def end_headers(self):
        self.handler.end_headers()

    @property
    def wfile(self):
        return self.handler.wfile

    @property
    def jobs(self):
        return getattr(self.handler, "jobs", None)

    @property
    def client_request_id(self):
        return getattr(self.handler, "client_request_id", None)

    @property
    def close_connection(self):
        return getattr(self.handler, "close_connection", False)

    @close_connection.setter
    def close_connection(self, value):
        self.handler.close_connection = value

    def event(self, name, data):
        """One named SSE event."""
        encoded = json.dumps(data, ensure_ascii=True, allow_nan=False)
        self.handler.wfile.write(f"event: {name}\ndata: {encoded}\n\n".encode("utf-8"))
        self.handler.wfile.flush()
        self.handler.last_write = time.monotonic()

    def sse(self, body):
        """A chat stream chunk, or the "[DONE]" end."""
        if body == "[DONE]":
            self.done()
        elif isinstance(body, dict):
            self.chunk(body)

    def send_json(self, status, body, headers=None):
        self.handler.send_json(status, self.reply(body) if status == 200 else body, headers)

    def chunk(self, body):
        raise NotImplementedError

    def done(self):
        raise NotImplementedError

    def reply(self, body):
        raise NotImplementedError


def stop_sequence(output):
    """The client stop string that ended the reply, or None."""
    stop_filter = getattr(output, "stop_filter", None)
    return getattr(stop_filter, "matched", None) if stop_filter is not None and stop_filter.stopped else None


def call_input(arguments):
    """A chat tool call's arguments (JSON text of an object) as an object."""
    try:
        value = json.loads(arguments)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}
