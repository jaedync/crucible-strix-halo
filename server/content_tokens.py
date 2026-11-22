"""RA-8: encode the client's content with special-token strings as plain text.

The chat template writes the control tokens of a prompt (<|im_start|>, <|im_end|>, <tool_call>, ...). The tokenizer
also parsed these strings inside client content, so a tool result such as a fetched page or a source file could forge
a turn or a tool call. Now each such string in client content becomes a private-use sentinel before the template
renders. The rendered text is split at the special-token strings that remain, which the template wrote; they keep
their IDs. Each piece between them gets the original strings back and is encoded with no added-token matching.

A prompt whose content has no special-token string is split the same way. Its pieces are encoded with the full
tokenizer, which gives the IDs of one encode of the whole text: the pieces run in parallel, and a piece that an
earlier prompt held comes from the piece cache.

Replay: the model can write an added token in its answer (a tool-call fallback, a stray tag). The server returns it as
text, and the client sends that text back on the next turn. That occurrence is the model's own output, not client
text, so it keeps its ID; otherwise the next prompt leaves the replay tail. The server remembers, for each returned
text that holds such a token, a hash of the text and the offsets of the generated tokens. Only an assistant message
whose content or reasoning equals that text exactly keeps them, and only at those offsets.
"""
from array import array
from collections import OrderedDict
import hashlib
from itertools import islice
import json
import re
import sys
import threading
from .api import ApiError, STOP_IDS

MODES = ("text", "parse")
# Returned texts with generated added tokens that the server remembers (a hash and offsets each, no text). Such
# texts are about 3% of agent turns, so this holds many sessions.
GENERATED_LIMIT = 4096
# More copies of one returned text in the output than this make its alignment ambiguous; nothing is kept then.
MAX_COPIES = 8
REPLAY_FIELDS = ("content", "reasoning_content")
# Supplementary Private Use Area-B: no tokenizer merge, template filter or JSON encoder changes these characters.
SENTINEL_FIRST, SENTINEL_LAST = 0x100000, 0x10FFFD
HIGH = re.compile("[\U00100000-\U0010FFFF]")
# Token IDs that the piece cache holds (4 bytes each): 16 MB, many long sessions.
PIECE_CACHE_TOKENS = 4_000_000
# Text that the piece cache holds (sys.getsizeof of each piece): few tokens can hold much text, so this bounds memory.
PIECE_CACHE_TEXT_BYTES = 64 << 20


class PieceCache:
    """Encoded prompt pieces, least recently used first out. A piece's IDs depend only on its text and the
    tokenizer kind, so a cached piece gives the same IDs as a new encode."""
    def __init__(self, limit=PIECE_CACHE_TOKENS, text_bytes=PIECE_CACHE_TEXT_BYTES):
        self.items, self.tokens, self.limit, self.lock = OrderedDict(), 0, limit, threading.Lock()
        self.text_bytes, self.text_limit = 0, text_bytes

    def encode(self, tokenizer, kind, pieces):
        """IDs of each piece: cached, or from one encode_batch of the pieces that are not."""
        found = {}
        with self.lock:
            for piece in pieces:
                ids = self.items.get((kind, piece))
                if ids is not None:
                    self.items.move_to_end((kind, piece))
                    found[piece] = ids
        missing = list(dict.fromkeys(piece for piece in pieces if piece not in found))
        if missing:
            encoded = [array("i", one.ids) for one in tokenizer.encode_batch(missing, add_special_tokens=False)]
            found.update(zip(missing, encoded))
            with self.lock:
                for piece, ids in zip(missing, encoded):
                    if (kind, piece) not in self.items:
                        self.items[(kind, piece)] = ids
                        self.tokens += len(ids)
                        self.text_bytes += sys.getsizeof(piece)
                while (self.tokens > self.limit or self.text_bytes > self.text_limit) and self.items:
                    (_, piece), ids = self.items.popitem(last=False)
                    self.tokens -= len(ids)
                    self.text_bytes -= sys.getsizeof(piece)
        return [found[piece] for piece in pieces]


class ContentEncoder:
    def __init__(self, tokenizer, tokenizer_path, stop_ids=STOP_IDS):
        added = tokenizer.get_added_tokens_decoder()
        for token in added.values():
            # The split below is plain leftmost-longest matching, which is what the tokenizer does for these flags.
            if token.normalized or token.lstrip or token.rstrip or token.single_word:
                raise ValueError(f"The added token {token.content!r} needs a matching rule that RA-8 does not model.")
        if tokenizer.padding or tokenizer.truncation:
            raise ValueError("encode_batch of the prompt pieces needs a tokenizer without padding or truncation.")
        self.pieces = PieceCache()
        self.ids = {token.content: token_id for token_id, token in added.items()}
        self.names = {token_id: token.content for token_id, token in added.items()}
        self.generated, self.generated_lock = OrderedDict(), threading.Lock()
        # Longest first: Python alternation takes the first alternative that matches at a position.
        self.pattern = re.compile("|".join(map(re.escape, sorted(self.ids, key=len, reverse=True)))) if self.ids else None
        self.tokenizer = tokenizer
        self.tokenizer_path = tokenizer_path
        self.stop_ids = frozenset(stop_ids)  # the model family's end-of-turn IDs: not output text
        self.lock, self.plain = threading.Lock(), None

    def text_tokenizer(self):
        """The same tokenizer with no added tokens: it encodes "<|im_end|>" as the characters it shows."""
        with self.lock:
            if self.plain is None:
                from tokenizers import Tokenizer
                with open(self.tokenizer_path, encoding="utf-8") as file:
                    spec = json.load(file)
                spec["added_tokens"] = []
                self.plain = Tokenizer.from_str(json.dumps(spec))
            return self.plain

    def strings(self, value, found):
        """Collect every text value and key of a JSON-like value."""
        if isinstance(value, str):
            found.append(value)
        elif isinstance(value, dict):
            for key, item in value.items():
                self.strings(key, found)
                self.strings(item, found)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self.strings(item, found)
        return found

    def has_special(self, text):
        return self.pattern is not None and self.pattern.search(text) is not None

    def sentinels(self, texts):
        """{special-token string: sentinel} for the strings that the texts hold; {} when they hold none."""
        found = sorted({match for text in texts if self.has_special(text) for match in self.pattern.findall(text)})
        if not found:
            return {}
        # Only the strings that the content holds need a sentinel (normally a few), so the free characters run out
        # only when the content holds almost every private-use character.
        taken = {character for text in texts for character in HIGH.findall(text)}
        free = list(islice((chr(code) for code in range(SENTINEL_FIRST, SENTINEL_LAST + 1) if chr(code) not in taken),
                           len(found)))
        if len(free) < len(found):
            raise ApiError("The content uses too many private-use characters (U+100000 to U+10FFFD) together with "
                           "special-token text. Remove some of them.", "messages")
        return dict(zip(found, free))

    def escape(self, messages, tools=()):
        """((messages, tools) with each special-token string replaced by a sentinel, {sentinel: string}).

        The values come back unchanged, with an empty table, when no text contains a special-token string. In an
        assistant message, the generated tokens of a remembered returned text stay (see the module text).
        """
        table = self.sentinels(self.strings((messages, tools), []))
        if not table:
            return (messages, tools), {}
        messages = type(messages)(self.replace_message(message, table) for message in messages)
        return (messages, self.replace(tools, table)), {char: string for string, char in table.items()}

    def replace_message(self, message, table):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            return self.replace(message, table)
        return {self.replace(key, table): self.replace_text(item, table, self.kept(item))
                if key in REPLAY_FIELDS and isinstance(item, str) else self.replace(item, table)
                for key, item in message.items()}

    def replace_text(self, text, table, kept=frozenset()):
        if not self.has_special(text):
            return text
        return self.pattern.sub(lambda match: match.group(0) if (match.start(), match.group(0)) in kept
                                else table[match.group(0)], text)

    def replace(self, value, table):
        if isinstance(value, str):
            return self.replace_text(value, table)
        if isinstance(value, dict):
            return {self.replace(key, table): self.replace(item, table) for key, item in value.items()}
        if isinstance(value, list):
            return [self.replace(item, table) for item in value]
        if isinstance(value, tuple):
            return tuple(self.replace(item, table) for item in value)
        return value

    def encode_text(self, text, kept=frozenset()):
        """IDs of one content text: the kept generated tokens keep their IDs, other special strings are text."""
        table = self.sentinels([text])
        return self.encode(self.replace_text(text, table, kept), {char: string for string, char in table.items()})

    @staticmethod
    def digest(text):
        return hashlib.sha256(text.encode("utf-8", "surrogatepass")).digest()

    def remember(self, text, kept):
        """Record one generation of a returned text. Two generations of the same text with different added tokens
        (for example one as plain pieces) leave frozenset(): the text is ambiguous, so it stays text."""
        key = self.digest(text)
        with self.generated_lock:
            entry = self.generated.get(key)
            if entry is not None and entry != (len(text), kept):
                kept = frozenset()
            self.generated[key] = (len(text), kept)
            self.generated.move_to_end(key)
            while len(self.generated) > GENERATED_LIMIT:
                self.generated.popitem(last=False)

    def kept(self, text):
        """The remembered generated tokens of this exact returned text, as {(offset, string)}; empty otherwise."""
        if not self.has_special(text):
            return frozenset()
        key = self.digest(text)
        with self.generated_lock:
            entry = self.generated.get(key)
            if entry is not None:
                self.generated.move_to_end(key)
        return entry[1] if entry is not None and entry[0] == len(text) else frozenset()

    def generated_strings(self, decode, ids, text):
        """{(offset, string)} of the special-token strings in `text` that the model wrote as added tokens in `ids`.

        The decoded output is rebuilt from runs of ordinary IDs and the added tokens between them, so the position of
        each generated added token is known. `text` is found in that output; if it is not there, or its copies
        disagree, nothing is kept (the RA-8 rule then applies).
        """
        pieces, starts, run, position = [], {}, [], 0
        for token in list(ids) + [None]:
            if token is not None and token in self.stop_ids:
                continue
            if token is None or token in self.names:
                piece = decode(run) if run else ""
                pieces.append(piece)
                position += len(piece)
                run = []
                if token is not None:
                    starts[position] = self.names[token]
                    pieces.append(self.names[token])
                    position += len(self.names[token])
            else:
                run.append(token)
        output = "".join(pieces)
        if output != decode(ids):
            return frozenset()  # a character split across an added token: the pieces do not align
        matches = [(match.start(), match.group(0)) for match in self.pattern.finditer(text)]
        results, index, copies = set(), output.find(text), 0
        while index >= 0 and copies < MAX_COPIES:
            results.add(frozenset((offset, string) for offset, string in matches
                                  if starts.get(index + offset) == string))
            copies += 1
            index = output.find(text, index + 1)
        if index >= 0:
            return frozenset()
        return next(iter(results)) if len(results) == 1 else frozenset()

    def encode(self, text, restore):
        """IDs of a rendered text whose content strings are sentinels: template tokens keep their IDs, and every
        piece between them is encoded as text after its sentinels get their strings back."""
        pieces, tokens, position = [], [], 0
        for match in self.pattern.finditer(text):
            pieces.append(text[position:match.start()])
            tokens.append(self.ids[match.group(0)])
            position = match.end()
        pieces.append(text[position:])
        if restore:
            back = re.compile("[" + "".join(restore) + "]")
            pieces = [back.sub(lambda match: restore[match.group(0)], piece) for piece in pieces]
        # Each piece between added tokens is encoded on its own, as the tokenizer does inside one encode() call.
        # Without restored strings no piece holds an added token, so the full tokenizer encodes them as text too.
        tokenizer, kind = (self.text_tokenizer(), "plain") if restore else (self.tokenizer, "full")
        encoded = self.pieces.encode(tokenizer, kind, [piece for piece in pieces if piece])
        ids, nonempty = [], iter(encoded)
        for index, piece in enumerate(pieces):
            if piece:
                ids.extend(next(nonempty))
            if index < len(tokens):
                ids.append(tokens[index])
        return ids
