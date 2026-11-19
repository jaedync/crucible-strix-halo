"""Hold possible stop prefixes so a stop sequence never leaks into SSE text."""
from .api import ApiError

MAX_SEQUENCES = 4
MAX_SEQUENCE_BYTES = 4096

def validate_stop(value):
    if value is None:
        return ()
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list) or len(values) > MAX_SEQUENCES:
        raise ApiError("stop must be text or a list of at most four strings.", "stop")
    for item in values:
        if not isinstance(item, str) or not item:
            raise ApiError("Each stop sequence must be nonempty text.", "stop")
        try:
            size = len(item.encode("utf-8"))
        except UnicodeError:
            raise ApiError("Each stop sequence must contain valid UTF-8 text.", "stop") from None
        if size > MAX_SEQUENCE_BYTES:
            raise ApiError("Each stop sequence must fit 4096 UTF-8 bytes.", "stop")
    return tuple(values)

def prefix_table(pattern):
    table = [0] * len(pattern)
    count = 0
    for index in range(1, len(pattern)):
        while count and pattern[index] != pattern[count]:
            count = table[count - 1]
        if pattern[index] == pattern[count]:
            count += 1
        table[index] = count
    return tuple(table)

def advance(pattern, table, count, character):
    while count and character != pattern[count]:
        count = table[count - 1]
    return count + (character == pattern[count])

class StopSequences:
    def __init__(self, sequences):
        self.patterns = tuple((value, prefix_table(value)) for value in sequences)
        self.counts = (0,) * len(sequences)
        self.pending = ""
        self.stopped = False
        self.matched = None  # the sequence that stopped the text (the one that starts first)

    def feed(self, text):
        if self.stopped:
            return ""
        if not self.patterns:
            return text
        value = self.pending + text
        # Prefix tables keep repeated-prefix input linear in the text length.
        for index, character in enumerate(text, len(self.pending)):
            self.counts = tuple(advance(pattern, table, count, character)
                                for (pattern, table), count in zip(self.patterns, self.counts))
            starts = [index + 1 - len(pattern) for (pattern, _), count in zip(self.patterns, self.counts)
                      if count == len(pattern)]
            if starts:
                self.stopped = True
                self.pending = ""
                first = min(starts)
                self.matched = next(pattern for (pattern, _), count in zip(self.patterns, self.counts)
                                    if count == len(pattern) and index + 1 - len(pattern) == first)
                return value[:first]
        held = max(self.counts, default=0)
        self.pending = value[-held:] if held else ""
        return value[:-held] if held else value

    def flush(self):
        value, self.pending = self.pending, ""
        return value
