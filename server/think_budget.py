"""Close a thinking block that runs too long, as Halogen 0.15 does, so a capped turn still ends with an answer.

Halogen's rule (serve_api.py, 0.11.0 to 0.15.3):
- The answer room is max(1024, 15% of max_tokens) unless the operator sets a number (0 turns it off).
- The think block closes after max(1, max_tokens - room) generated tokens, or after the request's own
  thinking budget (max_thinking_tokens and its aliases) when that is smaller.
- A request with no explicit budget whose limit above is below --min-thinking-tokens is served with thinking off
  (skips_thinking), because a close that early writes the budget sentence as the whole reasoning.
- The close is "\\n\\n" + Qwen's budget sentence + "\\n", then </think>, then "\\n\\n". It counts against max_tokens.
The count is on generated tokens, so the close lands on the same token with any decode mode or load.
"""
from dataclasses import dataclass

QWEN_BUDGET_SENTENCE = ("Considering the limited time by the user, I have to "
                        "give the solution based on the thinking directly now.")
THINK_END = "</think>"
ROOM_FLOOR = 1024
ROOM_PERCENT = 15
# --min-thinking-tokens default. Below it the forced close sentence (about 25 tokens) is a large part of the
# thinking, and real thinking has no room.
MIN_THINKING_DEFAULT = 256
SKIPPED_BY = "min_thinking_tokens"


@dataclass(frozen=True)
class ThinkCap:
    tokens: int
    by: str


def answer_room(max_tokens, setting=None):
    """Tokens of max_tokens kept for the answer. None selects Halogen's policy; 0 is off."""
    if setting is None:
        return max(ROOM_FLOOR, max_tokens * ROOM_PERCENT // 100)
    return setting


def think_cap(max_tokens, budget=None, room_setting=None):
    """The generated-token count at which the server closes the think block, or None."""
    cap = ThinkCap(budget, "max_thinking_tokens") if budget else None
    room = answer_room(max_tokens, room_setting)
    if room > 0:
        limit = max(1, max_tokens - room)
        if cap is None or cap.tokens > limit:
            cap = ThinkCap(limit, "answer_room")
    return cap


def skips_thinking(max_tokens, budget=None, room_setting=None, minimum=0):
    """True when the request is served with thinking off: no explicit thinking budget, and the answer room leaves
    fewer than `minimum` tokens (max_tokens - room, before think_cap's max(1, ...)) before the close.

    An explicit budget (the request's or --max-thinking-tokens) is the user's intent, so it never skips. Room 0 never
    closes the block, so it never skips. minimum 0 is the rule off.
    """
    if minimum <= 0 or budget:
        return False
    room = answer_room(max_tokens, room_setting)
    return room > 0 and max_tokens - room < minimum


def close_ids(tokenizer, end_tag=THINK_END, before="\n", after="\n\n"):
    """Token IDs of the forced close. The end tag is its own token, as the template writes it. The Hy4-flash template
    writes the answer directly after its end tag, so it uses before="" and after="" (hy4_format.THINK_END)."""
    end = tokenizer.token_to_id(end_tag)
    if end is None:
        raise ValueError("The tokenizer has no " + end_tag + " token.")
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids if text else []
    return tuple(encode("\n\n" + QWEN_BUDGET_SENTENCE + before)) + (end,) + tuple(encode(after))
