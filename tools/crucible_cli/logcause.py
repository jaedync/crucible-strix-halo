"""Why the last server stopped, from .crucible/server.log (E3). Read only: the server and the engine write the log."""
READY = "Crucible is ready at"   # server/__main__.py: the line of each start that loaded the model
CAUSES = ("The engine stopped", "server error:", "batch memory does not fit", "batched serve error", "HIP error",
          "out of memory", "prompt end failed")
TAIL_BYTES = 1 << 20


def last_stop(path, limit=3):
    """The last LIMIT cause lines after the last ready line of the log, or [] (no log, or the last run stopped
    without a cause: a normal stop writes none of these lines)."""
    try:
        with open(path, "rb") as log:
            log.seek(0, 2)
            log.seek(max(0, log.tell() - TAIL_BYTES))
            lines = log.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []
    ready = [i for i, line in enumerate(lines) if line.startswith(READY)]
    after = lines[ready[-1] + 1:] if ready else lines
    found = [line.strip() for line in after if any(word in line for word in CAUSES)]
    return [line[:400] for line in found[-limit:]]
