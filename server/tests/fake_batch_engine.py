"""CPU-only protocol fixture. It has no tokenizer or device state."""
import hashlib
import json
import os
import struct
import sys
import time

# FAKE_PREFILL_ROWS selects the reported chunk size; prompts shorter than it finish in one command.
rows = int(os.environ.get("FAKE_PREFILL_ROWS", "4096"))
# FAKE_SLOT_CONTEXTS selects slot classes; opens use the engine's smallest-fitting-free-slot rule.
contexts = [int(value) for value in os.environ.get("FAKE_SLOT_CONTEXTS", "128,128").split(",")]
# FAKE_STATE_DISK=1: the disk tier (engine/runtime/state_disk.h). FAKE_DISK_FILE holds the prompts that a stopping engine
# wrote; a stage command reads the longest one that a new prompt extends (FAKE_STAGE_READS polls), and the open
# restores it. End of input writes every finished prompt, after FAKE_FLUSH_DELAY seconds.
state_disk = os.environ.get("FAKE_STATE_DISK") == "1"
# A tier engine keeps running after SIGTERM until its flush ends (engine/serve/batched_serve.h KeepRunningOnTerm).
if os.environ.get("FAKE_IGNORE_TERM") == "1":
    import signal
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
disk_file = os.environ.get("FAKE_DISK_FILE")
ready = dict(event="ready", context=contexts[0], streams=len(contexts), prefill_rows=rows, slot_contexts=contexts,
             protocol="batch-v1", cache_tail=True, model=dict(arch="qwen4_exp", vocab=248320, eos=248044, layers=48, hidden=2560))
if state_disk: ready["state_disk"] = True
print(json.dumps(ready), flush=True)
disk = json.load(open(disk_file)) if state_disk and disk_file and os.path.exists(disk_file) else []
stage_polls = {}
staged = {}
finished_prompts = []
active = {}
prompts = {}
log = os.environ.get("FAKE_OPEN_LOG")
generation = 0
cache = set()
cache_enabled = os.environ.get("FAKE_CACHE") == "1"
# FAKE_CHECKPOINTS=1: each slot keeps its last prompt's end state (engine/runtime/slot_checkpoints.h). An open takes a free slot
# whose end state its prompt repeats, and resumes there; any later open on that slot replaces the state.
checkpoints = os.environ.get("FAKE_CHECKPOINTS") == "1"
# FAKE_END_RESUME=1 (re-send fix A): a prompt equal to a slot's end state resumes the whole prompt;
# its prefill command runs no row and publishes the first token.
end_resume = os.environ.get("FAKE_END_RESUME") == "1"
ends = {}
command_log = os.environ.get("FAKE_COMMAND_LOG")
rejection_mode = os.environ.get("FAKE_REJECTION_MODE", "safe")
error_op = os.environ.get("FAKE_ERROR_OP", "open")
pixel_log = os.environ.get("FAKE_PIXEL_LOG")
SLICES_PER_IMAGE = 3

def pixels_ok(image):
    """As the engine: the file holds height * width * 3 bytes whose digest matches."""
    try:
        with open(image["path"], "rb") as file: data = file.read()
    except OSError:
        return False
    shape = struct.pack("<II", image["height"], image["width"])
    return hashlib.sha256(shape + data).hexdigest() == image["digest"]


def log_pixels(ok):
    if pixel_log:
        with open(pixel_log, "a") as file: file.write(("ok" if ok else "bad") + "\n")


# FAKE_LIVE_DRAFTS=1: as engine/serve/batch_session.h BatchStepDraftFields, the first token event of a speculative
# step carries the request's cumulative counters. A spec_step proposes 3 and accepts 2; a verify_step proposes 2 and
# accepts 1.
live_drafts = os.environ.get("FAKE_LIVE_DRAFTS") == "1"


def step_counts(state, proposed, accepted):
    counts = state.setdefault("live", [0, 0, 0])
    counts[0] += proposed; counts[1] += accepted; counts[2] += 1
    return dict(zip(("draft_proposed", "draft_accepted", "spec_steps"), counts)) if live_drafts else {}


def done_event(id, state):
    if state_disk: finished_prompts.append(list(state["request"]["ids"]))
    event = dict(event="done", request_id=id, slot_generation=state["generation"],
                 prompt_tokens=len(state["request"]["ids"]), completion_tokens=state["count"],
                 finish_reason="length", prefill_seconds=0, decode_seconds=0)
    if os.environ.get("FAKE_DRAFT_STATS") == "1" and state["request"]["spec"] > 0 and state["request"]["max_tokens"] > 1:
        # As engine/serve/batch_session.h BatchDraftFields: a speculative request's counters, fixed numbers here.
        event.update(draft_proposed=40, draft_accepted=22, spec_steps=11, draft_passes=14, draft_ms=21.5,
                     draft_pos_proposed=[11, 9, 7, 5, 4, 2, 2], draft_pos_tested=[11, 9, 6, 4, 2, 1, 0],
                     draft_pos_accepted=[9, 6, 4, 2, 1, 0, 0], draft_len_hist=[0, 2, 2, 2, 1, 2, 0, 2])
        if state["request"].get("draft_detail") == 1:  # per-token counters, equal to the position arrays above: 71 is missed for 99
            event.update(draft_tally=[[70, 1, 11, 9], [71, 2, 9, 6], [72, 3, 6, 4], [73, 4, 4, 2], [74, 5, 2, 1], [75, 6, 1, 0]], draft_miss=[[71, 2, 99, 2]], draft_detail_cut=0,
                         draft_lookup_rounds=0)
    if live_drafts and "live" in state:
        # The done event's three counters equal the last step's (one counter set in the engine).
        event.update(zip(("draft_proposed", "draft_accepted", "spec_steps"), state["live"]))
    if cache_enabled:
        # This fixture checks metadata transport, not device snapshot exactness.
        event.update(cached_tokens=state["cached"], cache_class="shared")
        if state["request"].get("cache_control") != "bypass": cache.add(state["cache_key"])
    return event


line_limit = int(os.environ.get("FAKE_LINE_LIMIT", "0"))
for line in sys.stdin:
    if state_disk and json.loads(line).get("op") == "stage":
        if command_log:
            with open(command_log, "a") as file: file.write(json.dumps(dict(op="stage", n=len(json.loads(line)["request"]["ids"]))) + "\n")
        ids = json.loads(line)["request"]["ids"]
        best = max((saved for saved in disk if len(saved) < len(ids) and ids[:len(saved)] == saved), key=len, default=None)
        key = tuple(ids)
        stage_polls[key] = stage_polls.get(key, 0) + 1
        state = "none" if best is None else "reading" if stage_polls[key] <= int(os.environ.get("FAKE_STAGE_READS", "2")) else "ready"
        if state == "ready": staged[key] = len(best)
        print(json.dumps(dict(event="staged", state=state, tokens=len(best) if best else 0)), flush=True)
        print(json.dumps(dict(event="batch_end", op="stage")), flush=True)
        continue
    if line_limit and len(line) - 1 > line_limit:
        # As engine/serve/batched_serve.h: the op is unknown, so the error is not recoverable.
        print(json.dumps(dict(event="error", error="request exceeds the 2 MiB JSON-line limit", recoverable=False)), flush=True)
        print(json.dumps(dict(event="batch_end", op="")), flush=True)
        continue
    command = json.loads(line)
    if command_log:
        with open(command_log, "a") as file: file.write(json.dumps(command) + "\n")
    time.sleep(float(os.environ.get("FAKE_DELAY", "0")))
    op = command["op"]
    events = []
    cursor = None
    sliced = False  # an encoder slice names no prompt cursor, as the engine
    selected = command.get("request_ids", [command.get("request_id")])
    rejected = int(os.environ.get("FAKE_REJECT_TOKEN", "-1"))
    has_rejected = command["request"]["ids"][-1] == rejected if op == "open" else any(
        id in active and active[id]["request"]["ids"][-1] == rejected for id in selected)
    if op == error_op and has_rejected:
        error = dict(event="error", error=os.environ.get("FAKE_REJECT_TEXT", "fixture request rejected"))
        if rejection_mode != "missing-field":
            error["recoverable"] = False if rejection_mode == "unsafe" else 1 if rejection_mode == "integer-field" else True
        if rejection_mode == "opened-first":
            print(json.dumps(dict(event="opened", request_id=command["request_id"], slot_generation=1, slot_context=contexts[0])), flush=True)
        print(json.dumps(error), flush=True)
        if rejection_mode == "duplicate-error": print(json.dumps(error), flush=True)
        if rejection_mode == "missing-boundary": sys.exit(0)
        print(json.dumps(dict(event="batch_end", op="step" if rejection_mode == "wrong-boundary" else op)), flush=True)
        continue
    image_mode = os.environ.get("FAKE_IMAGE_ERROR_MODE", "")
    changed_id = command.get("request_id")
    changes = (changed_id in active and
               active[changed_id]["request"]["ids"][-1] == int(os.environ.get("FAKE_CHANGE_TOKEN", "-1")))
    if op == "spec_step" and image_mode == "op" and changes:
        # Out of protocol: a request error in a command other than prefill.
        state = active.pop(changed_id)
        print(json.dumps(dict(event="error", request_id=changed_id, slot_generation=state["generation"],
                              error="an image file does not match its digest", recoverable=True)), flush=True)
        print(json.dumps(dict(event="batch_end", op=op)), flush=True)
        continue
    # An End resume (FAKE_END_RESUME): the prefill command runs no row and names no cursor, as the engine.
    whole = op == "prefill" and command.get("request_id") in active and \
        active[command["request_id"]]["cursor"] == len(active[command["request_id"]]["request"]["ids"])
    # The engine runs no decode pass in a command that runs an encoder slice.
    slicing = op == "prefill" and changed_id in active and active[changed_id].get("slices")
    if op == "prefill" and command.get("request_ids") and not slicing:
        # Interleaved prefill: passes after layers 1..46 where (layer + 1 + every // 2) % every == 0, for a chunk
        # above 8 rows.
        state = active[command["request_id"]]
        chunk = min(rows, len(state["request"]["ids"]) - state["cursor"]) if rows else 8
        every = command["every"]
        passes = sum(1 for layer in range(1, 47) if (layer + 1 + every // 2) % every == 0) if chunk > 8 else 0
        live = list(command["request_ids"])
        for _ in range(passes):
            if not live: break
            part = []
            for id in list(live):
                decoder = active[id]
                decoder["token"] += 1
                decoder["count"] += 1
                part.append(dict(event="token", request_id=id, slot_generation=decoder["generation"], id=decoder["token"]))
                if decoder["count"] == decoder["request"]["max_tokens"]:
                    part.append(done_event(id, decoder))
                    del active[id]
                    live.remove(id)
            for event in part: print(json.dumps(event), flush=True)
            if os.environ.get("FAKE_PART_MODE") != "missing-boundary":
                print(json.dumps(dict(event="batch_part", op="verify_step")), flush=True)
    if op == "open":
        generation += 1
        request = command["request"]
        prefix = request.get("cache_prefix_tokens", 0)
        key = tuple(request["ids"][:prefix])
        cached = prefix if cache_enabled and prefix and key in cache and request.get("cache_control") == "auto" else 0
        need = len(request["ids"]) + request["max_tokens"] - 1
        used = [state["slot"] for state in active.values()]
        free = [slot for slot, size in enumerate(contexts) if slot not in used and size >= need]
        if not free:
            print(json.dumps(dict(event="error", error="no free sequence slot holds this request")), flush=True)
            print(json.dumps(dict(event="batch_end", op=op)), flush=True)
            continue
        repeats = lambda index: index in ends and (len(ends[index]) <= len(request["ids"]) if end_resume else
                                                   len(ends[index]) < len(request["ids"])) and \
            request["ids"][:len(ends[index])] == ends[index]
        slot = min(free, key=lambda index: (contexts[index], not (checkpoints and repeats(index)), index))
        if checkpoints:
            cached = len(ends[slot]) if repeats(slot) else 0
            ends[slot] = list(request["ids"])
        if state_disk: cached = max(cached, staged.pop(tuple(request["ids"]), 0))
        if log:
            with open(log, "a") as file: file.write(f"{command['request_id']} {contexts[slot]}\n")
        prompts[command["request_id"]] = request["ids"]
        images = request.get("images", [])
        active[command["request_id"]] = dict(request=request, token=request["ids"][-1], count=0, generation=generation,
                                             cursor=cached, slot=slot, cached=cached, cache_key=key,
                                             slices=SLICES_PER_IMAGE * len(images))
        opened = dict(event="opened", request_id=command["request_id"], slot_generation=generation,
                      slot_context=contexts[slot], cached_tokens=cached)
        if images: opened["vision_slices"] = SLICES_PER_IMAGE * len(images)
        events.append(opened)
    elif op == "spec_step":
        # A fixed window of three tokens, cut at max_tokens, stands in for draft and verify.
        id = command["request_id"]
        state = active[id]
        counts = step_counts(state, 3, 2)
        for _ in range(3):
            state["token"] += 1
            state["count"] += 1
            events.append(dict(event="token", request_id=id, slot_generation=state["generation"], id=state["token"]) |
                          counts)
            counts = {}
            if state["count"] == state["request"]["max_tokens"]:
                events.append(done_event(id, state))
                del active[id]
                break
    elif op == "close":
        del active[command["request_id"]]
    elif op == "prefill" and active[command["request_id"]]["slices"]:
        # One encoder slice; the first slice of each image reads its file, as the engine does.
        state = active[command["request_id"]]
        images = state["request"]["images"]
        done = SLICES_PER_IMAGE * len(images) - state["slices"]
        image = images[done // SLICES_PER_IMAGE]
        if done % SLICES_PER_IMAGE == 0 and state["request"]["ids"][-1] == int(os.environ.get("FAKE_CHANGE_TOKEN", "-1")):
            with open(image["path"], "ab") as file: file.write(b"x")  # the file changes after the open
        ok = pixels_ok(image) if done % SLICES_PER_IMAGE == 0 else True
        if done % SLICES_PER_IMAGE == 0: log_pixels(ok)
        if not ok:
            # As the engine: the request ends with its own error before any device work; the session continues.
            # FAKE_IMAGE_ERROR_MODE sends it outside that contract: another request ID or generation, or after
            # (or inside) a decode pass of the command's decoders.
            error = dict(event="error", request_id=command["request_id"] + 1000 * (image_mode == "id"),
                         slot_generation=state["generation"] + (image_mode == "generation"),
                         error="an image file is not a regular file of height * width * 3 bytes", recoverable=True)
            if image_mode in ("part-before", "part-inside") and command.get("request_ids"):
                for id in command["request_ids"]:
                    decoder = active[id]
                    decoder["token"] += 1
                    decoder["count"] += 1
                    print(json.dumps(dict(event="token", request_id=id, slot_generation=decoder["generation"],
                                          id=decoder["token"])), flush=True)
                if image_mode == "part-inside": print(json.dumps(error), flush=True)
                print(json.dumps(dict(event="batch_part", op="verify_step")), flush=True)
                if image_mode == "part-before": print(json.dumps(error), flush=True)
            else:
                events.append(error)
            del active[command["request_id"]]
        else:
            state["slices"] -= 1
        sliced = True
    elif op == "prefill" and rows and active[command["request_id"]]["cursor"] + rows < len(active[command["request_id"]]["request"]["ids"]):
        active[command["request_id"]]["cursor"] += rows
        cursor = active[command["request_id"]]["cursor"]
    else:
        ids = command["request_ids"] if op in ("step", "verify_step") else [command["request_id"]]
        for id in ids:
            state = active[id]
            # A verify step stands in for one accepted draft: speculative requests publish two tokens.
            speculative = op == "verify_step" and state["request"]["spec"] > 0 and state["request"]["max_tokens"] > 1
            counts = step_counts(state, 2, 1) if speculative else {}
            for _ in range(2 if speculative else 1):
                state["token"] += 1
                state["count"] += 1
                events.append(dict(event="token", request_id=id, slot_generation=state["generation"], id=state["token"]) |
                              counts)
                counts = {}
                if state["count"] == state["request"]["max_tokens"]:
                    events.append(done_event(id, state))
                    del active[id]
                    break
    for event in events: print(json.dumps(event), flush=True)
    boundary = dict(event="batch_end", op=op)
    # The engine names its prompt cursor after each prefill chunk (the final chunk ends the prompt).
    if op == "prefill" and os.environ.get("FAKE_NO_CURSOR") != "1" and not sliced and not whole:
        boundary["cursor"] = cursor if cursor is not None else len(prompts[command["request_id"]])
    print(json.dumps(boundary), flush=True)
    # FAKE_SETTLE_ERROR=1: the prompt end after a first token fails (engine/serve/batched_serve.h): one error
    # event after the boundary, then the engine stops.
    if os.environ.get("FAKE_SETTLE_ERROR") == "1" and op == "prefill" and any(e["event"] == "token" for e in events):
        print(json.dumps(dict(event="error", error="prompt end failed: fixture", recoverable=False)), flush=True)
        sys.exit(1)
if state_disk and disk_file:
    time.sleep(float(os.environ.get("FAKE_FLUSH_DELAY", "0")))
    with open(disk_file, "w") as file: json.dump(disk + finished_prompts, file)
