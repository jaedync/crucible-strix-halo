"""The taint rule of the leak guard (WP part 1): a file made from capture data carries the canary.

A derived file is tainted when any input is tainted. The canary comes from server/leak_canary.py.

  list shard   the first line is `#<CANARY>`. It has no space, so mtp_dump skips it (it skips a line without a space).
  ids file     the same first line. read_ids() skips a line that starts with `#`.
  JSON         the key "canary".
  safetensors  the metadata key "crucible_capture_canary".
  stk file     the canary bytes as a trailer after the fp16 rows (data.py reads only n rows).
  checkpoint   the key "canary" in the saved dict.

Text from public sources only (prep.py) is not tainted, so a draft head trained on it can ship. The rule is checked
by tests/test_mtp_tune_taint.py, and by the repository guard and the export scan.
"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from server.leak_canary import CANARY, CANARY_BYTES, refuse_inside_git_tree, shield_folder  # noqa: E402,F401

HEADER = "#" + CANARY
PUBLIC_HEADER = "#CRUCIBLE-PUBLIC-SOURCE-v1"   # first word of the first line of a list made from public text only (prep.py)
PROVENANCE = "PROVENANCE.json"
META_KEY = "crucible_capture_canary"
PUBLIC_META_KEY = "crucible_provenance"      # metadata value "public": the positive marker of a head
STK_TRAILER = CANARY_BYTES          # a tainted .stk file ends with these bytes, after n * W * 2 bytes of rows
STK_TRAILER_LEN = len(CANARY_BYTES)


def _first_line(path):
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.readline().rstrip("\n")


def file_has_canary(path):
    """True when the canary is on any line of the file (mtp_dump and the readers skip a `#` line, so the first line alone
    is not enough: outside check H7)."""
    with open(path, "rb") as handle:
        return any(CANARY_BYTES in line for line in handle)


def first_line_tainted(path):
    return _first_line(path) == HEADER


def list_tainted(path):
    return first_line_tainted(path) or file_has_canary(path)


def public_line(body):
    """The first line of a public list: the public mark and the SHA-256 of the rows (the bytes after this line). Rows
    added or changed later (cat public.list unknown.list) break the digest, so the list is no longer public (H7)."""
    return f"{PUBLIC_HEADER} sha256={hashlib.sha256(body).hexdigest()}"


def write_list(path, rows, tainted, public=False):
    """rows: (name, ids). One sequence per line: NAME comma-separated-ids.

    A tainted list starts with the canary line. A list from public text only (public=True) starts with the public
    line, which holds the digest of the rows. A list with neither line has unknown provenance, and everything made from it counts as tainted.
    """
    body = "".join(name + " " + ",".join(map(str, ids)) + "\n" for name, ids in rows)
    with open(path, "w") as handle:
        if tainted:
            handle.write(HEADER + "\n")
        elif public:
            handle.write(public_line(body.encode()) + "\n")
        handle.write(body)


def redact_path(path):
    """A path as a short name: the file name and 8 hex characters of a hash of the full path. No directory shows."""
    import hashlib

    path = str(path)
    return f"{os.path.basename(path) or '<root>'}~{hashlib.sha256(path.encode()).hexdigest()[:8]}"


def redact(text):
    """TEXT without any absolute path: each becomes redact_path(). Error messages go through this."""
    import re

    return re.sub(r"(?<![\w.~])/(?:[^\s:,;'\"()\[\]]+/)*[^\s:,;'\"()\[\]]*", lambda m: redact_path(m.group()), text)


def _ints(text, path):
    try:
        return [int(v) for v in text.split(",")]
    except ValueError:
        # int() would quote the bad value, which is file content: say only where.
        raise ValueError(f"malformed id list in {redact_path(path)}") from None


def read_list(path):
    with open(path) as handle:
        for line in handle:
            if line.startswith("#") or " " not in line:
                continue
            name, rest = line.rstrip("\n").split(" ", 1)
            yield name, _ints(rest, path)


def write_ids(path, ids, tainted):
    with open(path, "w") as handle:
        if tainted:
            handle.write(HEADER + "\n")
        handle.write(",".join(map(str, ids)) + "\n")


def read_ids(path):
    with open(path) as handle:
        body = "".join(line for line in handle if not line.startswith("#"))
    return _ints(body.strip(), path)


def list_public(path):
    """True only for a list whose first line is the public line WITH the digest of the rows that follow, and whose rows
    hold no line that starts with `#` and no canary. mtp_dump and read_list skip a `#` line, so a canary line could hide
    behind the public line. cat public.list tainted.list, or public.list unknown.list, is therefore not public (H7)."""
    with open(path, "rb") as handle:
        first = handle.readline().rstrip(b"\n")
        body = handle.read()
    if first.decode("utf-8", "replace") != public_line(body):
        return False
    return not (CANARY_BYTES in body or body.startswith(b"#") or b"\n#" in body)


def list_rows(path):
    """{name: ids} of the sequences of a list, as mtp_dump reads them (a line with a space that does not start with #)."""
    rows = {}
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if " " in line and not line.startswith("#"):
                name, rest = line.rstrip("\n").split(" ", 1)
                rows[name] = rest
    return rows


def list_names(path):
    """The sequence names of a list (the lines with a space that do not start with `#`), as mtp_dump reads them."""
    names = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if " " in line and not line.startswith("#"):
                names.append(line.split(" ", 1)[0])
    return names


def any_tainted(paths):
    return any(list_tainted(p) for p in paths)


def stk_file_tainted(path):
    """True when a .stk file ends with the canary trailer."""
    with open(path, "rb") as handle:
        handle.seek(0, 2)
        if handle.tell() < STK_TRAILER_LEN:
            return False
        handle.seek(-STK_TRAILER_LEN, 2)
        return handle.read() == STK_TRAILER


def dir_tainted(directory):
    """True when any .ids file, list shard or .stk file in DIRECTORY (top level) carries the canary."""
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if name.endswith((".ids", ".list")) and (_first_line(path) == HEADER or file_has_canary(path)):
            return True
        if name.endswith(".stk") and stk_file_tainted(path):
            return True
    return False


MARKER_VERSION = 1


def _digest(directory):
    """SHA-256 over the names and the full content of every file in DIRECTORY (the marker file excluded), streamed.

    Returns None when DIRECTORY holds anything that is not a plain file (a sub folder, a link): not clean.
    """
    import hashlib

    h = hashlib.sha256(f"crucible-provenance-v{MARKER_VERSION}\n".encode())
    for name in sorted(os.listdir(directory)):
        if name == PROVENANCE:
            continue
        path = os.path.join(directory, name)
        if os.path.islink(path) or not os.path.isfile(path):
            return None
        h.update(f"{name}\0{os.path.getsize(path)}\0".encode())
        with open(path, "rb") as handle:
            while chunk := handle.read(1 << 22):
                h.update(chunk)
    return h.hexdigest()


def mark_dir(directory, source):
    """Write the positive marker of a folder. Only source == "public" makes the folder clean (see dir_is_clean)."""
    import json

    with open(os.path.join(directory, PROVENANCE), "w") as handle:
        json.dump({"v": MARKER_VERSION, "source": source, "digest": _digest(directory)}, handle)


def dir_is_clean(directory):
    """True only with a positive marker: no canary in any file, a marker "public", and the files unchanged since.

    A folder without the marker (a direct mtp_dump call, a crash before the tagger, an old folder) is not clean.
    """
    import json

    if dir_tainted(directory):
        return False
    try:
        with open(os.path.join(directory, PROVENANCE)) as handle:
            doc = json.load(handle)
    except (OSError, ValueError):
        return False
    digest = _digest(directory)
    return (doc.get("v") == MARKER_VERSION and doc.get("source") == "public" and digest is not None
            and doc.get("digest") == digest)


def safetensors_tainted(path):
    """True when the metadata of a safetensors file holds the canary key. Reads the JSON header only."""
    import json

    with open(path, "rb") as handle:
        size = int.from_bytes(handle.read(8), "little")
        if size > 1 << 28:
            return True                      # a header that large is not a safetensors header: do not trust it
        try:
            header = json.loads(handle.read(size))
        except ValueError:
            return True
    meta = header.get("__metadata__", {})
    return META_KEY in meta or meta.get(PUBLIC_META_KEY) != "public"


def run_tainted(dirs=(), states=(), sidecar=None):
    """The fail-closed taint of a training or evaluation run: unknown provenance counts as tainted.

    DIRS are stack or list folders (each must be clean), STATES are loaded checkpoint dicts (init, resumed),
    SIDECAR is an exported head. A missing tag can make a run tainted, never clean.
    """
    if any(not dir_is_clean(d) for d in dirs):
        return True
    if any(s is not None and state_tainted(s) for s in states):
        return True
    return bool(sidecar) and safetensors_tainted(sidecar)


def source_label(tainted, public):
    """The provenance label of a folder that a tool writes: "tainted", "public" (a person said so with --public) or
    "unknown" (the default). Only "public" makes a folder clean. Both flags together are a mistake (H14)."""
    if tainted and public:
        raise ValueError("a source cannot be both tainted and public")
    return "tainted" if tainted else "public" if public else "unknown"


def guard_output(path, tainted, is_dir=True):
    """Call before a tool writes tainted output at PATH: refuse a git work tree (H13), make the folder, shield it.

    A public output may lie anywhere. For a file output (is_dir=False) PATH is the file and its folder is checked.
    """
    if not tainted:
        return
    folder = os.path.abspath(path if is_dir else os.path.dirname(os.path.abspath(path)))
    refuse_inside_git_tree(folder, "data made from capture data")
    os.makedirs(folder, mode=0o700, exist_ok=True)
    if is_dir:
        shield_folder(folder)


def _only_files_of_this_list(list_path, out_dir):
    rows = list_rows(list_path)
    made = {name + ext for name in list_names(list_path) for ext in (".stk", ".ids", ".f32", ".mtp")}
    since = os.path.getmtime(list_path)
    for name in sorted(os.listdir(out_dir)):
        if name == PROVENANCE:
            continue
        path = os.path.join(out_dir, name)
        stale = False
        if name.endswith(".ids") and name in made and os.path.isfile(path):
            stale = read_ids(path) != _ints(rows[name[:-4]], path)      # not the ids of this list: a leftover file
        if name not in made or not os.path.isfile(path) or os.path.islink(path) or os.path.getmtime(path) < since or stale:
            raise ValueError(f"{out_dir} holds a file that this list did not make ({redact_path(path)}): "
                             "use an empty output folder for each dump")


def tag_dump(list_path, out_dir, public=False):
    """After mtp_dump: when LIST_PATH is tainted, put the canary into every .stk (trailer) and .ids (first line) that
    the dump wrote to OUT_DIR. Returns the number of files changed (0 for an untainted list, or a second call).

    mtp_dump (bench/mtp_dump.hip) skips the header line of the list and writes its outputs without it. A run that reads
    the stacks would then look clean. A folder is clean only with the marker (dir_is_clean), so a dump that skips this
    tool is tainted by default. The dump scripts call this tool after every dump (`tag-dump` below). The
    --check outputs (.f32, .mtp) cannot carry the canary, so a tainted dump with them is refused.
    """
    if list_public(list_path) or (public and not first_line_tainted(list_path) and not file_has_canary(list_path)):
        # The one positive marker. It vouches for the whole folder, so the folder must hold only what THIS list made
        # (outside check H8): NAME.stk, NAME.ids (and NAME.f32, NAME.mtp) of its sequences, written after the list.
        _only_files_of_this_list(list_path, out_dir)
        mark_dir(out_dir, "public")
        return 0
    refuse_inside_git_tree(out_dir, "data made from capture data")   # a tainted dump never lies in a git work tree (H13)
    names = sorted(os.listdir(out_dir))
    extra = [n for n in names if n.endswith((".f32", ".mtp"))]
    if extra:
        raise ValueError(f"{out_dir} holds --check output ({extra[0]}, ...) made from a tainted list: "
                         "dump again without --check, and delete these files")
    changed = 0   # a tainted list, or one of unknown provenance (no header): over-taint, never trust
    for name in names:
        path = os.path.join(out_dir, name)
        if name.endswith(".stk") and not stk_file_tainted(path):
            with open(path, "ab") as handle:
                handle.write(STK_TRAILER)
            changed += 1
        elif name.endswith(".ids") and _first_line(path) != HEADER:
            with open(path) as handle:
                body = handle.read()
            tmp = path + ".tmp"
            with open(tmp, "w") as handle:
                handle.write(HEADER + "\n" + body)
            os.replace(tmp, path)
            changed += 1
    return changed


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="leak guard taint tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tag-dump", help="tag the output folder of mtp_dump when its LIST is tainted")
    t.add_argument("list")
    t.add_argument("out")
    t.add_argument("--public", action="store_true",
                   help="a person states that the list has public text only (an older list without the public line)")
    a = ap.parse_args(argv)
    try:
        n = tag_dump(a.list, a.out, a.public)
    except ValueError as error:
        print(f"taint: {error}", file=sys.stderr)
        return 1
    print(f"taint: {n} file(s) tagged in {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


def tag_doc(doc, tainted):
    return {**doc, "canary": CANARY} if tainted else dict(doc)


def tag_meta(meta, tainted):
    """Head metadata: the canary when tainted, else the positive public marker. No marker = unknown = tainted."""
    return {**meta, META_KEY: CANARY} if tainted else {**meta, PUBLIC_META_KEY: "public"}


def tag_checkpoint(state, tainted):
    """Checkpoint dict: the canary when tainted, else provenance "public". No marker = unknown = tainted."""
    return {**state, "canary": CANARY} if tainted else {**state, "provenance": "public"}


def state_tainted(state):
    """True unless the checkpoint dict carries the positive public marker and no canary."""
    return bool(state.get("canary")) or state.get("provenance") != "public"
