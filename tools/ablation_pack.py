#!/usr/bin/env python3
"""Build the 4.05 CRUCIBLE_ABLATE sidecar from the reviewed Heretic sidecar.

  repackage   The reviewed andrevp/Heretic sidecar, unchanged, with metadata
              for the 4.05 pack. A pin is looked up by sha256 alone, so the
              same bytes cannot serve two pack geometries; this rewrites only
              the header and keeps every tensor value byte for byte.

The input is third-party data. Read it, never execute it, and keep both the
input and the output out of git. The tool refuses to write inside the
repository.

Usage:
  tools/ablation_pack.py repackage SOURCE.safetensors OUT_DIR [--pack 405]

For the 4.05 pack the tool checks that the output has the sha256 that
kAblationPins (engine/layers/ablation.h) pins, and writes nothing otherwise.
Each written file prints:
  SIDECAR <path> sha256=<sha> bytes=<n> sites=<n> pack=<name> expert_k=<k> head_k=<k>
as one pin line in the format of kAblationPins.
"""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

TOOL = "tools/ablation_pack.py"

# The engine's geometry and tolerance (engine/ablation.h, engine/geometry.h).
HIDDEN = 2560
LAYERS = 48
SITES = ("abl_attn", "abl_mlp")
SITE_COUNT = LAYERS * len(SITES)
FORMAT = "crucible.ablate.1"

# Pack name and checkpoint geometry, as the pins carry them. The geometry
# comes from the pack's expert K and LM-head K (3.05bpw_h5 is 3 and 5).
PACKS = {
    "305": ("3.05bpw_h5_ng5", 3, 5),
    "405": ("4.05bpw_h6_ng6", 4, 6),
}

HERETIC_SHA256 = "05f77ae18082de58b31fb645f52ec5689dce22820add8bb81439e06f01998e5f"
HERETIC_SOURCE = ("andrevp/Qwen3.8-Flash-Next-exl3-3.05bpw-heretic@"
                  "da2b870f01da36c6601557bc386d0d45c51b36fc abliteration.safetensors")
HERETIC_LICENSE = "Qwen Community License 1.0"
# The sha256 that the repackaged file must have, per pack. The engine pins
# it (kAblationPins). The source file already is the 3.05 pin.
HERETIC_OUTPUT_SHA256 = {
    "405": "1bb0d4132a5798d2a7da7820b65af1e5d37e39a0cf5726f1986be924f2678baa",
}

NO_PLE_SLOT = ("ple_value blk.1 and token_embd have no slot: the sidecar covers "
               "sublayer outputs of layers 0..47 only")

REPO_ROOT = Path(__file__).resolve().parents[1]


class SidecarError(ValueError):
    """Bad input, or an output path this tool must not write."""


def site_key(layer, site, part):
    return f"model.language_model.layers.{layer}.{site}.{part}"


def _float_label(value):
    """Canonical label for a lambda: 1.25 stays 1.25, 1.50 becomes 1.5."""
    return f"{value:.2f}".rstrip("0").rstrip(".")


def scaled(values, factor):
    """Round each product to f32 once; the engine reads only the stored value."""
    return [struct.unpack("<f", struct.pack("<f", factor * value))[0] for value in values]


def _header_entry(offset, count):
    return {"dtype": "F32", "shape": [count],
            "data_offsets": [offset, offset + count * 4]}


def safetensors_bytes(tensors, metadata):
    """Write an F32 safetensors file with a leading __metadata__ block.

    The tensor order is the order of `tensors`, so the same inputs always give
    the same bytes and the same sha256.
    """
    header = {"__metadata__": {str(k): str(v) for k, v in metadata.items()}}
    payload = bytearray()
    for name, values in tensors:
        if len(values) != HIDDEN:
            raise SidecarError(f"{name}: {len(values)} values, expected {HIDDEN}")
        header[name] = _header_entry(len(payload), HIDDEN)
        payload += struct.pack(f"<{HIDDEN}f", *values)
    text = json.dumps(header, separators=(",", ":"), sort_keys=False).encode()
    return struct.pack("<Q", len(text)) + text + bytes(payload)


def parse_safetensors(data):
    """Split a safetensors file into (metadata, {name: raw tensor bytes})."""
    if len(data) < 8:
        raise SidecarError("file is too short for a safetensors header")
    (length,) = struct.unpack("<Q", data[:8])
    if length > len(data) - 8:
        raise SidecarError("header length exceeds the file size")
    header = json.loads(data[8:8 + length])
    payload = data[8 + length:]
    metadata = {}
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            metadata = {str(k): str(v) for k, v in info.items()}
            continue
        begin, end = info["data_offsets"]
        if end > len(payload):
            raise SidecarError(f"{name}: data_offsets lie outside the payload")
        tensors[name] = payload[begin:end]
    return metadata, tensors


def _refuse_in_repo(path):
    resolved = Path(path).resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise SidecarError(f"{resolved} is inside the repository {REPO_ROOT}; "
                           "sidecars stay out of git")


def _write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _split_key(name):
    """Split an ablation key into (layer, site, part); None if it is not one."""
    prefix = "model.language_model.layers."
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix):]
    digits, _, tail = rest.partition(".")
    if not digits.isdigit():
        return None
    site, _, part = tail.partition(".")
    if site not in SITES or part not in ("a", "b"):
        return None
    return int(digits), site, part


def build_heretic_repackage(source_bytes, pack_key):
    """Keep every tensor byte for byte; rename the pack in the metadata."""
    metadata, tensors = parse_safetensors(source_bytes)
    sha = hashlib.sha256(source_bytes).hexdigest()
    if sha != HERETIC_SHA256:
        raise SidecarError(f"source sha256 is {sha}, not the reviewed Heretic sidecar "
                           f"{HERETIC_SHA256}")
    parts = {}
    for name in tensors:
        split = _split_key(name)
        if split is None:
            raise SidecarError(f"source tensor {name} is not an ablation key")
        if len(tensors[name]) != HIDDEN * 4:
            raise SidecarError(f"source tensor {name} is not one row of {HIDDEN} f32")
        parts[split] = name
    pairs = {(layer, site) for (layer, site, _) in parts}
    missing = [f"{layer} {site}" for (layer, site) in pairs
               if (layer, site, "a") not in parts or (layer, site, "b") not in parts]
    if missing:
        raise SidecarError(f"source is missing a or b for: {', '.join(sorted(missing))}")
    kinds = sorted({site for (_, site) in pairs})
    layers = sorted({layer for (layer, _) in pairs})
    name, expert_k, head_k = PACKS[pack_key]
    fresh = {
        "format": FORMAT,
        "mode": "heretic-repackage",
        "pack": name,
        "expert_k": str(expert_k),
        "head_k": str(head_k),
        "layers": f"{len(layers)} of {LAYERS}",
        "sites": ",".join(kinds),
        "site_count": str(len(pairs)),
        "tensors": "values unchanged from the source sidecar",
        "original_sha256": sha,
        "original_source": HERETIC_SOURCE,
        "original_license": HERETIC_LICENSE,
        "generator": TOOL,
        "note": NO_PLE_SLOT,
    }
    # Reuse the source header entries so offsets and shapes stay identical.
    header = {"__metadata__": fresh}
    payload = bytearray()
    for name_key, raw in tensors.items():
        header[name_key] = _header_entry(len(payload), HIDDEN)
        payload += raw
    text = json.dumps(header, separators=(",", ":"), sort_keys=False).encode()
    data = struct.pack("<Q", len(text)) + text + bytes(payload)
    # The copy carries the same vectors, so check them against the source.
    _, written = parse_safetensors(data)
    if written != tensors:
        raise SidecarError("repacked tensors differ from the source sidecar")
    return data, len(pairs)


def pin_line(sha, pack_key, what):
    name, expert_k, head_k = PACKS[pack_key]
    return (f'    {{"{sha}",\n'
            f'     "{what}",\n'
            f'     {expert_k}, {head_k}}},')


def _report(path, data, pack_key, what, sites=SITE_COUNT):
    sha = hashlib.sha256(data).hexdigest()
    name, expert_k, head_k = PACKS[pack_key]
    print(f"SIDECAR {path} sha256={sha} bytes={len(data)} sites={sites} "
          f"pack={name} expert_k={expert_k} head_k={head_k}")
    print(pin_line(sha, pack_key, what))
    return sha


def command_repackage(args):
    out_dir = Path(args.out_dir)
    _refuse_in_repo(out_dir)
    source = Path(args.source)
    data, sites = build_heretic_repackage(source.read_bytes(), args.pack)
    expected = HERETIC_OUTPUT_SHA256.get(args.pack)
    if expected is not None and hashlib.sha256(data).hexdigest() != expected:
        raise SidecarError(f"output sha256 is {hashlib.sha256(data).hexdigest()}, not the "
                           f"pinned {expected}; nothing written")
    path = out_dir / f"heretic-andrevp-{PACKS[args.pack][0]}.safetensors"
    what = (f"andrevp Heretic sidecar values unchanged, {PACKS[args.pack][0]}, "
            f"repackage of {HERETIC_SHA256[:8]}")
    _report(path, data, args.pack, what, sites)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    repackage = sub.add_parser("repackage", help="Heretic sidecar for another pack")
    repackage.add_argument("source", help="path to the reviewed Heretic sidecar")
    repackage.add_argument("out_dir", help="output directory, outside the repository")
    repackage.add_argument("--pack", default="405", choices=sorted(PACKS))
    repackage.set_defaults(func=command_repackage)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SidecarError as error:
        print(f"ERROR {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
