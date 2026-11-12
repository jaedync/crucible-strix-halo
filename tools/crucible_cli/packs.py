"""The supported model packs (the same revisions as the README)."""
import json
from dataclasses import dataclass, field
from pathlib import Path

REPO = "turboderp/Qwen3.8-Flash-Next-exl3"
DEFAULT = "4.05"
REQUIRED = ("config.json", "tokenizer.json", "ngram_embedding.safetensors")


@dataclass(frozen=True)
class Pack:
    bpw: str
    revision: str
    download_gb: int
    disk_gb: int
    extra_env: dict = field(default_factory=dict)

    @property
    def folder(self):
        return f"Qwen3.8-Flash-Next-exl3-{self.revision}"


PACKS = {
    "4.05": Pack("4.05", "4.05bpw_h6_ng6", 107, 110),
    "3.05": Pack("3.05", "3.05bpw_h5_ng5", 85, 90, {}),
}


def verify(ckpt):
    """Return (state, detail). state is missing, incomplete or ok."""
    ckpt = Path(ckpt)
    if not ckpt.is_dir():
        return "missing", "not downloaded yet"
    problems = [n for n in REQUIRED if not (ckpt / n).is_file()]
    index = ckpt / "model.safetensors.index.json"
    if not index.is_file():
        problems.append("model.safetensors.index.json")
    else:
        try:
            shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        except (ValueError, KeyError):
            shards = []
            problems.append("model.safetensors.index.json (unreadable)")
        problems += [s for s in shards if not (ckpt / s).is_file()]
    partial = sorted(p.name for p in (ckpt / ".cache/huggingface/download").glob("*.incomplete")) \
        if (ckpt / ".cache/huggingface/download").is_dir() else []
    if partial:
        problems.append(f"{len(partial)} incomplete download file(s)")
    if problems:
        return "incomplete", "missing or incomplete: " + ", ".join(problems)
    return "ok", "all files are present"
