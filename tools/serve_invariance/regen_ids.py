#!/usr/bin/env python3
"""Re-make the frozen prompt ids of the serve invariance bundles from their request files and the public tokenizer.

    python3 tools/serve_invariance/regen_ids.py CHECKPOINT OUTPUT_DIR

Each bundle-*.request.json in bench/suite/serve_invariance is tokenized by the frontend tokenizer of the server
(`server.tokenization.ModelTokenizer`, the one that tools/serve_config_gate.py uses) and written as bundle-*.ids.json in
the format of the frozen files. CPU only. The leak guard allow entry of a bundle names this script as its generator
(tools/leak_remake.py re-makes the file and compares the bytes).
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(argv):
    from server.api import validate_request
    from server.tokenization import ModelTokenizer
    checkpoint, output = Path(argv[1]), Path(argv[2])
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = ModelTokenizer(checkpoint)
    for request_file in sorted((ROOT / "bench/suite/serve_invariance").glob("bundle-*.request.json")):
        request = validate_request(json.loads(request_file.read_text()), "chat", "crucible")
        ids = [int(token) for token in tokenizer.encode(request)]
        name = request_file.name.replace(".request.json", ".ids.json")
        (output / name).write_text(json.dumps(ids, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv)
