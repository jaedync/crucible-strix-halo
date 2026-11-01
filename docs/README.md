# Documentation

These documents are part of the public source tree.

| Document | Content |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | The modules, the data flow, the exactness rules, serving, caches and speculative decoding |
| [how-it-got-fast.md](how-it-got-fast.md) | Six ideas behind the speed, in plain words |
| [RESULTS.md](RESULTS.md) | Release data, method, hardware limits and the dated history of earlier tables |
| [RIVALS.md](RIVALS.md) | Our own measurements of Halogen and Gufo, with dates and packs |
| [QUALITY.md](QUALITY.md) | KL from the 16-bit model and next-token quality |
| [UPDATES.md](UPDATES.md) | Update, rollback and release watch |
| [NGRAM16.md](NGRAM16.md) | The opt-in 16-bit n-gram table |
| [BUILD.md](BUILD.md) | The build, the CPU tests and the GPU gates |
| [CALIBRATION.md](CALIBRATION.md) | The measured memory, latency and compute limits of gfx1151 |
| [BENCHMARKS.md](BENCHMARKS.md) | The benchmark suite, the ceiling models and the frozen results |
| [EXL3_FORMAT.md](EXL3_FORMAT.md) | The EXL3 storage and rounding rules |
| [FLASHNEXT_FORWARD.md](FLASHNEXT_FORWARD.md) | The forward pass of Qwen3.8-Flash-Next and its MTP head |
| [OPENAI_CLIENTS.md](OPENAI_CLIENTS.md) | How to connect an OpenAI-compatible client |
| [SESSION_REPLAY.md](SESSION_REPLAY.md) | The conversation cache and session replay |
| [THIRD_PARTY.md](THIRD_PARTY.md) | Third-party software and the model license |
| [oracle/](oracle/) | The fp64 oracle tables of Gate 1 |

The server guide is [server/README.md](../server/README.md).
Each module also has a charter `README.md` in its directory.
