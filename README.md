# Crucible

An inference engine for one model on one machine: Qwen3.8-Flash-Next (EXL3) on AMD Strix Halo, a Ryzen AI Max+ 395
with 128 GB. Its OpenAI-compatible server gives local coding agents 4 sessions of 262,144 tokens.

<!-- results:start charts -->
<p>
<picture><source media="(prefers-color-scheme: dark)" srcset="docs/img/prefill-by-context-dark.svg"><img src="docs/img/prefill-by-context-light.svg" width="412" alt="Line chart: Cold prefill by context length, in tokens per second: 4.05 bpw from 1,830 at 32K to 1,622 at 250K; 3.05 bpw from 1,855 at 32K to 1,648 at 250K. v0.33.0, Nov 3, 2026."></picture>
<picture><source media="(prefers-color-scheme: dark)" srcset="docs/img/decode-by-context-dark.svg"><img src="docs/img/decode-by-context-light.svg" width="412" alt="Line chart: Decode by context length, in tokens per second: 4.05 bpw from 71.7 at 32K to 62.2 at 250K; 3.05 bpw from 77.0 at 32K to 69.1 at 250K. v0.33.0, Nov 3, 2026."></picture>
</p>

<sub>One request at a time, default configuration, balanced power profile, v0.33.0 (Nov 3, 2026). No new run for v0.37.0.</sub>
<!-- results:end charts -->

## Install

<!-- The two ways below are whole blocks. To change their order, move one block above the other; no other edit. -->
**One command.** It downloads the release engine (about 25 MB) and checks its sha256.

```sh
curl -fsSL https://raw.githubusercontent.com/jaedync/crucible-strix-halo/main/install.sh | bash
```

**Build from source.** It takes about 16 minutes and needs the ROCm 7.2.4 compiler
([why](docs/QUICKSTART.md#step-4-build-the-engine)).

<!-- results:start clone -->
```sh
git clone --branch v0.37.0 https://github.com/jaedync/crucible-strix-halo.git
cd crucible-strix-halo
./crucible setup
```
<!-- results:end clone -->

Setup checks the machine, downloads the model (about 107 GB), starts the server and sends a test request. Then send
your own:

```sh
curl -sS http://127.0.0.1:8790/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"crucible","messages":[{"role":"user","content":"Say hello."}]}'
```

Read more: [setup guide](docs/QUICKSTART.md) · [service](docs/QUICKSTART.md#run-as-a-service) ·
[Docker](docs/DOCKER.md) · [clients](docs/OPENAI_CLIENTS.md) · [updates](docs/UPDATES.md)

## What you need

- **Machine:** Ryzen AI Max+ 395 with Radeon 8060S (gfx1151) and 128 GB of memory. Other machines are not supported.
- **System:** Linux with Python 3.12 or newer. The tested system is Arch Linux.
- **ROCm:** the release engine runs on ROCm 10.1 (recommended) and on 7.2.4. A source build needs the 7.2.4 compiler.
- **GPU memory:** two kernel parameters let the GPU use the system memory. Setup prints the change and applies it
  only after you type `yes`.
- **Disk:** an SSD with about 110 GB for the model, and up to 200 GiB for the conversation cache.

Read more: [step 1 and step 2 of the setup guide](docs/QUICKSTART.md#step-1-rocm-the-hip-compiler-and-gpu-access)

## Configurations

Setup installs the pack [`4.05bpw_h6_ng6`](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/4.05bpw_h6_ng6).
With `--pack 3.05` it installs [`3.05bpw_h5_ng5`](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/3.05bpw_h5_ng5),
the faster pack with the lower [quality](docs/QUALITY.md). `./crucible setup --layout NAME` sets the sessions:

- `default`: 4 sessions of 262,144 tokens.
- `long`: 1 session of 524,288 tokens and 2 of 262,144.
- `two-long`: 2 sessions of 524,288 tokens.

Each one needs about 28 GB of GPU memory for the sessions. A request that fits in 262,144 tokens gets the same tokens in
all three. A longer request uses YaRN. Measured in `long`:

<!-- results:start long -->
| `long`, 4.05 pack | 250,000 tokens | 490,000 tokens (YaRN) |
|---|---:|---:|
| Cold prompt: first token | 149.0 s | 340.4 s |
| Cold prompt: prefill | 1,677.6 tok/s | 1,436.2 tok/s |
| Decode, code | 67.7 tok/s | 59.8 tok/s |

<sub>v0.37.0, measured Nov 11, 2026: 1 run, balanced power profile.</sub>
<!-- results:end long -->

Read more: [long context](docs/LONG_CONTEXT.md)

## Speed by release

<!-- results:start releases -->
<p>
<picture><source media="(prefers-color-scheme: dark)" srcset="docs/img/prefill-by-release-dark.svg"><img src="docs/img/prefill-by-release-light.svg" width="412" alt="Line chart: Cold prefill of a 32K prompt, by release, in tokens per second: 4.05 bpw from 1,585 at v0.22.0 to 1,831 at v0.34.0; 3.05 bpw from 1,656 at v0.22.0 to 1,877 at v0.34.0. Oct 27 to Nov 7, 2026."></picture>
</p>

<sub>Each release ran with the settings of its own time, balanced power profile. No new run for v0.37.0.</sub>
<!-- results:end releases -->

<!-- results:start table -->
| v0.34.0 | 4.05 pack | 3.05 pack |
|---|---:|---:|
| Cold 32K prompt: first token | 17.1 s | 16.6 s |
| Cold 130K prompt: first token | 72.4 s | 71.1 s |
| Same 32K prompt again: first token | 0.019 s | 0.015 s |
| Decode at 32K, 1 session, code | 71.2 tok/s | 76.8 tok/s |
| Decode at 32K, 1 session, prose | 58.4 tok/s | 60.7 tok/s |
| Decode, 4 sessions, total | 118.3 tok/s | 129.0 tok/s |

<sub>v0.34.0, measured Nov 7, 2026: median of 2 runs, balanced power profile, default configuration. No new run for v0.37.0.</sub>
<!-- results:end table -->

Read more: [results and method](docs/RESULTS.md) · [history](docs/HISTORY.md) ·
[Halogen and Gufo on the same machine](docs/RIVALS.md) · [how it got fast](docs/how-it-got-fast.md)

## Features

- Its own ROCm/HIP kernels in C++20.
- Speculative decoding that is exact: the model checks every draft token, so only the speed changes.
- A conversation cache in memory and on the SSD. A later turn pays only for its new tokens, also after a restart.
- An OpenAI-compatible API with streaming, tool calls and image input, up to 8 sessions.
- One tool, `./crucible`, for setup, checks, the service, updates with rollback, and removal.
- [Kiln](https://github.com/jaedync/crucible-kiln), a dashboard and OpenAI-compatible gateway for Crucible.

## Read more

- [Architecture](docs/ARCHITECTURE.md), [server guide](server/README.md) and [all documents](docs/README.md).
- [Changelog](CHANGELOG.md): the changes and the known limits of each release.
- [Security](SECURITY.md) and [remote access](docs/REMOTE_ACCESS.md). The server listens on `127.0.0.1` with no key
  and no TLS. A key is required for an address that is not loopback, except in the Docker image.
- [Updates](docs/UPDATES.md): once in 24 hours the server asks GitHub if a newer release exists.
  `--no-update-check` turns this off.

This is research software for one GPU architecture. It is not a production service.

## Licenses

The engine uses Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). **The model weights use other terms:** the Qwen
Community License 1.0, with more conditions for commercial use. Read the
[model license note](docs/THIRD_PARTY.md#model-weights) before you download a pack. This repository does not include
the weights.
