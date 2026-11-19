# Third-party software and models

The engine license is Apache-2.0 ([LICENSE](../LICENSE), [NOTICE](../NOTICE)).
Dependency licenses and model licenses remain separate.
This inventory is not legal advice or a complete source-origin audit.

## Reference implementation

Crucible's EXL3 specification and CPU reference checks use ExLlamaV3 as a format reference.
The verified upstream source is commit `d3739fd393337b1ff4d6c2a342b12f0c87a9592f`.
Its [license is MIT](https://github.com/turboderp-org/exllamav3/blob/d3739fd393337b1ff4d6c2a342b12f0c87a9592f/LICENSE).
The optional tests extract selected upstream Python functions from a separate local clone.
They do not download that clone during CPU CI.

No separate vendored library tree was found in the inspected Crucible source.
This does not establish that every implementation fragment is independently authored.
Review source origins before distributing binaries.
Preserve this upstream notice for any copied or substantially adapted ExLlamaV3 material. [NOTICE](../NOTICE) keeps it:

```text
MIT License

Copyright (c) 2025 Turboderp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## External build and Python dependencies

These packages are installed separately, not copied into this repository.
Package distributions can contain additional third-party notices.
Preserve their full notices when distributing an environment or a linked binary.

| Component | Use | Principal license |
|---|---|---|
| ROCm HIP | GPU headers and runtime | MIT; inspect the selected ROCm distribution |
| LLVM/Clang | HIP compilation | Apache-2.0 with LLVM exceptions |
| GNU compiler runtime | Host C++ build | GPL with applicable runtime exceptions |
| Python | Tools and local HTTP frontend | Python Software Foundation License |
| NumPy | CPU numerical references | BSD-3-Clause, plus bundled notices |
| PyTorch CPU | Optional synthetic codec checks and oracle | BSD-3-Clause, plus bundled notices |
| safetensors | Python oracle weight access | Apache-2.0 |
| tokenizers | Checkpoint tokenization | Apache-2.0 |
| huggingface_hub | Optional model download | Apache-2.0 |
| Jinja2 | Sandboxed chat template | BSD-3-Clause |
| MarkupSafe | Jinja2 dependency | BSD-3-Clause |
| pytest | CPU test runner | MIT |
| GitHub checkout/setup-python actions | CPU CI setup | MIT |

Direct Python package metadata was checked in the release test environment on 2026-10-03.
NumPy's wheel also declares 0BSD, MIT, Zlib, and CC0-1.0 terms.
PyTorch's CPU wheel also declares Apache-2.0, LLVM exceptions, BSD-2-Clause, BSL-1.0, and MIT terms.
Dependency resolution is not locked yet. Recheck the actual installed distributions before shipping an environment.
This source release does not include model weights or a bundled Python environment.

## Model weights

All supported packs use **Qwen Community License 1.0**.
The license covers model artifacts separately from Crucible's engine code.
It is not Apache-2.0 or MIT.

| EXL3 pack | Checked immutable revision |
|---|---|
| `3.05bpw_h5_ng5` | `69e33439ae950f17bcbe95c98f117d80f759ab6d` |
| `4.05bpw_h6_ng6` | `55a732e0c4c3d4614bc42b68493bb930d9b02c0a` |
| `4.18bpw_h6_ng8` | `9796a106f998afbeb1cfd9adca0cdf057c7eae00` |

Sources checked on 2026-10-03:

- [3.05 pack license](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/blob/69e33439ae950f17bcbe95c98f117d80f759ab6d/LICENSE).
- [4.05 pack license](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/blob/55a732e0c4c3d4614bc42b68493bb930d9b02c0a/LICENSE).
- [4.18 pack license](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/blob/9796a106f998afbeb1cfd9adca0cdf057c7eae00/LICENSE) (checked on 2026-11-14).

The three LICENSE files have SHA-256 `a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08`.

The license requires preservation of its copyright and permission notice.
Large commercial products have model-name display requirements at the stated user or revenue thresholds.
Commercial Model-as-a-Service or AI Work Assistant businesses need a separate Qwen license, subject to the stated internal-use exception.
Read the full license for definitions, thresholds, exceptions, and applicable conditions.
Do not infer commercial permission from this short summary.

Download the complete model repository, including its LICENSE file.
Do not redistribute checkpoint shards under the engine's eventual license.
