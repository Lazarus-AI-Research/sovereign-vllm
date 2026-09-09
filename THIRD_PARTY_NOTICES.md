# Third-party notices

Sovereign Runtime images and release artifacts include or interoperate with
third-party components. Their own license terms continue to apply.

- vLLM — Apache License 2.0 — <https://github.com/vllm-project/vllm>
- llama.cpp — MIT License — <https://github.com/ggml-org/llama.cpp>
- CPython — Python Software Foundation License — <https://www.python.org/psf/license/>
- uv — Apache License 2.0 or MIT License — <https://github.com/astral-sh/uv>

The generated SPDX SBOMs distributed with each release enumerate the Python
packages and other components resolved for that release.

## Optional SlimServe source builds

The optional CUDA source image and native Metal installer use these exact
sources, not a published SlimServe wheel or image:

| Component | Revision | License |
| --- | --- | --- |
| [SlimServe](https://github.com/QuixiAI/SlimServe/tree/44a7d1a21851c7164c098c93fbc1baa12ab99847) (including its vLLM-derived code) | `44a7d1a21851c7164c098c93fbc1baa12ab99847` | [Apache-2.0](https://github.com/QuixiAI/SlimServe/blob/44a7d1a21851c7164c098c93fbc1baa12ab99847/LICENSE) |
| [QuixiCore-CUDA](https://github.com/QuixiAI/QuixiCore-CUDA/tree/08780aaa22cdc2d144b6beacb24df953134b34be), derived from ThunderKittens | `08780aaa22cdc2d144b6beacb24df953134b34be` | MIT, HazyResearch, 2024–2026 |
| [QuixiCore-Metal](https://github.com/QuixiAI/QuixiCore-Metal/tree/71a08cd4cbcdc622ce31b3fc91e1f505e144b516), derived from ThunderKittens | `71a08cd4cbcdc622ce31b3fc91e1f505e144b516` | MIT, HazyResearch and QuixiAI, 2024–2026 |

The QuixiCore revisions are **base revisions**, not claims that the compiled
kernels are unmodified upstream trees. SlimServe's pinned `csrc/quixicore`
contains modified and additional bindings, headers and kernels. The builder
retains that exact SlimServe snapshot, uses the independently fetched base tree
for byte-identical files, and records each source file's actual SHA256, origin,
and available base-file SHA256. Kernel versions identify both the base commit
and SlimServe modification commit. CUDA builds do not compile Metal or ROCm
kernels; native Metal builds do not compile CUDA or ROCm kernels.

### CUDA build-source dependencies

These are the pinned dependencies selected by SlimServe's CMake integration.
Some provide copied Python modules or headers rather than a compiled extension;
architecture gates determine which native components are produced. The build
provenance inventories the **actual installed** native artifacts separately.

| Source | Exact revision | License / attribution |
| --- | --- | --- |
| [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass/tree/da5e086dab31d63815acafdac9a9c5893b1c69e2) (`v4.4.2`) | `da5e086dab31d63815acafdac9a9c5893b1c69e2` | BSD-3-Clause for headers; `python/CuTeDSL` has separate NVIDIA terms |
| [Triton](https://github.com/triton-lang/triton/tree/0add68262ab0a2e33b84524346cb27cbb2787356) (`v3.5.1`, copied `triton_kernels`) | `0add68262ab0a2e33b84524346cb27cbb2787356` | MIT; Philippe Tillet 2018–2020, OpenAI 2020–2022 |
| [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM/tree/a6b593d2826719dcf4892609af7b84ee23aaf32a) | `a6b593d2826719dcf4892609af7b84ee23aaf32a` | MIT; DeepSeek 2025 |
| [MSA / fmha_sm100](https://github.com/vllm-project/MSA/tree/890aaa1a37a598ad17ccff0827fea21540d381fa) | `890aaa1a37a598ad17ccff0827fea21540d381fa` | MIT; MiniMax 2026 |
| [FlashMLA](https://github.com/vllm-project/FlashMLA/tree/a8f794d1251cbfd88a5011445dd5582289c727e4) | `a8f794d1251cbfd88a5011445dd5582289c727e4` | MIT; DeepSeek 2025 |
| [QuTLASS](https://github.com/IST-DASLab/qutlass/tree/e74319e3405ce6d71965732880f5dc1f52371f64) | `e74319e3405ce6d71965732880f5dc1f52371f64` | Apache-2.0 |
| [tml-fa4](https://github.com/vllm-project/tml-fa4/tree/b206834606ed5b5f21f8eed6b0683f528ea9cf7d) | `b206834606ed5b5f21f8eed6b0683f528ea9cf7d` | BSD-3-Clause; the authors 2022 |
| [FlashAttention](https://github.com/vllm-project/flash-attention/tree/ed4b7342bc8f0489dd9b649d5288867e35fc6a32) | `ed4b7342bc8f0489dd9b649d5288867e35fc6a32` | BSD-3-Clause; the authors 2022 |
| [fmt](https://github.com/fmtlib/fmt/tree/553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28) (DeepGEMM build input) | `553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28` | MIT with optional compiled-object exception |

Nested CUTLASS source snapshots are independently fetched at their exact
parent-recorded gitlink revisions, **without initializing submodules**:

- DeepGEMM: `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`.
- MSA: `eb61c911471867a5fd2466bfd8f29306cea6ebf8`.
- FlashMLA: `147f5673d0c1c3dcf66f78d677fd647e4a020219`.
- FlashAttention: `62750a2b75c802660e4894434dc55e839f322277`.

QuTLASS uses the explicitly supplied main CUTLASS headers. Its unused CUTLASS
gitlink and FlashAttention's ROCm-only gitlinks are not fetched. CUDA toolkit,
compiler, base-image and Python-wheel dependencies remain governed by their
own terms, including NVIDIA's proprietary components; they are not relicensed
as Apache-2.0 or MIT here.

The builder retains all source `LICENSE*`, `NOTICE*`, `COPYING*` and
`COPYRIGHT*` files, including nested notices and additional license terms,
under `/opt/sovereign-slimserve/licenses/`, preserving their component-relative
paths. Python distributions retain their wheel-provided license metadata.
`/opt/sovereign-slimserve/provenance.json` records the source pins, dependency
lock hash, installed package versions, actual source/artifact hashes and notice
file hashes. `docker/slimserve/provenance.json` is only the checked-in **source
lock**, not proof of a completed build. No binary build, GPU execution or
release qualification is asserted by these notices.

### QuixiCore-CUDA MIT notice (verbatim)

MIT License

Copyright (c) 2024-2026 HazyResearch

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

### QuixiCore-Metal MIT notice (verbatim)

MIT License

Copyright (c) 2024-2026 HazyResearch
Copyright (c) 2024-2026 QuixiAI

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
