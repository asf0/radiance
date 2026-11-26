# Aux device: a second card of another architecture (fork-only)

This fork ([asf0/radiance-strix-egpu](https://github.com/asf0/radiance-strix-egpu)) tracks upstream
[StillDeadcode/radiance](https://codeberg.org/StillDeadcode/radiance). Upstream radiance runs a model on
one or more *ranks*, cards of the same family that split the model by tensor parallelism. This fork adds
an **aux device**: a second card of a *different* architecture, with its own kernel library, beside the
rank's card. The rank's card runs the model. The aux device runs some of its work on its own stream.

The setup this was built for is a Strix Halo machine with an external GPU:

| role | card | arch | kernels | memory |
|---|---|---|---|---|
| rank (device 0) | Radeon AI PRO R9700, external over OCuLink | `gfx1201` | `libr4d` | 32 GB VRAM |
| aux (device 1) | Radeon 8060S, the Strix Halo iGPU | `gfx1151` | `libhalo` | unified with the 128 GB system RAM |

What runs on the aux device:

- **Qwen3.8-Flash-Next (`qwen4exp`)**: the routed experts are split. The R9700 holds N experts per
  layer (`aux-split`, the hottest ones with `RADIANCE_AUX_PERM`). The 8060S runs the rest straight from
  pinned host memory. A 113 GiB model can then run with every weight resident on a 32 GB card plus the
  iGPU, without streaming experts over the link per token.

Qwen3.8-27B runs on the R9700 alone. A dense feed-forward split onto the iGPU (`RADIANCE_AUX_FF`) cost
7-15% prefill for under 2% decode and was dropped at the 2.3.0 merge.

Everything else (attention, delta-net, shared experts, the head, MTP, KV cache) stays on the rank's
card.

## Build

Build both architectures and install:

```sh
cmake -S . -B build -G Ninja -DRAD_GPU_TARGETS="gfx1201;gfx1151" \
      -DCMAKE_INSTALL_PREFIX=$PWD/install
ninja -C build && cmake --install build
```

`libr4d` (gfx1201) and `libhalo` (gfx1151) both land in `install/share/radiance`.

## Configure

Three model keys, each also a command-line flag (`radiance --help` lists them under "local:"):

| key / flag | value | meaning |
|---|---|---|
| `aux-device` / `--aux-device` | `N` (HIP device index), default none | The aux card. It must be visible (`HIP_VISIBLE_DEVICES`) and must not be a rank's device. |
| `aux-kernels` / `--aux-kernels` | list, e.g. `libhalo` | Kernel libraries for the aux card, loaded into their own registry. Required with `aux-device`. |
| `aux-split` / `--aux-split` | `auto` (default), `0`, or `N` | Routed experts per layer that stay on the rank's card. `auto` takes what VRAM holds beside the KV cache, minus 512 MiB, rounded down to a multiple of 4. `0` puts every routed expert on the aux card. |

One server key:

| key / flag | meaning |
|---|---|
| `api-key-remote-only` / `--api-key-remote-only` | With `api-key` set, callers on loopback need no key. Every other address must send it. This lets a local proxy such as llama-swap reach the server keyless while the LAN still needs the key. Loopback callers still go through the local-operator checks (loopback socket, local Host, no foreign Origin). |

A Flash-Next entry, as the R9700 + 8060S fleet runs it (see `docs/MULTIMODEL.md` for fleets):

```yaml
server:
  kernels: libr4d,libavx,libref     # the rank card's libraries; libhalo goes in aux-kernels
memory:
  host-mib: 114688                  # pinned host memory: the aux experts live here
models:
  qwen3.8-flash-next-iqr:
    model: /models/qwen3.8-next-flash-iqr-v4.rad
    placement: expert_tiered
    host-pool-mib: 45056
    aux-device: 1
    aux-kernels: libhalo
    aux-split: 180                  # 180 of 512 experts a layer on the R9700
    max-num-seqs: 4
    max-model-len: 65536
    max-num-batched-tokens: 8192    # the aux pipe's MoE pass is half of this (4096 rows)
    checkpoint-interval: 4096
    num-speculative-tokens: 3
```

Run with both cards visible, the rank's first:

```sh
HIP_VISIBLE_DEVICES=0,1 \
RADIANCE_AUX_PERM=/models/aux-perm.txt \
install/bin/radiance --config flash-next.yaml
```

The startup log should show `aux device: 1 (gfx1151), 1 kernel library` and, with `auto`,
`aux split: auto picks N routed experts a layer`.

The memory goes to three places:

- **The R9700** holds the trunk, the KV cache and `aux-split` experts per layer.
- **Pinned host memory** (`host-mib`) holds the remaining experts, which the iGPU reads.
- **Fleet expert loans** are off beside an aux device. The R9700's expert slab doesn't grow into
  freed fleet memory.

### Hot-first placement (`RADIANCE_AUX_PERM`)

Upstream's expert ids say nothing about how hot each expert is. `RADIANCE_AUX_PERM` names a text file
with one line per trunk layer, a permutation of the layer's experts from coldest to hottest. The
last `aux-split` entries go to the rank's card. On Flash-Next this raised the R9700's share of routed
decode rows from 34% to 57%, and decode with MTP from 106 to 116 t/s.

To make the file:

1. Record routing statistics with every expert on one card, over prompts that look like your
   traffic. Add `--expert-stats stats.json` to a run (an iGPU-only run works, see below), send
   requests, then stop the server. The file is rewritten every 30 s and at shutdown.
2. `python3 tools/aux_perm.py aux-perm.txt stats.json[,more.json] [held-out.json ...]`. The
   script prints the rank card's share of routed rows before and after. It is written for
   Flash-Next's 48 layers × 512 experts. `R` at its top only changes the report, not the file.

The engine reorders the routers in place once, and again after every fleet restore.

## Environment variables

All of these are fork-only. Defaults are what the fleet runs.

| variable | default | effect |
|---|---|---|
| `RADIANCE_AUX_PERM` | unset | Hot-first expert permutation file (above). |
| `RADIANCE_WARMUP` | on | Before the server listens, run one synthetic prompt of a whole step plus one of 600 tokens, so the first request isn't cold (+3 s of startup). `0` skips it. |
| `RADIANCE_MOE_PASS_ROWS` | `max-num-batched-tokens / 2` | Rows per MoE pass in the aux pipe (a multiple of 64). Wider passes read the aux experts less often per token. |
| `RADIANCE_AUX_DMA_ROWS` | `64` | From this many rows up, the aux card's expert output is copied into VRAM by DMA on a separate lane instead of read over the link. `0` turns it off. |
| `RADIANCE_AUX_R_EXPERTS` | unset | Older override for `aux-split` on `qwen4exp`. |
| `RADIANCE_AUXPIPE_OFF` | unset | Turns off the pipelined prefill (two pieces per step, so the cards overlap). For debugging. |
| `RADIANCE_AUXPIPE_NOQP` / `RADIANCE_AUXPIPE_NOPRE` | unset | Turn off per-piece attention, or the early projections, in the aux pipe. For debugging. |
| `RADIANCE_AUX_MOE_OFF` | unset | Keeps the aux device loaded but runs the MoE on the rank's card. For debugging. |
| `RADIANCE_AUX_NOCHAIN` | unset | Turns off chained launches beside an aux device. |
| `HALO_MOE_I8` | `1` | `libhalo`: the w4nl expert down projection against an int8 activation runs on int8 WMMA, with the codebook at libr4d's int8 levels. That is 17-22% faster than f16 at the kernel. `0` selects the exact f16 levels and upstream's T ≤ 4 decode GEMV. |

## The iGPU alone

Flash-Next also runs on the 8060S by itself (`HIP_VISIBLE_DEVICES=1`, `kernels: libhalo,libavx,libref`,
no aux keys). Measured 2026-11-21, int8 vs `HALO_MOE_I8=0`:

| | int8 (default) | exact f16 + GEMV |
|---|---|---|
| decode, no MTP | 34.3 t/s | 34.4 t/s |
| decode, MTP 3, one stream | 51.9 t/s (acceptance 0.477) | 54.8 t/s (0.499) |
| 4 streams, MTP 3 | 113.5 t/s | 111.5 t/s |
| prefill 3K / 15K | 1356 / 1490 t/s | 1323 / 1451 t/s |

With every expert at int8 levels, draft acceptance drops about 4.5%. For a single-stream iGPU-only server,
`HALO_MOE_I8=0` is the better choice. In the split, where only the aux part runs int8, the two are equal.

## Performance (R9700 + 8060S)

Qwen3.8-Flash-Next IQR-V4 in the fleet (`aux-split: 180`, hot-first, MTP 3 with suffix drafting):

- prefill: about 4.9K / 4.8K / 4.7K t/s at 3K / 15K / 60K tokens
- decode: about 117 t/s for one stream; about 223 t/s aggregate at 4 and 8 streams
- back on the card in about 6 s after a 27B in the same fleet; a cold load takes 29 s

For comparison, Flash-Next on the R9700 alone streams its experts over the link: about 22 t/s decode
and 370 t/s prefill. Against that run as the reference, the split's KLD is about 0.008 on the short
and long corpora.

## Known issues

- **A GPU memory fault at startup, about 1 start in 14.** It happens in `r4dchain_copy16` on device
  0, during the warm-up's 8192-token prompt. It is most likely the lane-1 DMA of the aux experts'
  output (`MoeQ::pass_aux`). Restarting fixes it. Not yet fixed.
- Only `qwen4exp` (routed experts, with shared experts and rotated experts) has an aux path. Other models ignore `aux-device`.
- One aux card per process. A fleet can hold several models, but only those that set `aux-device`
  use the card.

## For plugin authors

The fork adds these to the plugin ABI (`abi/rad_runtime.h`, upstream's ABI 44), so the arch plugins
here need this fork's core:

- `rad_aux_device`, `rad_aux_split`, `rad_aux_window`
- `rad_buf_aux_local`
- `rad_lane_mark`, `rad_lane_wait_mark`, `rad_lane_stream`
- `rad_weights_epoch`

An aux op is issued on lane 2, the aux card's stream. Buffers the aux card reads must be in the host
domain, because it never reads the rank's VRAM.

## Keeping up with upstream

`main` here is upstream plus the fork's commits, kept by merging (not rebasing) upstream's `main`:

```sh
git remote add upstream https://codeberg.org/StillDeadcode/radiance.git
git fetch upstream && git merge upstream/main
```

Upstream changes to `libhalo/halo_moe_gemm_q.hip`, `arch/common/rad_block_moe_q.h`,
`arch/qwen4exp_fp8/` and `core/engine_bringup.cpp` are where conflicts usually show up.
