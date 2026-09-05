# Engineering notes: Qwen3.8-Flash-Next-FP8 on two DGX Sparks with SGLang

These are the working notes from getting `Qwen/Qwen3.8-Flash-Next-FP8` serving across two NVIDIA
DGX Sparks. They are written for someone who owns two Sparks, has a ConnectX-7 cable between them,
and wants to reproduce this or push past it. Every number below is a measurement taken on this
cluster on 2026-09-04, not a spec-sheet figure, unless marked otherwise. Where a figure was not
captured, it is marked `[[NUMBER NEEDED: ...]]` rather than guessed.

## 1. Purpose and result

The goal was to serve the FP8 checkpoint of Qwen3.8-Flash-Next, a roughly 125B-total/6B-active
MoE that does not fit on one 128 GB Spark, across two Sparks joined by their ConnectX-7 ports.
Five attempts on vLLM failed to reach a stable serving state for this model on this hardware.
SGLang, on the fifth attempt, worked. The cluster has been serving continuously since 2026-09-04
at 19:39 CDT (00:39 UTC on 2026-09-05) under a pair of systemd units, one per box, both restart-safe. Measured
single-stream throughput is 36 to 41 tok/s (median 40.3), context is 65,536 tokens, and the KV
cache is BF16 (FP8 KV was believed not possible on this GPU for this model at the time this
section was written: this was revised on 2026-09-05, see section 12). That is roughly 15 to 21
percent slower than the community's best published NVFP4 numbers on the same hardware pair, at
less than a quarter of the context length, in exchange for staying at 8-bit weights end to end.

## 2. Hardware and interconnect

### The two boxes

Two NVIDIA DGX Spark units, each with a 20-core ARM big.LITTLE CPU (10 Cortex-X925 performance
cores plus 10 Cortex-A725 efficiency cores), an NVIDIA GB10 Grace-Blackwell GPU with unified
memory (no fixed VRAM partition, so `nvidia-smi --query-gpu=memory.total` returns `N/A` by design),
and 128 GB of unified RAM shared between CPU and GPU (the kernel reports roughly 119 to 122 GiB
after firmware reserve, and this varies slightly per unit). Driver and CUDA versions differed
slightly between the two boxes at the time of this build (580.142/CUDA 13.0 on one, 580.173.02/
CUDA 13.0 on the other). This did not cause a problem, but it is worth checking `nvidia-smi` on
both boxes before starting, when chasing a driver-specific bug.

### The single QSFP cable and the "twins" trap

The two Sparks are joined by one QSFP cable between their ConnectX-7 ports. NVIDIA's own
documentation for connecting two Sparks states that a single cable delivers full bandwidth: a
second cable adds nothing, because both physical ports front the same PCIe 5.0 x4 link pair.

The trap: each physical CX7 port shows up in Linux as two Ethernet interfaces and two
RoCE devices, one pair per PCIe x4 link:

- Ethernet: `enp1s0f1np1` and `enP2p1s0f1np1`
- RoCE: `rocep1s0f1` and `roceP2p1s0f1`

Both "twins" report a 200000 Mb/s link when the cable is connected. They look like two separate
100+ Gb links. They are not: they are two views of the same physical port. Bonding them (802.3ad,
`balance-rr`, anything) does not double the bandwidth, it creates an ARP black hole, and
`Destination Host Unreachable` shows up on both sides.

The correct configuration is one IP address on `enp1s0f1np1` only, nothing on its twin. Full
bandwidth to NCCL comes a different way: NCCL is told to use both RoCE twins together with
`NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`, so the collective library spreads traffic across both PCIe
x4 links itself, at the RDMA layer, rather than at the IP layer. Do not try to replicate this with
IP bonding. Source for the two-Spark cabling story: NVIDIA's `dgx-spark-playbooks` repository,
`connect-two-sparks` recipe.

### The one-IP rule and netplan/NetworkManager gist

Each box gets a static IP on `enp1s0f1np1` only, in a private /24 dedicated to this link (this
cluster used the last-octet convention 10 for the head node and 11 for the worker node), MTU 9000,
IPv6 disabled on that interface. On a distribution that renders network config through
NetworkManager rather than netplan directly, a connection profile created with `nmcli` (bound to
`enp1s0f1np1`, static IPv4, MTU 9000, IPv6 method `disabled`) is sufficient. Delete any leftover
bond profile from an earlier bonding attempt, since a stale bond definition competing for the
same interface is a common source of "it worked yesterday" confusion.

### Verifying the link before touching NCCL

Two checks, in order:

1. Ping, including an 8972-byte no-fragment ping (`ping -M do -s 8972`), to confirm MTU 9000 is
   actually negotiated end to end and not silently capped somewhere.
2. `iperf3 -P 4` for 8 seconds. This cluster measured 110 Gbit/s aggregate over TCP, zero
   retransmits. That is the TCP ceiling on this link, not what NCCL will actually achieve (NCCL
   uses RDMA over the RoCE twins, not TCP), but it is the fastest way to prove the cable, the MTU,
   and the switch-free direct connection are all sound before bringing a 173 GB model anywhere
   near it.

Also confirm RoCEv2 GID population on both boxes (`show_gids` or the equivalent under
`/sys/class/infiniband`): GID index 3 (RoCEv2 with the assigned IPv4 address) should populate
automatically on `rocep1s0f1` once the interface has an IP. The twin, `roceP2p1s0f1`, will only
ever show link-local GIDs (indexes 0 and 1) because it carries no IP itself. This asymmetry
matters later (see the GID pinning failure in the attempt ledger).

### CX7 hot-plug on newer DGX OS builds

One of these two Sparks (whichever one has the newer DGX OS build, driven by the package
`dgx-spark-mlnx-hotplug`) will remove the ConnectX-7 device from the PCI bus entirely when no
cable is inserted. This is not a fault, it is a udev rule plus an ACPI platform device
(`MTKP0001:00`, driver `cx7-pcie-hotplug`) intentionally hot-unplugging the card. It looks exactly
like dead hardware to anyone who does not know to check for it:

- `lspci` shows the two root ports (`0000:00:00.0` and `0002:00:00.0`) with an empty bus 01-0f
  behind them, i.e. nothing enumerated where the CX7's four PCI functions should be.
- The root port's own link status regresses: `LnkSta 2.5GT/s x4` against a `LnkCap 32GT/s x4`,
  meaning the root port itself trained down to its slowest possible link because there is nothing
  on the other end to negotiate with.

The diagnostic is a single sysfs read:

```
cat /sys/devices/platform/MTKP0001:00/pcie_hotplug/debug_state
```

`0` means no cable, `1` means cable present. Plugging the cable in flips this to `1` and
re-enumerates all four CX7 PCI functions without a reboot. A config gate file,
`/etc/nvidia/cx7-hotplug-enabled`, controls whether this behavior is active at all. Check for its
presence on both boxes if one Spark behaves this way and the other does not (the older DGX OS
build used by the other Spark in this cluster does not have this hot-plug node at all, and its CX7
device stays enumerated whether or not a cable is present).

The practical upshot: unplugging the interconnect cable for any reason on the newer-build box
makes the device vanish from `lspci` until the cable is plugged back in. This is recoverable and
requires no driver reload, just the cable.

### Copying the weights across the link

The FP8 checkpoint (173 GB, 131 safetensors shards) has to exist on the local NVMe of both nodes,
because SGLang's multi-node tensor parallelism reads weights from local disk on every node, not
over a shared filesystem. Copying it over the RoCE link with rsync measured 406 MB/s, 7 minutes
15 seconds for the full 173 GB. Confirm the copy landed correctly with a size diff plus a sampled
sha256 across a subset of shards before trusting it, rather than assuming rsync's own checksum
behavior covers everything that matters.

## 3. The model

| Fact | Value |
|---|---|
| Checkpoint size on disk | 172.78 GiB (173 GB), 131 safetensors shards |
| Total parameters | approximately 125B |
| Active parameters per token | approximately 6B |
| N-gram embedding table | approximately 51B parameters, separate from the MoE body |
| Transformer layers | 48 total |
| Layers with standard attention KV | 12 |
| Layers with Gated-DeltaNet (recurrent, no KV) | 36 |
| Native context | 262,144 tokens (1,048,576 via YaRN) |
| Architecture family | ultra-sparse multimodal MoE |

The 12-of-48 split matters directly for the FP8 KV cache decision later: only a quarter of the
layers carry a conventional attention KV cache at all, and the other 36 carry a fixed recurrent
state instead (Gated-DeltaNet), which is not KV cache in the vLLM/SGLang sense and is not
affected by the `--kv-cache-dtype` flag either way.

### Why two Sparks

After expert-parallel placement (see the FP8-block-vs-sharding arithmetic in section 4), each node
ends up holding about 94 GB of resident weights plus roughly 1.8 GB for the MTP draft model.
Neither figure fits comfortably inside one Spark's roughly 119 GiB usable pool alongside an OS,
a KV cache, and CUDA graph buffers, hence the two-node split with expert parallelism rather than a
single-node deployment with CPU offload.

### What the community had already published

Before this build, no one had published a working FP8 deployment of this model across two DGX
Sparks. Every completed dual-Spark report used NVFP4 (4-bit) instead:

- NVIDIA Developer Forums thread 381428, "Qwen3.8 Flash Next NVFP4 on 2x DGX Spark, full
  multimodal, 70 tok/s peak, 47 typical."
- `tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark` (community repo backing the forum thread above).
- `MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks` (a vLLM-based NVFP4 recipe: TP2+EP,
  `gpu_memory_utilization` 0.835, `kv-cache-dtype auto`, MTP-3, `--enforce-eager`, reporting 44 to
  56 tok/s single-stream).
- SGLang discussion #36891, reporting one independent user running the official FP8 checkpoint
  with FP8 weights, FP8 KV, and native NEXTN MTP at TP=2 (a single, brief report, not a full
  writeup).
- The vendor's own recipe page, `https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next`.

A second forum thread, 381440, existed but carried no numbers and answered no open question. It
is listed in the sources below for completeness but contributed nothing to this build beyond
confirming that others were also attempting this.

The absence of a working FP8-on-two-Sparks report was itself a data point: it suggested the FP8
path was harder than NVFP4 on this hardware, which turned out to be correct (see the attempt
ledger), but not impossible.

## 4. Attempt ledger

Ten attempts, in order. "Engine" is vLLM or SGLang. Failure signatures are quoted verbatim from
the logs.

| # | Engine | Key config | Failure signature | Root cause | Fix or decision |
|---|---|---|---|---|---|
| 1 | vLLM | TP2, `--distributed-executor-backend` native PyTorch (not Ray) | `Engine core initialization failed` | Worker node had no model weights on local disk yet | Rsync the checkpoint to the worker (see section 2) |
| 2 | vLLM | `--kv-cache-dtype fp8` | `NotImplementedError: Qwen4Exp QSA requires a BF16 main KV cache` | Model's QSA attention implementation hard-requires BF16 KV in vLLM | Drop `--kv-cache-dtype`, fall back to `auto` |
| 3 | vLLM | `gpu_memory_utilization 0.85`, `max_model_len 131072`, `max_num_batched_tokens 16384` | `Worker proc VllmWorker-1 died unexpectedly (exit code: None)` (worker), head unreachable over SSH on both network paths | Head thrashed into swap during `torch.compile`/profiling on unified memory: a hard hang, not a clean crash | Brad power-cycled the head node (journal shows a hard cut, no clean shutdown) |
| 4 | vLLM | `max_model_len 65536`, `max_num_batched_tokens 4096`, `max_num_seqs 8`, multimodal profiling disabled, `util 0.85` | (aborted before completion, at shard 1 of 131) | Superseded before running: community research on GB10-specific vLLM pitfalls landed first | Aborted to apply the research findings below |
| 5 | vLLM | `util 0.80`, `max_model_len 65536`, batched 4096, `max_num_seqs 8`, `--enforce-eager`, `--no-enable-prefix-caching`, BF16 KV | (aborted at shard 16 of 131) | Engine decision changed to SGLang mid-run | Aborted, vLLM retired for this model |
| 6 | SGLang | TP2 only (no EP) | `ValueError: The output_size of gate's and up's weight = 320 is not divisible by weight quantization block_n = 128` | FP8 128-block quantization vs. tensor-parallel sharding arithmetic (see subsection below) | Add `--ep-size 2` |
| 7 | SGLang | TP2/EP2, `NCCL_IB_GID_INDEX=3` pinned | `ibv_modify_qp failed with 61 No data available, on dev roceP2p1s0f1:1 ... local GID index 3, local GID ::` | GID index 3 exists only on `rocep1s0f1`, not its twin: pinning it breaks the expert-parallel NCCL communicator | Remove the `NCCL_IB_GID_INDEX` pin entirely, let NCCL auto-select |
| 8 | SGLang | `--mem-fraction-static 0.80` | `Loaded weights leave no GPU memory for the KV cache under --mem-fraction-static=0.8. Raise --mem-fraction-static above 0.872 (minimum viable = 0.8714)` | Static weight footprint (about 94 GB main plus about 1.8 to 2.0 GB MTP draft) leaves too little of the roughly 119 GiB pool at 0.80 | Raise to `--mem-fraction-static 0.90` |
| 9 | SGLang | `--kv-cache-dtype fp8_e4m3` | `ValueError: unsupported SM121 QSA call: expected BF16 D=256, 12:1 GQA, TP1 24Q/2KV or TP2 12Q/1KV, bs<=128, and selected KV<=2055` | SM121 QSA attention kernel on GB10 had no dequant path for FP8 KV in this image, not a hardware limit (see section 12) | Use `--kv-cache-dtype auto` (BF16), revisited and resolved 2026-09-05 |
| 10 | SGLang | Full final flag set (section 5), `--kv-cache-dtype auto` | none: serving | none | Serving since 2026-09-04 19:39 CDT (00:39 UTC on 2026-09-05) |

### 4a. FP8 128-block quantization vs. TP sharding arithmetic (attempt 6)

The FP8 checkpoint quantizes MoE expert weights in blocks of 128 along the output dimension
(`block_n = 128`). Each expert's gate/up projection here has an output width of 640. With plain
tensor parallelism at TP=2, SGLang tries to split that 640-wide dimension across two ranks, giving
320 per rank. But 320 is not divisible by 128, so the quantization block boundaries do not align
with the shard boundaries, and model construction fails outright rather than producing a subtly
wrong result. The fix is expert parallelism (`--ep-size 2`) instead of plain tensor sharding for
the MoE layers: EP keeps each expert whole on one rank rather than splitting an individual
expert's weight matrix across ranks. 640 divides evenly by the block size (640 = 5 x 128) as long
as no single expert's own weight gets cut mid-block, which EP guarantees and plain TP does not.

### 4b. GID table asymmetry (attempt 7)

RoCEv2 exposes a table of GIDs per RDMA device, indexed by number, each entry describing an
address family and network. GID index 3 on this cluster's addressing scheme is the RoCEv2 entry
carrying the assigned IPv4 address for the interface. But only one of the two "twin" RoCE
devices actually has an IP address (`rocep1s0f1`, since only `enp1s0f1np1` got the static IP in
section 2). Its twin, `roceP2p1s0f1`, has no IP of its own and therefore only ever populates the
link-local GID entries (indexes 0 and 1). Pinning `NCCL_IB_GID_INDEX=3` to force a specific GID
(a reasonable-looking troubleshooting move if NCCL seems to pick an unexpected GID) forces
that same index onto both twins uniformly. On `roceP2p1s0f1` there is no entry at index 3, so any
NCCL communicator that tries to open a queue pair over that twin fails immediately. This showed up
specifically when the expert-parallel communicator was created (a second NCCL group beyond the
original TP group), because the TP group had happened to route around the missing twin by luck of
initialization order, and the EP group did not. The fix is to never pin the GID index at all: let
NCCL auto-select the correct entry per device.

### 4c. Memory floor arithmetic (attempt 8)

At `--mem-fraction-static 0.80`, SGLang computed a required floor of about 87.2 percent
(minimum viable 0.8714) to fit the resident weights before any KV cache could be allocated at all.
The measured static footprint per node was roughly 93.47 GB of main FP8 weights plus about 1.99 GB
for the MTP draft model (later runs measured 93.9 to 94.8 GB main plus 1.74 to 1.99 GB draft,
varying slightly by node and run). Against a usable pool of roughly 119 GiB, 0.80 of that pool
(about 95.2 GiB) was consumed almost entirely by the weights themselves, leaving effectively
nothing for KV. Raising to 0.90 (about 107.1 GiB) freed roughly 11 GB above the weight floor for
the actual KV pool, cache buffers, and CUDA graph memory, which proved sufficient at the chosen
context length and batch settings (section 6).

### 4d. SM121 QSA kernel constraint and the cost of skipping it (attempt 9)

The attention kernel implementing this model's QSA (the sparse-attention mechanism over the 12
layers that carry real KV) is compiled specifically for SM121 (the GB10's streaming multiprocessor
architecture) and only accepts a fixed set of shapes: BF16 data, head dimension 256, a 12:1
grouped-query-attention ratio, and either TP1 with 24 query heads to 2 KV heads or TP2 with 12
query heads to 1 KV head, batch size at most 128, and a selected-KV count at most 2055. FP8 KV
falls outside this kernel's accepted dtype outright, and the kernel raises rather than silently
degrading. Separately, the FP8-KV attempt logged that the checkpoint ships no KV quantization
scaling factors at all ("no scaling factors provided, defaulting to 1.0"), which would have been a
correctness concern even if the kernel had accepted the dtype. The upside foregone by staying at
BF16 KV is small: since only 12 of 48 layers carry attention KV in the first place (the rest are
Gated-DeltaNet, with no KV cache to shrink), moving that fraction from BF16 to FP8 would have saved
on the order of 1 GB, not a meaningful fraction of the roughly 94 GB weight footprint.

This was revised on 2026-09-05: the SM121 QSA kernel's BF16-only constraint was a missing dequant
path, not a hardware limit. See section 12.

### 4e. Why vLLM was abandoned rather than tuned further (attempts 1 through 5)

vLLM's failure mode on this hardware was not one bad flag, it was a pattern the wider community
had already run into on GB10's unified-memory architecture: CUDA graphs and `torch.compile`
profiling passes can lock up the unified memory pool rather than failing cleanly, because the
memory profiler (`cudaMemGetInfo`) does not correctly account for reclaimable host page cache on
a unified-memory system (this is the same non-determinism documented separately for this box's
earlier single-node vLLM deployments, where KV pool sizing varied by 2 to 3x between otherwise
identical restarts). The community's accepted workaround is `--enforce-eager`, trading away CUDA
graph and compiled-kernel speedups to avoid the hang entirely. Community reports also converged on
`gpu_memory_utilization 0.80` as the practical ceiling rather than 0.85, since 0.85 was repeatedly
reported to drift into swap. A further known vLLM issue on this GPU, prefix caching hitting
`vllm-project/vllm#54173` (an illegal memory access on `sm_121`), meant prefix caching had to be
disabled as well. Given that Brad's ruling was to move to SGLang with MTP speculative decoding
before attempt 4 finished loading, and that SGLang's own community report (SGLang discussion
#36891) showed a path to FP8 plus native NEXTN MTP without any of vLLM's GB10-specific
workarounds, the decision was to switch engines rather than spend further attempts tuning around
vLLM's unified-memory profiling hang.

## 5. Final configuration, flag by flag

### Container image

`lmsysorg/sglang:qwen38flashnext` (`sglang 0.0.0.dev1+g593134d17`, `torch 2.13.0+cu130`, CUDA
13.0.3), with one local patch applied per box (`Dockerfile` in this repo):

- The SM121 QSA attention guard (the check that routes this model's attention kernel to the
  correct SM121-specific implementation) is already present upstream in this image tag, under the
  function name `is_sm120()`. The Dockerfile verifies this with a `grep` that fails the build if
  the guard is missing, rather than assuming it is there.
- The one patch actually required is an M-RoPE (multimodal rotary position embedding) fix in
  `rotary_triton.py`: the partial-rotary mask line `t_mask = ~(h_mask | w_mask)` is changed to
  `t_mask = ~(h_mask | w_mask) & (cos_offsets < half_rd)`, adding a bound that was missing for
  this model's partial-rotary configuration.

A community image, `radixark/sglang-qwen38flashnext:sm121-qsa-mrope1`, already carries both
patches pre-applied, but it is a private repository and the pull was denied, hence building the
one-line patch locally instead.

### `launch.sh` flags

| Flag | Why |
|---|---|
| `--tp-size 2` | Tensor-parallel the attention and dense layers across the two GPUs (one per node) |
| `--ep-size 2` | Expert-parallel the MoE layers instead of tensor-sharding them: required because FP8's 128-wide quantization blocks do not divide evenly across a plain TP2 shard of this model's 640-wide expert projections (section 4a) |
| `--nnodes 2` | Two physical nodes participate in the same model instance |
| `--dist-init-addr $HEAD_IP:29511` | Rendezvous address. The head node (rank 0) listens here, the rank-1 node connects to it |
| `--dist-timeout 600` | 10-minute rendezvous timeout, generous enough that either node can be started first (weight loading alone takes 7 to 8 minutes) without one node timing out waiting for the other |
| `--quantization fp8` | Use the checkpoint's native FP8 weights |
| `--kv-cache-dtype auto` | BF16 KV cache. FP8 KV is rejected outright by the SM121 QSA kernel for this model (section 4d) |
| `--page-size 64` | KV cache page size in tokens, as used throughout this build. Not independently varied |
| `--speculative-algorithm NEXTN` | MTP-style next-token speculative decoding. Note the server reports this back as `speculative_algorithm: EAGLE` in `/get_server_info`, because NEXTN is implemented on SGLang's EAGLE speculative-decoding code path: it is not a separate algorithm label at the API level |
| `--speculative-num-steps 3` | Draft model proposes 3 steps ahead per verification pass |
| `--speculative-eagle-topk 1` | Single top candidate per step (no branching tree) |
| `--speculative-num-draft-tokens 4` | 4 draft tokens verified per pass |
| `--chunked-prefill-size 2048` | Prefill is chunked in 2048-token pieces rather than done in one pass, bounding peak memory during long-prompt prefill |
| `--max-running-requests 4` | Concurrency cap, set to fit inside the memory floor described in section 4c |
| `--context-length 65536` | Maximum context per request, well short of the checkpoint's native 262K, chosen to fit inside the measured memory floor at `mem-fraction-static 0.90` |
| `--mem-fraction-static 0.90` | Fraction of the unified memory pool reserved for weights plus KV cache plus CUDA graph buffers. 0.80 was insufficient (section 4c) |
| `--cuda-graph-max-bs 4` | Largest batch size captured into a CUDA graph, matched to `--max-running-requests` |
| `--disable-cuda-graph-padding` | Do not pad smaller batches up to the captured graph size, avoiding wasted compute on partially-filled batches |
| `--ple-offload-embedding` | Intended to offload the (large) embedding/N-gram table to host memory. Measured weight footprint was about 94 GB resident either way, so this flag's effect in this build could not be confirmed and may be a no-op here (open question, section 10) |
| `--disable-radix-cache` | Disable SGLang's prefix-caching radix tree. Not exercised as a performance lever in this build, disabled defensively given vLLM's own prefix-caching issue on this GPU family (section 4e), though SGLang's implementation is separate code and was not itself observed to fail |
| `--sampling-backend pytorch` | Use the PyTorch sampling backend rather than SGLang's other sampling implementations |
| `--reasoning-parser qwen3` | Parses `<think>` content into the `reasoning_content` field of the API response |
| `--tool-call-parser qwen3_coder` | The tool-call parser that actually matches this model's tool-call output format in practice, per community reports for this model family |
| `--trust-remote-code` | Required to load this model's custom modeling code |
| `--enable-metrics` | Serves Prometheus metrics at `/metrics` (added when the cluster moved under systemd, see section 8. Every other flag is unchanged from the original launch) |

### Environment variables

| Variable | Why |
|---|---|
| `NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1` | Tells NCCL to use both RoCE twins on the physical port together, which is how full interconnect bandwidth is achieved without IP bonding (section 2) |
| `NCCL_IB_DISABLE=0` | Explicitly keep InfiniBand/RoCE transport enabled |
| `NCCL_SOCKET_IFNAME=enp1s0f1np1` | Restrict NCCL's TCP bootstrap/out-of-band traffic to the interconnect interface, not the management NIC |
| `GLOO_SOCKET_IFNAME=enp1s0f1np1` | Same restriction for Gloo, used for non-NCCL collective operations |
| `NCCL_DEBUG=INFO` | Verbose NCCL logging, left on for this build to make communicator setup issues (like the GID failure in section 4b) visible in the logs |
| `SGLANG_DEEPGEMM=0` | Disable SGLang's DeepGEMM block-FP8 kernel selection path |
| `SGL_ENABLE_JIT_DEEPGEMM=0` | Disable DeepGEMM's JIT compilation path as well | 

Both DeepGEMM variables are set defensively: DeepGEMM's block-scaled FP8 GEMM kernels have shown
hardware-level failures on GB10 in unrelated deployments on this same hardware (a separate model,
using vLLM's DeepGEMM path, hit `Assertion error ... Unknown SF transformation` and, after forcing
a non-DeepGEMM kernel, a CUDA `cudaErrorIllegalAddress` from the Cutlass block-scaled path instead,
both pointing at a GB10-level gap in Tensor Memory Accelerator support for these two kernel
implementations). Disabling DeepGEMM selection for this SGLang deployment avoids exercising that
code path at all rather than hoping SGLang's own DeepGEMM integration is unaffected.

`--kv-cache-dtype auto` (BF16) and `--context-length 65536` were the production values as of this
section. Both changed on 2026-09-05: see section 12 for the FP8 KV cache patch and the promotion
to 131,072 context.

One more operational step, run at the top of `launch.sh` before every start: dropping the page
cache (`sync && echo 3 > /proc/sys/vm/drop_caches`). This is defensive housekeeping against stale
cached pages competing with a mostly-loaded unified memory pool at start time, not something with
an isolated before/after measurement in this build.

## 6. Startup profile and memory

| Stage | Measured duration |
|---|---|
| Main weight load | 472 to 483 s (about 8 minutes) |
| MTP draft model load | 45 to 52 s |
| CUDA graph capture | approximately 40 s (39.6 s target/verify pass, plus 2.9 s and 0.6 s for smaller passes) |
| Total, cold start to ready | approximately 10 minutes |

KV cache pool, BF16, confirmed identical on both nodes:

- At first measurement: 124,480 tokens (K 0.71 GB + V 0.71 GB).
- After the systemd cutover (section 8): 114,688 tokens (K 0.66 GB + V 0.66 GB).

Resident memory, measured after warmup:

- `nvidia-smi` (compute apps): approximately 70.9 GB for the scheduler process per node, plus a
  roughly 170 to 196 MiB helper process.
- `free -g` on the head node: total 119, used 113, free 1, shared 34, buff/cache 40, available 6.
- `free -g` on the worker node: total 121, used 112, free 1, shared 34, buff/cache 43, available 9.
- Five samples taken over five minutes on the head node were identical, i.e. no drift once warmed
  up, and the endpoint kept answering correctly throughout.

The thin margin here is the operationally important fact: the head node runs with only about 6 GB
of unified memory available after warmup. Do not raise `--context-length`,
`--max-running-requests`, or `--cuda-graph-max-bs` without re-measuring `free -g` during startup
on the head node afterward. There is no cushion in this configuration for a casual bump to any of
those three flags.

## 7. Measured performance

| Metric | Value |
|---|---|
| Single-stream, 256-token completions | 36 to 41 tok/s, median 40.3 tok/s (three runs: 40.3, 36.3, 41.2) |
| Single-stream, 512-token completions | 34.7 to 38.8 tok/s |
| Time to first token | 0.18 s |
| 2 concurrent streams, aggregate | 42.9 tok/s |
| 4 concurrent streams, per-stream | 22 to 24 tok/s each |
| 4 concurrent streams, aggregate | 88 to 98.5 tok/s (two runs, per-request latency 10.0 to 10.4 s) |
| Long-prompt summarization | 3,561-token prompt summarized correctly in 5.5 s |

Correctness probes: thinking mode returns `reasoning_content` with no token-0 `"!!!!"` loop, a
`qwen3_coder`-parsed tool call (`get_weather({"city": "Denver"})`) returned with
`finish_reason: tool_calls`, and three factual probes (capital of Australia, three noble gases,
17 x 23) were answered correctly (the arithmetic answer was cut off mid-explanation by a 64-token
budget in one probe, not answered incorrectly).

### Honest comparison to the community's NVFP4 numbers

The published NVFP4 dual-Spark result (forum thread 381428) reports 70 tok/s peak and 47 tok/s
typical, at up to 262K context with full multimodal support. This FP8 deployment measures 15 to
21 percent slower on single-stream throughput than that 47 tok/s typical figure, runs at 65,536
tokens of context rather than 262,144 (a quarter, chosen for the memory-floor reasons in section
4c, not a hard ceiling of the checkpoint), and keeps the KV cache in BF16 rather than whatever the
NVFP4 report used. No side-by-side output-quality comparison between the FP8 and NVFP4 checkpoints
was performed as part of this build. The tradeoff recorded here is entirely a throughput-and-
context tradeoff, not a quality claim in either direction.

## 8. Operations

### systemd unit design

One `sglang-cluster.service` unit per box (`systemd/sglang-cluster.rank0.service` and
`sglang-cluster.rank1.service` in this repo, differing only in `NODE_RANK` and the rank number in
`Description`). Key fields:

- `Type=simple`, running `docker run` in the foreground (no `-d`), so systemd can supervise the
  container process directly rather than supervising a detached background container it cannot
  see the exit status of.
- `Restart=always`, `RestartSec=30`.
- `TimeoutStartSec=900` (15 minutes), generous enough to cover the roughly 10-minute cold start
  from section 6 with margin.
- `ExecStartPre=-/usr/bin/docker stop sglang_node` and `-/usr/bin/docker rm sglang_node` (the
  leading `-` makes these non-fatal if no such container exists), clearing any leftover container
  from a previous run before starting fresh.
- The head node is the rendezvous master (`--dist-init-addr $HEAD_IP:29511`, `--dist-timeout 600`
  from section 5), so either node may boot first on a cold start of the whole cluster. The
  10-minute timeout covers the case where one node's weight load finishes well before the other's.
- `launch.sh` requires `HEAD_IP` with no default, so both units load
  `EnvironmentFile=-/etc/sglang-cluster.env`. Create that file with `HEAD_IP=<head node
  interconnect address>` (and optionally `MODEL_DIR`, `HF_CACHE`) before `systemctl enable --now`.
  The same file works for a manual launch: `set -a; . /etc/sglang-cluster.env; set +a` before
  running `launch.sh`.

### Measured recovery test

The worker node's service was restarted directly (`systemctl restart sglang-cluster`). The head
node's process exited cleanly on NCCL peer loss shortly afterward (a clean `SystemExit 0`, not a
crash), and its own unit auto-restarted it about 30 seconds later, as designed. Total time back to
fully ready: 14 minutes, including a fresh weight load on the head node's restarted process. One
benign NCCL warning appeared twice during re-rendezvous: `NET/IB : roceP2p1s0f1:1 GID table
changed`. `ncclCommInitRank` succeeded both times despite the warning, and no error followed it.

### Maintenance stop

```
sudo systemctl stop sglang-cluster
```

on both boxes. Do not run `docker rm -f sglang_node` alone to stop the service: with
`Restart=always` in effect, systemd will simply relaunch the container within `RestartSec`
seconds, undoing the stop.

### Logs

`docker logs sglang_node` is the only log source, and it is lost the moment the container is
removed (including by systemd's own `ExecStartPre` cleanup on the next start). If a failure needs
to be preserved for later diagnosis, capture it before that happens:

```
docker logs sglang_node > last-run.log 2>&1
```

`journalctl -u sglang-cluster` captures the unit's own start/stop/restart history and exit codes,
but not the container's own stdout once the container is gone (that only lives in `docker logs`
while the container still exists).

### The 12-minute rule

Do not judge a launch as failed before about 12 minutes have passed, unless `docker logs
sglang_node` already shows a Python traceback. The combination of an 8-minute weight load plus a
40-second CUDA graph capture plus normal variance between runs means a launch that is still
working at minute 9 or 10 is very likely still working, not stuck.

## 9. Observability notes

`/metrics` on port 8000 serves Prometheus text format, every family under the `sglang:` prefix.
Family count is not fixed: idle, immediately after startup, this build reported 58 families.
After the first completed request, that grew to 67 to 69, because several histogram families
(anything keyed on a completed request's latency or token counts) only register their first
`_bucket`/`_sum`/`_count` series once at least one observation has occurred. A scrape taken
immediately after a cold start that shows fewer families than expected calls for one real
completion through the endpoint before concluding metrics are broken.

There is no inter-token-latency histogram in this build. `time_to_first_token_seconds` (TTFT) is
the closest substitute available for a per-token latency signal on a dashboard.

Metric families relevant to a dashboard, all under the `sglang:` prefix:

- `e2e_request_latency_seconds`
- `time_to_first_token_seconds`
- `gen_throughput`
- `generation_tokens_total`
- `prompt_tokens_total`
- `token_usage`
- `kv_used_tokens`
- `num_running_reqs`
- `num_queue_reqs`
- `cache_hit_rate`
- `spec_accept_rate`
- `spec_accept_length`
- `weight_memory_usage_gb`
- `startup_time_seconds`

A DGX GPU Prometheus exporter running alongside this one has a memory-utilization metric that is a ratio,
not a byte count: `nvidia_smi_utilization_memory_ratio`. There is no
`nvidia_smi_memory_used_bytes` metric in that exporter. A memory-based alert has to be written
against the ratio metric, not a bytes threshold, which is an easy mistake to carry over from a
discrete-GPU dashboard template.

## 10. Open questions and untested ideas

- **FP8 KV on SM121.** Answered on 2026-09-05: not a hard architectural limit, see section 12 for
  the patch that resolves it. The remaining open question is calibrated KV scales, since the
  checkpoint ships none and the patch defaults to a descale of 1.0.
- **`--ple-offload-embedding`.** Its effect in this build could not be confirmed: measured weight
  footprint was about 94 GB resident either way, with or without expecting the embedding table to
  be offloaded. It may be a no-op in this SGLang build, or the offload may be happening without
  changing the number being measured. Worth instrumenting directly (host RAM usage specifically
  attributable to the embedding table) before relying on it.
- **NVFP4 versus FP8 output quality.** No comparison was run. If the roughly 15 to 21 percent
  throughput gap and quarter of the context length matter less than output quality for a given
  use case, this is the open question that would settle which checkpoint to run.
- **Raising context or concurrency.** Section 6's roughly 6 GB memory margin on the head node has
  not been pushed. A careful, re-measured attempt at a higher `--context-length` or
  `--max-running-requests`, watching `free -g` through the full startup sequence rather than only
  at steady state, is the natural next experiment.
- **SGLang CUDA graph max batch size.** `--cuda-graph-max-bs 4` was chosen to match
  `--max-running-requests 4`, not independently tuned. Whether a larger captured graph batch (with
  a correspondingly higher memory floor) trades favorably against the margin in section 6 is
  untested.
- **Speculative decoding acceptance under load.** The community reports referenced in section 3
  cite MTP acceptance rates as part of their throughput story, but `spec_accept_rate` on this
  cluster was not captured under sustained multi-stream load, only observed as an available metric
  family. Neither `spec_accept_rate` nor `spec_accept_length` under sustained load was measured in
  this build.
- **Forum thread 381440's unanswered question.** This thread was found during the community
  research pass and is listed in the sources below, but its own open question was never resolved
  by anyone, including this build. Left here for anyone following up.

## 12. FP8 KV cache, revisited (2026-09-05)

Section 4d and section 10 both recorded FP8 KV cache as failing outright on the SM121 QSA kernel,
with the checkpoint's missing KV scale factors noted as a separate concern that would have mattered
even if the kernel had accepted the dtype. Both points turned out to be about a missing dequant
path rather than a hardware ceiling.

**The pointer.** Forum user jahnclawdmonet replied on NVIDIA Developer Forums thread 382435 (the
thread covering this build, listed in section 11) pointing at upstream sglang PR #36644, "[Qwen3.8]
Fix FP8 KV cache support in QSA" (author LingZ315, open and unreviewed at the time it was applied
here, stacked on PR #36497, already validated upstream on RTX 5090 and RTX PRO 6000 with the NVFP4
checkpoint). The diff is 879 lines, Python only, touching `qsa/kernel.py`, `qsa/sparse_attn.py`,
`qwen_sparse_attn_backend.py`, and one test file. sha256 of the diff as applied here:
`2730b1659fdc384ece80e8f51039fb950968a0e71e799b5f35283be0302c9fd5`.

**Applying it.** The patch applied cleanly with `git apply` against this image's sglang commit
(`g593134d17`), the same commit recorded in section 5. It was added to the build as a third patch
step in `Dockerfile`, after the existing M-RoPE fix: the diff is copied into the build context,
applied with `git apply`, and verified with a `grep -q k_scale` guard against `qsa/kernel.py` that
fails the build if the patch did not land.

**Mechanism.** The patch dequantizes the FP8 KV scratch into the query dtype with a per-layer
descale factor before the SM121 QSA kernel call, rather than requiring the kernel to consume FP8
directly. This is why the kernel's fixed shape list in section 4d (`TP1 24Q/2KV or TP2 12Q/1KV`)
was never actually the FP8 blocker: the kernel still runs in BF16 internally, the KV cache storage
just happens to be FP8 on either side of it. Since the checkpoint ships no KV scale factors, the
descale defaults to 1.0, and the server logs the same "Using FP8 KV cache but no scaling factors
provided. Defaulting to scaling factors of 1.0" message as before at startup, now informational
rather than a precursor to the attempt-9 crash.

**Test at 65,536 context.** With the patch applied and `--kv-cache-dtype fp8_e4m3` restored, the KV
pool held 178,624 tokens (0.51 GB K plus 0.51 GB V), against 114,688 to 124,480 tokens under BF16
at the same context (0.71 GB K plus 0.71 GB V each side). Benchmarked with thinking mode off and
512-token completions: time to first token 0.17 to 0.18 s, single-stream 31.3 to 36.6 tok/s,
4-stream aggregate 90.3 to 91.0 tok/s. Needle-in-context recall was correct at roughly 4.6K, 20.6K,
and 52.2K prompt tokens, with no token-id-0 corruption observed in any test run.

**Promotion to 131,072 context and the memory guard.** Given the roughly 33 percent smaller KV pool
footprint at a given context (12 of 48 layers carry attention KV at all, per section 3, so the FP8
savings apply only to that fraction), the natural next step was doubling context rather than
banking the memory savings unused. At `--context-length 131072`, the KV pool holds 242,944 tokens
(0.70 GB K plus 0.70 GB V), with 9.98 GB of available GPU memory remaining on the head node after
CUDA graph capture, comfortably inside the memory floor discussed in section 4c and section 6.
Weight load took 483.78 s, the MTP draft head 52.53 s, both consistent with the section 6 baseline.
Needle recall was independently verified correct at 16,817, 101,066, and 125,596 prompt tokens.
Measured throughput: single-stream 35.7 to 37.0 tok/s, 2-stream aggregate 59.4 tok/s, 4-stream
aggregate 92.8 tok/s, time to first token 0.15 to 0.19 s, all neutral against the BF16 65K baseline
in section 7. The KV pool re-sizes with `--context-length` at the same 0.90 `--mem-fraction-static`
recorded in section 4c, with no other flag changed to reach this figure.

**Promotion and rollback procedure.** Before rebuilding, the pre-patch image was retagged
(`docker tag sglang-spark:fp8 sglang-spark:fp8-bf16kv`) and the pre-patch `launch.sh` kept as a
backup, on both boxes. Reverting to BF16 KV at 65,536 context is a retag plus
`systemctl restart sglang-cluster`, restoring the flags `--kv-cache-dtype auto` and
`--context-length 65536` under the same `sglang-spark:fp8` tag the systemd unit expects, with no
change to the unit file itself in either direction.

**Descale caveat.** The 1.0 default descale is a known approximation, not a measured calibration:
the checkpoint was never shipped with FP8 KV scale factors, so there is no calibrated value to fall
back to short of computing one from the model's own activation statistics. This is recorded as the
open question in section 10 going forward, in place of the resolved "is FP8 KV possible" question.

**Benchmark-contention pitfall.** Running two benchmark clients against the same endpoint
concurrently roughly halves both the per-client and the aggregate measured throughput. Every number
in this section was captured with a single benchmark client running alone: a multi-stream figure
(the "2 concurrent streams" and "4 concurrent streams" rows above) means multiple streams from one
client, not multiple clients.

## 13. Sources

- NVIDIA Developer Forums thread 381428, "Qwen3.8 Flash Next NVFP4 on 2x DGX Spark, full
  multimodal, 70 tok/s peak, 47 typical":
  https://forums.developer.nvidia.com/t/qwen3-8-flash-next-nvfp4-on-2x-dgx-spark-full-multimodal-70-tok-s-peak-47-typical/381428
- NVIDIA Developer Forums thread 381440 (no numbers, unresolved question):
  https://forums.developer.nvidia.com/t/381440
- `tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark`:
  https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark
- `MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks`:
  https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks
- SGLang discussion #36891: https://github.com/sgl-project/sglang/discussions/36891
- vLLM recipe page, `Qwen/Qwen3.8-Flash-Next`: https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next
- vLLM issue #54173 (prefix caching illegal memory access on `sm_121`):
  https://github.com/vllm-project/vllm/issues/54173
- NVIDIA `dgx-spark-playbooks`, `connect-two-sparks` recipe (two-Spark cabling and bandwidth
  guidance): https://github.com/NVIDIA/dgx-spark-playbooks (under `nvidia/connect-two-sparks`)
- NVIDIA Developer Forums thread on this run, "FP8 Qwen3.8-Flash-Next on 2x DGX Spark via
  SGLang: 37-40 tok/s":
  https://forums.developer.nvidia.com/t/fp8-qwen3-8-flash-next-on-2x-dgx-spark-via-sglang-37-40-tok-s/382435
- Forum user jahnclawdmonet's reply on the thread above (2026-09-05), pointing at sglang PR #36644
  for FP8 KV cache support: https://github.com/sgl-project/sglang/pull/36644
