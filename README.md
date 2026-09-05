# dgx-spark-qwen38-flash-next-fp8

Launch assets for running `Qwen/Qwen3.8-Flash-Next-FP8` across two NVIDIA DGX Sparks with SGLang, using tensor parallel plus expert parallel across nodes, speculative decoding (MTP/NEXTN), and systemd-managed startup.

Every published dual-Spark result for this model uses the NVFP4 checkpoint. That was the first thing I had to decide whether to accept or argue with.

The reasoning behind NVFP4 is sound: it is smaller, it fits the 128 GB unified memory budget on each GB10 with room to spare, and the community has already done the work of tuning it. But NVFP4 is a re-quantization, not the vendor's own release, and I wanted to know what the official 8-bit checkpoint could do on hardware nobody had run it on. Qwen/Qwen3.8-Flash-Next-FP8 is 172.78 GiB across 131 safetensors shards: an ultra-sparse multimodal Mixture-of-Experts model, roughly 125B total parameters with 6B active per token, plus a separate 51B-parameter N-gram embedding table, at 262K native context. It does not fit on one Spark. It barely fits on two.

## The hardware, and the "twins"

The setup is two NVIDIA DGX Spark units (I had one already but bought the 2nd one specifically to run this model and do this testing), connected directly by a single ConnectX-7 200G QSFP cable with no switch between them. The detail that cost me the most time to understand is that each physical CX7 port presents as two Ethernet interfaces and two RoCE interfaces, one pair per PCIe 5.0 x4 link underneath: `enp1s0f1np1` and `enP2p1s0f1np1` on the Ethernet side, `rocep1s0f1` and `roceP2p1s0f1` on the RoCE side. These are not redundant paths to bond. Bonding the twins produces an ARP black hole (the two names are just two kernel-level identities for the same physical port, so a bond enslaves the same link twice and ARP never resolves). The correct topology puts exactly one IP on `enp1s0f1np1` and lets NCCL address both RoCE twins directly through `NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`. Measured with iperf3, the TCP link runs 110 Gbit/s with zero retransmits, which is the TCP ceiling on this cable, not the RDMA ceiling NCCL actually uses.

One more wrinkle worth recording: on the newer DGX OS build, the CX7 disappears from the PCI bus entirely when no cable is inserted. A hot-plug driver keyed on cable presence (visible in a sysfs `debug_state` flag, 0 or 1) removes the device rather than just marking the link down. The first time I saw this, it looked exactly like dead hardware. It is not. It is a driver doing what it was told.

Because the model needs both boxes, the weights have to live on each node's local NVMe, which means moving 173 GB over the RoCE link before anything else can happen: rsync managed about 406 MB/s, 7 minutes 15 seconds end to end. After expert-parallel placement, each node holds about 94 GB of weights plus a 1.8 GB multi-token-prediction draft head.

## What the community had already found

Before this run, every public dual-Spark number I could find was on NVFP4. NVIDIA Developer Forums [thread 381428](https://forums.developer.nvidia.com/t/qwen3-8-flash-next-nvfp4-on-2x-dgx-spark-full-multimodal-70-tok-s-peak-47-typical/381428) reports SGLang on NVFP4 at TP2 with four-step MTP and CUDA graphs: 47 tok/s typical, 70 peak, and about 20 without speculative decoding, backed by [tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark). [MiaAI-Lab's repo](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks) runs vLLM with NVFP4 experts and FP8 PLE (the N-gram embedding table) at TP2 plus EP with three-step MTP, 44 to 56 tok/s. An [sglang GitHub discussion](https://github.com/sgl-project/sglang/discussions/36891), #36891, describes FP8 weights with FP8 KV cache and MTP at TP2 on unspecified hardware, and a second forum thread, [381440](https://forums.developer.nvidia.com/t/381440), simply asked whether FP8 worked on dual Sparks at all: no flags, no numbers, no answer. The [vendor's own vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next) for this model states that TP2 is the minimum viable FP8 deployment on GB300, and that tensor-and-expert parallelism (TEP) is required on H200 because plain tensor parallelism is incompatible with the checkpoint's 128-wide quantization blocks. There is no INT8 W8A8 release of this model to fall back on. The discussion of this run lives on the NVIDIA Developer Forums: https://forums.developer.nvidia.com/t/fp8-qwen3-8-flash-next-on-2x-dgx-spark-via-sglang-37-40-tok-s/382435

The question was open. I set out to answer it.

## The six walls

The engine is SGLang, running the day-0 image `lmsysorg/sglang:qwen38flashnext` (sglang `0.0.0.dev1+g593134d17`, torch `2.13.0+cu130`), with exactly one local patch: an M-RoPE partial-rotary fix in `rotary_triton.py` (`t_mask = ~(h_mask | w_mask) & (cos_offsets < half_rd)`). The SM121 QSA guard (`is_sm120()`) was already upstream, which was a pleasant surprise. Getting there took six failures, in order, each with a fix that only makes sense after the error has been seen.

1. **vLLM refuses FP8 KV outright.** The `eugr/spark-vllm-docker` toolkit, native PyTorch distributed TP2, with `--kv-cache-dtype fp8`: `NotImplementedError: Qwen4Exp QSA requires a BF16 main KV cache`. Not a configuration problem, a hard requirement.
2. **vLLM with BF16 KV takes down the head node.** At `gpu-memory-utilization 0.85`, after loading 87.5 GiB of weights per node, the head thrashed into swap during `torch.compile` and profiling (about 16 GiB of RAM free after weights). The worker logged `Worker proc VllmWorker-1 died unexpectedly (exit code: None)`, and the head needed a power cycle. The community attributes this to CUDA graphs and `torch.compile` fighting GB10's unified memory model (their workaround is `--enforce-eager`), and reports that 0.80 is the safe ceiling where 0.85 drifts into swap. I abandoned vLLM here rather than chase it further.
3. **SGLang's tensor parallelism collides with the quantization block size.** `--tp-size 2` alone: `ValueError: The output_size of gate's and up's weight = 320 is not divisible by weight quantization block_n = 128`. The FP8 blocks are 128 wide, and sharding 640-wide experts across two ranks under plain TP does not divide cleanly. Fix: `--ep-size 2`, so experts stay whole on one rank (640 = 5 x 128).
4. **A pinned GID index breaks NCCL on exactly one twin.** With `NCCL_IB_GID_INDEX=3` set in the environment: `ibv_modify_qp failed with 61 No data available, on dev roceP2p1s0f1:1 ... local GID index 3, local GID ::`, then `NCCL error: unhandled system error`. GID index 3 (RoCEv2 over the IPv4 address) exists only on the twin whose netdev actually carries the IP. The other twin has nothing but link-local GIDs at that index. Fix: do not pin the index. Let NCCL select per HCA.
5. **The memory budget for the KV cache was simply too tight.** `--mem-fraction-static 0.80`: `Loaded weights leave no GPU memory for the KV cache under --mem-fraction-static=0.8. Raise --mem-fraction-static above 0.872 (minimum viable = 0.8714)`. Measured static footprint was 93.47 GB of main weights plus 1.99 GB of MTP draft per node. Fix: 0.90, paired with a 65536 context, four running requests, 2048-token chunked prefill, and a CUDA graph max batch size of 4.
6. **FP8 KV loads fine and then fails on the first decode.** `--kv-cache-dtype fp8_e4m3` allocated the KV pool without complaint (209,472 tokens, 1.3 GB), but the first decode through the SM121 sparse-attention kernel raised `ValueError: unsupported SM121 QSA call: expected BF16 D=256, 12:1 GQA, TP1 24Q/2KV or TP2 12Q/1KV, bs<=128, and selected KV<=2055`. The checkpoint also ships no KV scale factors at all (`Using FP8 KV cache but no scaling factors provided. Defaulting to scaling factors of 1.0`), which would have been its own problem. Fix: `--kv-cache-dtype auto`, meaning BF16. The cost turned out smaller than I expected, about 1 GB, because only 12 of the model's 48 layers carry attention KV at all (the other 36 are Gated-DeltaNet, running a fixed recurrent state instead). The BF16 pool held 124,480 tokens at first (0.71 GB K plus 0.71 GB V), settling to 114,688 after the systemd cutover described below.

The full launch configuration lives in `launch.sh` in this repository, with the environment variables for the RoCE twins, `SGLANG_DEEPGEMM=0` (DeepGEMM's block-FP8 path is known broken on GB10), and the page-cache drop that precedes every launch.

## What it measures

| Metric | Result |
|---|---|
| Single-stream, 256-token completion | 36 to 41 tok/s (median 40.3) |
| Single-stream, 512-token completion | 34.7 to 38.8 tok/s |
| Time to first token (short prompt, streaming) | 0.18 s |
| 4 concurrent 512-token streams | 22 to 24 tok/s each, 88 to 98.5 tok/s aggregate (two runs) |
| 3,561-token prompt, summarized | 5.5 s |
| Resident memory after warmup | about 70.9 GB per node (nvidia-smi), 6 GB free on the head, 9 GB on the worker, stable over five minutes |

Thinking mode and `qwen3_coder` tool calls both work end to end. Three factual probes I ran came back correct, which is not a benchmark, just a sanity check I wanted on record.

Startup is not instant: about 8 minutes of weight loading (483 s on the head, 472 s on the worker), 45 to 52 seconds for the draft head, and roughly 40 seconds of CUDA graph capture, for a total of about 10 minutes to a ready server. Oddly, the server reports its own speculative algorithm as EAGLE even though the configuration requests NEXTN, apparently because NEXTN runs on the EAGLE code path internally. And `--ple-offload-embedding` may simply be a no-op in this build: resident weights still measured about 94 GB whether or not it was set, which I cannot fully explain and am not going to pretend I can.

## Making it survive a reboot

Each node runs the container under a systemd unit (`Type=simple`, foreground `docker run --rm`, `Restart=always`, `RestartSec 30`, `TimeoutStartSec 900`), enabled at boot. Because the head node is the NCCL rendezvous master with a 600-second distributed-init timeout (`--dist-timeout 600`), either node is free to come up first without a hand-holding startup order. I tested this by restarting the worker's service: the head detected the NCCL peer loss and exited, its own unit restarted it 30 seconds later, and the full two-node cluster was back and serving in 14 minutes with no operator action. The only trace of the event was one benign NCCL warning, "GID table changed", during re-rendezvous. Prometheus scrapes the SGLang `/metrics` endpoint directly (69 `sglang:*` metric families show up once traffic has flowed, though the latency histograms specifically don't populate until the first request completes).

## The comparison

This FP8 run is 15 to 21 percent slower single-stream than the published NVFP4 dual-Spark numbers (40 and 37 tok/s here against 47 typical there), holds a quarter of the context (65K against 262K), and keeps the KV cache in BF16 rather than FP8. In exchange, the weights are the vendor's own 8-bit checkpoint with no re-quantization step of my own in the loop. Whether that trade, an unmodified checkpoint against roughly 20 percent more throughput and four times the context on NVFP4, is worth making is a question about output quality under each quantization, and I have not measured that here. It would be remiss to not at least mention that this rests on a single hardware pair and a handful of runs, not a rigorous benchmark suite.

What I'm still turning over is whether the FP8 KV wall (failure six above) is a permanent SM121 kernel limitation or just an unwritten kernel variant. The shape constraints in that error message (`TP1 24Q/2KV or TP2 12Q/1KV`) read like a kernel that was written for specific configurations rather than a general one. If someone gets FP8 KV working on this hardware, I would like to know what they changed.

## Quick start

### Build (run on each box separately)
```bash
cd /opt/sglang-cluster
docker build -t sglang-spark:fp8 -f Dockerfile .
```
Each box builds its own local image derivative from the public `lmsysorg/sglang:qwen38flashnext` base. Image ids will differ between boxes because of build timestamps.

### Launch
`launch.sh` drops the page cache, removes any existing `sglang_node` container, and runs `python3 -m sglang.launch_server` with `--tp-size 2 --ep-size 2 --nnodes 2 --node-rank N --dist-init-addr $HEAD_IP:29511` over the ConnectX-7 RoCE link, speculative decoding (NEXTN, 3 steps, topk 1, 4 draft tokens), fp8 quantization with BF16 KV cache (`--kv-cache-dtype auto`), and `--context-length 65536`. `MODEL_DIR` and `HF_CACHE` are environment variables with sane defaults. Override them for a different layout if needed. `HEAD_IP` has no default: set it to the head node's address on the cluster's fast interconnect. Order matters at startup: start rank 0 (the rendezvous master) within about 30 seconds before or after rank 1.

### Install as a systemd service (one-time, per box)
Before enabling either unit, create `/etc/sglang-cluster.env` with `HEAD_IP=<head node interconnect address>` (and optionally `MODEL_DIR`, `HF_CACHE`), which the units load via `EnvironmentFile`.
```bash
sudo cp systemd/sglang-cluster.rank0.service /etc/systemd/system/sglang-cluster.service   # Spark #1 (rank 0)
sudo cp systemd/sglang-cluster.rank1.service /etc/systemd/system/sglang-cluster.service   # Spark #2 (rank 1)
sudo systemctl daemon-reload
sudo systemctl enable --now sglang-cluster.service
```

### Manual run, outside systemd
The same `/etc/sglang-cluster.env` file works for manual launches: `set -a; . /etc/sglang-cluster.env; set +a` before running `launch.sh`.
```bash
NODE_RANK=0 IMAGE=sglang-spark:fp8 DETACH=1 bash launch.sh
```

### Stop for maintenance (on both boxes)
```bash
sudo systemctl stop sglang-cluster
```
Do not run `docker rm -f sglang_node` to stop the service: with `Restart=always` systemd will simply relaunch it.

## Author

Brad Smith
