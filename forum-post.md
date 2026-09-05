Posted 2026-09-04 in DGX Spark / GB10: https://forums.developer.nvidia.com/t/fp8-qwen3-8-flash-next-on-2x-dgx-spark-via-sglang-37-40-tok-s/382435

FP8 Qwen3.8-Flash-Next on 2x DGX Spark via SGLang: 37-40 tok/s

Every dual-Spark result I could find for this model uses the NVFP4 checkpoint (thread 381428: 47 tok/s typical on SGLang, TP2, MTP4 plus CUDA graphs). Nobody had posted numbers for the official FP8 release, and one thread (381440) asked directly and got no answer. So I ran it: two GB10 units, 128 GB unified memory each, one ConnectX-7 200G cable, no switch, SGLang (`lmsysorg/sglang:qwen38flashnext`) at TP2/EP2, `--kv-cache-dtype auto`.

Result: 36 to 41 tok/s single-stream (256 tokens, median 40.3), 34.7 to 38.8 tok/s at 512 tokens, 0.18 s TTFT, 88 to 98.5 tok/s aggregate across four concurrent streams. 15 to 21 percent slower than the published NVFP4 numbers (40 and 37 tok/s here against 47 typical), on a quarter of the context (65K against 262K), KV cache staying BF16. The weights are the vendor's own 8-bit checkpoint, no re-quantization.

Four walls stood between a clean SGLang launch and that number:

1. **Plain TP collides with the FP8 block size.** `--tp-size 2` alone: `output_size of gate's and up's weight = 320 is not divisible by weight quantization block_n = 128`. Fix: `--ep-size 2`.
2. **A pinned NCCL GID index breaks one RoCE twin.** `NCCL_IB_GID_INDEX=3`: `ibv_modify_qp failed with 61 No data available ... local GID index 3, local GID ::`. Only one of the CX7's two RoCE interfaces per port carries the IPv4 GID. Fix: don't pin it.
3. **The static memory fraction was too tight for the KV cache.** `--mem-fraction-static 0.80`: `Raise ... above 0.872 (minimum viable = 0.8714)`. Measured: 93.47 GB weights plus 1.99 GB MTP draft per node. Fix: 0.90.
4. **FP8 KV cache loads, then fails on first decode.** `--kv-cache-dtype fp8_e4m3`: `unsupported SM121 QSA call: expected BF16 D=256, 12:1 GQA, TP1 24Q/2KV or TP2 12Q/1KV, bs<=128`. No KV scale factors shipped either. Fix: `--kv-cache-dtype auto`, costing about 1 GB since 36 of 48 layers are Gated-DeltaNet with no attention KV.

Launch script, environment variables, and systemd setup for both nodes (14-minute unattended recovery after a restarted worker, tested) at github.com/bradsmithmba/dgx-spark-qwen38-flash-next-fp8.

Two questions for the room. Is the FP8-versus-NVFP4 quality gap worth what you give up in throughput and context, or is NVFP4 close enough that the official checkpoint is a curiosity? And has anyone gotten FP8 KV cache past the SM121 QSA kernel's shape constraints? The error reads like a kernel written for specific configurations, not a hard wall.

Tags: llm (the forum only accepts pre-existing tags)

---

Reply posted 2026-09-05 to post 2 (jahnclawdmonet), thread post 3: https://forums.developer.nvidia.com/t/fp8-qwen3-8-flash-next-on-2x-dgx-spark-via-sglang-37-40-tok-s/382435/3

Thank you for the pointer. It worked, and here are the numbers.

I applied PR #36644 to the same image (sglang g593134d17, `git apply` took it clean, no conflicts) and relaunched with `--kv-cache-dtype fp8_e4m3` on the official FP8 checkpoint. The SM121 QSA error is gone. CUDA graph capture completes, first decode is fine, and I've now run it at both 65K and 131K context on the two Sparks.

What changed, measured:

1. KV pool at 65,536 context: 178,624 tokens (0.51 GB K + 0.51 GB V), against 114,688 to 124,480 tokens under BF16 (0.71 + 0.71 GB). Same 0.90 static memory fraction.
2. KV pool at 131,072 context: 242,944 tokens (0.70 + 0.70 GB). The pool re-sizes with the context flag in this build, and the head node still had 9.98 GB free after graph capture.
3. Throughput at 131K, FP8 KV, thinking off, 512-token completions: 35.7 to 37.0 tok/s single-stream, 92.8 tok/s aggregate at 4 concurrent streams, TTFT 0.15 to 0.19 s. Within noise of my BF16 numbers at 65K (34.7 to 38.8 single, 88 to 98.5 aggregate). Speed was never the goal here... the pool size was.
4. Needle recall was correct at 16,817, 101,066, and 125,596 prompt tokens, and I saw no token-id-0 runs in any test, including prompts above the 2,048 chunked-prefill size (the chunk-prefill kernel path you flagged).

Note: the checkpoint ships no KV scales, so the PR's per-layer descale runs at 1.0 and the server says so at startup. Correctness held in every test I ran, but that is three needle prompts and a benchmark script, not a quality evaluation. If anyone has a calibration recipe for k_scale/v_scale on this model, that is the piece I'm missing.

I didn't touch the FlashInfer routing (#36649/#36806). Everything above is on the paged-attention fallback path with the PR's dequantized scratch, so the 120K-plus corruption reports for the FlashInfer route do not apply to this configuration, and my 125K result should not be read as evidence about that route either.

Repo is updated with the patch step in the Dockerfile, the new launch flags, and a notes section on the promotion and rollback: github.com/bradsmithmba/dgx-spark-qwen38-flash-next-fp8

---

Comment posted 2026-09-05 on sglang PR #36644: https://github.com/sgl-project/sglang/pull/36644#issuecomment-5553857175

Data point from GB10 (SM121), which is not in the PR's validation matrix: 2x NVIDIA DGX Spark, TP2/EP2 over a ConnectX-7 200G direct link, the official `Qwen/Qwen3.8-Flash-Next-FP8` checkpoint (not NVFP4), image `lmsysorg/sglang:qwen38flashnext` (sglang `0.0.0.dev1+g593134d17`, torch 2.13.0+cu130), NEXTN speculative decoding (3 steps, 4 draft tokens).

Before the patch, `--kv-cache-dtype fp8_e4m3` allocated the pool and then failed at CUDA graph capture with `ValueError: unsupported SM121 QSA call: expected BF16 D=256, 12:1 GQA, TP1 24Q/2KV or TP2 12Q/1KV, bs<=128`.

With the PR applied (`git apply` against g593134d17, clean):

- FP8 KV allocates, graph capture completes, decode works. KV pool 178,624 tokens at 65,536 context and 242,944 tokens at 131,072 context (0.90 static memory fraction) against 114,688 to 124,480 under BF16 at 65,536.
- Needle recall correct at 16,817, 101,066, and 125,596 prompt tokens, with `--chunked-prefill-size 2048` so the chunk-prefill path is exercised. No token-id-0 output in any test.
- Throughput unchanged within noise: 35.7 to 37.0 tok/s single-stream, 92.8 tok/s aggregate at 4 streams, TTFT 0.15 to 0.19 s (512-token completions, thinking off), against 34.7 to 38.8 single and 88 to 98.5 aggregate with BF16 KV at 65K.
- The checkpoint ships no KV scales, so the descales run at the 1.0 default and the startup warning fires. Correctness held, but I have not run a quality eval.

This is the paged-attention fallback path (the FlashInfer/trtllm decode route stays excluded for SM121 after #36806). Dockerfile, launch flags, and notes: https://github.com/bradsmithmba/dgx-spark-qwen38-flash-next-fp8
