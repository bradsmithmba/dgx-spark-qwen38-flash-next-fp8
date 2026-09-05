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

Tags: dgx-spark, sglang, fp8, moe, quantization
