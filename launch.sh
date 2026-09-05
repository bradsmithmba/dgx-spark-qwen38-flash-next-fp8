#!/bin/bash
set -e
NODE_RANK=${NODE_RANK:?}
IMAGE=${IMAGE:?}
# Rendezvous head address for --dist-init-addr below: the head node's
# address on your cluster's fast interconnect. No default, set it per cluster.
HEAD_IP=${HEAD_IP:?set HEAD_IP to the head node interconnect address}
MODEL_DIR=${MODEL_DIR:-/opt/models}
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
docker rm -f sglang_node 2>/dev/null || true
DETACH_FLAG=""
if [ "${DETACH:-0}" = "1" ]; then
  DETACH_FLAG="-d"
fi
docker run $DETACH_FLAG --rm --name sglang_node --network host --gpus all --ipc=host --privileged \
  -v "$MODEL_DIR":/models:ro \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 -e NCCL_IB_DISABLE=0 \
  -e NCCL_SOCKET_IFNAME=enp1s0f1np1 -e GLOO_SOCKET_IFNAME=enp1s0f1np1 \
  -e NCCL_DEBUG=INFO -e SGLANG_DEEPGEMM=0 -e SGL_ENABLE_JIT_DEEPGEMM=0 \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path /models/Qwen3.8-Flash-Next-FP8 \
    --served-model-name qwen3.8-flash-next-fp8 \
    --host 0.0.0.0 --port 8000 \
    --tp-size 2 --nnodes 2 --node-rank "$NODE_RANK" \
    --ep-size 2 \
    --dist-init-addr "$HEAD_IP":29511 --dist-timeout 600 \
    --quantization fp8 \
    --kv-cache-dtype fp8_e4m3 \
    --page-size 64 \
    --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --chunked-prefill-size 2048 \
    --max-running-requests 4 \
    --context-length 131072 \
    --mem-fraction-static 0.90 \
    --cuda-graph-max-bs 4 --disable-cuda-graph-padding \
    --ple-offload-embedding \
    --disable-radix-cache \
    --sampling-backend pytorch \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
    --trust-remote-code \
    --enable-metrics
