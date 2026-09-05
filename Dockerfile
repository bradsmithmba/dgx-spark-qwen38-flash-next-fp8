FROM lmsysorg/sglang:qwen38flashnext

# Patch 1 (SM121/GB10 attention guard): already present upstream in this image
# under the name is_sm120() rather than is_sm120_supported(). Verify it exists
# and fail the build if it does not (no working SM121 guard at all).
RUN grep -n "is_sm120" /sgl-workspace/sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py

# Patch 2 (M-RoPE t_mask fix): add the cos_offsets < half_rd bound.
RUN grep -n "t_mask = ~(h_mask | w_mask)$" /sgl-workspace/sglang/python/sglang/kernels/ops/attention/rotary_triton.py
RUN sed -i 's/t_mask = ~(h_mask | w_mask)$/t_mask = ~(h_mask | w_mask) \& (cos_offsets < half_rd)/' /sgl-workspace/sglang/python/sglang/kernels/ops/attention/rotary_triton.py
RUN grep -n "t_mask = ~(h_mask | w_mask) & (cos_offsets < half_rd)" /sgl-workspace/sglang/python/sglang/kernels/ops/attention/rotary_triton.py
