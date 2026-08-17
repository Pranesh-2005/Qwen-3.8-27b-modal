# Qwen3.8-27B (4-bit) on Modal — low cost

Serves `cyankiwi/Qwen3.8-27B-AWQ-INT4` (dense 27B vision-language, reasoning, apache-2.0, ~21GB INT4 weights) via vLLM on a **single A100-80GB**.

## Benchmarks

Charts built from the official [model-card](https://huggingface.co/Qwen/Qwen3.8-27B) scores (Qwen3.8-27B vs its predecessor Qwen3.6-27B and Opus4.6 Max). Regenerate with `python make_benchmark_charts.py`.

![Text and agent benchmarks](benchmarks_text.svg)

![Vision-language benchmarks](benchmarks_vl.svg)

## Prerequisites

```
pip install modal openai
modal setup
```

No HF secret needed — the model repo is public.

## Steps

1. **Download weights into a Modal Volume** (CPU only, no GPU cost):
   ```
   modal run download_model.py
   ```
   Idempotent, safe to re-run if interrupted.

2. **Deploy the vLLM server** (1x A100-80GB):
   ```
   modal deploy serve.py
   ```
   Copy the printed endpoint URL (`https://<workspace>--qwen38-27b-serve-serve.modal.run`).

3. **Chat**:
   ```
   python chat.py <endpoint-url>
   ```
   OpenAI-compatible API, so any OpenAI client works (`base_url = <endpoint>/v1`, model `qwen3.8-27b`).

## Cost

| Item | Cost |
|---|---|
| 1x A100-80GB | **~$2.50/hr** (still ~7x cheaper than the 4xH200 GLM-5.2 deploy) |
| Cold boot | a few minutes (weight load + cudagraph capture) |
| Volume storage | free at this size (21GB) |

`max_containers=1` and no keep-warm → scales to zero after 10 min idle; you only pay while serving. Two free volumes persist between runs: `qwen38-27b-awq` (weights, read-only) and `qwen38-27b-vllm-cache` (compile cache, faster reboots).

**Want cheaper?** A 48GB L40S ($1.95/hr) OOMs at default settings — this VL model's live footprint is ~42GB before KV cache. To force it onto an L40S anyway, set `gpu="L40S"` and add `--enforce-eager --max-num-seqs 4 --max-model-len 8192 --limit-mm-per-prompt '{"image":1,"video":0}'` (see comment in `serve.py`). Tighter and slower, but works.

## Notes / knobs

- **Context**: capped at 64K (`MAX_MODEL_LEN` in `serve.py`). Native is 262K; KV cache is ~17GB at 64K and ~34GB at 128K, and the live footprint is already ~42GB of the 80GB — so 128K OOMs on one A100.
- **Reasoning / tools**: tool calling is on (`--enable-auto-tool-choice --tool-call-parser qwen3_xml`) so OpenAI-style `tools` + `tool_choice:"auto"` work. Parser must be `qwen3_xml`, not `hermes` — this model's chat template emits XML calls (`<tool_call><function=f><parameter=p>`), and `hermes` tries `json.loads` on that and fails every call.
- **Thinking is off** by default here, via `--default-chat-template-kwargs '{"enable_thinking": false}'` in `serve.py` (the model thinks by default — thinking tokens are GPU seconds you pay for). Turn it back on:
  - **per request** (wins over the server default): `extra_body={"chat_template_kwargs": {"enable_thinking": True}}`
  - **server-wide**: delete that flag from `serve.py` and redeploy. Then add `--reasoning-parser qwen3` if you want thinking split into `reasoning_content` instead of streamed inline.

  Recommended sampling differs per mode — non-thinking: `temperature=0.7, top_p=0.8, top_k=20`; thinking: use the model card's thinking values. Also `reasoning_effort` (`low`/`medium`/`xhigh`) controls depth when thinking is on.
- **Vision**: it's a VL model; vLLM accepts images via the OpenAI `image_url` content type out of the box.
- **vLLM version**: pinned to `0.25.1`. If boot fails with an unknown-architecture error, bump the pin in `serve.py` to a release that supports Qwen3.8.
- **FlashInfer sampler**: disabled via `VLLM_USE_FLASHINFER_SAMPLER=0` — it JIT-compiles a CUDA kernel at boot and crashes on the slim image (`Could not find nvcc`). Native Torch sampler is used instead. If another nvcc JIT error appears, switch the image base to `nvidia/cuda:12.8.1-devel` + `CUDA_HOME` (see comment in `serve.py`).

## Important

**Stop the app when done** to avoid idle GPU burn:
```
modal app stop qwen38-27b-serve
```
