import modal

app = modal.App("qwen38-27b-serve")

MODEL_REPO = "cyankiwi/Qwen3.8-27B-AWQ-INT4"
VOLUME_NAME = "qwen38-27b-awq"
MODELS_DIR = "/models"
MODEL_PATH = f"{MODELS_DIR}/{MODEL_REPO.split('/')[-1]}"
SERVED_NAME = "qwen3.8-27b"
# Native context is 262144; capped here so the KV cache fits alongside the
# ~21GB weights on one 48GB L40S. Raise if you have headroom / need long ctx.
MAX_MODEL_LEN = 32768

# AWQ (compressed-tensors) uses precompiled Marlin kernels — no nvcc/DeepGEMM
# needed. FlashInfer's *sampler*, though, JIT-compiles a CUDA kernel at boot and
# aborts on this slim image ("Could not find nvcc"). Disable it → vLLM uses the
# native Torch top-k/top-p sampler (no JIT). If any *other* nvcc JIT error shows
# up later, swap this base for nvidia/cuda:12.8.1-devel + CUDA_HOME instead.
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    # Pinned to the version proven on the GLM-5.2 deploy. If vLLM rejects the
    # Qwen3.8 arch on boot, bump this pin to the release that adds it.
    "vllm==0.25.1",
    "huggingface_hub",
).env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

# vLLM cudagraph/torch.compile cache — persisting it makes reboots faster.
compile_cache = modal.Volume.from_name("qwen38-27b-vllm-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",  # ~$2.50/hr. This VL model's live footprint (INT4 LLM +
    # bf16 vision tower + profiling workspace) is ~42GB *before* KV cache, which
    # OOMs a 48GB L40S. 80GB leaves room for KV + cudagraphs. To force it onto a
    # cheaper 48GB L40S instead, set gpu="L40S" and add these to the cmd below:
    #   "--enforce-eager", "--max-num-seqs", "4", "--max-model-len", "8192",
    #   "--limit-mm-per-prompt", '{"image":1,"video":0}'   (tighter, slower)
    volumes={
        MODELS_DIR: volume.read_only(),
        "/root/.cache/vllm": compile_cache,
    },
    timeout=60 * 60,
    scaledown_window=600,
    # One replica: stops the autoscaler adding a second GPU during slow boots.
    max_containers=1,
    # ponytail: no min_containers/keep_warm — scales to zero when idle so you
    # only pay while serving. Cold boot is a few minutes for a 27B.
)
@modal.concurrent(max_inputs=16)
@modal.web_server(port=8000, startup_timeout=15 * 60)
def serve():
    import subprocess

    cmd = [
        "vllm", "serve", MODEL_PATH,
        "--served-model-name", SERVED_NAME,
        "--max-model-len", str(MAX_MODEL_LEN),
        "--host", "0.0.0.0",
        "--port", "8000",
        # Tool calling: clients (e.g. turnloop) send tool_choice="auto", which
        # vLLM rejects unless these are set. Qwen3 uses the hermes parser.
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
        # To also split thinking into reasoning_content, add:
        #   "--reasoning-parser", "qwen3"
    ]
    subprocess.Popen(cmd)
