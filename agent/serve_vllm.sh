#!/bin/bash
#
# vLLM Server with LoRA Support
#
# 启动一个兼容OpenAI API的vLLM服务器，支持动态加载LoRA适配器。
#
# Usage:
#     # 使用LoRA适配器
#     ./serve_vllm.sh --lora out/qwen3-4b-lora-medical
#     
#     # 仅使用基础模型
#     ./serve_vllm.sh --no-lora
#     
#     # 自定义端口
#     ./serve_vllm.sh --port 8080

# =========================
# 可配置参数（默认值）
# =========================
BASE_MODEL_PATH="model/qwen"
LORA_ADAPTER_PATH="out/qwen3-4b-lora"
SERVED_MODEL_NAME="qwen"
HOST="localhost"
PORT=8001
MAX_LORA_RANK=16
DTYPE="bfloat16"
GPU_MEMORY_UTILIZATION=0.9
NO_LORA=false
TENSOR_PARALLEL_SIZE=2
MAX_MODLE_LEN=8192
MAX_NUM_SEQS=1

# =========================
# 帮助信息
# =========================
show_help() {
    cat << EOF
Start vLLM server with LoRA support

Usage: $(basename "$0") [OPTIONS]

Options:
    --model PATH              Base model path (default: $BASE_MODEL_PATH)
    --lora PATH               LoRA adapter path (default: $LORA_ADAPTER_PATH)
    --port PORT               Server port (default: $PORT)
    --host HOST               Server host (default: $HOST)
    --no-lora                 Disable LoRA loading, use base model only
    --max-lora-rank RANK      Maximum LoRA rank (default: $MAX_LORA_RANK)
    --dtype TYPE              Model data type: float16, bfloat16, float32, auto (default: $DTYPE)
    --gpu-memory-utilization  GPU memory utilization (default: $GPU_MEMORY_UTILIZATION)
    -h, --help                Show this help message

Examples:
    $(basename "$0")                                    # 使用默认LoRA
    $(basename "$0") --lora out/my-lora-adapter        # 指定LoRA路径
    $(basename "$0") --no-lora                          # 不使用LoRA
    $(basename "$0") --model /path/to/model --port 8080 # 自定义模型和端口
EOF
    exit 0
}

# =========================
# 参数解析
# =========================
# while [[ $# -gt 0 ]]; do
#     case $1 in
#         --model)
#             BASE_MODEL_PATH="$2"
#             shift 2
#             ;;
#         --lora)
#             LORA_ADAPTER_PATH="$2"
#             shift 2
#             ;;
#         --port)
#             PORT="$2"
#             shift 2
#             ;;
#         --host)
#             HOST="$2"
#             shift 2
#             ;;
#         --no-lora)
#             NO_LORA=true
#             shift
#             ;;
#         --max-lora-rank)
#             MAX_LORA_RANK="$2"
#             shift 2
#             ;;
#         --dtype)
#             DTYPE="$2"
#             shift 2
#             ;;
#         --gpu-memory-utilization)
#             GPU_MEMORY_UTILIZATION="$2"
#             shift 2
#             ;;
#         -h|--help)
#             show_help
#             ;;
#         *)
#             echo "Unknown option: $1"
#             echo "Use --help for usage information"
#             exit 1
#             ;;
#     esac
# done

# =========================
# 构建vLLM启动参数
# =========================
SERVER_ARGS=(
    --model "$BASE_MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --host "$HOST"
    --port "$PORT"
    --trust-remote-code
    --dtype "$DTYPE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --max-model-len "$MAX_MODLE_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --enable-auto-tool-choice
    --tool-call-parser "hermes"
)

# 如果启用LoRA
if [[ "$NO_LORA" == false ]] && [[ -n "$LORA_ADAPTER_PATH" ]]; then
    SERVER_ARGS+=(
        --enable-lora
        --lora-modules "lora-adapter=$LORA_ADAPTER_PATH"
        --max-lora-rank "$MAX_LORA_RANK"
    )
    echo "LoRA enabled: $LORA_ADAPTER_PATH"
    echo "  - Use model='lora-adapter' in API calls to use LoRA"
    echo "  - Use model='$SERVED_MODEL_NAME' to use base model"
else
    echo "LoRA disabled, using base model only"
fi

# =========================
# 启动服务器
# =========================
echo "============================================================"
echo "Starting vLLM Server"
echo "============================================================"
echo "Model: $BASE_MODEL_PATH"
echo "Host: $HOST"
echo "Port: $PORT"
echo "API Base URL: http://$HOST:$PORT/v1"
echo "============================================================"
echo "Command: python -m vllm.entrypoints.openai.api_server ${SERVER_ARGS[*]}"
echo "============================================================"

# 启动服务器
exec python -m vllm.entrypoints.openai.api_server "${SERVER_ARGS[@]}"
