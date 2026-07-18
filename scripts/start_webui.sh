#!/bin/bash
# ============================================================
# 启动 LLaMA Board Web UI
# 在浏览器中可视化配置、启动训练、监控和推理
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "  LLaMA-Factory Web UI 启动脚本"
echo "  项目路径: $PROJECT_DIR"
echo "=========================================="

cd "$PROJECT_DIR"

ENV_PREFIX=/home/hyl/miniconda3/envs/qwen_fintune
export PATH="$ENV_PREFIX/bin:$PATH"
export PYTHONPATH=""

echo "[INFO] 使用环境: qwen_fintune"
echo "[INFO] 启动 LLaMA Board Web UI..."
echo "[INFO] 访问地址: http://localhost:7860"
echo ""

"$ENV_PREFIX/bin/llamafactory-cli" webui \
    --host 0.0.0.0 \
    --port 7860
