#!/bin/bash
# ============================================================
# 启动 LLaMA Board Web UI
# 在浏览器中可视化配置、启动训练、监控和推理
# ============================================================

set -e

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "  LLaMA-Factory Web UI 启动脚本"
echo "  项目路径: $PROJECT_DIR"
echo "=========================================="

# 激活 conda 环境
# 请根据实际情况修改环境名称
ENV_NAME="llamafactory"

if command -v conda &> /dev/null; then
    # 尝试 conda run
    echo "[INFO] 使用 conda 环境: $ENV_NAME"
else
    echo "[WARN] conda 未找到，尝试直接使用 python3"
fi

# 进入项目目录
cd "$PROJECT_DIR"

# 启动 LLaMA Board Web UI
# --share: 生成 Gradio 公开链接 (可选)
# --host 0.0.0.0: 局域网内其他设备可访问
echo "[INFO] 启动 LLaMA Board Web UI..."
echo "[INFO] 访问地址: http://localhost:7860"
echo ""

if command -v conda &> /dev/null; then
    conda run -n "$ENV_NAME" llamafactory-cli webui \
        --host 0.0.0.0 \
        --port 7860
else
    llamafactory-cli webui \
        --host 0.0.0.0 \
        --port 7860
fi
