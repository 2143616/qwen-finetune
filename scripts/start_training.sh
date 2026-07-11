#!/bin/bash
# ============================================================
# 命令行一键启动训练脚本
# 无需 GUI，直接在终端运行训练
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "  LLaMA-Factory CLI 训练启动脚本"
echo "  配置: config/train_qwen_qlora.yaml"
echo "  输出: output/qwen-qlora-financial"
echo "=========================================="

cd "$PROJECT_DIR"

ENV_NAME="llamafactory"

if command -v conda &> /dev/null; then
    echo "[INFO] 使用 conda 环境: $ENV_NAME"
    echo "[INFO] 开始训练..."
    conda run -n "$ENV_NAME" llamafactory-cli train config/train_qwen_qlora.yaml
else
    echo "[INFO] 开始训练..."
    llamafactory-cli train config/train_qwen_qlora.yaml
fi

echo ""
echo "=========================================="
echo "  训练完成！"
echo "  检查点保存在: $PROJECT_DIR/output/"
echo "=========================================="
