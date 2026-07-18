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

ENV_PREFIX=/home/hyl/miniconda3/envs/qwen_fintune
export PATH="$ENV_PREFIX/bin:$PATH"
export PYTHONPATH=""
export LLAMAFACTORY_VERBOSITY=ERROR     # 只输出 ERROR 日志

echo "[INFO] 使用环境: qwen_fintune"
echo "[INFO] 开始训练..."
"$ENV_PREFIX/bin/llamafactory-cli" train config/train_qwen_qlora.yaml

echo ""
echo "=========================================="
echo "  训练完成！"
echo "  检查点保存在: $PROJECT_DIR/output/"
echo "=========================================="
