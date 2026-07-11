#!/bin/bash
# ============================================================
# 启动 GPU 资源监控 (nvidia-smi 实时刷新)
# 在另一个终端中运行，方便观察显存和 GPU 利用率
# ============================================================

echo "=========================================="
echo "  NVIDIA GPU 监控 (每 2 秒刷新)"
echo "  按 Ctrl+C 退出"
echo "=========================================="

watch -n 2 nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv
