#!/bin/bash
# 激活 qwen_fintune 环境 (env bin 直跑，绕过 conda 26.5.3 activate bug)
ENV_PREFIX=/home/hyl/miniconda3/envs/qwen_fintune
export PATH="$ENV_PREFIX/bin:$PATH"
export CONDA_PREFIX="$ENV_PREFIX"
export CONDA_DEFAULT_ENV=qwen_fintune
export CONDA_PREFIX_1=""
export PYTHONPATH=""
export LLAMAFACTORY_VERBOSITY=ERROR     # 只输出 ERROR 日志
echo "✅ qwen_fintune 环境已激活"
echo "   Python: $($ENV_PREFIX/bin/python --version)"
echo "   PyTorch: $($ENV_PREFIX/bin/python -c 'import torch; print(torch.__version__)')"
echo "   CUDA: $($ENV_PREFIX/bin/python -c 'import torch; print(torch.cuda.is_available())')"
echo "   CUDA 版本: $($ENV_PREFIX/bin/python -c 'import torch; print(torch.version.cuda)')"
