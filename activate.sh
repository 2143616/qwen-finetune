#!/bin/bash
# 激活 llamafactory 环境
source /home/hyl/miniconda3/etc/profile.d/conda.sh
export CONDA_PREFIX_1=""
export PYTHONPATH=""
conda activate llamafactory
echo "✅ llamafactory 环境已激活"
echo "   Python: $(python --version)"
echo "   PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "   CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "   CUDA 版本: $(python -c 'import torch; print(torch.version.cuda)')"
