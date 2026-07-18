#!/bin/bash
# ============================================================
# 模型导出脚本：合并 LoRA + 转换为 GGUF
# 步骤0：检查/初始化 llama.cpp
# 步骤1：合并 adapter 到基座模型
# 步骤2：转换为 GGUF 格式 (Q4_K_M 量化)
# 输出目录：output/qwen-financial-gguf
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

ENV_PREFIX=/home/hyl/miniconda3/envs/qwen_fintune
export PATH="$ENV_PREFIX/bin:$PATH"
export PYTHONPATH=""
export LLAMAFACTORY_VERBOSITY=ERROR

echo "=========================================="
echo "  模型导出：合并 LoRA → GGUF"
echo "=========================================="

# 步骤0：检查/初始化 llama.cpp
LLAMA_CPP="$PROJECT_DIR/llama.cpp"
if [ ! -f "$LLAMA_CPP/convert_hf_to_gguf.py" ]; then
    echo ""
    echo "[步骤 0/3] 初始化 llama.cpp..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$LLAMA_CPP"
fi
if [ ! -f "$LLAMA_CPP/build/bin/llama-quantize" ]; then
    echo "  编译 llama-quantize..."
    cmake -B "$LLAMA_CPP/build" -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF
    cmake --build "$LLAMA_CPP/build" -j$(nproc) --target llama-quantize
fi

# 步骤1：合并 LoRA
echo ""
echo "[步骤 1/3] 合并 LoRA 权重..."
"$ENV_PREFIX/bin/llamafactory-cli" export config/export_ollama.yaml

# 步骤2：转换为 GGUF (先 f16，再量化到 Q4_K_M)
echo ""
echo "[步骤 2/3] 转换为 GGUF (f16)..."
MERGED_DIR="$PROJECT_DIR/output/qwen-financial-merged"
GGUF_F16="$PROJECT_DIR/output/qwen-financial-gguf/qwen2.5-7b-financial-f16.gguf"
GGUF_OUT="$PROJECT_DIR/output/qwen-financial-gguf/qwen2.5-7b-financial-q4_k_m.gguf"
mkdir -p "$(dirname "$GGUF_OUT")"

PYTHONPATH="$LLAMA_CPP:$PYTHONPATH" "$ENV_PREFIX/bin/python" "$LLAMA_CPP/convert_hf_to_gguf.py" \
    "$MERGED_DIR" \
    --outtype f16 \
    --outfile "$GGUF_F16"

echo ""
echo "[步骤 3/3] 量化到 Q4_K_M..."
"$LLAMA_CPP/build/bin/llama-quantize" "$GGUF_F16" "$GGUF_OUT" q4_k_m
rm -f "$GGUF_F16"

# 复制 Modelfile 并修正 FROM 路径
cp "$MERGED_DIR/Modelfile" "$(dirname "$GGUF_OUT")/Modelfile"
sed -i 's|^FROM .*|FROM ./qwen2.5-7b-financial-q4_k_m.gguf|' "$(dirname "$GGUF_OUT")/Modelfile"

echo ""
echo "=========================================="
echo "  导出完成！"
echo "  合并模型：$MERGED_DIR"
echo "  GGUF 模型：$GGUF_OUT"
echo "=========================================="
