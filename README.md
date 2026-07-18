# Qwen2.5-7B QLoRA 微调项目

在 **NVIDIA DGX Spark** 上使用 **LLaMA-Factory** 框架，通过 **QLoRA (4-bit)** 技术微调 **Qwen2.5-7B-Instruct** 模型，使其具备金融领域问答能力。

---

## 📋 目录

- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [数据集](#数据集)
- [训练](#训练)
- [推理与评估](#推理与评估)
- [导出与部署](#导出与部署)
- [配置说明](#配置说明)
- [常见问题](#常见问题)

---

## 项目结构

```
qwen-finetune/
├── config/                          # 配置文件
│   ├── train_qwen_qlora.yaml        # QLoRA 训练配置
│   ├── inference.yaml               # 推理配置
│   └── export_ollama.yaml           # LoRA 合并导出配置
├── data/                            # 数据集
│   ├── financial_qa.json            # 金融问答数据集 (Alpaca 格式)
│   └── dataset_info.json            # 数据集注册文件
├── scripts/                         # 工具脚本
│   ├── start_webui.sh               # 启动 LLaMA Board Web UI
│   ├── start_training.sh            # 命令行一键训练
│   ├── export_ollama.sh             # 一键导出（合并 LoRA → GGUF 量化）
│   ├── monitor_gpu.sh               # GPU 资源监控
│   ├── download_belle_finance.py    # BELLE 数据集下载
│   ├── extract_belle_finance.py     # 金融数据提取 (v1)
│   ├── extract_belle_finance_v2.py  # 金融数据提取 (v2)
│   └── filter_belle_finance.py      # 金融数据过滤
├── LLaMA-Factory/                   # LLaMA-Factory 框架 (子目录集成)
├── llama.cpp/                       # GGUF 转换工具 (export_ollama.sh 自动克隆)
├── output/                          # 训练输出 (自动生成)
│   ├── qwen-qlora-financial/        # LoRA adapter + checkpoint
│   ├── qwen-financial-merged/       # 合并后的完整 HF 模型 + Modelfile
│   └── qwen-financial-gguf/         # GGUF 量化模型 + Modelfile
├── models/                          # 模型权重 (本地缓存, 不上传)
├── activate.sh                      # 激活 conda 环境脚本
├── README.html                      # 详细教学文档（推荐阅读）
└── README.md                        # 本文件
```

---

## 快速开始

### 环境要求

| 组件 | 要求 |
|------|------|
| GPU 显存 | ≥ 8 GB（QLoRA 4-bit）|
| CUDA | ≥ 11.6（推荐 12.2+）|
| Python | ≥ 3.11 |
| PyTorch | ≥ 2.0.0（推荐 2.13.0）|

### 环境搭建

```bash
# 1. 创建 conda 环境
conda create -n qwen_fintune python=3.11 -y

# 2. 安装 PyTorch (CUDA 12.4)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# 3. 安装 FlashAttention-2 (强烈推荐)
pip install flash-attn --no-build-isolation

# 4. 安装 LLaMA-Factory
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ..

# 5. 激活环境（conda activate 有 bug 时用 source activate.sh 代替）
source activate.sh
```

> 💡 **DGX Spark 用户**：Grace Blackwell 架构原生支持 bf16 和 FlashAttention-2。128GB 统一内存可安全提高 batch size 到 4-8。

---

## 数据集

### 数据来源

来自 [BELLE](https://github.com/LianjiaTech/BELLE) 项目 `BelleGroup/train_0.5M_CN` 数据集，通过金融关键词（银行、贷款、利率、投资、保险、KYC、Basel 等）筛选出 **63 条**高质量金融问答数据。

### 数据格式 (Alpaca)

每条数据包含三个字段：

```json
{
  "instruction": "解释以下金融术语的含义。\n基金、股票、投资回报率",
  "input": "",
  "output": "基金是由多个投资者合资形成的一种集合型投资工具..."
}
```

### 提取更多数据

`scripts/` 下提供了数据提取脚本，可调整参数从全部 51.9 万条数据中提取更多金融样本：

```bash
# 调整 extract_belle_finance_v2.py 中的 max_scan 和 max_count 参数
python scripts/extract_belle_finance_v2.py
```

---

## 训练

### 方式一：命令行训练（推荐）

```bash
source activate.sh
bash scripts/start_training.sh
```

### 方式二：Web UI 训练

```bash
bash scripts/start_webui.sh
```

浏览器访问 `http://localhost:7860`，在 LLaMA Board 界面中配置参数并启动训练。

### 核心训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | Qwen2.5-7B-Instruct (本地) | 阿里千问 2.5，7B 参数，bf16 原始精度 |
| 微调方法 | LoRA | 低秩适配，仅训练 0.1%-1% 参数 |
| 量化位数 | 4-bit (QLoRA) | 运行时 bitsandbytes 量化，显存降低约 75% |
| LoRA 秩 (rank) | 8 | 效果与效率的平衡点 |
| LoRA Alpha | 16 | 缩放因子（通常为 rank×2）|
| 学习率 | 2e-4 | LoRA 常用学习率 |
| Batch Size | 2 (×4 梯度累积) | 等效 batch = 8 |
| 训练轮数 | 5 | 63 条数据约 35 步 |
| 精度 | bf16 | DGX Spark 原生支持 |
| 最大长度 | 2048 | 大多数场景足够 |

### 训练效果

训练约 2 分钟完成（111 秒），train loss 从 ~1.9 下降到 ~0.81（平均 1.15），eval loss 约 2.04。

> ⚠️ 验证 loss（2.04）比训练 loss（1.15）高不少，说明 63 条数据存在过拟合。建议扩大数据集到 1000+ 条。

### 训练监控

```bash
# 另一终端运行
bash scripts/monitor_gpu.sh
```

---

## 推理与评估

### 交互式对话

```bash
llamafactory-cli chat config/inference.yaml
```

### API 服务部署

```bash
API_PORT=8000 llamafactory-cli api config/inference.yaml
```

调用示例：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-7B-Financial",
    "messages": [
      {"role": "user", "content": "什么是量化宽松政策？"}
    ]
  }'
```

### 推荐测试问题

- 解释以下金融术语的含义：基金、股票、投资回报率
- 什么是区块链技术以及它的主要应用场景？
- 什么是金融风险管理？它对银行业的重要性是什么？
- 什么是 Basel III？对银行有什么影响？

---

## 导出与部署

### 一键导出（推荐）

```bash
bash scripts/export_ollama.sh
```

这个脚本自动完成四步：初始化 llama.cpp（首次自动克隆编译）→ 合并 LoRA → 转 GGUF f16 → Q4_K_M 量化，最后在 `output/qwen-financial-gguf/` 生成可直接部署的 GGUF 模型和 Modelfile。

### 导出流程

```
Qwen2.5-7B (基座 bf16) → QLoRA 训练 → 合并 adapter 到基座 → GGUF 转换 (Q4_K_M) → Ollama 部署
```

中间产物 `output/qwen-financial-merged/` 是合并后的完整 HuggingFace 模型（含 Modelfile，`FROM .` 指向 safetensors），也能直接用 `ollama create` 导入，15GB。`output/qwen-financial-gguf/` 是 Q4_K_M 量化版（含 Modelfile，`FROM ./xxx.gguf`），4.4GB。

### 分步操作（可选）

```bash
# 1. 合并 LoRA 到基座（产出一个完整 HuggingFace 模型）
llamafactory-cli export config/export_ollama.yaml

# 2. 转 GGUF（llama.cpp 在项目根目录，脚本会自动克隆编译）
python llama.cpp/convert_hf_to_gguf.py output/qwen-financial-merged --outtype f16 --outfile model-f16.gguf
llama.cpp/build/bin/llama-quantize model-f16.gguf model-q4_k_m.gguf q4_k_m
```

### Ollama 部署

```bash
# 方式一：GGUF 量化版（4.4GB，推荐）
cd output/qwen-financial-gguf
ollama create qwen-financial -f Modelfile

# 方式二：完整 safetensors 版（15GB）
cd output/qwen-financial-merged
ollama create qwen-financial -f Modelfile

# 运行
ollama run qwen-financial
```

两个 Modelfile 的 TEMPLATE、SYSTEM、PARAMETER 完全一样，唯一区别是 `FROM` 指向的模型文件不同——一个指 GGUF，一个指 safetensors。

---

## 配置说明

### 训练配置

`config/train_qwen_qlora.yaml` — 包含所有训练超参数，关键项：

- `do_train: true` — **必须显式开启**，否则不会训练
- `quantization_bit: 4` — 4-bit QLoRA，运行时加载 bf16 原始模型后动态量化
- `lora_target: all` — 对所有线性层（q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj）应用 LoRA
- **不要设 `device_map`** — bitsandbytes 量化时会自动管理，手动设置会冲突 OOM

### 推理配置

`config/inference.yaml` — 使用 LoRA adapter 进行对话，基座模型和 adapter 分离加载，每次推理时动态合并。

### 导出配置

`config/export_ollama.yaml` — 将 LoRA adapter 永久合并到基座模型（`merge_and_unload`），产出完整 HuggingFace 格式模型。

### 显存配置速查

| 显存 | 推荐配置 |
|------|----------|
| 4 GB | QLoRA + batch=1 + cutoff=512 + rank=4 |
| 8 GB | **本实验配置**: QLoRA + batch=2 + cutoff=2048 + rank=8 |
| 16 GB | QLoRA + batch=4 + cutoff=4096 + rank=16 |
| 24 GB+ | LoRA (8-bit) + batch=8 + cutoff=4096 |
| 128 GB (DGX Spark) | QLoRA + batch=8 + cutoff=4096 + rank=16 |

---

## 常见问题

<details>
<summary><b>CUDA Out of Memory (OOM)</b></summary>

减小 `per_device_train_batch_size` 到 1，或减小 `cutoff_len` 到 1024，确认已设置 `quantization_bit: 4`。
</details>

<details>
<summary><b>Dataset 加载失败</b></summary>

确认 `data/dataset_info.json` 文件存在且路径正确，`dataset_dir` 指向包含 `dataset_info.json` 的目录。
</details>

<details>
<summary><b>模型下载太慢</b></summary>

设置镜像站加速：`export HF_ENDPOINT=https://hf-mirror.com`，或先手动下载到本地路径。
</details>

<details>
<summary><b>conda activate 报错</b></summary>

conda 26.5.3 的 activate 有 bug，用 `source activate.sh` 代替。该脚本直接设置 PATH 和 PYTHONPATH，绕过 conda activate。
</details>

---

## 许可证

本项目基于 [Apache-2.0](LICENSE) 许可证开源。LLaMA-Factory 框架遵循其自身许可证。
