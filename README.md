# 🦙 Qwen2.5-7B QLoRA 微调项目

[![LLaMA-Factory](https://img.shields.io/badge/框架-LLaMA--Factory-blueviolet)](https://github.com/hiyouga/LLaMA-Factory)
[![Model](https://img.shields.io/badge/模型-Qwen2.5--7B--Instruct-blue)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
[![Method](https://img.shields.io/badge/方法-QLoRA%204--bit-cyan)](https://github.com/huggingface/peft)
[![Hardware](https://img.shields.io/badge/硬件-NVIDIA%20DGX%20Spark-green)]()

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
│   └── export_ollama.yaml           # Ollama 导出配置
├── data/                            # 数据集
│   ├── financial_qa.json            # 金融问答数据集 (Alpaca 格式)
│   └── dataset_info.json            # 数据集注册文件
├── scripts/                         # 工具脚本
│   ├── start_webui.sh               # 启动 LLaMA Board Web UI
│   ├── start_training.sh            # 命令行一键训练
│   ├── monitor_gpu.sh               # GPU 资源监控
│   ├── download_belle_finance.py    # BELLE 数据集下载
│   ├── extract_belle_finance.py     # 金融数据提取 (v1)
│   ├── extract_belle_finance_v2.py  # 金融数据提取 (v2)
│   └── filter_belle_finance.py      # 金融数据过滤
├── LLaMA-Factory/                   # LLaMA-Factory 框架 (子目录集成)
├── output/                          # 训练输出 (自动生成)
├── models/                          # 模型权重 (本地缓存, 不上传)
├── activate.sh                      # 激活 conda 环境脚本
└── 模型训练.html                    # 完整教程文档
```

---

## 快速开始

### 环境要求

| 组件 | 要求 |
|------|------|
| GPU 显存 | ≥ 8 GB（QLoRA 4-bit）|
| CUDA | ≥ 11.6（推荐 12.2+）|
| Python | ≥ 3.11 |
| PyTorch | ≥ 2.0.0（推荐 2.6.0）|

### 环境搭建

```bash
# 1. 创建 conda 环境
conda create -n llamafactory python=3.11 -y
conda activate llamafactory

# 2. 安装 PyTorch (CUDA 12.4)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# 3. 安装 FlashAttention-2 (强烈推荐)
pip install flash-attn --no-build-isolation

# 4. 安装 LLaMA-Factory
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ..

# 5. 激活环境
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
conda activate llamafactory
bash scripts/start_training.sh
```

或直接指定配置文件：

```bash
llamafactory-cli train config/train_qwen_qlora.yaml
```

### 方式二：Web UI 训练

```bash
bash scripts/start_webui.sh
```

浏览器访问 `http://localhost:7860`，在 LLaMA Board 界面中配置参数并启动训练。

### 核心训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | Qwen/Qwen2.5-7B-Instruct | 阿里千问 2.5，7B 参数 |
| 微调方法 | LoRA | 低秩适配，仅训练 0.1%-1% 参数 |
| 量化位数 | 4-bit (QLoRA) | 显存降低约 75% |
| LoRA 秩 (rank) | 8 | 效果与效率的平衡点 |
| LoRA Alpha | 16 | 缩放因子（通常为 rank×2）|
| 学习率 | 2e-4 | LoRA 常用学习率 |
| Batch Size | 2 (×4 梯度累积) | 等效 batch = 8 |
| 训练轮数 | 5 | 63 条数据约 315 步 |
| 精度 | bf16 | DGX Spark 原生支持 |
| 最大长度 | 2048 | 大多数场景足够 |

### 训练监控

```bash
# 另一终端运行
bash scripts/monitor_gpu.sh
```

训练约 5-10 分钟完成，loss 从 ~2.5 下降到 ~0.1 以下。

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

### 导出为 GGUF + Ollama

```bash
# 1. 导出
llamafactory-cli export config/export_ollama.yaml

# 2. 创建 Ollama 模型
cd output/qwen-financial-gguf
ollama create qwen-financial -f Modelfile

# 3. 运行
ollama run qwen-financial
```

导出流程：

```
Qwen2.5-7B (基座模型) → QLoRA 微调 (金融 QA 数据集) → GGUF 导出 + Modelfile → Ollama 部署
```

---

## 配置说明

各配置文件的详细参数说明见 [config/train_qwen_qlora.yaml](config/train_qwen_qlora.yaml) 中的注释。

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

---

## 许可证

本项目基于 [Apache-2.0](LICENSE) 许可证开源。LLaMA-Factory 框架遵循其自身许可证。
