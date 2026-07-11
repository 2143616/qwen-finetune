#!/usr/bin/env python3
"""从 BELLE train_0.5M_CN 数据集中提取金融相关问答（流式处理大文件）"""

import json
import os
import re

# 金融相关关键词
FINANCE_KEYWORDS = [
    "银行", "存款", "贷款", "利率", "LPR", "房贷", "信用", "信用卡",
    "理财", "基金", "股票", "债券", "保险", "税务", "审计",
    "KYC", "AML", "Basel", "巴塞尔", "资本充足", "流动性",
    "投资", "ROI", "IRR", "回报率", "收益率", "久期", "Duration",
    "贴现", "再贴现", "量化宽松", "QE", "缩表", "QT",
    "通胀", "通缩", "CPI", "GDP", "货币政策", "财政政策",
    "A股", "H股", "沪港通", "深港通", "北向", "南向",
    "外汇", "汇率", "人民币", "美元", "黄金",
    "财务报表", "利润表", "资产负债表", "现金流量表",
    "资产", "负债", "权益", "净利润", "营收", "现金流",
    "IPO", "上市", "并购", "M&A", "估值",
    "金融", "复利", "年金", "期权", "期货",
    "个人征信", "征信", "财富管理", "资产负债率",
    "市盈率", "PE", "PB", "市净率", "股息",
    "区块链", "比特币", "数字货币", "支付", "FinTech",
    "反洗钱", "合规", "风险控制", "风控", "不良贷款",
    "拨备", "准备金", "央行", "准备金率", "降准", "加息",
    "降息", "逆回购", "MLF", "SLF", "公开市场",
]

def is_finance_related(text):
    if not text:
        return False
    for kw in FINANCE_KEYWORDS:
        if kw in text:
            return True
    return False

# 查找缓存中的 BELLE 数据文件
cache_dir = "/media/hyl/DataDisk/cache/huggingface/hub/datasets--BelleGroup--train_0.5M_CN"
snapshots_dir = os.path.join(cache_dir, "snapshots")
if os.path.exists(snapshots_dir):
    snapshots = os.listdir(snapshots_dir)
    if snapshots:
        data_file = os.path.join(snapshots_dir, snapshots[0], "Belle_open_source_0.5M.json")
    else:
        data_file = None
else:
    data_file = None

if not data_file or not os.path.exists(data_file):
    print("错误：找不到 BELLE 数据集文件")
    print(f"查找路径: {cache_dir}")
    exit(1)

print(f"数据文件: {data_file}")
print(f"文件大小: {os.path.getsize(data_file) / 1024 / 1024:.1f} MB")

output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_path = os.path.join(output_dir, "data", "financial_qa.json")

results = []
count = 0
total = 0

print("\n正在扫描并提取金融相关问答...")

with open(data_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue

        instruction = row.get("instruction", "")
        inp = row.get("input", "")
        output = row.get("output", "")

        # 只提取 instruction 或 input 中包含金融关键词的（过滤 output 避免噪音）
        if is_finance_related(instruction) or is_finance_related(inp):
            item = {
                "instruction": instruction,
                "input": inp if inp else "",
                "output": output
            }
            if item not in results:
                results.append(item)
                count += 1
                if count % 10 == 0:
                    print(f"  已找到 {count} 条...")

        # 限制最大提取量，避免太多
        if count >= 100:
            break

        # 进度提示
        if total % 50000 == 0:
            print(f"  已扫描 {total} 条...")

print(f"\n扫描完成！共扫描 {total} 条，找到 {count} 条金融相关问答。")

# 写入 JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"已保存到: {output_path}")

# 打印前 10 条预览
print(f"\n{'='*60}")
print(f"提取的金融数据集预览（前 10 条）:")
print(f"{'='*60}")
for idx, item in enumerate(results[:10]):
    print(f"\n--- 第 {idx+1} 条 ---")
    print(f"指令: {item['instruction'][:100]}")
    if item['input']:
        print(f"输入: {item['input'][:100]}")
    print(f"输出: {item['output'][:120]}...")
