#!/usr/bin/env python3
"""从 BELLE train_0.5M_CN 中提取高质量的金融问答（严格筛选）"""

import json
import os
import re

# 严格的金融关键词——要求 instruction 明显涉及金融概念
STRICT_FINANCE_KEYWORDS = [
    # 银行/金融核心概念
    "银行", "存款", "贷款", "利率", "LPR", "房贷", "信用", "信用卡",
    "理财", "基金", "股票", "债券", "保险", "税务", "审计",
    "KYC", "AML", "Basel", "巴塞尔", "资本充足率", "流动性覆盖率",
    # 投资分析
    "投资", "ROI", "IRR", "回报率", "收益率", "久期",
    "贴现", "再贴现", "量化宽松", "QE", "缩表", "QT",
    # 宏观经济
    "通胀", "通缩", "CPI", "GDP", "货币政策", "财政政策",
    "A股", "H股", "沪港通", "深港通", "北向资金",
    "外汇", "汇率", "人民币汇率",
    # 公司金融
    "财务报表", "利润表", "资产负债表", "现金流量表",
    "净利润", "营收", "现金流",
    "IPO", "上市", "并购", "估值",
    # 金融术语
    "金融", "复利", "年金", "期权", "期货", "区块链", "比特币",
    "个人征信", "财富管理", "资产负债率", "市盈率",
    "PE", "PB", "市净率", "股息",
    # 金融政策
    "反洗钱", "合规", "风险控制", "风控", "不良贷款",
    "拨备", "准备金", "央行", "准备金率", "降准", "加息",
    "降息", "逆回购", "MLF", "SLF", "公开市场操作",
    "工资", "税金", "个人所得税", "增值税",
]

def is_strictly_finance(text):
    """严格判断文本是否明显涉及金融概念"""
    if not text:
        return False
    for kw in STRICT_FINANCE_KEYWORDS:
        if kw in text:
            return True
    return False

# 需要排除的干扰项——
EXCLUDE_PATTERNS = [
    "角色介绍", "请续写", "对话内容", "写一个故事", "写一段对话",
    "编程", "代码", "算法", "Python", "Java", "C++",
    "健身", "减肥", "穿搭", "美食", "旅游攻略",
    "翻译", "英文", "email", "邮件",
]

def is_noise(text):
    if not text:
        return False
    for pat in EXCLUDE_PATTERNS:
        if pat in text:
            return True
    return False

# 定位缓存文件
cache_dir = "/media/hyl/DataDisk/cache/huggingface/hub/datasets--BelleGroup--train_0.5M_CN"
snapshots_dir = os.path.join(cache_dir, "snapshots")
snapshots = os.listdir(snapshots_dir)
data_file = os.path.join(snapshots_dir, snapshots[0], "Belle_open_source_0.5M.json")

output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_path = os.path.join(output_dir, "data", "financial_qa.json")

results = []
count = 0
total = 0

print("正在从 BELLE train_0.5M_CN 提取高质量金融问答...\n")

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

        # 严格筛选：instruction 必须明显涉及金融
        if not is_strictly_finance(instruction):
            continue

        # 排除噪音（角色扮演、非问答等）
        if is_noise(instruction) or is_noise(inp):
            continue

        item = {
            "instruction": instruction,
            "input": inp if inp else "",
            "output": output
        }
        if item not in results:
            results.append(item)
            count += 1
            if count <= 30:
                print(f"  [{count:2d}] {instruction[:80]}")

        if count >= 100:
            break

        if total % 100000 == 0:
            print(f"  已扫描 {total} 条...")

print(f"\n扫描完成！共扫描 {total} 条，提取到 {len(results)} 条高质量金融问答。")

# 写入
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"已保存: {output_path}")
print(f"文件大小: {os.path.getsize(output_path) / 1024:.1f} KB")
