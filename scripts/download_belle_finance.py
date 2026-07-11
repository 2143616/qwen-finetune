#!/usr/bin/env python3
"""从 BELLE train_0.5M_CN 数据集中提取金融相关问答"""

import json
import re
import os

# 金融相关关键词（用于筛选）
FINANCE_KEYWORDS = [
    # 银行/金融
    "银行", "存款", "贷款", "利率", "LPR", "房贷", "信用", "信用卡",
    "理财", "基金", "股票", "债券", "保险", "税务", "审计",
    "KYC", "AML", "Basel", "巴塞尔", "资本充足", "流动性",
    # 投资
    "投资", "ROI", "IRR", "回报率", "收益率", "久期", "Duration",
    "贴现", "再贴现", "量化宽松", "QE", "缩表", "QT",
    # 市场/经济
    "通胀", "通缩", "CPI", "GDP", "货币政策", "财政政策",
    "A股", "H股", "沪港通", "深港通", "北向", "南向",
    "外汇", "汇率", "人民币", "美元", "黄金",
    # 公司金融
    "财务报表", "利润表", "资产负债表", "现金流量表",
    "资产", "负债", "权益", "净利润", "营收", "现金流",
    "IPO", "上市", "并购", "M&A", "估值",
    # 其他金融术语
    "金融", "经济", "复利", "年金", "期权", "期货",
    "个人征信", "征信", "财富管理", "资产负债率",
    "市盈率", "PE", "PB", "市净率", "股息",
]

def is_finance_related(text):
    """检查文本是否与金融相关"""
    if not text:
        return False
    text_lower = text.lower()
    for kw in FINANCE_KEYWORDS:
        if kw.lower() in text_lower:
            return True
    return False

def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "financial_qa.json")

    print("正在加载 BELLE train_0.5M_CN 数据集...")
    print("首次加载需要下载数据，请稍候...")

    try:
        from datasets import load_dataset
    except ImportError:
        print("未安装 datasets 库，正在安装...")
        os.system("pip install datasets -q")
        from datasets import load_dataset

    # 只加载前 10000 条，从中筛选金融相关
    # train_0.5M_CN 有 519k 条，我们扫描前 50000 条
    dataset = load_dataset("BelleGroup/train_0.5M_CN", split="train", streaming=True)

    results = []
    count = 0
    max_scan = 50000  # 扫描 5 万条
    target = 30       # 目标提取 30 条

    print(f"正在扫描前 {max_scan} 条数据，筛选金融相关问答...")

    for i, row in enumerate(dataset):
        if i >= max_scan:
            break

        instruction = row.get("instruction", "")
        inp = row.get("input", "")
        output = row.get("output", "")

        # 检查是否与金融相关
        if is_finance_related(instruction) or is_finance_related(inp):
            item = {
                "instruction": instruction,
                "input": inp if inp else "",
                "output": output
            }
            if item not in results:
                results.append(item)
                count += 1
                print(f"  [{count}] 找到: {instruction[:60]}...")

        if count >= target:
            break

    print(f"\n扫描完成！共扫描 {i+1} 条，找到 {len(results)} 条金融相关问答。")

    # 写入 JSON 文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"已保存到: {output_path}")
    print(f"数据集大小: {len(results)} 条")

    # 打印前几条的概览
    print("\n=== 提取的数据预览 ===")
    for idx, item in enumerate(results[:5]):
        print(f"\n--- 第 {idx+1} 条 ---")
        print(f"指令: {item['instruction'][:80]}...")
        print(f"输入: {item['input'][:60] if item['input'] else '(无)'}...")
        print(f"输出: {item['output'][:80]}...")

if __name__ == "__main__":
    main()
