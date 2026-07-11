#!/usr/bin/env python3
"""对 BELLE 提取结果进行第二次严格过滤，只保留真正高质量的金融问答"""

import json
import os
import re

# 载入初步提取的结果
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_path = os.path.join(base_dir, "data", "financial_qa.json")
output_path = input_path  # 覆盖写回

with open(input_path, "r", encoding="utf-8") as f:
    items = json.load(f)

print(f"原始数据: {len(items)} 条")

# 强排除规则——这些类型的数据不适合作为金融问答数据集
STRONG_EXCLUDE = [
    # 角色扮演对话
    "角色介绍", "请续写", "对话内容", "续写他们",
    "John:", "小明：", "小刚：", "Tom:", "Jack:", "Emily:",
    "Peter:", "Cathy:", "李老板", "小红", "老板", "投资人",
    # 通用文章/摘要生成
    "生成一篇", "写一篇短文", "写一篇关于", "写一段",
    "生成一个50字的摘要", "生成一篇短文章",
    # 写代码类（不是金融问答）
    "编写程序", "编写一个函数", "写下一个获得",
    # 通用润色
    "帮我重新润色", "润色下面这段",
    # 通用翻译
    "翻译", "英文句子", "中文",
    # 非金融主题
    "我最喜欢的电影", "肖申克", "智能手机",
    "健身", "减肥", "穿搭", "美食",
    # 太宽泛的 AI 文章
    "人工智能对于世界的影响", "人工智能技术",
    # 太泛的 resume 类
    "提取一份简历",
]

def is_excluded(text):
    if not text:
        return False
    for pat in STRONG_EXCLUDE:
        if pat in text:
            return True
    return False

# 强包含规则——必须明确谈论金融概念
STRONG_INCLUDE = [
    "银行", "存款", "贷款", "利率", "LPR", "房贷", "信用",
    "理财", "基金", "股票", "债券", "保险", "税务",
    "投资", "回报率", "收益率", "久期",
    "量化宽松", "QE", "缩表", "QT",
    "通胀", "通缩", "CPI", "GDP", "货币政策", "财政政策",
    "A股", "H股", "沪港通", "外汇", "汇率",
    "财务报表", "利润表", "资产负债表", "现金流量表",
    "净利润", "营收", "现金流", "估值",
    "区块链", "比特币", "数字货币",
    "工资", "税金", "个人所得税", "税后工资",
    "期权", "期货", "复利", "年金",
    "金融", "Basel", "巴塞尔", "KYC", "反洗钱",
    "风控", "风险控制", "合规", "不良贷款",
    "央行", "准备金率", "降准", "加息", "降息",
    "市盈率", "市净率", "股息",
    "投资组合", "保险", "保单",
]

filtered = []
removed = []

for item in items:
    inst = item.get("instruction", "")
    inp = item.get("input", "")
    combined = inst + " " + inp

    # 先检查排除规则
    if is_excluded(combined):
        removed.append((inst[:60], "排除: 角色/对话/通用文章"))
        continue

    # 检查包含规则
    has_finance = False
    for kw in STRONG_INCLUDE:
        if kw in combined:
            has_finance = True
            break

    if not has_finance:
        removed.append((inst[:60], "排除: 不涉及明确金融概念"))
        continue

    # 检查输出长度——太短的输出没有训练价值
    output = item.get("output", "")
    if len(output) < 30:
        removed.append((inst[:60], "排除: 输出太短"))
        continue

    filtered.append(item)

print(f"\n过滤后: {len(filtered)} 条")
print(f"移除: {len(removed)} 条")
print(f"\n=== 移除明细 ===")
for inst, reason in removed:
    print(f"  {reason} | {inst}")

# 写入过滤后的结果
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(filtered, f, ensure_ascii=False, indent=2)

print(f"\n=== 最终数据集 ({len(filtered)} 条) ===")
for idx, item in enumerate(filtered):
    print(f"\n[{idx+1:2d}] 指令: {item['instruction'][:80]}")
    if item['input']:
        print(f"     输入: {item['input'][:80]}")
    print(f"     输出: {item['output'][:100]}...")
