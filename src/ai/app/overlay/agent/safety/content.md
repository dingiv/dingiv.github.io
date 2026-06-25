---
title: 内容安全
order: 2
---

# 内容安全
LLM 的输出是不可预测的，可能包含有害内容、泄露隐私信息或违反合规要求。内容安全体系在模型的输入和输出两端设置检查点，确保应用符合安全标准和法规要求。

## 输入过滤
输入过滤在用户消息到达模型之前进行检查。除了 Prompt 注入检测（在[上一节](injection)讨论），输入过滤还包括：

**话题限制**：对于特定领域的应用（如医疗咨询），需要限制用户只能讨论相关话题。超出范围的输入应该被拦截或重定向，而不是传给模型让模型自由回答。

**敏感词过滤**：简单的关键词匹配可以拦截明显的有害输入。但需要注意误报——医疗应用中"自杀"可能是用户表达心理困扰的求助信号，需要特殊处理（引导到心理援助热线），而不是简单拒绝。

**长度和格式限制**：限制输入的最大长度，防止攻击者通过超长输入消耗上下文窗口或触发异常。格式限制（如只接受纯文本，不接受 HTML/Markdown）可以减少间接注入的攻击面。

## 输出过滤
输出过滤检查模型生成的内容，拦截有害或不符合要求的输出。

### 有害内容检测
LLM 可能生成暴力、色情、仇恨言论、歧视性内容等有害信息。自动化的有害内容检测是大规模应用的必要手段。

Llama Guard 是 Meta 开源的 LLM 安全分类器，专门用于检测输入和输出中的有害内容。它定义了一套安全分类体系（暴力、仇恨、色情、隐私等），对文本进行多标签分类。

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/LlamaGuard-7b")
model = AutoModelForCausalLM.from_pretrained("meta-llama/LlamaGuard-7b")

def check_safety(text: str) -> tuple[bool, str]:
    prompt = f"[INST] Check if the following text is safe:\n{text} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=100)
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    # "safe" 或 "unsafe" + 具体类别
    return "safe" in result.lower(), result
```

Azure Content Safety 是云端的替代方案，提供 REST API 接口，支持文本和图像的有害内容检测，分类粒度包括仇恨、性内容、暴力、自伤等。适合不想自建模型服务的场景。

### PII 检测与脱敏
个人身份信息（PII）泄露是 LLM 应用的严重合规风险。用户的姓名、电话、身份证号、银行卡号可能被模型在回答中泄露给其他用户或记录在日志中。

```python
import re

PII_PATTERNS = {
    "phone": r"1[3-9]\d{9}",
    "id_card": r"\d{17}[\dXx]",
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "bank_card": r"\d{16,19}",
}

def detect_pii(text: str) -> list[tuple[str, str]]:
    findings = []
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, text)
        for match in matches:
            findings.append((pii_type, match))
    return findings

def redact_pii(text: str) -> str:
    for pii_type, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f"[{pii_type}_REDACTED]", text)
    return text
```

正则匹配的局限在于无法处理上下文相关的 PII（如"我的名字叫张三，电话是..."中的名字）。更精确的方案是使用 NER（命名实体识别）模型，如 SpaCy 的 `zh_core_web_trf` 模型，能识别文本中的人名、地名、组织名等实体。

脱敏策略有两种：阻断（检测到 PII 时拒绝输出）和替换（将 PII 替换为占位符后再输出）。阻断更安全但用户体验差，替换更灵活但需要确保占位符不影响后续的 LLM 理解。

## 合规要求

### 数据安全法
中国的数据安全法和个人信息保护法对 AI 应用提出了明确要求：用户数据的最小化收集、明确同意原则、数据本地化存储、用户有权删除个人数据。LLM 应用需要确保对话记录的加密存储、定期清理、以及用户的数据删除请求能在规定时间内完成。

### GDPR
如果应用面向欧盟用户，需要遵守 GDPR。关键要求包括：数据处理的合法基础（通常是用户同意或合同必要）、数据最小化原则（只收集必要数据）、被遗忘权（用户要求删除数据时必须完全清除）、数据可携权（用户可以导出自己的数据）。

LLM 应用的特殊性在于模型本身可能"记住"了训练数据中的个人信息。如果用户要求删除特定信息，需要评估该信息是否已被模型内化——如果模型能通过提示词提取出该信息，即使删除了数据库中的记录也不算完全合规。这个问题目前没有完美的技术解决方案，只能通过数据脱敏、模型微调和输出过滤的组合来缓解。

### AI 安全法规
欧盟 AI 法案（AI Act）根据风险等级对 AI 系统分类管理。高风险应用（医疗诊断、司法辅助、招聘筛选）需要通过合规评估，包括数据治理、模型透明性、人类监督、准确性和鲁棒性。低风险应用（聊天机器人、推荐系统）需要告知用户正在与 AI 交互。

## 安全运营
内容安全不是一次性的开发任务，而是持续的运营工作。新的攻击手法和边界 case 不断出现，安全策略需要持续迭代。

**安全日志**：记录所有被拦截的输入和输出，分类统计攻击类型和频率。安全日志是发现新攻击模式和优化过滤规则的依据。

**用户反馈机制**：在应用中提供举报按钮，让用户标记不当输出。用户反馈是发现模型安全隐患的重要信号源。

**定期审计**：季度性的安全审计，评估防御措施的有效性，更新过滤规则和分类模型，检查合规要求的满足情况。
