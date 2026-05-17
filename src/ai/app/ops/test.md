---
title: 测试策略
order: 2
---
# 测试策略
LLM 应用的测试比传统软件测试更困难——输出是概率性的，"正确"可能有多种形式，而且 Prompt 的微小修改可能导致输出大幅变化。但测试仍然是质量保障的基石，关键在于选择合适的测试粒度和自动化策略。

## Prompt 回归测试
Prompt 是 LLM 应用的"源代码"，每次修改都可能影响输出质量。Prompt 回归测试确保修改后的 Prompt 在已有测试用例上不退化。

**测试集构建**：从生产日志中抽取 50-200 个代表性问题，人工标注期望的输出特征（不是精确文本，而是关键信息点、格式要求、约束条件）。测试集应该覆盖简单和复杂场景、不同长度和类型的输入、已知的边界 case。

**评分机制**：由于 LLM 输出不是确定性的，精确匹配不适用。更好的方式是检查输出是否包含期望的关键信息点（信息点覆盖率）、是否符合格式要求（格式合规率）、是否违反约束（约束违反率）。DeepEval 的 assertion-style 测试适合这种场景。

```python
# Prompt 回归测试示例
test_cases = [
    {
        "input": "北京今天天气怎么样",
        "must_contain": ["温度", "天气状况"],
        "must_not_contain": ["上海", "深圳"],
        "format": "自然语言段落"
    },
    {
        "input": "用 JSON 格式返回北京和上海的温度",
        "must_contain": ["beijing", "shanghai", "temperature"],
        "format": "valid_json"
    }
]

def run_regression_test(new_prompt):
    failures = []
    for case in test_cases:
        output = llm.invoke(new_prompt, case["input"])
        # 检查必须包含的信息
        for keyword in case["must_contain"]:
            if keyword.lower() not in output.lower():
                failures.append(f"缺少关键词: {keyword}")
        # 检查格式
        if case["format"] == "valid_json":
            try:
                json.loads(output)
            except json.JSONDecodeError:
                failures.append("输出不是有效的 JSON")
    return len(failures) == 0, failures
```

**CI/CD 集成**：将回归测试集成到 CI 流水线中。每次修改 Prompt 相关文件时自动触发测试，测试失败则阻止合并。测试结果应该包含具体的失败用例和对比信息，方便开发者快速定位问题。

## RAG Pipeline 测试
RAG 系统的端到端测试验证从数据摄入到检索到生成的完整链路。每个环节都可能成为瓶颈，需要分层测试。

**数据摄入测试**：验证文档解析的正确性——原始 PDF/Word 文件是否被正确提取为文本，表格是否保留了结构，编码是否统一。这可以通过对比解析输出和人工标注的期望输出来验证。

**检索质量测试**：给定一组测试问题和期望命中的文档 ID，验证检索结果是否包含正确的文档。召回率（正确文档是否被检索到）和精确率（检索到的文档中正确文档的比例）是核心指标。

```python
def test_retrieval_quality():
    queries = [
        {"query": "年假怎么休", "expected_docs": ["holiday_policy_v2.pdf"]},
        {"query": "报销流程", "expected_docs": ["expense_guide.pdf", "finance_rules.pdf"]},
    ]

    for case in queries:
        results = retriever.invoke(case["query"])
        retrieved_ids = [r.metadata["source"] for r in results]
        hit = any(doc in retrieved_ids for doc in case["expected_docs"])
        assert hit, f"查询'{case['query']}'未检索到期望文档: {case['expected_docs']}"
```

**生成质量测试**：给定问题和检索到的文档，验证生成的回答是否忠实于文档内容、是否完整回答了问题。这里需要用 LLM-as-Judge 或人工评估。

## 对抗测试
对抗测试（Adversarial Testing）模拟攻击者和刁钻用户，测试系统在极端输入下的表现。

**边界 case**：空输入、超长输入、特殊字符、多语言混合、代码片段、Base64 编码内容。这些边界输入不应导致系统崩溃或异常行为。

**Prompt 注入测试**：使用已知的注入攻击模式批量测试系统，验证输入过滤和指令隔离的有效性。测试用例应该定期更新，包含最新的攻击变体。

**多轮对话退化测试**：长时间的多轮对话可能导致上下文溢出、逻辑不一致、重复内容等问题。设计 20+ 轮的对话测试脚本，验证系统在长对话中的稳定性。

## 自动化评测流水线
将评估、测试和部署串联为自动化的 CI/CD 流水线：

```
代码/Prompt 变更
  → 单元测试（格式校验、工具调用测试）
  → Prompt 回归测试（50-200 用例）
  → RAG Pipeline 测试（检索质量 + 生成质量）
  → 对抗测试（注入攻击 + 边界 case）
  → 通过 → 灰度发布（5% 流量 → 20% → 100%）
  → 失败 → 阻止合并 + 发送通知
```

流水线的关键参数是"通过阈值"。不是所有测试用例都需要 100% 通过——LLM 的非确定性意味着总有一些边缘情况。设定合理的阈值（如信息点覆盖率 ≥ 80%、格式合规率 ≥ 95%、注入防御率 ≥ 98%），在质量和迭代速度之间找到平衡。

灰度发布阶段配合 [可观测性](monitor) 工具，实时监控新版本的质量指标。如果灰度期间评估指标显著低于旧版本（如准确率下降 5% 以上），自动回滚到旧版本。
