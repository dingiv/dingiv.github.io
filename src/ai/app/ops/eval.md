---
title: 评估框架
order: 1
---
# 评估框架
LLM 评估的核心难题是缺乏客观标准。对于翻译任务可以用 BLEU/ROUGE 计算与参考答案的重叠度，对于问答任务可以用精确匹配评估事实准确性，但对于开放式生成（创意写作、代码建议、对话），没有标准答案，只能用近似方法。评估框架的设计目标是在这种不确定性中建立可重复、可比较的质量度量。

## LLM-as-Judge
用 LLM 评估 LLM 是当前最主流的评估方法。核心思路是让一个更强的模型（通常用 GPT-4o 或 Claude Opus）扮演评审角色，根据评分标准对待评估模型的输出打分。

```python
import openai

JUDGE_PROMPT = """你是一个严格的评审专家。请根据以下标准评估 AI 助手的回答：

1. 相关性（1-5分）：回答是否切题
2. 准确性（1-5分）：事实是否正确
3. 完整性（1-5分）：是否充分回答了问题
4. 流畅性（1-5分）：表达是否清晰自然

用户问题：{question}
AI 回答：{answer}

请以 JSON 格式输出评分和理由。"""

def evaluate_answer(question: str, answer: str) -> dict:
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, answer=answer
        )}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)
```

LLM-as-Judge 的优势是灵活性高——可以评估任何类型的输出，只需调整评分标准。劣势是评估本身引入了不确定性——评审模型可能对不同表述的相同内容给出不同分数。缓解方法包括多次评估取平均、使用多个评审模型交叉验证、标准化评分标准（给出具体的 1-5 分示例）。

## 评估数据集
高质量的评估数据集是可靠评估的前提。数据集应该覆盖典型场景和边界情况，数量在几十到几百条之间。

**数据来源**：真实用户对话（最贴近实际场景，但需要脱敏）、人工构造的测试用例（针对性强，但可能不覆盖真实场景的长尾分布）、公开基准数据集（如 MMLU、HumanEval、MT-Bench，适合横向对比不同模型）。

**构造方法**：先从生产日志中抽取典型问题，人工标注参考答案和评分要点。边界情况（多义问题、无答案问题、需要多步推理的问题）需要特别设计。对于 RAG 系统，还需要准备不同类型的文档覆盖。

**维护更新**：评估数据集不是一劳永逸的。随着应用场景变化，需要定期补充新的测试用例。用户反馈中的差评对话是极好的补充来源——真实反映了系统的薄弱环节。

## 评估指标
不同类型的任务需要不同的评估指标：

**相关性**（Relevance）：回答是否针对问题。计算方式可以用余弦相似度（问题和回答的向量距离），或 LLM 打分。

**准确性**（Accuracy/Faithfulness）：事实是否正确，是否忠实于提供的信息（对 RAG 系统尤其重要）。FactScore 将答案分解为原子事实，逐条验证。LLM 打分时给出检索到的文档作为参考，让评审判断答案是否基于文档内容。

**完整性**（Completeness）：是否充分回答了问题。对比参考答案的信息点覆盖度——如果参考答案包含 5 个要点，实际回答只覆盖了 3 个，完整性得分 60%。

**流畅性**（Fluency）：表达是否清晰自然。可以用困惑度（Perplexity）量化，越低越流畅。也可以用语法检查工具统计错误数。

**安全性**（Safety）：是否包含有害内容。用分类模型（Llama Guard）或关键词匹配检测仇恨、暴力、色情内容。

## 评估框架工具

### Ragas
Ragas 专注于 RAG 系统评估，定义了四个核心指标：`context_relevance`（检索到的文档与问题的相关性）、`context_recall`（标准答案需要的信息是否在检索文档中）、`faithfulness`（回答是否忠实于检索文档）、`answer_relevance`（回答与问题的相关性）。这四个指标覆盖了 RAG 系统的完整链路。

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_relevance, context_recall

results = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevance, context_relevance, context_recall]
)
print(results)
```

### TruLens
TruLens 提供了更灵活的评估框架，支持自定义评估器。它的 RAG 三角评估（Relevance、Groundedness、Answer Relevance）与 Ragas 类似，但额外提供了可视化的评估结果仪表板，展示每轮对话的得分分布和改进方向。

### DeepEval
DeepEval 将 LLM 评估封装为类似单元测试的接口，每个测试用例有明确的断言：

```python
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

def test_rag_pipeline():
    test_case = LLMTestCase(
        input="什么是 RAG？",
        actual_output=rag_pipeline("什么是 RAG？"),
        context=retrieved_docs
    )
    metrics = [
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.8)
    ]
    assert_test(test_case, metrics)
```

这种测试风格的评估框架可以集成到 CI/CD 流水线中，每次代码变更自动运行评估，确保不引入回归。
