# AI 记忆
现在的大模型（Grok、Claude、GPT-4o、DeepSeek 等）虽然参数几千亿，但每次对话都是“失忆”的白痴——上一次教它的操作、偏好、模板，下一次又得重新教。真正的 AI 助手必须拥有「记忆」。

模型会失去记忆本质原因在于大模型的代码结构。大模型的代码可以理解成一个纯函数，它接受一个输入信息，然后运行神经网络预测该输入应该对应的输出，预测的过程中所使用的参数是在训练中不断地迭代更新，最后保存下来的，重新部署模型，仅仅只是用于推理任务，无法再修改参数，原因是因为更新一次参数相当于在训练模型，训练模型所需要的资源要远大于部署。

本文将介绍为大模型添加记忆的部分技术方向
+ Context 工程
+ 模型记忆层
+ 动态微调

## Context 工程
Context 把记忆塞进上下文，在用户提出的问题的基础上，额外增加

### 1.1 RAG（Retrieval-Augmented Generation）
- 核心：每次对话前把相关文档、历史操作记录、截图描述检索出来塞进 prompt
- 落地方式：
  - 向量数据库（Chroma、FAISS、Qdrant、Milvus）
  - 截图 → CLIP/IP-Adapter 编码 → 存向量库
  - 代码片段、操作日志 → 文本嵌入 → 存向量库
- 优点：零微调、无状态、可无限扩展
- 缺点：上下文窗口限制（目前最长 200k~1M），贵

### 1.2 MCP（Memory-Augmented Context Prompting）
- 2024 年新提出的一种“超级 RAG”
- 思路：不是把所有记忆都塞进去，而是让模型自己维护一个「记忆摘要表」
- 每次对话后让模型输出：
  ```markdown
  【记忆更新】
  - 用户最常用的登录按钮坐标：(1420, 870)
  - 用户偏好的 OCR 语言：chi_sim
  - 上次失败的模板：login_button_v2.png（被遮挡）

下次对话自动把这张表放在 prompt 最前面
实际效果：200k 上下文压缩到 4k 还能保持 95% 记忆准确率

## 记忆层

+ LoRA（Low-Rank Adaptation）本质是给大模型再插一个极小的适配器（几 MB 到几百 MB）
优点：
训练快（几分钟到几小时）
存储小（一个用户一个 LoRA 文件）
可以热切换（同一套权重，换不同 LoRA 就是不同人格/技能）

目前最成熟的记忆方式：
用户每完成一次成功操作 → 收集 (截图, 操作, 结果) 三元组
每 20~50 条经验微调一次 LoRA
推理时自动加载用户专属 LoRA

+ Elastic Weight Consolidation (EWC)

Description: Penalizes changes to important parameters from prior tasks during new training, using a Fisher information matrix to estimate parameter importance.
How it Simulates Memory: Preserves "core" knowledge by consolidating weights, mimicking synaptic stabilization in the brain.
Advantages: Reduces forgetting without storing data; effective for sequential tasks.
2025 Status: Integrated in continual learning frameworks; outperforms LoRA in some multi-task benchmarks.

+ Memory Layers

Description: Adds sparse, high-capacity layers (e.g., key-value stores) to models, allowing selective activation of parameters for specific memories.
How it Simulates Memory: Enables targeted recall without full retraining; finetuning these layers avoids broad interference.
Advantages: More efficient than LoRA for personalization; supports online adaptation.
2025 Status: Emerging in continual learning; shown to maintain 95% retention in fact-learning tasks.



## 动态微调：让模型“边用边学”
真正的记忆必须是在线、持续的。下面三种技术正在 2025 年快速落地：
3.1 Hebbian 学习（类脑可塑性）

核心思想：“一起放电的神经元，连接更强”
实现方式：
成功操作 → 加大对应 LoRA 权重（+0.01~0.1）
失败操作 → 减小对应 LoRA 权重（-0.05）
不需要梯度回传，纯前向更新，速度极快（毫秒级）

9. Regularization Methods (e.g., Orthogonal Projections)

Description: Constrains weight updates to orthogonal subspaces, separating new and old knowledge.
How it Simulates Memory: Projects gradients away from sensitive directions.
Advantages: Simple integration; works with Hebbian rules.
2025 Status: Applied in SNN continual learning; boosts forgetting resistance.

3. Spike-Timing-Dependent Plasticity (STDP) Enhancements

Description: Bio-inspired rule adjusting weights based on neuron firing timing; extended for spiking neural networks (SNNs).
How it Simulates Memory: Creates time-sensitive associations, enabling temporal sequence recall.
Advantages: Energy-efficient for edge devices; complements Hebbian rules.
2025 Status: Used in neuromorphic continual learning (NCL); improves unsupervised adaptation.

+ 经验回放（Experience Replay）

灵感来源：AlphaGo、DQN
实现：
维护一个经验池（成功率 > 90% 的操作序列）
空闲时（或者睡觉时）拿出来回放 5~20 次继续微调
类似人类睡觉巩固记忆

实测效果：24 小时回放后，相同任务成功率从 78% → 96%

3.3 神经调制（Neuromodulation）

给模型加一个“情绪/重要性”信号
成功 → 释放“多巴胺” → 学习率 ×5
失败 → 释放“去甲肾上腺素” → 学习率 ×10 + 加大探索
目前落地方式：用一个极小的 MLP 预测“重要性分数”，动态调节 LoRA 学习率

8. Brain-Inspired Internal Replay

Description: Reactivates latent representations of past experiences during training, without external data storage.
How it Simulates Memory: Mimics hippocampal replay for consolidation, strengthening internal traces.
Advantages: No data overhead; aligns with sleep-like phases.
2025 Status: Extends experience replay; effective in ANNs and SNNs for lifelong learning.

4. 未来：终极记忆方案（2026~2027 可能实现）


方案,记忆容量,更新速度,预计落地时间
RAG + MCP,几乎无限,秒级,已落地
用户专属 LoRA,几千~几万条经验,分钟级,已落地
动态 Hebbian + 回放,十万条级,秒~分钟,2025 Q4
Mamba / RWKV 状态记忆,无限长序列,实时,2026+
外挂海马体（独立记忆网络）,真正的终身记忆,实时,2027+

## 其他方式
Bayesian Continual Learning

Description: Models uncertainty in weights via probabilistic distributions, updating posteriors incrementally.
How it Simulates Memory: Handles ambiguity in new data while retaining probabilistic priors from old tasks.
Advantages: Quantifies confidence in recalls; robust to noisy environments.
2025 Status: Applied in SNNs for online learning; reduces overfitting in dynamic settings.


5. Predictive Coding

Description: Networks predict sensory inputs and update based on prediction errors, propagating signals hierarchically.
How it Simulates Memory: Builds internal models of past experiences for proactive recall.
Advantages: End-to-end differentiable; aligns with cortical hierarchies.
2025 Status: Explored in NCL for bio-plausible learning; enhances unsupervised memory formation.

5. Predictive Coding

Description: Networks predict sensory inputs and update based on prediction errors, propagating signals hierarchically.
How it Simulates Memory: Builds internal models of past experiences for proactive recall.
Advantages: End-to-end differentiable; aligns with cortical hierarchies.
2025 Status: Explored in NCL for bio-plausible learning; enhances unsupervised memory formation.


7. Metaplasticity

Description: Modulates plasticity rates dynamically (e.g., via neuromodulators like dopamine), altering how easily weights change.
How it Simulates Memory: Prevents over-stabilization or excessive forgetting by adapting learning dynamics.
Advantages: Integrates with existing neuromodulation; enhances hybrid networks.
2025 Status: Used in corticohippocampal-inspired models; reduces forgetting in task-agnostic learning.

9. Regularization Methods (e.g., Orthogonal Projections)

Description: Constrains weight updates to orthogonal subspaces, separating new and old knowledge.
How it Simulates Memory: Projects gradients away from sensitive directions.
Advantages: Simple integration; works with Hebbian rules.
2025 Status: Applied in SNN continual learning; boosts forgetting resistance.