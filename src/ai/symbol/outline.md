# 学习路线
1. 基础知识（必备前提）

数理逻辑基础
命题逻辑（Propositional Logic）：真值表、蕴涵、等价。
一阶谓词逻辑（First-Order Logic）：量词、全称/存在、谓词、函数。
推理规则：演绎、归纳、溯因。

自动推理简介
可满足性（SAT）。
定理证明（Theorem Proving）。


2. 知识表示（Knowledge Representation）

经典表示方法
生产规则（Production Rules）：IF-THEN规则。
语义网络（Semantic Networks）：节点+边表示关系。
框架（Frames）：带槽位的对象表示（Minsky提出）。
脚本（Scripts）和情景（Schemas）。

本体（Ontology）和描述逻辑
OWL（Web Ontology Language）基础。
描述逻辑（Description Logics）。


3. 推理引擎与搜索

前向链（Forward Chaining）和后向链（Backward Chaining）。
冲突解决策略（Conflict Resolution）。
搜索算法在推理中的应用
深度优先、广度优先、A*。
处理不确定性：Certainty Factors（MYCIN）、贝叶斯方法。


4. 经典系统与案例

专家系统（Expert Systems）
MYCIN（医疗诊断）。
DENDRAL（分子结构分析）。
XCON（计算机配置）。

通用问题求解器（General Problem Solver, GPS）。
SHRDLU（自然语言理解块世界）。

5. 编程实现（动手实践，必学）

Prolog语言（核心）
事实、规则、查询。
递归、列表处理、回溯。
切断（cut）、否定即失败。
推荐：SWI-Prolog安装，写简单专家系统。

Lisp基础（可选但推荐）
符号表达式（S-expression）。
早期AI程序多用Lisp实现。

其他工具
CLIPS或Jess（生产规则系统）。
SOAR或ACT-R（认知架构，含符号成分）。


6. 局限性与批判

知识获取瓶颈（Knowledge Acquisition Bottleneck）。
资格问题（Qualification Problem）。
框架问题（Frame Problem）。
常识知识问题（Commonsense Knowledge）。

7. 现代延伸（强烈推荐，连接当下热点）

神经符号AI（Neuro-Symbolic AI）
神经网络 + 符号推理结合方式。
Logic Tensor Networks (LTN)。
DeepProbLog、Scallop等框架。
AlphaGeometry（几何证明案例）。

可解释AI（XAI）中的符号方法。
大语言模型的符号增强（LLM + Knowledge Graph）。



符号神经系统，依赖关系全部转化为查找和模式匹配，显式逻辑命题转变为概念声明，并且永不说死一个命题，基于图和节点，来构建搜索算法