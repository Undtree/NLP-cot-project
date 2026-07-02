// ============================================================
//  基于思维链推理的策略拓展实验报告
// ============================================================
#set text(font: ("Libertinus Serif", "Noto Serif SC"))
#show raw: set text(font: ("Cascadia Code", "Noto Serif SC"), weight: 400)

= 基于思维链推理的策略拓展实验报告

== 一、摘要

本报告整合了四项围绕思维链（Chain-of-Thought, CoT）推理的拓展实验：自一致性推理（Self-Consistency）、基于 Verifier 的加权投票推理（Verifier CoT）、多智能体辩论推理（Multi-Agent Debate），以及基于检索增强的交错式思维链推理（Interleaving Retrieval with CoT, IRCoT）。前三项实验基于 AQuA-RAT 数学选择题数据集，分别使用 Qwen2.5-Coder-32B-Instruct-AWQ 与 Qwen2.5-Coder-7B-Instruct 两款模型；IRCoT 实验基于 HotpotQA 多跳问答数据集，使用 Qwen2.5-0.5B-Instruct 模型配合 BM25 检索器。实验结果表明，Self-Consistency 通过多次采样与多数投票将 32B 模型的准确率从约 81% 提升至最高 87.01%；Verifier CoT 通过对不同推理路径的评分加权，在 7B 模型上达到了 76.77% 的准确率；Multi-Agent Debate 通过角色分工与反思机制，在三智能体设定下取得了 78.74% 的最佳结果；IRCoT 在 20 条 HotpotQA 样本上取得了 0.350 的 EM 和 0.432 的 F1，明显优于无检索 CoT 和一次性检索增强 CoT，验证了"推理一步、检索一步"交替策略在多跳问答任务上的有效性。

== 二、引言与背景

大语言模型在数学推理任务上的表现近年来取得了长足进步。结合思维链提示（Chain-of-Thought prompting），模型能够在输出最终答案之前先生成逐步推理过程，从而显著提升复杂推理任务上的准确率。然而，单条推理路径存在固有缺陷：一旦模型在早期步骤中出现理解偏差、计算错误或逻辑谬误，后续推理往往会沿着错误方向延续，最终导致答案错误。

针对这一问题，研究者们提出了多种改进方案。本报告涉及的三个方向分别从不同角度入手：

- *自一致性推理（Self-Consistency）*：对同一问题多次采样，利用采样随机性产生多条推理路径，再通过多数投票聚合最终答案。
- *Verifier 加权投票*：通过设计不同角色的 Solver 生成多样化推理路径，再引入 Verifier 对每条路径的推理质量进行评分，最终以评分为权重进行加权投票。
- *多智能体辩论（Multi-Agent Debate）*：让多个扮演不同角色的 Agent 独立推理、互相审视、反思修正，最后由 Judge 汇总裁决。
- *检索增强思维链（IRCoT）*：将检索步骤嵌入 CoT 推理过程中，模型每生成一步中间推理就触发一次检索，用新召回的证据补充下一步推理，适用于需要组合多个外部知识片段的多跳问答任务。

下面先介绍统一的实验环境与数据集，再分章节详述每种方法的设计、实验与结果。

== 三、实验环境与数据集

前三项实验（Self-Consistency、Verifier CoT、Multi-Agent Debate）共享以下实验环境与数据集。IRCoT 实验使用不同的数据集和模型，详见第八章。

=== 数据集

前三项实验使用 AQuA-RAT 数据集的测试集（`aqua_test.json`），共 254 道英语数学选择题。每道题包含自然语言题目描述、五个候选选项（A 到 E）、人工撰写的推理解释（`rationale`）以及标准答案（`correct`）。实验中仅将题目文本与候选选项输入模型，不使用人工解释。

=== 模型与推理服务

实验涉及两款模型，均通过 vLLM 提供 OpenAI 兼容接口，在单卡环境下部署：

- *Qwen2.5-Coder-32B-Instruct-AWQ*：32B 参数，AWQ 量化，部署于 V100 32GB。用于 Self-Consistency 实验。
- *Qwen2.5-Coder-7B-Instruct*：7B 参数，部署于 V100 16GB 或 T4 16GB。用于 Verifier CoT 与 Multi-Agent Debate 实验。

IRCoT 实验使用 Qwen2.5-0.5B-Instruct，通过 Hugging Face Transformers 后端运行，详见第八章。

=== 统一评测框架

项目实现了一套模块化 Harness 框架（`harness/`），包含数据集加载器、LLM 客户端、答案抽取器、评测器与实验日志系统。前三种策略均以独立 Task 类的形式接入框架，运行命令示例：

```bash
python main.py --strategy self_consistency --paths 5 --temperature 0.7
python main.py --strategy verifier_cot --max-concurrent 8
python main.py --strategy three_agent_debate --temperature 0.3
```

框架自动完成答案抽取（通过 `quick_extract` 匹配最终答案行 `The answer is: X`）、正确性判定与结果持久化（CSV、JSONL、Summary JSON）。

== 四、自一致性推理（Self-Consistency CoT）

=== 方法设计

普通 CoT 每次只生成一条推理链，一旦链中某个步骤出错，最终答案往往随之错误。Self-Consistency 的核心思想是把"生成一条"改为"生成多条再投票"。

具体而言，对同一个问题 $x$，模型在非零温度下采样 $K$ 次，得到 $K$ 条推理路径：

$
y_1, y_2, ..., y_K ~ p(y | x)
$

从每条路径中抽取最终选项：

$
a_i = "extract"(y_i), quad a_i in {A, B, C, D, E}
$

最终答案取票数最多的选项：

$
a^* = limits("argmax")_a sum_(i=1)^K bold(1)[a_i = a]
$

策略的核心不在于手工设计 $K$ 个不同 prompt，而在于利用采样随机性让模型对同一题目产生多条可能的推理。理想情况下，错误路径之间的错误答案较为分散，而正确答案在多条路径中更容易获得相对多数。

=== Prompt 与采样

每条路径使用同一套模板，要求模型逐步推理并在末尾输出固定格式 `The answer is: X`。每条路径附加编号提示 `This is reasoning path #N. Use your own independent reasoning.`，但多样性主要来源于 temperature 和多次采样。

核心参数见 @tbl:sc-params。

#figure(
  table(
    columns: 3,
    [*参数*], [*含义*], [*baseline 默认值*],
    [`paths`], [采样路径数], [$5$],
    [`temperature`], [采样温度], [$0.7$],
    [`max_tokens`], [单次最大 token 数], [$2048$],
  ),
  caption: [Self-Consistency 核心参数],
) <tbl:sc-params>

=== 并发优化

若每道题的 $K$ 条路径串行生成，运行时间将接近普通 CoT 的 $K$ 倍。实现中在单样本内部使用 `ThreadPoolExecutor` 并发采样多条路径，并发数受 `LLM_MAX_CONCURRENT` 控制。并发不改变算法逻辑，仅将同题的多条请求并行发出以缩短总耗时。

=== 实验结果

==== 主实验

baseline 使用 `paths=5, temperature=0.7` 在 AQuA test 全量 254 题上运行，模型为 Qwen2.5-Coder-32B-Instruct-AWQ。结果如下：

#figure(
  table(
    columns: 2,
    [*指标*], [*数值*],
    [样本数], [$254$],
    [正确数], [$217$],
    [错误数], [$37$],
    [准确率], [$85.43%$],
    [答案提取失败], [$0$],
    [平均耗时], [$11.61$ 秒/样本],
  ),
  caption: [Self-Consistency baseline 结果 (`paths=5`, `temperature=0.7`)],
) <tbl:sc-baseline>

答案提取失败数为 0，说明固定输出格式与抽取器配合稳定。

==== 答案分布

预测分布整体较均衡，未出现明显的选项塌缩：

#figure(
  table(
    columns: 3,
    [*选项*], [*标准答案数*], [*模型预测数*],
    [A], [$63$], [$54$],
    [B], [$58$], [$57$],
    [C], [$46$], [$51$],
    [D], [$53$], [$49$],
    [E], [$34$], [$43$],
  ),
  caption: [Self-Consistency baseline 答案分布],
) <tbl:sc-dist>

模型对 E 的预测略多于标准分布，可能说明在一部分题目上模型更容易被靠后的干扰选项吸引。

==== 消融实验

为分析路径数量和采样温度的影响，进行了四组消融实验。除 `paths` 和 `temperature` 外其余设置保持一致，所有实验均无提取失败。

#figure(
  table(
    columns: 5,
    [*实验设置*], [`paths`], [`temperature`], [*正确/总数*], [*准确率*],
    [单路径近似 CoT], [$1$], [$0.7$], [$206/254$], [$81.10%$],
    [轻量 Self-Consistency], [$3$], [$0.7$], [$220/254$], [$86.61%$],
    [baseline], [$5$], [$0.7$], [$217/254$], [$85.43%$],
    [低温采样], [$5$], [$0.3$], [$221/254$], [$87.01%$],
    [高温采样], [$5$], [$1.0$], [$217/254$], [$85.43%$],
  ),
  caption: [Self-Consistency 消融实验结果],
) <tbl:sc-ablation>

=== 分析

从路径数量看，`paths=1`（等价于单次 CoT）准确率仅 81.10%，`paths=3` 提升至 86.61%（+5.51 pp），说明多路径投票能显著提升推理稳定性。但 `paths=5, temperature=0.7`（85.43%）反而略低于 `paths=3`，说明路径数量并非越多越好：当新增路径的质量不稳定，或多条路径都受到同一类错误思路影响时，投票结果仍可能出错。

从温度看，在 `paths=5` 设定下，`temperature=0.3` 达到最高准确率 87.01%。较低温度使单条推理更稳定，同时 5 条路径仍能提供一定多样性。`temperature=1.0` 基本无增益，推测高温引入了更多无效或偏离题意的推理路径。

Self-Consistency 的主要优势在于稳健性。单路径 CoT 一次生成定胜负，而多次采样 + 投票允许模型的独立尝试之间互相纠错。但这一策略只能应对推理的随机波动，不能修正模型本身的能力缺陷——如果模型对某类题形成了系统性的错误认知，多数投票同样救不回来。此外，`paths=5` 意味着每道题需要 5 次模型调用，计算成本是普通 CoT 的约 5 倍。

== 五、基于 Verifier 的加权投票推理（Verifier CoT）

=== 研究动机

本实验参考 _Making Large Language Models Better Reasoners with Step-Aware Verifier_ 一文的思想，尝试在无法训练专用 Verifier 模型的约束下，用同一 LLM 充当评分员。核心思路是：先生成多样化的推理路径，再用 Verifier 对每条路径的推理质量进行评分，最后以评分为权重进行加权投票。

与原论文的主要区别在于：论文中的 Verifier 是通过自动标注数据训练得到的专用模型，而本实验受限于条件，直接使用同一 LLM 配合评分提示词充当 Verifier。

=== 方法设计

==== 整体流程

整个算法分为三个阶段：

+Phase 1: 多样化路径生成+。三个 Solver（Baseline、Skeptic、DoubleChecker）各自独立推理，产生三条推理路径。

+Phase 2: Verifier 评分+。LLM 以 Verifier 角色对每条路径的推理质量进行 0--1 分的评分，并输出 JSON 格式的评分结果。

+Phase 3: 加权投票+。以 Verifier 评分作为权重进行加权求和，选出总权重最高的选项。

优化设计：当三条路径的答案全部一致时，直接返回该答案，跳过评分阶段。

==== 多样化推理路径

三个 Solver 共用同一份用户模板（与 baseline 相同的逐步推理格式），仅在系统提示词上有所区分：

#figure(
  table(
    columns: 3,
    [*Solver*], [*角色定位*], [*核心推理策略*],
    [Baseline], [通用数学助手], [标准的逐步推理，与 baseline 保持一致],
    [Skeptic], [批判性推理者], [逐一分析每个选项，寻找其可能的错误原因，幸存者即为正确答案],
    [DoubleChecker], [先解后选], [先不看选项独立求解，再将结果与每个选项逐一比对，选出匹配项],
  ),
  caption: [三个 Solver 的角色与策略],
) <tbl:vc-solvers>

==== Verifier 评分

Verifier 的提示词规定了 0.0--1.0 的五级评分标准：0.9--1.0 为逻辑完美、步骤全对；0.7--0.8 为大体正确、仅有小问题；0.4--0.6 为部分正确但存在明显缺口；0.1--0.3 为显著错误或逻辑谬误；0.0 为完全无意义或空输出。Verifier 被要求仅输出一行 JSON，如 `{"score": 0.85, "extracted_answer": "B", "critique": "brief one-sentence assessment"}`。

Verifier 使用 `temperature=0.0` 以保证评分相对稳定。

==== 加权投票

加权投票以 Verifier 评分为权重：

$
op("Weight")(X) = sum_(i=1)^N "score"_i dot bold(1)["answer"_i = X]
$

其中 $N=3$，$"score"_i in [0, 1]$，$X$ 为选项。选择总权重最高的选项作为最终答案。

=== 实验设置

本实验使用 Qwen2.5-Coder-7B-Instruct 模型。Solver 和 Verifier 的温度均设为 0.0，Solver 的 `max_tokens` 为 2048，Verifier 的 `max_tokens` 为 512。并发数设为 8。

=== 实验结果

==== 准确率

#figure(
  table(
    columns: 3,
    [*策略*], [*正确数*], [*准确率*],
    [BaselineCoT], [$181/254$], [$71.26%$],
    [VerifierCoT], [$195/254$], [$76.77%$],
  ),
  caption: [VerifierCoT 与 Baseline 准确率对比],
) <tbl:vc-accuracy>

VerifierCoT 相比 baseline 提升了 5.51 个百分点。

==== 三个 Solver 独立表现

#figure(
  table(
    columns: 3,
    [*Solver*], [*正确数*], [*准确率*],
    [Baseline], [$181/254$], [$71.3%$],
    [Skeptic], [$188/254$], [$74.0%$],
    [DoubleChecker], [$190/254$], [$74.8%$],
  ),
  caption: [三个 Solver 独立准确率],
) <tbl:vc-solver-individual>

VerifierCoT 的最终准确率（76.77%）高于任一 Solver 的独立表现，说明 Verifier 加权投票能在一定程度上纠正单条路径的错误。

==== 典型样本分析

以样本 `aqua_0125` 为例。题目要求找出一个三位数，已知各位数字之和为 17、各位数字平方和为 109，且该数减去 495 后得到逆序数。正确选项为 A。

三个 Solver 中，Baseline 和 DoubleChecker 均选了 C（错误），仅 Skeptic 选了 A（正确）。如果采用简单多数投票，C 以 2:1 胜出，最终会答错。但 Verifier 对三条路径评分后，A 所获得的权重超过了 C 的两条路径的权重和，加权投票最终选出了正确答案。

这个样本说明：即便正确答案在路径数上不占多数，只要 Verifier 能合理评判推理质量，高评分路径仍有机会翻盘。

==== 资源消耗

#figure(
  table(
    columns: 4,
    [*指标*], [*Baseline*], [*VerifierCoT*], [*倍数*],
    [总 tokens], [$166,095$], [$801,075$], [$4.8×$],
    [总时间], [$60$ min], [$208$ min], [$3.5×$],
    [每题平均 tokens], [$654$], [$3,154$], [$4.8×$],
    [每题平均时间], [$14.2$ s], [$49.0$ s], [$3.5×$],
    [每题 LLM 调用], [$1$ 次], [$6$ 次 (3 Solver + 3 Verifier)], [$6×$],
  ),
  caption: [VerifierCoT 资源消耗对比],
) <tbl:vc-cost>

VerifierCoT 的资源开销在 3--6 倍之间。由于 Solver 与 Verifier 调用可在各路径间并行，实际耗时约为 baseline 的 3.5 倍，在可接受范围内。

=== 分析

VerifierCoT 的核心优势在于区分了推理路径的质量。简单多数投票将所有路径一视同仁，而实际场景中不同路径的推理质量可能差异很大。Verifier 评分提供了细粒度的质量信号，让更好的推理获得更大话语权。此外，每条路径都有评分，可以输出答案的权重和作为置信度参考。

局限同样明显。第一，Verifier 自身的能力决定了评分上限——如果 Verifier 无法可靠区分推理质量，加权投票的增益就很有限。第二，路径多样性是前提条件，如果三条路径高度相似，即使有 Verifier 也无法发挥作用。第三，每条路径需要额外一次 Verifier 调用，扩展到更多路径时开销会进一步增大。

== 六、多智能体辩论推理（Multi-Agent Debate CoT）

=== 方法设计

本实验探索了基于多智能体协作的 CoT 推理。核心思路是让多个扮演不同角色的 Agent 共同参与推理，通过角色分工、交叉验证、反思修正和最终裁决来弥补单路径 CoT 的不足。

==== 三种策略

实现了三种递增复杂度的策略：

#figure(
  table(
    columns: 5,
    [*策略名*], [*Agent 数*], [*反思轮数*], [*说明*],
    [`debate`], [$2$], [$0$], [Analyst 与 Verifier 独立推理，Judge 最终裁决],
    [`reflective_debate`], [$2$], [$1$], [两 Agent 先独立推理，再读对方观点并修正，Judge 裁决],
    [`three_agent_debate`], [$3$], [$1$], [增加 Skeptic Agent，引入更强的错误检查与多样性],
  ),
  caption: [三种 Multi-Agent Debate 策略],
) <tbl:ma-strategies>

==== Agent 角色

为避免多个 Agent 生成高度相似的推理路径，设计了四个不同职责的角色：

#figure(
  table(
    columns: 3,
    [*Agent*], [*角色定位*], [*Prompt 关注点*],
    [Analyst], [主解题者], [从数学公式和直接计算出发，给出逐步推理],
    [Verifier], [验证者], [独立求解后检查单位、计算和候选项匹配],
    [Skeptic], [质疑者], [寻找题目陷阱、隐藏条件和易错选项],
    [Judge], [裁决者], [比较不同 Agent 的推理质量，输出最终答案],
  ),
  caption: [Multi-Agent 角色分工],
) <tbl:ma-roles>

==== 基础 Debate（`debate`）

两个 Agent（Analyst 和 Verifier）独立生成推理链，Judge 阅读两条推理链后比较推理质量并输出最终答案。核心机制是并行生成 + 汇总裁决，不涉及 Agent 之间的信息交互。

==== Reflective Debate（`reflective_debate`）

在基础 Debate 之上加入一轮反思。每个 Agent 首先生成初始推理，然后阅读其他 Agent 的推理内容，判断自己或对方是否存在错误，并选择保持或修正答案。Judge 最终能看到完整的讨论过程（初始推理 + 反思修正）。

==== 三智能体反思 Debate（`three_agent_debate`）

在 Reflective Debate 基础上加入 Skeptic Agent。Skeptic 专门负责寻找潜在陷阱与错误选项，增加推理路径多样性。这一策略的计算成本最高，但也提供了最充分的交叉检查。

=== 实验设置

实验使用 Qwen2.5-Coder-7B-Instruct 模型。主要参数：`max_tokens=4096`，主要对比温度为 0.3（同时保留 `temperature=0.0` 下的 Debate 结果以分析采样多样性的影响）。

对比方法包括：BaselineCoT（单 Agent Zero-shot CoT）、MultiAgentDebate、ReflectiveDebate、ThreeAgentReflectiveDebate。

=== 实验结果

==== 主实验

#figure(
  table(
    columns: 6,
    [*方法*], [*Temperature*], [*Agent 数*], [*反思轮数*], [*正确/总数*], [*Accuracy*],
    [BaselineCoT], [$0.0$], [$1$], [$0$], [$192/254$], [$75.59%$],
    [MultiAgentDebate], [$0.0$], [$2$], [$0$], [$189/254$], [$74.41%$],
    [BaselineCoT], [$0.3$], [$1$], [$0$], [$191/254$], [$75.20%$],
    [MultiAgentDebate], [$0.3$], [$2$], [$0$], [$193/254$], [$75.98%$],
    [ReflectiveDebate], [$0.3$], [$2$], [$1$], [$195/254$], [$76.77%$],
    [ThreeAgentReflectiveDebate], [$0.3$], [$3$], [$1$], [$200/254$], [$78.74%$],
  ),
  caption: [Multi-Agent Debate 主实验结果],
) <tbl:ma-results>

在 `temperature=0.3` 设置下，多智能体方法整体优于同温度的 BaselineCoT。`ThreeAgentReflectiveDebate` 表现最佳，达到 78.74%，相比同温度 BaselineCoT 提升 3.54 个百分点。

==== 计算成本

#figure(
  table(
    columns: 4,
    [*方法*], [*总耗时*], [*平均每题耗时*], [*平均每题 tokens*],
    [BaselineCoT, temp=0.0], [$0.90$ h], [$12.80$ s], [$681.4$],
    [MultiAgentDebate, temp=0.0], [$1.83$ h], [$25.95$ s], [$2428.9$],
    [BaselineCoT, temp=0.3], [$0.91$ h], [$12.85$ s], [$677.8$],
    [MultiAgentDebate, temp=0.3], [$1.80$ h], [$25.50$ s], [$2381.8$],
    [ReflectiveDebate, temp=0.3], [$2.37$ h], [$33.53$ s], [$5091.0$],
    [ThreeAgentReflectiveDebate, temp=0.3], [$4.35$ h], [$61.63$ s], [$9105.9$],
  ),
  caption: [Multi-Agent Debate 计算成本],
) <tbl:ma-cost>

以 `ThreeAgentReflectiveDebate` 为例，准确率最高，但平均每题耗时约为 BaselineCoT 的 4.8 倍，平均 token 消耗约为 13.4 倍。

==== 样本级对比

以 `temperature=0.3` 的 BaselineCoT 为参照，统计样本级此消彼长：

#figure(
  table(
    columns: 4,
    [*方法*], [*Baseline 错而本方法对*], [*Baseline 对而本方法错*], [*净增正确数*],
    [MultiAgentDebate], [$29$], [$27$], [$+2$],
    [ReflectiveDebate], [$27$], [$23$], [$+4$],
    [ThreeAgentReflectiveDebate], [$27$], [$18$], [$+9$],
  ),
  caption: [Multi-Agent Debate 与 Baseline 样本级对比 (`temp=0.3`)],
) <tbl:ma-sample>

基础 Debate 能够修正部分 baseline 错误，但也引入了新的错误（29 vs 27）。加入反思和第三个 Agent 后，新引入的错误数下降（27 vs 18），净收益更加明显。

=== 分析

Multi-Agent Debate 的价值不限于"多个答案投票"。它的核心机制包括：角色分工带来更丰富的推理视角（Analyst 重计算、Verifier 重验证、Skeptic 重陷阱识别）；反思机制允许 Agent 阅读彼此的推理后修正自身错误；Judge 汇总全部信息做最终裁决。

三个发现值得注意。第一，低温确定性解码下（`temperature=0.0`），MultiAgentDebate 的准确率（74.41%）反而低于 BaselineCoT（75.59%）。原因是多个 Agent 使用同一模型，在确定性解码下推理路径高度相似，此时 Judge 裁决不仅没有带来增益，反而可能引入额外的判断错误。第二，反思机制确实有效——`ReflectiveDebate` 相比基础 `MultiAgentDebate` 提升了约 0.79 个百分点，且新引入错误数下降。第三，增加第三个 Agent（Skeptic）后收益最明显，说明在数学推理任务中，"专门找茬"的角色对提升整体正确率有实质帮助。

局限性方面：Judge 的质量直接影响最终结果，如果 Judge 错误采纳了较差的推理，正确答案可能被否决。此外，计算成本随着 Agent 数量和反思轮数的增加而快速增长，实际使用时需要在准确率与成本之间权衡。最后，所有 Agent 使用同一个 7B 模型模拟，虽通过 prompt 赋予不同角色，但 Agent 之间仍存在较强相关性。

== 七、三种方法的综合对比与讨论

=== 准确率总览

由于三种方法使用的模型规格不同，直接横向比较准确率数值并不公平。下表按模型分开汇总：

#figure(
  table(
    columns: 4,
    [*方法*], [*模型*], [*准确率*], [*相对 Baseline 提升*],
    // 32B 模型
    [BaselineCoT (paths=1)], [32B], [$81.10%$], [---],
    [Self-Consistency (paths=3)], [32B], [$86.61%$], [+$5.51$ pp],
    [Self-Consistency (paths=5, temp=0.3)], [32B], [$87.01%$], [+$5.91$ pp],
    [Self-Consistency (paths=5, temp=0.7)], [32B], [$85.43%$], [+$4.33$ pp],
    // 7B 模型
    [BaselineCoT (temp=0.0)], [7B], [$75.59%$], [---],
    [VerifierCoT], [7B], [$76.77%$], [+$1.18$ pp vs 7B Baseline\@0.0],
    [MultiAgentDebate (temp=0.3)], [7B], [$75.98%$], [+$0.78$ pp vs 7B Baseline\@0.3],
    [ReflectiveDebate (temp=0.3)], [7B], [$76.77%$], [+$1.57$ pp vs 7B Baseline\@0.3],
    [ThreeAgentReflectiveDebate (temp=0.3)], [7B], [$78.74%$], [+$3.54$ pp vs 7B Baseline\@0.3],
  ),
  caption: [三种方法准确率汇总（按模型分组）],
) <tbl:compare-accuracy>

=== 方法维度对比

三种方法虽然都是在"多条推理路径"上做文章，但出发点和机制有明显差异：

- *Self-Consistency*：最朴素的方法，不改变 prompt 结构，纯粹依靠采样随机性 + 多数投票。优势是实现简单、不需要设计多个角色；劣势是计算成本与路径数线性增长，且多数投票无法利用路径间的质量差异。

- *Verifier CoT*：在路径生成阶段引入了角色多样性（三种 Solver），在聚合阶段引入了质量评估（Verifier 评分 + 加权投票）。优势是能区分路径质量，让优质推理获得更大权重；劣势是 Verifier 本身未经训练，评分可靠性存疑，且额外增加了评分调用。

- *Multi-Agent Debate*：在路径生成、信息交互和最终裁决三个层面都做了设计。Agent 之间可以互相阅读推理内容并反思修正，而非像 Self-Consistency 那样各路径完全独立。优势是信息利用率最高，反思机制有实质收益；劣势是 prompt 设计复杂、计算成本最高，且对采样多样性有较高依赖。

=== 共同的经验教训

三条实验路径在各自探索中也揭示了一些共性问题：

*路径数量与质量的权衡*。多路径并不总是越多越好。Self-Consistency 中 `paths=5` 低于 `paths=3`，Multi-Agent Debate 中也需要反思机制才能让多 Agent 的增益超过噪声。额外的路径如果质量参差不齐，反而可能拉低投票或裁决的结果。

*采样多样性是关键前提*。Multi-Agent Debate 在 `temperature=0.0` 时反而不如 BaselineCoT，Self-Consistency 在 `temperature=1.0` 的高温下也没有进一步收益。多样性不足时多路径方法失去了存在的意义，多样性过高又可能导致无效推理。找到合适的温度区间对于这些方法的实际效果至关重要。

*聚合机制的可靠性与生成质量同样重要*。无论是多数投票、Verifier 评分还是 Judge 裁决，聚合阶段的决策质量直接决定了整个方法的"天花板"。如果聚合机制本身不可靠（如 Verifier 评分不准、Judge 判断失误），即使某条 Solver 路径已经给出了正确答案，最终也可能被错误否决。

*计算成本不可忽视*。三种方法的计算开销分别是 Baseline 的约 3--13 倍。在资源受限的实际场景中，需要在准确率收益与推理成本之间找到合适的平衡点。

== 八、基于检索增强的交错式思维链推理（IRCoT）

=== 研究动机

前述三种方法都在不同程度上试图提升推理路径的多样性与可靠性，但它们共享一个隐含假设：模型仅依靠自身参数化知识完成推理。在需要组合多条外部证据才能回答的多跳问答（Multi-hop QA）场景中，这一假设往往不成立——例如"Scott Derrickson 和 Ed Wood 是同一国籍吗"这样的问题，模型需要先分别查证两人的国籍信息，再进行比较。仅仅依靠参数记忆或单次检索往往不够。

本实验参考论文 _Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions_，在 HotpotQA 多跳问答数据集上实现了检索增强思维链推理。实验的核心问题是：模型能否通过"推理一步、检索一步"的交替过程逐步补充证据，从而获得比纯 CoT 或一次性 RAG-CoT 更好的答案。

=== 数据集与实验环境

本实验在数据集和模型两个维度上与前三个实验有所不同。数据集方面，使用 HotpotQA（distractor / validation 子集），这是一个多跳开放域问答数据集，每题需要组合多个 Wikipedia 段落的信息才能回答。实验从 validation 集中选取 20 条样本作为主实验和消融实验的评测集合。每条样本自带了 context 段落列表作为检索语料，避免了索引完整 Wikipedia 的高额成本。

模型方面，使用 Qwen2.5-0.5B-Instruct，通过 Hugging Face Transformers 后端在 float32 精度下运行。检索器采用轻量级 BM25，`top_k` 设为 5。相比于前三个实验使用的 7B/32B 模型和 vLLM 推理服务，本实验的模型规模更小、工程实现更轻量，适合课程项目资源约束下的快速迭代。

=== 方法设计

==== 三种对照方法

本实验实现了三类推理方法，分别承担不同的对照作用：

#figure(
  table(
    columns: 3,
    [*方法*], [*检索方式*], [*作用*],
    [No Retrieval CoT], [无检索], [纯 CoT baseline，观察模型不依赖外部证据时的表现],
    [One-step RAG-CoT], [问题一次性检索 top-5 段落], [观察外部证据本身是否提升效果],
    [IRCoT], [推理与检索交替进行], [核心方法，观察交替策略是否优于一次性检索],
  ),
  caption: [IRCoT 实验中三种对照方法],
) <tbl:ircot-methods>

==== No Retrieval CoT

无检索 CoT 不引入任何外部段落，仅将问题输入模型并要求逐步推理。提示词形式为 `Question: ... Think step by step. End with exactly one final line: So the answer is: <answer>`。该方法用于衡量模型内部知识和基础推理能力的上限。

==== One-step RAG-CoT

一次检索 RAG-CoT 先用原始问题作为查询，从 BM25 语料库检索 top-5 段落，再将这些段落与问题一同输入模型。核心流程为：`passages = retriever.search(question, top_k=5); answer = LLM(question, passages)`。

==== IRCoT

IRCoT 的核心思想是让检索和推理交替进行。初始时使用问题检索一批证据段落；随后模型基于已有证据生成下一步推理线索；系统再将该推理句作为新的查询进行检索，并合并新证据。循环结束后，模型基于全部证据和中间推理生成最终答案。

伪代码如下：

```python
passages = retriever.search(question, top_k=5)
chain = []

for step in range(max_steps):
    next_clue = LLM(question, passages, chain)   # 只生成检索线索，不输出最终答案
    chain.append(next_clue)
    new_passages = retriever.search(next_clue, top_k=5)
    passages = merge_unique(passages, new_passages)

final_answer = LLM(question, passages, chain)     # 综合所有证据输出最终答案
```

实现中有两个关键设计。第一，中间步骤只生成下一轮检索所需的线索、桥接事实或查询语句，不输出最终答案也不做答案早停，保证 `max_steps` 对应真实的推理-检索轮数。第二，如果 `max_steps = k`，则 IRCoT 需要 `k+1` 次 LLM 调用（`k` 次中间线索生成 + 1 次最终答案生成）。

=== 评价指标

本实验使用五类指标，兼顾答案质量和证据覆盖：

#figure(
  table(
    columns: 2,
    [*指标*], [*含义*],
    [EM], [预测答案与标准答案完全匹配的比例],
    [F1], [预测答案与标准答案 token 级别的重合程度],
    [Title Recall], [检索段落标题覆盖 supporting facts 标题的比例],
    [Avg LLM Calls], [每个样本平均 LLM 调用次数],
    [Avg Retrieved], [每个样本平均保留的检索段落数],
  ),
  caption: [IRCoT 实验评价指标],
) <tbl:ircot-metrics>

=== 实验结果

==== 主对比实验

三类方法在 20 条 HotpotQA validation 样本上的结果如下：

#figure(
  table(
    columns: 6,
    [*方法*], [*EM*], [*F1*], [*Title Recall*], [*Avg LLM Calls*], [*Avg Retrieved*],
    [No Retrieval CoT], [$0.000$], [$0.101$], [$0.000$], [$1.0$], [$0.00$],
    [One-step RAG-CoT], [$0.050$], [$0.095$], [$0.775$], [$1.0$], [$5.00$],
    [IRCoT (`steps=3`)], [$0.350$], [$0.432$], [$0.875$], [$4.0$], [$7.75$],
  ),
  caption: [IRCoT 主对比实验结果（20 条样本）],
) <tbl:ircot-main>

IRCoT 在所有指标上表现最好。与 No Retrieval CoT 相比，EM 从 0.000 提升到 0.350，F1 从 0.101 提升到 0.432；与 One-step RAG-CoT 相比，EM 从 0.050 提升到 0.350，F1 从 0.095 提升到 0.432。

从证据召回角度看，No Retrieval CoT 没有检索过程，Title Recall 为 0。One-step RAG-CoT 的 Title Recall 达到 0.775，说明用问题直接检索已经能找回大量 supporting titles。IRCoT 进一步提升到 0.875，说明中间检索线索确实帮助补充了更多关键证据。

值得注意的是，One-step RAG-CoT 的 F1（0.095）略低于 No Retrieval CoT（0.101）。这不表示检索无效——其 EM 和 Title Recall 均高于 No Retrieval CoT。更合理的解释是 0.5B 小模型在单次检索上下文中存在答案抽取和输出格式问题，常常生成长篇推理开头而非简洁答案，导致 F1 受影响。

成本方面，No Retrieval CoT 和 One-step RAG-CoT 每题只调用 1 次 LLM，而 `max_steps=3` 的 IRCoT 平均调用 4 次。IRCoT 的效果优势是以更高的推理时间为代价的。

==== IRCoT 步数消融实验

为分析推理-检索轮数的影响，固定模型和 top_k，仅改变 `max_steps`，结果如下：

#figure(
  table(
    columns: 6,
    [*`max_steps`*], [*EM*], [*F1*], [*Title Recall*], [*Avg LLM Calls*], [*Avg Retrieved*],
    [$1$], [$0.300$], [$0.374$], [$0.850$], [$2.0$], [$6.70$],
    [$3$], [$0.350$], [$0.432$], [$0.875$], [$4.0$], [$7.75$],
    [$5$], [$0.250$], [$0.339$], [$0.875$], [$6.0$], [$7.75$],
  ),
  caption: [IRCoT 步数消融实验结果],
) <tbl:ircot-ablation>

`max_steps=3` 在 EM 和 F1 上表现最好。与 `max_steps=1` 相比，EM 从 0.300 提升到 0.350，F1 从 0.374 提升到 0.432，Title Recall 也从 0.850 提升到 0.875。这说明适度增加推理-检索轮数能够帮助模型补充更多证据。

但继续增加到 `max_steps=5` 后，EM 和 F1 反而下降，虽然 Title Recall 仍保持 0.875。这表明更多轮次并不一定带来更好答案：额外检索可能引入冗余或干扰证据，中间线索也可能逐步偏离原问题，导致最终 reader 阶段更难抽取正确答案。因此 `max_steps=3` 是本实验设定下的较优折中点。

==== 案例分析

选取三个典型样本来说明 IRCoT 的工作机制。

样本一（IRCoT 通过补充证据正确回答）：问题是"Scott Derrickson 和 Ed Wood 是否同一国籍"，标准答案为"yes"。No Retrieval CoT 只生成了一句推理开头便停止，EM 和 F1 均为 0。One-step RAG-CoT 已成功找回 supporting titles（`Scott Derrickson` 和 `Ed Wood`），但模型最终没有完成答案抽取，输出停留在"Step 1: Identify the nationality of Scott Derrickson"。IRCoT 的中间线索多次聚焦于"Scott Derrickson is an American director"，并结合 Ed Wood 相关证据，最终简洁回答"Yes"，EM 和 F1 均为 1.0。

样本二（IRCoT 补充中间实体）：问题是"什么科学奇幻青少年系列小说采用第一人称叙述，并有配套丛书讲述被奴役世界和外星物种的故事"，标准答案为"Animorphs"。IRCoT 的中间线索明确写出了 `Animorphs` 以及 `Animorphs is a science fantasy young adult series told in first person`，后续检索也找回了 Animorphs 和 companion books 相关证据，最终答案完全匹配标准答案。相比之下，No Retrieval CoT 完全无法回答该类开放式知识问题（模型虚构了"The Book Thief"），One-step RAG-CoT 虽已检索到 Animorphs 段落，但答案抽取仍失败。

样本三（IRCoT 失败样本）：问题是"在电影 Kiss and Tell 中饰演 Corliss Archer 的女性担任过什么政府职位"，标准答案为"Chief of Protocol"。IRCoT 输出为"Secretary of State for Constitutional Affairs"。分析发现，该样本的 Title Recall 只有 0.5，且模型在早期检索到错误政府职位段落后，中间线索持续重复"Secretary of State for Constitutional Affairs"，后续推理完全围绕错误方向展开。这个案例说明 IRCoT 的效果高度依赖中间检索线索的质量——如果中间线索一开始就偏离正确实体，更多轮次反而会强化错误方向。

=== 分析

==== 方法优越性

IRCoT 相比纯 CoT 和一次性 RAG-CoT 的优势主要体现为三点。第一，它显式引入了外部知识，减少了模型仅凭参数记忆回答导致的错误，尤其适合多跳问答中需要组合多条证据的场景。第二，它的动态证据补充机制不局限于原始问题的词面——中间推理可以识别出问题中未直接出现的实体或关系，引导后续检索覆盖更多关键段落。第三，IRCoT 保存了完整的中间推理链和检索历史，可解释性强于单纯黑盒生成。

==== 局限性

IRCoT 的局限同样明显。检索器方面，BM25 依赖词面匹配，对实体别名、语义改写和复杂关系的处理能力有限。模型方面，本实验使用的 Qwen2.5-0.5B-Instruct 在答案抽取和格式遵循上不够稳定，部分样本中即便检索到了正确证据，最终输出仍可能因为格式问题被判为错误。此外，中间线索的质量对整体效果有决定性影响——线索一旦偏航，后续检索和推理都会被带偏。计算开销也值得注意：`max_steps=3` 时每题平均调用 4 次 LLM，虽然远低于 Multi-Agent Debate 的 6+ 次，但仍明显高于单次 CoT。

==== 与前三项实验的关系

IRCoT 在方法论层面与前三项实验形成互补。Self-Consistency、Verifier CoT 和 Multi-Agent Debate 的思路都是在"生成端"做文章——通过多次采样、质量评分或多角色协作来提升推理的可靠性。IRCoT 的思路则是在"输入端"做文章——通过检索机制动态引入外部证据，弥补模型自身知识的不足。这两种方向并不互斥：例如可以让 Self-Consistency 的每条采样路径共享同一份检索上下文（Self-Consistency + RAG），或者让 Multi-Agent Debate 中的每个 Agent 在推理过程中主动查询外部知识库。这种组合可能是后续进一步提升推理质量的可行路线。

== 九、总结与展望

本报告整合了围绕思维链推理的四种拓展策略的实验结果。从不同模型和数据集来看：

- Self-Consistency 在 32B 模型和 AQuA 数学选择题上展示了多路径投票的稳健性，最优配置（`paths=5`, `temperature=0.3`）达到 87.01%。
- Verifier CoT 在 7B 模型上通过加权投票纠正了简单多数投票可能犯的错误，达到 76.77%。
- Multi-Agent Debate 通过角色分工与反思机制，在三智能体设定下取得 78.74%，是同模型规模下的最佳结果。
- IRCoT 在 0.5B 模型和 HotpotQA 多跳问答上验证了交替式检索-推理策略的有效性，EM 达 0.350、F1 达 0.432，明显优于无检索和一次性检索方法。

四种方法从不同角度回答了同一个问题：如何让模型的推理不止于"一条路走到黑"。前三者聚焦于生成侧的多样性提升与质量评估，IRCoT 则聚焦于输入侧的外部知识注入。它们分别对应了不同的设计哲学，各自有其适用场景和局限性。

后续工作可在以下方向展开：一是将检索增强与前述生成侧方法结合（如 Self-Consistency + RAG、Multi-Agent Debate + RAG），验证外部知识对推理稳定性的增量贡献；二是将 IRCoT 的 BM25 检索器替换为 dense retriever 或混合检索，提升语义召回能力；三是探索异构模型协作（不同规模、不同训练来源的模型分别扮演不同 Agent），降低 Agent 之间的推理相关性；四是研究更高效的聚合与检索调度机制，在准确率与计算成本之间取得更好的平衡。

== 九、总结与展望

本报告整合了围绕思维链推理的三种拓展策略的实验结果。从 32B 和 7B 两个规模的 Qwen2.5-Coder 模型来看：

- Self-Consistency 在 32B 模型上展示了多路径投票的稳健性，最优配置（`paths=5`, `temperature=0.3`）达到 87.01%。
- Verifier CoT 在 7B 模型上通过加权投票纠正了简单多数投票可能犯的错误，达到 76.77%。
- Multi-Agent Debate 通过角色分工与反思机制，在三智能体设定下取得 78.74%，是同模型规模下的最佳结果。

三种方法从不同角度回答了同一个问题：如何让模型的推理不止于"一条路走到黑"。它们分别对应了不同的设计哲学——随机采样 + 多数投票、角色多样化 + 质量评分、多智能体协作 + 反思修正——各自有其适用场景和局限性。

后续工作可在以下方向展开：一是将 RAG 与现有方法结合，验证外部知识对推理稳定性的增量贡献；二是探索异构模型协作（不同规模、不同训练来源的模型分别扮演不同 Agent），降低 Agent 之间的推理相关性；三是研究更高效的聚合机制（如学习式投票权重、基于置信度的自适应路径数），在准确率与计算成本之间取得更好的平衡。
