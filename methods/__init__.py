"""
方法模块 (Methods)
------------------
组员各自实现的 CoT 策略文件放在此目录下。

每个策略类必须:
1. 继承 harness.base_task.BaseTask
2. 实现 solve(question: str) -> str 方法
3. (可选) 实现 solve_sample_with_trace(sample) -> CoTTrace 以支持 Sample 接口

已有策略:
- baseline_cot.py      : 基础思维链 (Zero-shot CoT)
- self_consistency.py  : 多路径采样 + 投票
- verifier_cot.py      : Verifier 加权投票
- multi_agent_debate.py: 多角色辩论 (Debate / Reflective / ThreeAgent)
- ircot_method.py      : 检索增强交错式 CoT (NoRetrieval / OneStepRAG / IRCoT)
"""
