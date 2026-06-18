"""
方法模块 (Methods)
------------------
组员各自实现的 CoT 策略文件放在此目录下。

每个策略类必须:
1. 继承 harness.base_task.BaseTask
2. 实现 solve(question: str) -> str 方法

示例策略:
- baseline_cot.py  : 基础思维链 (Zero-shot CoT)
- self_consistency.py : 多路径采样 + 投票 (组员 A)
- verifier_agent.py   : Step-aware Verifier (组员 B)
- rag_cot.py          : 检索增强 CoT / IRCoT (组员 C)
- multi_agent_debate.py : 多角色辩论 (组员 D)
"""
