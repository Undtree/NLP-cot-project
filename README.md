# CoT Harness Engineering — 思维链推理评测框架

> **选题二：基于大语言模型的思维链推理 (Chain-of-Thought Reasoning)**
>
> 华为云部署 LLM + 本地 Harness 评测管理的端云协同架构

---

## 项目概述

本项目是一个**模块化、可扩展的思维链 (CoT) 推理评测框架**。

核心架构为 **"端云协同"**：

```
┌─────────────────────────┐         HTTP (OpenAI API)        ┌──────────────────────────┐
│   华为云 ECS (4×T4 GPU)  │ ◄─────────────────────────────► │   本地 Harness 框架        │
│   vLLM + Qwen2.5-Coder   │                                 │   数据集 → 策略 → 评估    │
│   纯算力引擎 / API Server │                                 │   控制面 / 评测逻辑        │
└─────────────────────────┘                                 └──────────────────────────┘
```

- **云端**：华为云 ECS 裸机部署 Qwen2.5-Coder-32B-Instruct，通过 vLLM 暴露 OpenAI 兼容 API
- **本地**：Harness Engineering 框架负责数据集管理、策略调度、答案提取、正确率计算、结果导出

---

## 项目结构

```
cot_harness_project/
├── data/                          # 数据集目录
│   └── aqua_test.json             #   示例数据集 (AQuA 格式)
│
├── harness/                       #   核心框架
│   ├── __init__.py                #   统一导出接口
│   ├── dataset.py                 #   数据集加载器：下载/清洗/格式化
│   ├── llm_client.py              #   云端 API 客户端：重试/超时/并发控制
│   ├── base_task.py               #   抽象基类 BaseTask + CoTTrace 追踪
│   ├── evaluator.py               #   评估器：8 种答案提取模式 + 正确率
│   └── logger.py                  #   日志系统：JSONL/CSV/汇总报告
│
├── methods/                       # 策略实现目录（组员各自建文件）
│   ├── __init__.py
│   └── baseline_cot.py            #   基线：Zero-shot CoT（参考模板）
│
├── results/                       # 实验结果自动保存
│   ├── run_<timestamp>_<strategy>.jsonl
│   ├── run_<timestamp>_<strategy>.csv
│   └── all_runs_summary.jsonl     #   全局汇总（横向对比）
│
├── main.py                        # 统一启动入口 (CLI)
├── requirements.txt               # Python 依赖
└── README.md                      # 本文件
```

---

## 如何运行这个项目

### 1. 环境准备

```bash
# 克隆项目
git clone <your-repo-url>
cd cot_harness_project

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置云端 API 地址

```bash
# 方式一：环境变量（推荐）
export LLM_BASE_URL="http://<华为云ECS公网IP>:8000/v1"
export LLM_MODEL_NAME="qwen2.5-coder-32b"

# 方式二：每次命令行指定
python main.py --base-url http://<IP>:8000/v1 --strategy baseline
```

### 3. 检查云端连接

```bash
python main.py --check-health
```

### 4. 运行实验

```bash
# 查看所有可用策略
python main.py --list-strategies

# 运行基线 CoT（快速测试 10 个样本）
python main.py --strategy baseline --max-samples 10 --verbose

# 运行完整评测
python main.py --strategy baseline

# 使用自定义数据集
python main.py --strategy baseline --dataset data/my_data.json
```

> `main.py` 每次只运行**一个**策略。每个组员的策略需要单独运行、单独评估。如需依次运行所有已注册策略，使用 `--run-all`。

---

## 组员接入

### 第一步：创建策略文件

在 `methods/` 下创建你的 Python 文件，继承 `BaseTask` 并实现 `solve` 方法：

```python
# methods/my_strategy.py
from harness.base_task import BaseTask, CoTTrace

class MyCoTStrategy(BaseTask):
    def __init__(self, client, **kwargs):
        super().__init__(client, name="MyCoTStrategy", **kwargs)

    def solve(self, question: str) -> str:
        # 1. 构建你自己的 Prompt
        prompt = f"请逐步推理以下问题：\n{question}\n\n请一步步思考。"
        # 2. 调用云端 LLM
        response = self.client.chat(
            user_message=prompt,
            temperature=self.extra_config.get("temperature", 0.0),
            max_tokens=self.extra_config.get("max_tokens", 4096),
        )
        # 3. 返回原始输出（Evaluator 会自动提取答案）
        return response.content

    def solve_with_trace(self, question: str) -> CoTTrace:
        # （推荐）实现带推理路径追踪的版本
        trace = CoTTrace(question=question)
        trace.final_answer = self.solve(question)
        trace.add_step("使用了某 CoT 策略")
        return trace
```

### 第二步：注册策略

在 `main.py` 的 `STRATEGY_REGISTRY` 中添加一行：

```python
STRATEGY_REGISTRY = {
    "baseline": "methods.baseline_cot.BaselineCoT",
    "my_strategy": "methods.my_strategy.MyCoTStrategy",   # ← 添加这行
}
```

### 第三步：运行你的策略

```bash
python main.py --strategy my_strategy --max-samples 10 --verbose
```

---

## 核心模块

### `harness/dataset.py` — 数据集加载器

| 功能 | 说明 |
|------|------|
| 格式支持 | JSON 列表 / JSON 嵌套 / JSONL 逐行 / 字典 |
| 自动清洗 | 无效样本跳过 + 告警，不中断流程 |
| 选项拼接 | 自动构建 "A. ... B. ..." 格式的完整问题 |
| 唯一 ID | 每个样本自动生成 MD5 唯一标识 |
| 自动下载 | 可从 GitHub 自动下载 AQuA 数据集 |

```python
from harness.dataset import load_dataset, download_aqua_dataset

# 自动下载
path = download_aqua_dataset("data")

# 加载数据集
dataset = load_dataset("data/aqua_test.json")
print(f"共 {len(dataset)} 个样本")
print(dataset[0]["question"])
print(dataset[0]["ground_truth"])
```

### `harness/llm_client.py` — 云端 API 客户端

| 特性 | 实现 |
|------|------|
| 自动重试 | 指数退避 (1s → 2s → 4s)，最多 3 次 |
| 超时控制 | 连接超时 30s + 读取超时 300s（适应长 CoT） |
| 并发控制 | BoundedSemaphore，默认最多 8 并发 |
| 连接复用 | Session + HTTPAdapter 连接池 |
| 健康检查 | `check_health()` 验证 API 可用性 |

```python
from harness.llm_client import LLMClient

client = LLMClient(
    base_url="http://<IP>:8000/v1",
    model_name="qwen2.5-coder-32b",
    timeout=300,
    max_concurrent=8,
)

# 单轮对话
resp = client.chat("请回答：1+1等于几？")
print(resp.content)

# 多轮对话（Agent 场景）
resp = client.chat_multi_turn([
    {"role": "system", "content": "你是数学助手"},
    {"role": "user", "content": "2+2=?"},
])
```

### `harness/evaluator.py` — 评估器

**8 种答案提取模式**（按优先级级联匹配）：

| # | 模式 | 匹配示例 |
|---|------|----------|
| 1 | `answer is X` | "The answer is: A" |
| 2 | `therefore/thus/so X` | "Therefore, the answer is B" |
| 3 | `choose/select/pick X` | "I choose C" |
| 4 | `correct option is X` | "The correct option is D" |
| 5 | 中文答案格式 | "答案是：E" |
| 6 | Markdown 格式 | "#### A" |
| 7 | LaTeX 格式 | "\boxed{B}" |
| 8 | 行首孤立字母 | `C` |

```python
from harness.evaluator import Evaluator, quick_extract

evaluator = Evaluator(verbose=True)
report = evaluator.evaluate(
    raw_outputs=["... The answer is: B", "..."],
    ground_truths=["B", "A"],
)
print(f"准确率: {report.accuracy:.2%}")
```

### `harness/base_task.py` — 抽象基类

```python
from harness.base_task import BaseTask, CoTTrace

class BaseTask(ABC):
    @abstractmethod
    def solve(self, question: str) -> str:
        """组员必须实现：输入问题 → 返回答案"""
        pass

    def solve_with_trace(self, question: str) -> CoTTrace:
        """推荐实现：记录完整推理路径"""
        pass
```

### `harness/logger.py` — 日志系统

每次运行自动生成 **3 种格式**：

| 文件 | 内容 |
|------|------|
| `run_*.jsonl` | 逐样本完整信息（含推理路径） |
| `run_*.csv` | 表格格式，方便 Excel 查看 |
| `run_*_summary.json` | 汇总统计（准确率/耗时等） |
| `all_runs_summary.jsonl` | 全局汇总，横向对比所有策略 |

---

## 查看实验结果

```bash
# 运行实验
python main.py --strategy baseline

# 查看所有历史运行的对比表（终端直接打印）
python main.py --strategy baseline  # 运行完后自动打印对比表
```

输出示例：

```
================================================================================
策略名称                            准确率     正确/总数     提取失败 时间
--------------------------------------------------------------------------------
BaselineCoT                        65.00%    13/20             2 2026-06-19T10:30:00
SelfConsistency                    72.50%    29/40             1 2026-06-19T11:15:00
VerifierAgent                      80.00%    16/20             0 2026-06-19T14:00:00
================================================================================
```

CSV 和 JSONL 文件可直接用于论文绘图（matplotlib / Excel 图表）。

---

## 🔄 设计理念

### 控制面与数据面解耦

```
控制面 (本地 Git)              数据面 (华为云 ECS)
┌──────────────────┐          ┌─────────────────────┐
│ main.py           │  HTTP   │ vLLM API Server     │
│ harness/*         │ ◄─────► │ Qwen2.5-Coder-32B   │
│ methods/*         │         │ 4×T4 Tensor Parallel │
│ git push / pull   │         │ tmux 持久化运行      │
└──────────────────┘          └─────────────────────┘
```

- **控制面**：所有业务逻辑、评测框架在本地通过 Git 协作
- **数据面**：云端纯粹作为算力引擎，24/7 提供 API 服务
- 组员写策略、调 Prompt 完全不消耗云端显存

### 控制变量法

全组使用**同一个模型权重**（Qwen2.5-Coder-32B-Instruct），由架构师统一在华为云部署为 API。这样在答辩时，性能差异可以干净地归因于 **"策略优劣"** 而非 **"模型强弱"**。

---

## 环境变量参考

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `LLM_BASE_URL` | 云端 API 地址 | `http://localhost:8000/v1` |
| `LLM_MODEL_NAME` | 模型名称 | `qwen2.5-coder-32b` |
| `LLM_API_KEY` | API 密钥 | `EMPTY` |
| `LLM_TIMEOUT` | 请求超时 (秒) | `300` |
| `LLM_MAX_CONCURRENT` | 最大并发数 | `8` |

---

## 依赖说明

```
openai>=1.0.0          # OpenAI 兼容客户端
requests>=2.31.0       # HTTP 请求
urllib3>=2.0.0         # HTTP 连接池
tqdm>=4.66.0           # 进度条（可选）
pandas>=2.0.0          # 数据处理（可选）
```

云端 ECS 上（已配置）：
```
vllm>=0.5.0            # LLM 推理加速引擎
torch>=2.4.0           # PyTorch (CUDA 12.1)
transformers>=4.44.0   # HuggingFace Transformers
modelscope>=1.16.0     # 魔搭社区（下载模型）
lm-format-enforcer>=0.10.0  # 格式化输出约束
```

---

## License

本项目仅供学术课程使用。
