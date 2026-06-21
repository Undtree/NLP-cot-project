# CoT — 思维链推理评测框架

> **选题二：基于大语言模型的思维链推理 (Chain-of-Thought Reasoning)**
>
> 分布式 Notebook 协作模式 —— 每人独立实例、统一代码框架

---

## 项目概述

本项目是一个**模块化、可扩展的思维链 (CoT) 推理评测框架**。

协作模式为 **"代码共享 + 独立运行"**：

```
┌──────────────────────────────┐     ┌──────────────────────────────┐
│  组员 A: ModelArts Notebook   │     │  组员 B: ModelArts Notebook   │
│  ┌────────────────────────┐  │     │  ┌────────────────────────┐  │
│  │ vLLM + Qwen2.5-Coder   │  │     │  │ vLLM + Qwen2.5-Coder   │  │
│  │ (localhost:8000)       │  │     │  │ (localhost:8000)       │  │
│  └────────────────────────┘  │     │  └────────────────────────┘  │
│  ┌────────────────────────┐  │     │  ┌────────────────────────┐  │
│  │ Harness 框架 + 自己策略 │  │     │  │ Harness 框架 + 自己策略 │  │
│  └────────────────────────┘  │     │  └────────────────────────┘  │
└──────────────────────────────┘     └──────────────────────────────┘
         │                                     │
         └────────── Git Push/Pull ─────────────┘
                      代码 & 结果汇总
```

- **每人一个 Notebook 实例**：独立 GPU、独立 vLLM、独立 API（`localhost:8000`）
- **统一的 Harness 框架**：通过 Git 共享代码，`.env` 文件管理各自配置
- **结果汇总对比**：各自跑完实验后通过 Git 或打包提交 `results/` 目录

---

## 开始前：GPU 显存规格

组员申请 Notebook 时务必确认 GPU 规格：

| GPU 规格 | 推荐显存 | 推荐模型 | 说明 |
|----------|------|----------|------|
| V100 32GB | 32 GB | `Qwen/Qwen2.5-Coder-32B-Instruct-AWQ` | 可跑 32B AWQ 量化模型 |
| V100 16GB / T4 | 16 GB | `Qwen/Qwen2.5-Coder-7B-Instruct` | **只能跑 7B 模型** |
| T4 16GB | 16 GB | `Qwen/Qwen2.5-Coder-7B-Instruct` | 同上，32B 会 OOM |

> **⚡ 关键提醒**：如果你只有 16GB 显存的卡，绝对不要尝试加载 32B 模型（含 AWQ 量化），否则会直接 `CUDA out of memory`！
> 可以尝试一下 ModelScope 上免费实例的 24GB 显存能否加载 32B-AWQ 模型。

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
│   ├── llm_client.py              #   API 客户端：重试/超时/并发控制（.env 配置）
│   ├── base_task.py               #   抽象基类 BaseTask + CoTTrace 追踪
│   ├── evaluator.py               #   评估器：9 种答案提取模式 + 正确率
│   └── logger.py                  #   日志系统：JSONL/CSV/汇总报告
│
├── methods/                       # 策略实现目录（组员各自建文件）
│   ├── __init__.py
│   └── baseline_cot.py            #   基线：Zero-shot CoT（参考模板）
│
├── results/                       # 实验结果自动保存（已 gitignore）
│   ├── run_<timestamp>_<strategy>.jsonl
│   ├── run_<timestamp>_<strategy>.csv
│   └── all_runs_summary.jsonl     #   全局汇总（横向对比）
│
├── main.py                        # 统一启动入口 (CLI)
├── requirements.txt               # Python 依赖
├── .env.example                   # 环境变量模板（复制为 .env 后修改）
└── README.md                      # 本文件
```

---

## 快速开始（组员操作指南）

### 第一步：克隆仓库并安装依赖

```bash
git clone <your-repo-url>
cd cot_harness_project
pip install -r requirements.txt
```

### 第二步：配置你的 .env 文件

```bash
# 从模板创建你的本地配置
cp .env.example .env

# 编辑 .env，根据你的 Notebook 实例修改
# 最低配置（7B 模型 / 16GB 显存）：
```

**.env 示例（7B 模型 / 16GB 显存卡）**：

```bash
LLM_API_BASE="http://localhost:8000/v1"
LLM_API_KEY="EMPTY"
LLM_MODEL_NAME="qwen2.5-coder-7b"
LLM_MAX_CONCURRENT=2    # 16GB 显存建议保守并发
```

**.env 示例（32B-AWQ / V100 32GB 卡）**：

```bash
LLM_API_BASE="http://localhost:8000/v1"
LLM_API_KEY="EMPTY"
LLM_MODEL_NAME="qwen2.5-coder-32b-awq"
LLM_MAX_CONCURRENT=4
```

> `.env` 已加入 `.gitignore`，不会被提交到 Git。每人独立配置互不影响。

### 第三步：启动 vLLM 推理服务

在 Notebook 的 **Terminal（终端）** 中执行（不要在 Python 代码中执行）：

```bash
# 国内网络加速：从 ModelScope 下载模型
export VLLM_USE_MODELSCOPE=True

# 7B 模型（16GB 显存卡）
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --port 8000 --host 127.0.0.1

# 32B AWQ 量化模型（32GB 显存卡）
vllm serve Qwen/Qwen2.5-Coder-32B-Instruct-AWQ --port 8000 --host 127.0.0.1
```

> **首次启动须知**：vLLM 首次启动时会**自动下载模型权重**（约 15~20 GB），请耐心等待。下载完成后模型会缓存到系统目录（`~/.cache/huggingface/` 或 `~/.cache/modelscope/`），后续启动无需重新下载，秒级即可就绪。
>
> **注意**：vLLM 启动后会占用该终端。不要关闭它！另开一个终端或直接在 Notebook 的 Python 环境中运行实验。

### 第四步：检查连接

```bash
python main.py --check-health
```

看到 `✓ API 连接正常！` 即可继续。

### 第五步：运行实验

```bash
# 查看所有可用策略
python main.py --list-strategies

# 运行基线 CoT（快速测试 10 个样本）
python main.py --strategy baseline --max-samples 10 --verbose

# 运行完整评测
python main.py --strategy baseline

# 运行你自己的策略
python main.py --strategy my_strategy

# 使用自定义数据集
python main.py --strategy baseline --dataset data/my_data.json
```

> `main.py` 每次只运行**一个**策略。每个组员的策略需要单独运行、单独评估。如需依次运行所有已注册策略，使用 `--run-all`。

---

## 组员开发自己的策略

### 第一步：创建策略文件

在 `methods/` 下创建你的 Python 文件，继承 `BaseTask` 并实现 `solve` 方法：

```python
# methods/my_strategy.py
from harness.base_task import BaseTask, CoTTrace

class MyCoTStrategy(BaseTask):
    def __init__(self, client, **kwargs):
        super().__init__(client, name="MyCoTStrategy", **kwargs)

    def solve(self, question: str) -> str:
        prompt = f"请逐步推理以下问题：\n{question}\n\n请一步步思考。"
        response = self.client.chat(
            user_message=prompt,
            temperature=self.extra_config.get("temperature", 0.0),
            max_tokens=self.extra_config.get("max_tokens", 4096),
        )
        return response.content

    def solve_with_trace(self, question: str) -> CoTTrace:
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

### 第三步：运行

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

path = download_aqua_dataset("data")
dataset = load_dataset("data/aqua_test.json")
print(f"共 {len(dataset)} 个样本")
```

### `harness/llm_client.py` — API 客户端

| 特性 | 实现 |
|------|------|
| 配置方式 | `.env` 文件 / 环境变量 / 命令行参数 |
| 自动重试 | 指数退避 (1s → 2s → 4s)，最多 3 次 |
| 超时控制 | 连接超时 30s + 读取超时 300s（适应长 CoT） |
| 并发控制 | BoundedSemaphore，默认 4 并发 |
| 连接复用 | Session + HTTPAdapter 连接池 |
| 健康检查 | `check_health()` 验证 API 可用性 |

```python
from harness.llm_client import LLMClient

# 自动从 .env 读取配置
client = LLMClient()

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

**9 种答案提取模式**（按优先级级联匹配）：

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
| 9 | 最后一个选项字母（回退） | 自动匹配 |

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

每次运行自动生成 **4 种格式**，全部保存到 `results/` 目录：

| 文件 | 内容 |
|------|------|
| `run_*.jsonl` | 逐样本完整信息（含推理路径） |
| `run_*.csv` | 表格格式，方便 Excel 查看 |
| `run_*_summary.json` | 汇总统计（准确率/耗时等） |
| `all_runs_summary.jsonl` | 全局汇总，横向对比所有策略 |

---

## 查看与汇总实验结果

每次运行后在终端自动打印对比表：

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

### 汇总组员结果

各组成员将自己的 `results/` 目录通过 Git 提交或打包发给组长：

```bash
# 组员：提交 result（注意 results/ 默认被 gitignore，需单独提交）
git add results/run_*_<your_strategy>.csv results/run_*_<your_strategy>_summary.json
git commit -m "提交 XX 策略实验结果"
git push
```

组长收到后，利用 `results/all_runs_summary.jsonl` 即可横向对比所有策略。

---

## 设计理念

### 分布式 Notebook 协作

```
组员 A 的 Notebook                组员 B 的 Notebook
┌──────────────────────┐          ┌──────────────────────┐
│ vLLM :8000 (本地)     │          │ vLLM :8000 (本地)     │
│ Harness + 策略代码     │          │ Harness + 策略代码     │
│ .env (A 的配置)       │          │ .env (B 的配置)       │
└──────┬───────────────┘          └──────┬───────────────┘
       │                                 │
       └──── Git Push/Pull 代码 ─────────┘
            + 提交 results/ 结果文件
```

- **代码统一**：所有组员使用同一套 Harness 框架，Git 同步
- **配置独立**：`.env` 文件各自管理，互不冲突
- **结果可对比**：统一的数据集、统一的评估标准，确保公平比较

### 控制变量法

全组使用**同一模型家族**（Qwen2.5-Coder，按显存选 7B 或 32B-AWQ）。答辩时性能差异可干净归因于 **"策略优劣"** 而非 **"模型强弱"**。

> **建议**：最终答辩时标注每个策略使用的具体模型和 GPU 规格，确保透明度。

---

## 环境变量参考（.env 文件）

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `LLM_API_BASE` | API 地址 | `http://localhost:8000/v1` |
| `LLM_MODEL_NAME` | 模型名称 | `qwen2.5-coder-7b` |
| `LLM_API_KEY` | API 密钥 | `EMPTY` |
| `LLM_TIMEOUT` | 请求超时 (秒) | `300` |
| `LLM_MAX_CONCURRENT` | 最大并发数 | `4` |

---

## ModelScope 加速（国内网络）

直接拉取 Hugging Face 模型可能较慢。使用魔搭社区镜像加速：

```bash
# 终端中设置（启动 vLLM 前）
export VLLM_USE_MODELSCOPE=True

# 或在 Python 脚本最顶端添加
import os
os.environ["VLLM_USE_MODELSCOPE"] = "True"
```

这样 vLLM 会自动从 ModelScope 下载 Qwen 模型，下载速度可达几十 MB/s。

---

## 运行方式对照

| 运行方式 | 说明 | 代码改动 |
|----------|------|----------|
| **方案 A：vLLM API 模式**（推荐） | Notebook 终端启动 `vllm serve`，代码用 OpenAI 客户端调用 `localhost:8000` | **零改动** |
| **方案 B：vLLM 离线批处理** | 代码中直接调 `vllm.LLM` 引擎，适合跑 Benchmark | 需要改写策略代码 |

> 本项目当前所有代码基于**方案 A** 实现。如果你的 Notebook 无法后台运行 vLLM serve，可考虑方案 B。

方案 B 示例（供参考）：

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-Coder-7B-Instruct", tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

prompts = ["请写一个快排", "什么是思维链推理？"]
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

---

## 路径管理约定

代码中所有数据集和输出路径均使用**相对路径**，确保不同 Notebook 实例间兼容：

- 数据集：`./data/` 下的相对路径
- 输出：`./results/` 自动创建（`os.makedirs(exist_ok=True)`）
- 配置文件：`.env`（不提交 Git）

---

## 依赖说明

核心依赖（`pip install -r requirements.txt`）：

```
python-dotenv>=1.0.0    # .env 环境变量管理
openai>=1.0.0           # OpenAI 兼容客户端
requests>=2.31.0        # HTTP 请求
urllib3>=2.0.0          # HTTP 连接池
tqdm>=4.66.0            # 进度条（可选）
pandas>=2.0.0           # 数据处理（可选）
```

Notebook 环境额外依赖（按需安装）：

```
vllm>=0.5.0             # LLM 推理加速引擎
torch>=2.4.0            # PyTorch
transformers>=4.44.0    # HuggingFace Transformers
modelscope>=1.16.0      # 魔搭社区（加速下载）
```

---

## 常见问题

**Q: 启动 vLLM 报 `CUDA out of memory`？**
A: 你的 GPU 显存不足以加载当前模型。切换到更小的模型（如 7B），或检查是否有其他进程占用显存。

**Q: `python main.py` 报连接错误？**
A: 确认 vLLM 服务已在另一个终端启动且未关闭。检查 `.env` 中的 `LLM_API_BASE` 是否正确。

**Q: 模型下载很慢？**
A: 设置 `export VLLM_USE_MODELSCOPE=True` 使用魔搭社区镜像。

**Q: 不同组员用的模型不一样，结果能对比吗？**
A: 尽量统一。建议全组用同一模型（如 `Qwen2.5-Coder-7B-Instruct`）。如果显存不同导致模型不同，答辩时需说明。

---

## License

本项目仅供学术课程使用。
