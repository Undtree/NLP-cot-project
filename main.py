"""
统一启动入口 (Main Entry Point)
--------------------------------
Harness Engineering 的主入口脚本。

⚠️ 设计原则：每次运行只跑一个策略，互不干扰。
   每个组员的策略需要单独运行、单独评估、单独出报告。

负责:
1. 检查 API 连接
2. 加载数据集
3. 实例化【一个】组员的策略类
4. 运行实验循环（逐样本调用 solve -> 评估 -> 记录）
5. 生成最终评估报告

使用方式:
    # 配置 .env 文件后直接运行
    python main.py --strategy baseline
    python main.py --strategy baseline --max-samples 10 --verbose

    # 依次跑所有已注册的策略（用于最终汇总对比）
    python main.py --run-all

    # 指定 API 地址（覆盖 .env 配置）
    python main.py --strategy baseline --base-url http://localhost:8000/v1

    # 快速检查 API 连接
    python main.py --check-health
"""

import os
import sys
import time
import argparse
import logging
from typing import Optional

# 加载 .env 文件（若存在）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv 未安装时忽略

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness.dataset import load_dataset, download_aqua_dataset
from harness.llm_client import LLMClient, create_client_from_env
from harness.evaluator import Evaluator, quick_extract
from harness.logger import ExperimentLogger
from harness.base_task import BaseTask

# 配置全局日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


# ============================================================
# 策略注册表
# ============================================================
# 组员写完自己的策略后，在这里注册即可。
# 注意：main.py 每次只跑一个策略（通过 --strategy 指定）。
#       如需依次跑所有策略做横向对比，使用 --run-all。

STRATEGY_REGISTRY = {
    "baseline": "methods.baseline_cot.BaselineCoT",
    # === 组员写完策略后，取消下面相应行的注释 ===
    "self_consistency": "methods.self_consistency.SelfConsistencyTask",
    # "verifier": "methods.verifier_agent.VerifierAgent",
    # "rag_cot": "methods.rag_cot.RAGCoT",
    "debate": "methods.multi_agent_debate.DebateTask",
    "reflective_debate": "methods.multi_agent_debate.ReflectiveDebateTask",
    "three_agent_debate": "methods.multi_agent_debate.ThreeAgentReflectiveDebateTask",
    "verifier_cot": "methods.verifier_cot.VerifierCoT",
}


def import_strategy(strategy_name: str):
    """
    动态导入策略类。

    Args:
        strategy_name: 策略名称（注册表中的 key）

    Returns:
        策略类对象
    """
    if strategy_name not in STRATEGY_REGISTRY:
        available = ", ".join(STRATEGY_REGISTRY.keys())
        raise ValueError(
            f"未知策略: '{strategy_name}'。可用策略: {available}"
        )

    module_path, class_name = STRATEGY_REGISTRY[strategy_name].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# ============================================================
# 核心实验运行函数
# ============================================================

def run_experiment(
    task: BaseTask,
    dataset,
    evaluator: Optional[Evaluator] = None,
    logger_instance: Optional[ExperimentLogger] = None,
    max_samples: Optional[int] = None,
    use_trace: bool = True,
    verbose: bool = False,
):
    """
    运行一次完整的实验。

    Args:
        task: 策略实例（继承 BaseTask）
        dataset: 数据集（AQuADataset 或 list[dict]）
        evaluator: 评估器实例
        logger_instance: 日志记录器实例
        max_samples: 最大测试样本数（用于快速调试）
        use_trace: 是否使用 solve_with_trace 记录推理路径
        verbose: 是否打印详细过程

    Returns:
        (评估报告, 实验日志记录器)
    """
    if evaluator is None:
        evaluator = Evaluator(verbose=verbose)
    if logger_instance is None:
        logger_instance = ExperimentLogger(results_dir="results")

    # 限制样本数
    samples = list(dataset)
    if max_samples and max_samples < len(samples):
        samples = samples[:max_samples]
        logger.info(f"限制样本数为 {max_samples}（总共 {len(dataset)}）")

    total = len(samples)
    logger.info(f"开始运行实验: 策略={task.name}, 样本数={total}")

    # 进度条（可选 tqdm）
    try:
        from tqdm import tqdm
        sample_iter = tqdm(samples, desc=f"[{task.name}]", unit="sample")
    except ImportError:
        logger.info("提示: 安装 tqdm 可获得进度条显示 (pip install tqdm)")
        sample_iter = samples

    raw_outputs = []
    ground_truths = []
    traces = []

    start_time = time.time()

    for i, sample in enumerate(sample_iter):
        question = sample["question"]
        ground_truth = sample["ground_truth"]

        try:
            if use_trace:
                trace = task.solve_with_trace(question)
                traces.append(trace)
                raw_outputs.append(trace.final_answer)
            else:
                answer = task.solve(question)
                raw_outputs.append(answer)

            ground_truths.append(ground_truth)

            if verbose and not use_trace:
                extracted = quick_extract(raw_outputs[-1])
                is_correct = (extracted == ground_truth)
                status = "✓" if is_correct else "✗"
                logger.info(
                    f"  [{i+1}/{total}] {status} "
                    f"预测={extracted or '?'} 标准={ground_truth}"
                )

        except Exception as e:
            logger.error(f"样本 {sample.get('id', i)} 处理失败: {e}")
            raw_outputs.append(f"[ERROR] {str(e)}")
            ground_truths.append(ground_truth)

    elapsed = time.time() - start_time
    logger.info(f"实验完成，耗时 {elapsed:.1f}s ({elapsed/total:.1f}s/样本)")

    # 评估
    sample_ids = [s.get("id", f"sample_{i:04d}") for i, s in enumerate(samples)]
    questions = [s.get("question", "") for s in samples]
    eval_report = evaluator.evaluate(
        raw_outputs=raw_outputs,
        ground_truths=ground_truths,
        sample_ids=sample_ids,
        questions=questions,
    )

    # 记录日志
    run_id = logger_instance.log_run(
        eval_report=eval_report,
        strategy_name=task.name,
        config=task.get_config(),
        dataset_name=dataset.name if hasattr(dataset, "name") else "unknown",
        traces=traces if traces else None,
    )

    # 打印结果
    print(f"\n{'='*60}")
    print(f"  实验完成!")
    print(f"  策略: {task.name}")
    print(f"  准确率: {eval_report.accuracy:.2%} "
          f"({eval_report.correct_count}/{eval_report.total_samples})")
    print(f"  答案提取失败: {eval_report.extraction_failures}")
    print(f"  Run ID: {run_id}")
    print(f"{'='*60}\n")

    return eval_report, logger_instance


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="CoT Harness Engineering - 思维链推理评测框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py                           # 默认运行 baseline 策略
  python main.py --strategy baseline       # 运行 baseline
  python main.py --strategy self_consistency --paths 5   # 运行自洽性策略
  python main.py --strategy baseline --max-samples 10 --verbose
  python main.py --run-all                 # 依次运行所有已注册策略
  python main.py --strategy baseline --base-url http://localhost:8000/v1
  python main.py --check-health            # 检查云端连接
        """,
    )

    # 必需参数
    parser.add_argument(
        "--strategy", "-s",
        type=str,
        default=None,
        help="要运行的策略名称。不指定则运行 'baseline'。"
             "可用: {}".format(", ".join(STRATEGY_REGISTRY.keys())),
    )

    # 数据集参数
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="data/aqua_test.json",
        help="数据集文件路径",
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=None,
        help="最大测试样本数（用于快速调试）",
    )

    # API 参数
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="API 地址 (默认从 .env 的 LLM_API_BASE 读取)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="模型名称 (默认: qwen2.5-coder-7b)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API 密钥 (默认从 .env 的 LLM_API_KEY 读取；vLLM 默认为 EMPTY)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="请求超时时间 (秒，默认从 .env 的 LLM_TIMEOUT 读取)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="最大并发请求数 (默认从 .env 的 LLM_MAX_CONCURRENT 读取)",
    )

    # 策略参数
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="采样温度 (默认使用各策略内置值)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="最大生成 token 数",
    )
    parser.add_argument(
        "--paths",
        type=int,
        default=5,
        help="Self-Consistency 采样的推理路径数量",
    )

    # 其他参数
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="打印详细过程",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="不记录推理路径追踪（仅记录最终答案）",
    )
    parser.add_argument(
        "--check-health",
        action="store_true",
        help="仅检查云端 API 连接状态，不运行实验",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="实验结果保存目录",
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="列出所有可用策略",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="依次运行所有已注册的策略（用于最终横向对比）。"
             "与 --strategy 互斥，不可同时使用。",
    )

    args = parser.parse_args()

    # --run-all 和 --strategy 互斥检查
    if args.run_all and args.strategy is not None:
        parser.error(
            "--run-all 和 --strategy 不可同时使用。\n"
            "  • 使用 --run-all 会依次运行所有已注册策略\n"
            "  • 使用 --strategy <name> 只运行指定的一个策略"
        )

    # 列出可用策略
    if args.list_strategies:
        print("\n可用策略（每次只跑一个，用 --strategy <name> 指定）:")
        for name, path in STRATEGY_REGISTRY.items():
            print(f"  {name:<25} -> {path}")
        print()
        return

    # ---- 初始化 LLM 客户端 ----
    # 优先使用命令行参数，其次环境变量 (.env)，最后代码默认值
    base_url = args.base_url or os.environ.get("LLM_API_BASE")
    model_name = args.model_name or os.environ.get("LLM_MODEL_NAME")

    client = LLMClient(
        base_url=base_url,
        model_name=model_name,
        api_key=args.api_key,
        timeout=args.timeout,
        max_concurrent=args.max_concurrent,
    )

    # 健康检查
    if args.check_health:
        print(f"正在检查 API ({client.base_url}) ...")
        if client.check_health():
            print("✓ API 连接正常！")
        else:
            print("✗ API 连接失败，请检查:")
            print(f"  1. vLLM 服务是否已启动: vllm serve <model> --port 8000")
            print(f"  2. API 地址是否正确: {client.base_url}")
            print(f"  3. 如使用远程服务，检查网络和防火墙")
        return

    # 健康检查（跑实验前快速验证）
    if not client.check_health():
        logger.error(
            f"API 不可用 ({client.base_url})。"
            f"使用 --check-health 单独检查连接。"
        )
        logger.warning("继续尝试运行，但可能会失败...")

    # ---- 加载数据集 (只加载一次) ----
    dataset_path = args.dataset
    if not os.path.exists(dataset_path):
        logger.warning(f"数据集文件不存在: {dataset_path}")
        logger.info("尝试下载 AQuA 数据集...")
        try:
            dataset_path = download_aqua_dataset("data")
        except Exception:
            logger.error(
                "自动下载失败。请手动将数据集放到 data/ 目录。\n"
                "下载地址: https://raw.githubusercontent.com/deepmind/AQuA/master/test.json"
            )
            sys.exit(1)

    dataset = load_dataset(dataset_path)
    logger.info(f"数据集加载完成: {len(dataset)} 个样本")

    # ---- 确定要运行的策略列表 ----
    if args.run_all:
        # 依次运行所有已注册策略
        strategies_to_run = list(STRATEGY_REGISTRY.keys())
        if not strategies_to_run:
            logger.error("STRATEGY_REGISTRY 中没有注册任何策略！")
            sys.exit(1)
        print(f"\n{'='*60}")
        print(f"  --run-all 模式：将依次运行 {len(strategies_to_run)} 个策略")
        print(f"  策略列表: {', '.join(strategies_to_run)}")
        print(f"{'='*60}\n")
    else:
        # 只运行用户指定的一个策略（未指定则默认 baseline）
        strategy_name = args.strategy or "baseline"
        strategies_to_run = [strategy_name]

    # ---- 初始化共享的 Evaluator 和 Logger ----
    evaluator = Evaluator(verbose=args.verbose)
    experiment_logger = ExperimentLogger(results_dir=args.results_dir)

    # ---- 逐个运行策略 ----
    all_ok = True
    for idx, strategy_name in enumerate(strategies_to_run):
        if len(strategies_to_run) > 1:
            print(f"\n{'#'*60}")
            print(f"  [{idx+1}/{len(strategies_to_run)}] 正在运行策略: {strategy_name}")
            print(f"{'#'*60}")

        # 导入策略类
        try:
            StrategyClass = import_strategy(strategy_name)
        except Exception as e:
            logger.error(f"无法导入策略 '{strategy_name}': {e}")
            all_ok = False
            continue

        # 实例化策略
        strategy_kwargs = {
            "client": client,
            "max_tokens": args.max_tokens,
        }
        if args.temperature is not None:
            strategy_kwargs["temperature"] = args.temperature
        if strategy_name == "self_consistency":
            strategy_kwargs["paths"] = args.paths

        task = StrategyClass(**strategy_kwargs)
        logger.info(f"策略实例化: {task}")

        # 运行实验
        try:
            run_experiment(
                task=task,
                dataset=dataset,
                evaluator=evaluator,
                logger_instance=experiment_logger,
                max_samples=args.max_samples,
                use_trace=not args.no_trace,
                verbose=args.verbose,
            )
        except KeyboardInterrupt:
            logger.warning(f"策略 '{strategy_name}' 被用户中断")
            all_ok = False
            break  # 用户中断则停止后续策略
        except Exception as e:
            logger.error(f"策略 '{strategy_name}' 运行出错: {e}", exc_info=True)
            all_ok = False
            # 继续运行下一个策略（不因一个失败而全部中断）

    # ---- 清理 ----
    client.close()

    # ---- 打印汇总 ----
    if len(strategies_to_run) > 1:
        print(f"\n{'='*60}")
        print(f"  全部策略运行完毕！")
        print(f"  共 {len(strategies_to_run)} 个策略，{'全部成功' if all_ok else '部分失败'}")
        print(f"{'='*60}")

    experiment_logger.print_comparison_table()


if __name__ == "__main__":
    main()
