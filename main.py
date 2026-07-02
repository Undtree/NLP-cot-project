"""
统一启动入口 (Main Entry Point)
--------------------------------
Harness Engineering 的主入口脚本。

⚠️ 设计原则：每次运行只跑一个策略，互不干扰。
   每个组员的策略需要单独运行、单独评估、单独出报告。

负责:
1. 检查 API 连接
2. 加载数据集（支持 AQuA 和 HotpotQA）
3. 实例化【一个】组员的策略类
4. 运行实验循环（逐样本调用 solve -> 评估 -> 记录）
5. 生成最终评估报告

使用方式:
    # AQuA 数学选择题（默认）
    python main.py --strategy baseline
    python main.py --strategy self_consistency --paths 5 --temperature 0.7

    # HotpotQA 多跳问答
    python main.py --task-type hotpotqa --strategy ircot --max-steps 3
    python main.py --task-type hotpotqa --strategy no_retrieval_cot
    python main.py --task-type hotpotqa --strategy one_step_rag

    # 依次跑所有已注册的策略（用于最终汇总对比）
    python main.py --run-all

    # 快速检查 API 连接
    python main.py --check-health
"""

import os
import sys
import time
import argparse
import logging
from typing import Optional, List

# 加载 .env 文件（若存在）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv 未安装时忽略

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness.dataset import load_dataset, download_aqua_dataset
from harness.retrieval import SimpleBM25
from harness.llm_client import LLMClient
from harness.evaluator import Evaluator, quick_extract, QAMatchEvaluator
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

STRATEGY_REGISTRY = {
    # ---- AQuA 数学策略 ----
    "baseline": "methods.baseline_cot.BaselineCoT",
    "self_consistency": "methods.self_consistency.SelfConsistencyTask",
    "debate": "methods.multi_agent_debate.DebateTask",
    "reflective_debate": "methods.multi_agent_debate.ReflectiveDebateTask",
    "three_agent_debate": "methods.multi_agent_debate.ThreeAgentReflectiveDebateTask",
    "verifier_cot": "methods.verifier_cot.VerifierCoT",
    # ---- HotpotQA / IRCoT 策略 ----
    "no_retrieval_cot": "methods.ircot_method.NoRetrievalCoT",
    "one_step_rag": "methods.ircot_method.OneStepRAGCoT",
    "ircot": "methods.ircot_method.IRCoTTask",
}

# 按任务类型分组的策略
TASK_TYPE_STRATEGIES = {
    "aqua": ["baseline", "self_consistency", "debate", "reflective_debate",
             "three_agent_debate", "verifier_cot"],
    "hotpotqa": ["no_retrieval_cot", "one_step_rag", "ircot"],
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
    evaluator=None,
    logger_instance=None,
    max_samples: Optional[int] = None,
    use_trace: bool = True,
    verbose: bool = False,
    task_type: str = "aqua",
):
    """
    运行一次完整的实验。

    Args:
        task: 策略实例（继承 BaseTask）
        dataset: 数据集
            - task_type="aqua": AQuADataset (iterable of dict)
            - task_type="hotpotqa": List[Sample]
        evaluator: 评估器实例
            - task_type="aqua": Evaluator (默认)
            - task_type="hotpotqa": QAMatchEvaluator (默认)
        logger_instance: 日志记录器实例
        max_samples: 最大测试样本数
        use_trace: 是否使用 solve_sample_with_trace 记录推理路径
        verbose: 是否打印详细过程
        task_type: 任务类型 ("aqua" 或 "hotpotqa")

    Returns:
        (评估报告, 实验日志记录器)
    """
    # ---- 根据 task_type 设置默认 evaluator ----
    if evaluator is None:
        if task_type == "hotpotqa":
            evaluator = QAMatchEvaluator(verbose=verbose)
        else:
            evaluator = Evaluator(verbose=verbose)

    if logger_instance is None:
        logger_instance = ExperimentLogger(results_dir="results")

    # ---- 准备样本 ----
    samples = list(dataset)
    if max_samples and max_samples < len(samples):
        samples = samples[:max_samples]
        logger.info(f"限制样本数为 {max_samples}（总共 {len(dataset)}）")

    total = len(samples)
    logger.info(f"开始运行实验: 策略={task.name}, 任务类型={task_type}, 样本数={total}")

    # 进度条
    try:
        from tqdm import tqdm
        sample_iter = tqdm(samples, desc=f"[{task.name}]", unit="sample")
    except ImportError:
        logger.info("提示: 安装 tqdm 可获得进度条显示 (pip install tqdm)")
        sample_iter = samples

    start_time = time.time()

    # ---- 分路径运行 ----
    if task_type == "hotpotqa":
        # HotpotQA 路径：使用 Sample 对象 + QAMatchEvaluator
        predictions = []
        ground_truths = []
        retrieved_list = []
        supporting_titles_list = []
        llm_calls_list = []
        traces = []

        for i, sample in enumerate(sample_iter):
            try:
                if use_trace:
                    trace = task.solve_sample_with_trace(sample)
                    traces.append(trace)
                    predictions.append(trace.final_answer)
                    retrieved_list.append(trace.metadata.get("retrieved", []))
                    supporting_titles_list.append(sample.supporting_titles)
                    llm_calls_list.append(trace.metadata.get("llm_calls", 0))
                else:
                    answer = task.solve_sample(sample)
                    predictions.append(answer)
                    retrieved_list.append([])
                    supporting_titles_list.append(sample.supporting_titles)
                    llm_calls_list.append(0)

                ground_truths.append(str(sample.ground_truth))

            except Exception as e:
                logger.error(f"样本 {sample.id} 处理失败: {e}")
                predictions.append(f"[ERROR] {str(e)}")
                ground_truths.append(str(sample.ground_truth))
                retrieved_list.append([])
                supporting_titles_list.append(sample.supporting_titles)
                llm_calls_list.append(0)

        elapsed = time.time() - start_time
        logger.info(f"实验完成，耗时 {elapsed:.1f}s ({elapsed/total:.1f}s/样本)")

        # QA 评估
        sample_ids = [s.id for s in samples]
        questions = [s.question for s in samples]
        eval_report = evaluator.evaluate(
            predictions=predictions,
            ground_truths=ground_truths,
            retrieved_list=retrieved_list,
            supporting_titles_list=supporting_titles_list,
            llm_calls_list=llm_calls_list,
            sample_ids=sample_ids,
            questions=questions,
        )

        # 记录日志
        dataset_name = "hotpotqa"
        run_id = logger_instance.log_run_qa(
            eval_report=eval_report,
            strategy_name=task.name,
            config=task.get_config() if hasattr(task, "get_config") else {},
            dataset_name=dataset_name,
            traces=traces if traces else None,
        )

        # 打印结果
        print(f"\n{'='*60}")
        print(f"  实验完成!")
        print(f"  策略: {task.name}")
        print(f"  EM: {eval_report.em:.3f}  F1: {eval_report.f1:.3f}  "
              f"Title Recall: {eval_report.title_recall:.3f}")
        print(f"  平均 LLM 调用: {eval_report.avg_llm_calls:.1f}")
        print(f"  平均检索段落: {eval_report.avg_retrieved:.1f}")
        print(f"  Run ID: {run_id}")
        print(f"{'='*60}\n")

    else:
        # AQuA 路径（保持向后兼容）
        raw_outputs = []
        ground_truths = []
        traces = []

        for i, sample in enumerate(sample_iter):
            # 兼容 dict 和 Sample 两种格式
            if hasattr(sample, "question"):
                question = sample.question
                gt = sample.ground_truth
            else:
                question = sample["question"]
                gt = sample["ground_truth"]

            try:
                if use_trace:
                    trace = task.solve_with_trace(question)
                    traces.append(trace)
                    raw_outputs.append(trace.final_answer)
                else:
                    answer = task.solve(question)
                    raw_outputs.append(answer)

                ground_truths.append(gt)

                if verbose and not use_trace:
                    extracted = quick_extract(raw_outputs[-1])
                    is_correct = (extracted == gt)
                    status = "✓" if is_correct else "✗"
                    logger.info(
                        f"  [{i+1}/{total}] {status} "
                        f"预测={extracted or '?'} 标准={gt}"
                    )

            except Exception as e:
                sid = sample.get("id", i) if isinstance(sample, dict) else getattr(sample, "id", i)
                logger.error(f"样本 {sid} 处理失败: {e}")
                raw_outputs.append(f"[ERROR] {str(e)}")
                ground_truths.append(gt)

        elapsed = time.time() - start_time
        logger.info(f"实验完成，耗时 {elapsed:.1f}s ({elapsed/total:.1f}s/样本)")

        # AQuA 评估
        if hasattr(samples[0], "id"):
            sample_ids = [s.id for s in samples]
            questions = [s.question for s in samples]
        else:
            sample_ids = [s.get("id", f"sample_{i:04d}") for i, s in enumerate(samples)]
            questions = [s.get("question", "") for s in samples]

        eval_report = evaluator.evaluate(
            raw_outputs=raw_outputs,
            ground_truths=ground_truths,
            sample_ids=sample_ids,
            questions=questions,
        )

        # 记录日志
        dataset_name = getattr(dataset, "name", "unknown")
        run_id = logger_instance.log_run(
            eval_report=eval_report,
            strategy_name=task.name,
            config=task.get_config() if hasattr(task, "get_config") else {},
            dataset_name=dataset_name,
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
  # AQuA 数学选择题
  python main.py --strategy baseline
  python main.py --strategy self_consistency --paths 5 --temperature 0.7
  python main.py --strategy three_agent_debate --temperature 0.3

  # HotpotQA 多跳问答
  python main.py --task-type hotpotqa --strategy ircot --max-steps 3
  python main.py --task-type hotpotqa --strategy no_retrieval_cot
  python main.py --task-type hotpotqa --strategy one_step_rag

  # 其他
  python main.py --run-all                 # 依次运行所有策略
  python main.py --check-health            # 检查云端连接
  python main.py --list-strategies         # 列出可用策略
        """,
    )

    # ---- 任务类型 ----
    parser.add_argument(
        "--task-type", "-t",
        type=str,
        default="aqua",
        choices=["aqua", "hotpotqa"],
        help="任务类型: aqua (数学选择题, 默认) | hotpotqa (多跳问答)",
    )

    # ---- 策略 ----
    parser.add_argument(
        "--strategy", "-s",
        type=str,
        default=None,
        help="要运行的策略名称。可用: {}".format(", ".join(STRATEGY_REGISTRY.keys())),
    )

    # ---- 数据集 ----
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=None,
        help="数据集文件路径（默认: aqua=data/aqua_test.json, hotpotqa=data/hotpot_validation.parquet）",
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=None,
        help="最大测试样本数（用于快速调试）",
    )

    # ---- API 参数 ----
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--max-concurrent", type=int, default=None)

    # ---- 策略参数 ----
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--paths", type=int, default=5,
                        help="Self-Consistency 采样路径数")
    parser.add_argument("--max-steps", type=int, default=3,
                        help="IRCoT 最大推理-检索轮数")
    parser.add_argument("--top-k", type=int, default=5,
                        help="IRCoT 每次检索返回的段落数")

    # ---- 其他参数 ----
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--check-health", action="store_true")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--list-strategies", action="store_true")
    parser.add_argument("--run-all", action="store_true")

    args = parser.parse_args()

    # ---- 互斥检查 ----
    if args.run_all and args.strategy is not None:
        parser.error("--run-all 和 --strategy 不可同时使用。")

    # ---- 列出策略 ----
    if args.list_strategies:
        print("\n可用策略（按任务类型分组）:")
        for ttype, names in TASK_TYPE_STRATEGIES.items():
            print(f"\n  [{ttype}]")
            for name in names:
                path = STRATEGY_REGISTRY[name]
                print(f"    {name:<25} -> {path}")
        print()
        return

    # ---- 初始化 LLM 客户端 ----
    base_url = args.base_url or os.environ.get("LLM_API_BASE")
    model_name = args.model_name or os.environ.get("LLM_MODEL_NAME")

    client = LLMClient(
        base_url=base_url,
        model_name=model_name,
        api_key=args.api_key,
        timeout=args.timeout,
        max_concurrent=args.max_concurrent,
    )

    # ---- 健康检查 ----
    if args.check_health:
        print(f"正在检查 API ({client.base_url}) ...")
        if client.check_health():
            print("✓ API 连接正常！")
        else:
            print("✗ API 连接失败，请检查 vLLM 服务状态")
        return

    if not client.check_health():
        logger.error(f"API 不可用 ({client.base_url})。使用 --check-health 单独检查。")
        logger.warning("继续尝试运行，但可能会失败...")

    # ---- 加载数据集 ----
    task_type = args.task_type

    if task_type == "hotpotqa":
        # HotpotQA 路径
        dataset_path = args.dataset or "data/hotpot_validation.parquet"
        if not os.path.exists(dataset_path):
            logger.error(f"HotpotQA 数据集不存在: {dataset_path}")
            logger.info("请将 HotpotQA 数据放置到 data/ 目录，或使用 --dataset 指定路径")
            sys.exit(1)

        samples, corpus = load_dataset(dataset_path, dataset_type="hotpotqa",
                                       max_samples=args.max_samples)
        # 构建 BM25 检索器
        retriever = SimpleBM25(corpus)
        logger.info(f"BM25 检索器已构建: {len(retriever)} 个段落")

        # HotpotQA 专用参数
        extra_task_kwargs = {
            "retriever": retriever,
            "top_k": args.top_k,
        }
        if args.max_steps is not None:
            extra_task_kwargs["max_steps"] = args.max_steps

        dataset_for_run = samples

    else:
        # AQuA 路径（默认）
        dataset_path = args.dataset or "data/aqua_test.json"
        if not os.path.exists(dataset_path):
            logger.warning(f"数据集文件不存在: {dataset_path}")
            logger.info("尝试下载 AQuA 数据集...")
            try:
                dataset_path = download_aqua_dataset("data")
            except Exception:
                logger.error("自动下载失败。请手动将数据集放到 data/ 目录。")
                sys.exit(1)

        dataset_for_run = load_dataset(dataset_path, dataset_type="aqua")
        extra_task_kwargs = {}
        retriever = None
        logger.info(f"AQuA 数据集加载完成: {len(dataset_for_run)} 个样本")

    # ---- 确定要运行的策略列表 ----
    if args.run_all:
        strategies_to_run = TASK_TYPE_STRATEGIES.get(task_type, list(STRATEGY_REGISTRY.keys()))
        if not strategies_to_run:
            logger.error(f"任务类型 '{task_type}' 没有注册任何策略！")
            sys.exit(1)
        print(f"\n{'='*60}")
        print(f"  --run-all 模式 [{task_type}]: 依次运行 {len(strategies_to_run)} 个策略")
        print(f"  策略列表: {', '.join(strategies_to_run)}")
        print(f"{'='*60}\n")
    else:
        strategy_name = args.strategy
        if strategy_name is None:
            # 自动选择该任务类型的第一个策略
            defaults = TASK_TYPE_STRATEGIES.get(task_type, [])
            strategy_name = defaults[0] if defaults else "baseline"
        strategies_to_run = [strategy_name]

    # ---- 初始化 Evaluator 和 Logger ----
    if task_type == "hotpotqa":
        evaluator = QAMatchEvaluator(verbose=args.verbose)
    else:
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

        # 构造策略实例
        strategy_kwargs = {
            "client": client,
            "max_tokens": args.max_tokens,
        }
        if args.temperature is not None:
            strategy_kwargs["temperature"] = args.temperature
        if strategy_name == "self_consistency":
            strategy_kwargs["paths"] = args.paths
        # IRCoT 策略需要 retriever
        if strategy_name in ("ircot", "no_retrieval_cot", "one_step_rag"):
            strategy_kwargs.update(extra_task_kwargs)
        if strategy_name == "ircot" and args.max_steps is not None:
            strategy_kwargs["max_steps"] = args.max_steps

        task = StrategyClass(**strategy_kwargs)
        logger.info(f"策略实例化: {task}")

        # 运行实验
        try:
            run_experiment(
                task=task,
                dataset=dataset_for_run,
                evaluator=evaluator,
                logger_instance=experiment_logger,
                max_samples=args.max_samples,
                use_trace=not args.no_trace,
                verbose=args.verbose,
                task_type=task_type,
            )
        except KeyboardInterrupt:
            logger.warning(f"策略 '{strategy_name}' 被用户中断")
            all_ok = False
            break
        except Exception as e:
            logger.error(f"策略 '{strategy_name}' 运行出错: {e}", exc_info=True)
            all_ok = False

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
