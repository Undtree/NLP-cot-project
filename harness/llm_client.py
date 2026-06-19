"""
LLM 客户端封装 (LLM Client Wrapper)
------------------------------------
封装对华为云 vLLM API 的调用逻辑，提供:
  - OpenAI 兼容的 Chat Completions 接口
  - 自动重试 (Retry with Exponential Backoff)
  - 超时控制 (Timeout)
  - 并发控制 (Semaphore-based Rate Limiting)
  - 连接池复用 (Session-based)

云端 API 地址:
    http://<华为云ECS公网IP>:8000/v1

我们的服务器公网 IP 为 124.70.101.1

使用方式:
    from harness.llm_client import LLMClient

    client = LLMClient(base_url="http://<IP>:8000/v1")
    response = client.chat("你好，请回答一个问题...")
"""

import time
import threading
import logging
from typing import List, Dict, Optional, Any, Generator
from dataclasses import dataclass, field

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("harness.llm_client")


@dataclass
class LLMResponse:
    """标准化的 LLM 响应"""
    content: str                # 模型生成的文本内容
    model: str = ""             # 模型名称
    finish_reason: str = ""     # 结束原因: "stop", "length", etc.
    usage: Dict[str, int] = field(default_factory=dict)  # token 使用量
    raw_response: Any = None    # 原始 API 响应 (调试用)


class RateLimiter:
    """
    基于信号量的并发控制。
    防止组员的多 Agent 同时疯狂发请求把云端节点打挂。
    """

    def __init__(self, max_concurrent: int = 8):
        self._semaphore = threading.BoundedSemaphore(max_concurrent)
        self._max_concurrent = max_concurrent

    def acquire(self) -> bool:
        """获取信号量（阻塞直到有空位）"""
        self._semaphore.acquire()
        return True

    def release(self):
        """释放信号量"""
        self._semaphore.release()

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent


class LLMClient:
    """
    云端 LLM API 客户端。

    特性:
    - 与 OpenAI API 格式兼容，可直接对接 vLLM 的 OpenAI 兼容 Server
    - 内置指数退避重试 (exponential backoff)
    - 信号量并发控制，防止过载
    - Session 连接池复用

    Args:
        base_url: 云端 API 地址，例如 "http://10.0.0.1:8000/v1"
        model_name: 模型名称，例如 "qwen2.5-coder-32b"
        api_key: API 密钥 (vLLM 默认不需要，可设 "EMPTY")
        max_retries: 最大重试次数
        timeout: 每次请求的超时时间 (秒)
        max_concurrent: 最大并发请求数
    """

    def __init__(
        self,
        base_url: str = "http://124.70.101.1:8000/v1",
        model_name: str = "qwen2.5-coder-32b",
        api_key: str = "EMPTY",
        max_retries: int = 3,
        timeout: int = 300,
        max_concurrent: int = 8,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout

        # 并发控制
        self._rate_limiter = RateLimiter(max_concurrent)

        # 创建带重试机制的 Session
        self._session = self._create_session(max_retries)

        logger.info(
            f"LLMClient 初始化完成: base_url={self.base_url}, "
            f"model={self.model_name}, timeout={timeout}s, "
            f"max_concurrent={max_concurrent}, max_retries={max_retries}"
        )

    def _create_session(self, max_retries: int) -> requests.Session:
        """创建带重试策略的 HTTP Session"""
        session = requests.Session()

        # 配置重试策略
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1.0,  # 重试间隔: 1s, 2s, 4s, ...
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    @property
    def chat_endpoint(self) -> str:
        """Chat Completions API 端点"""
        return f"{self.base_url}/chat/completions"

    @property
    def models_endpoint(self) -> str:
        """Models API 端点"""
        return f"{self.base_url}/models"

    def _build_headers(self) -> Dict[str, str]:
        """构建请求头"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """构建请求体"""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        if stop:
            payload["stop"] = stop
        payload.update(kwargs)
        return payload

    def _parse_response(self, response_data: Dict[str, Any]) -> LLMResponse:
        """解析 API 响应为统一的 LLMResponse 对象"""
        try:
            choice = response_data["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason", "unknown")
        except (KeyError, IndexError) as e:
            logger.error(f"解析 API 响应失败: {e}, 原始数据: {response_data}")
            raise ValueError(f"无法解析 API 响应: {e}")

        usage = response_data.get("usage", {})
        model = response_data.get("model", self.model_name)

        return LLMResponse(
            content=content,
            model=model,
            finish_reason=finish_reason,
            usage=usage,
            raw_response=response_data,
        )

    def chat(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        发送单轮对话请求。

        Args:
            user_message: 用户消息内容
            system_message: 系统提示 (可选)
            temperature: 采样温度 (0 为贪婪解码)
            max_tokens: 最大生成 token 数
            stop: 停止词列表
            **kwargs: 其他 OpenAI 兼容参数

        Returns:
            LLMResponse 对象

        Raises:
            requests.RequestException: 网络请求失败且重试耗尽
            ValueError: 响应格式异常
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})

        return self._request_with_retry(messages, temperature, max_tokens, stop, **kwargs)

    def chat_multi_turn(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        发送多轮对话请求（用于 Agent 多轮交互）。

        Args:
            messages: 消息列表，每项含 role 和 content
            temperature: 采样温度
            max_tokens: 最大生成 token 数
            stop: 停止词列表
            **kwargs: 其他参数

        Returns:
            LLMResponse 对象
        """
        return self._request_with_retry(messages, temperature, max_tokens, stop, **kwargs)

    def _request_with_retry(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]],
        **kwargs,
    ) -> LLMResponse:
        """
        带重试和并发控制的请求核心逻辑。
        """
        payload = self._build_payload(messages, temperature, max_tokens, stop, **kwargs)
        headers = self._build_headers()

        # 获取并发槽位
        self._rate_limiter.acquire()
        try:
            last_exception = None
            for attempt in range(3):  # requests 库自身的重试 + 额外手动重试
                try:
                    response = self._session.post(
                        self.chat_endpoint,
                        json=payload,
                        headers=headers,
                        timeout=(30, self.timeout),  # (connect_timeout, read_timeout)
                    )
                    response.raise_for_status()
                    return self._parse_response(response.json())

                except requests.exceptions.ReadTimeout:
                    last_exception = TimeoutError(
                        f"请求超时 (timeout={self.timeout}s)，云端推理可能仍在进行中"
                    )
                    logger.warning(
                        f"请求超时 (attempt {attempt + 1}/3): "
                        f"已等待 {self.timeout}s。"
                        f"如果是长 CoT 推理，可适当增大 max_tokens 或 timeout。"
                    )
                    if attempt < 2:
                        wait_time = 2 ** attempt * 5
                        logger.info(f"等待 {wait_time}s 后重试...")
                        time.sleep(wait_time)

                except requests.exceptions.ConnectionError as e:
                    last_exception = e
                    logger.warning(
                        f"连接错误 (attempt {attempt + 1}/3): {e}"
                    )
                    if attempt < 2:
                        time.sleep(2 ** attempt * 5)

                except requests.exceptions.HTTPError as e:
                    status_code = e.response.status_code if e.response else None
                    logger.error(
                        f"HTTP 错误 (attempt {attempt + 1}/3): "
                        f"status={status_code}, detail={e}"
                    )
                    # 4xx 错误不重试（客户端错误）
                    if status_code and 400 <= status_code < 500:
                        raise
                    last_exception = e
                    if attempt < 2:
                        time.sleep(2 ** attempt * 5)

            # 所有重试耗尽
            raise RuntimeError(
                f"请求失败，已耗尽所有重试机会。最后错误: {last_exception}"
            )
        finally:
            self._rate_limiter.release()

    def check_health(self) -> bool:
        """
        检查云端 API 是否可用。
        通过请求 /v1/models 端点验证。

        Returns:
            True 如果服务正常
        """
        try:
            resp = self._session.get(
                self.models_endpoint,
                headers=self._build_headers(),
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("data", [])
                logger.info(f"云端 API 可用，已加载 {len(models)} 个模型")
                return True
            else:
                logger.warning(f"健康检查返回非 200 状态码: {resp.status_code}")
                return False
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return False

    def close(self):
        """关闭 HTTP Session，释放连接资源"""
        self._session.close()
        logger.info("LLMClient 已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ---------- 便捷函数 ----------

def create_client_from_env() -> LLMClient:
    """
    从环境变量创建 LLMClient。
    支持的环境变量:
    - LLM_BASE_URL: 云端 API 地址
    - LLM_MODEL_NAME: 模型名称
    - LLM_API_KEY: API 密钥
    - LLM_TIMEOUT: 超时时间 (秒)
    - LLM_MAX_CONCURRENT: 最大并发数
    """
    import os

    base_url = os.environ.get("LLM_BASE_URL", "http://124.70.101.1:8000/v1")
    model_name = os.environ.get("LLM_MODEL_NAME", "qwen2.5-coder-32b")
    api_key = os.environ.get("LLM_API_KEY", "EMPTY")
    timeout = int(os.environ.get("LLM_TIMEOUT", "300"))
    max_concurrent = int(os.environ.get("LLM_MAX_CONCURRENT", "8"))

    return LLMClient(
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        timeout=timeout,
        max_concurrent=max_concurrent,
    )
