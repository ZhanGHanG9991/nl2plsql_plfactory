"""
实验专用 LLM 调用模块
特点：
1. 只使用单个指定的模型
2. 重试时不切换模型
3. 简化的配置和调用逻辑
"""
import time
from typing import Callable, TypeVar
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from langchain_openai import ChatOpenAI
from langchain.chat_models.base import BaseChatModel

T = TypeVar('T')


def init_llm(model_config: dict, llm_call_config: dict) -> BaseChatModel:
    """
    根据模型配置初始化 LLM
    
    Args:
        model_config: 模型配置，包含 type, model, api_key, base_url
        llm_call_config: LLM 调用配置，包含 temperature, max_tokens, top_p
        
    Returns:
        初始化好的 LLM 实例
        
    Raises:
        ValueError: 当配置不正确时
    """
    provider_type = model_config.get("type", "openai")
    model = model_config.get("model")
    api_key = model_config.get("api_key")
    base_url = model_config.get("base_url")
    
    if not api_key:
        raise ValueError(f"API key for {provider_type} is not set")
    
    if not model:
        raise ValueError(f"Model name is not set")
    
    # 获取调用参数
    temperature = llm_call_config.get("temperature", 0.0)
    max_tokens = llm_call_config.get("max_tokens", 8192)
    top_p = llm_call_config.get("top_p", 0.3)
    
    # 检查是否为推理模型（o-系列模型不支持 temperature 和 top_p）
    is_reasoning_model = model.startswith("o") and ("-mini" in model or "-preview" in model or model.startswith("o1") or model.startswith("o3") or model.startswith("o4"))
    
    # 构建 ChatOpenAI 参数
    llm_params = {
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "max_tokens": max_tokens,
    }
    
    # 推理模型不支持 temperature 和 top_p
    if not is_reasoning_model:
        llm_params["temperature"] = temperature
        llm_params["top_p"] = top_p
    
    # 使用 ChatOpenAI（兼容大多数 OpenAI API 格式的服务）
    return ChatOpenAI(**llm_params)


def call_llm_with_retry(
    llm_func_factory: Callable[[BaseChatModel], T],
    model_config: dict,
    llm_call_config: dict,
    max_retries: int = 3,
    timeout: float = 120.0,
    verbose: bool = True
) -> T:
    """
    调用 LLM 并支持重试（不切换模型）
    
    Args:
        llm_func_factory: 接受 LLM 实例并返回调用结果的函数
        model_config: 模型配置
        llm_call_config: LLM 调用配置
        max_retries: 最大重试次数
        timeout: 每次调用的超时时间（秒）
        verbose: 是否打印详细信息
        
    Returns:
        LLM 调用结果
        
    Raises:
        Exception: 当所有重试都失败时，抛出最后一次的异常
    
    Example:
        >>> def my_llm_call(llm):
        >>>     return llm.invoke(messages)
        >>> 
        >>> result = call_llm_with_retry(
        >>>     llm_func_factory=my_llm_call,
        >>>     model_config=model_cfg,
        >>>     llm_call_config=llm_cfg,
        >>>     max_retries=3
        >>> )
    """
    model_name = model_config.get("model", "unknown")
    last_exception = None
    
    # 初始化 LLM（只初始化一次）
    try:
        llm = init_llm(model_config, llm_call_config)
        if verbose:
            print(f"[INFO] 使用模型: {model_config.get('type')} - {model_name}")
    except Exception as e:
        raise ValueError(f"初始化 LLM 失败: {e}")
    
    for attempt in range(max_retries):
        executor = None
        try:
            if verbose:
                print(f"[INFO] 尝试 {attempt + 1}/{max_retries}...")
            
            start_time = time.time()
            
            # 创建线程池执行调用
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(llm_func_factory, llm)
            
            try:
                result = future.result(timeout=timeout)
                elapsed_time = time.time() - start_time
                
                if verbose:
                    print(f"[成功] 调用完成，耗时 {elapsed_time:.2f}秒")
                
                return result
                
            except FuturesTimeoutError:
                elapsed_time = time.time() - start_time
                error_msg = f"调用超时，耗时 {elapsed_time:.2f}秒 (限制: {timeout}秒)"
                
                if verbose:
                    print(f"[超时] {error_msg}")
                
                # 尝试取消任务
                future.cancel()
                last_exception = TimeoutError(error_msg)
                
        except Exception as e:
            last_exception = e
            if verbose:
                print(f"[错误] 尝试 {attempt + 1} 失败: {str(e)}")
        
        finally:
            # 确保线程池被正确关闭
            if executor is not None:
                try:
                    executor.shutdown(wait=False)
                except Exception as shutdown_error:
                    if verbose:
                        print(f"[警告] 关闭线程池时出错: {shutdown_error}")
        
        # 如果不是最后一次尝试，等待后重试
        if attempt < max_retries - 1:
            wait_time = min(2 ** attempt, 10)  # 指数退避，最多等待10秒
            if verbose:
                print(f"[INFO] 等待 {wait_time} 秒后重试...")
            time.sleep(wait_time)
    
    # 所有重试都失败
    error_msg = f"所有 {max_retries} 次尝试都失败"
    if verbose:
        print(f"[失败] {error_msg}")
    
    if last_exception:
        raise last_exception
    else:
        raise RuntimeError(error_msg)

