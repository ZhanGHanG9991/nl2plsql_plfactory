"""
LLM工具模块
提供统一的多provider LLM初始化和管理功能
"""
import random
import time
from typing import Dict, Any, List, Callable, TypeVar
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from langchain_openai import ChatOpenAI
from langchain.chat_models.base import BaseChatModel

T = TypeVar('T')


def init_llm_from_provider(provider: Dict[str, Any], model_config: Dict[str, Any]) -> BaseChatModel:
    """
    根据provider配置初始化LLM
    
    Args:
        provider: provider配置，包含type、model、api_key等信息
            - type: provider类型，如 openai、deepseek、gemini等
            - model: 模型名称
            - api_key: API密钥
            - base_url: (可选) API base URL
        model_config: 模型通用配置，包含temperature、max_tokens等
            - temperature: 温度参数
            - max_tokens: 最大token数
            - top_p: top_p参数
    
    Returns:
        初始化好的LLM实例
    
    Raises:
        ValueError: 当API key未设置时
        NotImplementedError: 当provider类型尚未实现时
    """
    provider_type = provider.get("type", "openai")
    model = provider.get("model")
    api_key = provider.get("api_key")
    base_url = provider.get("base_url", "xxx")
    
    if not api_key:
        raise ValueError(f"API key for {provider_type} is not set")
    
    # 获取通用配置参数
    temperature = model_config.get("temperature", 0.0)
    max_tokens = model_config.get("max_tokens", 8192)
    top_p = model_config.get("top_p", 0.3)
    
    # 根据provider类型初始化不同的LLM
    if provider_type == "openai":
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
    elif provider_type == "deepseek":
        # Deepseek使用OpenAI兼容的API
        base_url = provider.get("base_url", "https://api.deepseek.com")
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
    elif provider_type == "gemini":
        # 预留Gemini支持
        # 使用时需要安装: pip install langchain-google-genai
        # from langchain_google_genai import ChatGoogleGenerativeAI
        # return ChatGoogleGenerativeAI(
        #     model=model,
        #     google_api_key=api_key,
        #     temperature=temperature,
        #     max_output_tokens=max_tokens,
        #     top_p=top_p,
        # )
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
    elif provider_type == "anthropic":
        # 预留Anthropic Claude支持
        # 使用时需要安装: pip install langchain-anthropic
        # from langchain_anthropic import ChatAnthropic
        # return ChatAnthropic(
        #     model=model,
        #     anthropic_api_key=api_key,
        #     temperature=temperature,
        #     max_tokens=max_tokens,
        #     top_p=top_p,
        # )
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
    elif provider_type == "glm":
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
    elif provider_type == "qwen3":
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
    else:
        raise ValueError(f"Unsupported provider type: {provider_type}")


def init_llm_with_random_provider(model_config: Dict[str, Any], 
                                   model_name: str = "unknown",
                                   verbose: bool = True) -> BaseChatModel:
    """
    从配置的多个providers中随机选择一个可用的provider来初始化LLM
    
    Args:
        model_config: 模型配置字典，应包含providers列表和通用配置
            - providers: provider配置列表
            - temperature: 温度参数
            - max_tokens: 最大token数
            - top_p: top_p参数
        model_name: 模型名称，用于日志输出
        verbose: 是否打印详细信息
    
    Returns:
        初始化好的LLM实例
    
    Raises:
        ValueError: 当没有配置providers或没有可用的provider时
    """
    providers = model_config.get("providers", [])
    
    if not providers:
        raise ValueError(f"No providers configured for {model_name}")
    
    # 过滤出api_key有效的providers
    available_providers = [p for p in providers if p.get("api_key")]
    
    if not available_providers:
        raise ValueError(f"No available providers with valid API keys for {model_name}")
    
    # 随机选择一个provider
    selected_provider = random.choice(available_providers)
    
    if verbose:
        print(f"[INFO] {model_name} - Selected provider: {selected_provider.get('type')} - {selected_provider.get('model')}")
    
    # 初始化并返回LLM
    return init_llm_from_provider(selected_provider, model_config)


def get_available_providers(model_config: Dict[str, Any]) -> List[str]:
    """
    获取配置中所有可用的provider类型列表
    
    Args:
        model_config: 模型配置字典
    
    Returns:
        可用的provider类型列表
    """
    providers = model_config.get("providers", [])
    available_providers = [
        p.get("type") 
        for p in providers 
        if p.get("api_key") and p.get("type")
    ]
    return available_providers


def init_llm_with_fallback(model_config: Dict[str, Any], 
                           preferred_provider: str = None,
                           model_name: str = "unknown",
                           verbose: bool = True) -> BaseChatModel:
    """
    使用指定的provider初始化LLM，如果失败则自动fallback到其他可用provider
    
    Args:
        model_config: 模型配置字典
        preferred_provider: 优先使用的provider类型
        model_name: 模型名称，用于日志输出
        verbose: 是否打印详细信息
    
    Returns:
        初始化好的LLM实例
    """
    providers = model_config.get("providers", [])
    available_providers = [p for p in providers if p.get("api_key")]
    
    if not available_providers:
        raise ValueError(f"No available providers with valid API keys for {model_name}")
    
    # 如果指定了优先provider，先尝试使用它
    if preferred_provider:
        for provider in available_providers:
            if provider.get("type") == preferred_provider:
                try:
                    if verbose:
                        print(f"[INFO] {model_name} - Using preferred provider: {provider.get('type')} - {provider.get('model')}")
                    return init_llm_from_provider(provider, model_config)
                except Exception as e:
                    if verbose:
                        print(f"[WARNING] {model_name} - Failed to initialize preferred provider {preferred_provider}: {e}")
                    break
    
    # 如果没有指定优先provider或优先provider失败，随机选择一个
    return init_llm_with_random_provider(model_config, model_name, verbose)


def call_llm_with_timeout(llm_func: Callable[[], T], 
                          timeout: float = 120.0,
                          model_name: str = "LLM") -> T:
    """
    使用超时机制调用LLM函数
    
    Args:
        llm_func: 要执行的LLM调用函数（无参数）
        timeout: 超时时间（秒），默认120秒
        model_name: 模型名称，用于日志输出
    
    Returns:
        LLM调用结果
    
    Raises:
        TimeoutError: 当调用超时时
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(llm_func)
        try:
            result = future.wait(timeout=timeout)
            return result
        except FuturesTimeoutError:
            future.cancel()
            raise TimeoutError(f"{model_name} call timed out after {timeout} seconds")


def call_llm_with_retry(llm_func_factory: Callable[[BaseChatModel], T],
                        model_config: Dict[str, Any],
                        max_retries: int = 3,
                        timeout: float = 120.0,
                        model_name: str = "unknown",
                        verbose: bool = True) -> T:
    """
    使用超时和重试机制调用LLM，超时后自动切换provider重试
    
    Args:
        llm_func_factory: 接受LLM实例并返回调用结果的函数
        model_config: 模型配置字典
        max_retries: 最大重试次数，默认3次
        timeout: 每次调用的超时时间（秒），默认120秒
        model_name: 模型名称，用于日志输出
        verbose: 是否打印详细信息
    
    Returns:
        LLM调用结果
    
    Raises:
        Exception: 当所有重试都失败时，抛出最后一次的异常
    
    Example:
        >>> def my_llm_call(llm):
        >>>     return llm.invoke("Hello, world!")
        >>> 
        >>> result = call_llm_with_retry(
        >>>     llm_func_factory=my_llm_call,
        >>>     model_config=config["generation_model"],
        >>>     max_retries=3,
        >>>     timeout=120.0,
        >>>     model_name="generation_model"
        >>> )
    """
    providers = model_config.get("providers", [])
    available_providers = [p for p in providers if p.get("api_key")]
    
    if not available_providers:
        raise ValueError(f"No available providers with valid API keys for {model_name}")
    
    last_exception = None
    used_providers = set()
    
    for attempt in range(max_retries):
        executor = None
        try:
            # 选择一个未使用过的provider，如果都用过了就随机选
            remaining_providers = [p for p in available_providers 
                                  if p.get("type") not in used_providers]
            
            if not remaining_providers:
                # 所有provider都试过了，重置并继续
                used_providers.clear()
                remaining_providers = available_providers
            
            selected_provider = random.choice(remaining_providers)
            used_providers.add(selected_provider.get("type"))
            
            if verbose:
                print(f"[INFO] {model_name} - Attempt {attempt + 1}/{max_retries} - "
                      f"Using provider: {selected_provider.get('type')} - {selected_provider.get('model')}")
            
            # 初始化LLM
            llm = init_llm_from_provider(selected_provider, model_config)
            
            # 使用超时机制调用
            start_time = time.time()
            
            # 每次都创建新的线程池，确保隔离
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(llm_func_factory, llm)
            
            try:
                result = future.result(timeout=timeout)
                elapsed_time = time.time() - start_time
                
                if verbose:
                    print(f"[SUCCESS] {model_name} - Call completed in {elapsed_time:.2f}s")
                
                return result
                
            except FuturesTimeoutError:
                elapsed_time = time.time() - start_time
                error_msg = f"Call timed out after {elapsed_time:.2f}s (limit: {timeout}s)"
                
                if verbose:
                    print(f"[TIMEOUT] {model_name} - {error_msg}")
                
                # 尝试取消任务（虽然可能无法立即生效）
                future.cancel()
                last_exception = TimeoutError(error_msg)
                
        except Exception as e:
            last_exception = e
            if verbose:
                print(f"[ERROR] {model_name} - Attempt {attempt + 1} failed: {str(e)}")
        
        finally:
            # 确保线程池被正确关闭，即使超时也要强制关闭
            if executor is not None:
                try:
                    # 立即关闭线程池，不等待正在执行的任务
                    executor.shutdown(wait=False)
                    if verbose and last_exception and isinstance(last_exception, TimeoutError):
                        print(f"[INFO] {model_name} - Force shutdown executor for timed out task")
                except Exception as shutdown_error:
                    if verbose:
                        print(f"[WARNING] {model_name} - Error during executor shutdown: {shutdown_error}")
        
        # 如果不是最后一次尝试，等待一小段时间再重试
        if attempt < max_retries - 1:
            wait_time = min(2 ** attempt, 10)  # 指数退避，最多等待10秒
            if verbose:
                print(f"[INFO] {model_name} - Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
    
    # 所有重试都失败
    error_msg = f"All {max_retries} attempts failed for {model_name}"
    if verbose:
        print(f"[FAILED] {error_msg}")
    
    if last_exception:
        raise last_exception
    else:
        raise RuntimeError(error_msg)

